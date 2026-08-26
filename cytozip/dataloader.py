#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataloader.py — Fast window-by-window single-cell cytosine loader for deep
learning on single-cell DNA methylation ``.cz`` data.

The core use case: for one chromosome, stream every single cell's cytosine
methylation (``mc`` / ``cov``) in fixed-size genomic windows (bins), so a
training loop can consume ``(n_cells, n_sites_in_window)`` matrices window by
window without materialising the whole genome.

Inputs
------
* ``reference`` — reference ``.cz`` (built by ``czip build_ref`` /
  ``allc2cz`` reference): supplies the coordinate (``pos``) axis and the
  per-chromosome row count. Every per-cell file is row-aligned to this axis.
* ``cells`` — the single-cell data, given as ANY of:

  - a directory containing many per-cell ``*.cz`` files,
  - a single ``catcz``'d ``.cz`` (``chunk_dims=['chrom', 'cell_id', ...]``),
  - a list / comma-separated string of ``.cz`` paths,
  - a cell-table text file (one ``.cz`` path per line).

* ``index`` — optional context index ``.cz`` (e.g. ``*.CGN.cz`` /
  ``*.CHN.cz``) restricting the returned rows to CG / CH sites.
* ``chrom`` — one chromosome name (or a list of them).
* ``binsize`` — genomic window width in bp.

Public API
----------
* :class:`CzWindowLoader` — open the reference, index and cells **once** at
  construction, then call :meth:`CzWindowLoader.iter_windows` (or
  :meth:`CzWindowLoader.load_chrom`) for any chromosome without re-opening
  files. The recommended entry point for a training loop.
* :func:`load_chrom_matrix` — load the whole (optionally context-filtered)
  chromosome into ``(cell_ids, pos, mc, cov)`` once. ``mc`` / ``cov`` are
  ``(n_cells, n_sites)`` arrays. A one-shot wrapper around
  :class:`CzWindowLoader`.
* :func:`CzWindowDataset` — thin :class:`CzWindowLoader` wrapper usable directly
  as a PyTorch ``IterableDataset`` (torch is optional and only imported when
  ``to_torch=True``).

Alignment requirement
---------------------
Per-cell ``.cz`` files must be row-aligned to ``reference`` (one row per
reference cytosine, in reference order) — i.e. reference-less ``mc`` / ``cov``
files produced by ``allc2cz`` with a ``reference=``. A window then maps to a
fixed row-index range shared by every cell, so a positional window is O(window)
rather than requiring a per-cell coordinate search. Cells whose chunk is
missing or has a mismatched length contribute zeros (same policy as
:mod:`cytozip.dmr`).
"""
from __future__ import annotations

import os
import glob
import collections

from .cz import Reader, _make_np_dtype, _find_pos_col, np


Window = collections.namedtuple("Window", ["chrom", "start", "end",
                                           "pos", "mc", "cov"])


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def _resolve_cell_paths(cells, ext=".cz"):
    """Resolve the ``cells`` argument into a list of ``.cz`` paths.

    Accepts a directory (globbed for ``*.cz``), a single ``.cz`` file (which
    may be a ``catcz``'d multi-cell file), a list/tuple of paths, a
    comma-separated string of paths, or a cell-table text file (one ``.cz``
    path per line; the first whitespace/comma column is used).
    """
    def _abspath(p):
        return os.path.abspath(os.path.expanduser(p))

    if isinstance(cells, (list, tuple)):
        return [_abspath(p) for p in cells]

    if not isinstance(cells, str):
        raise TypeError(f"Cannot interpret `cells` argument: {cells!r}")

    path = _abspath(cells)
    # Directory -> glob every .cz inside it.
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, f"*{ext}")))
        if not found:
            raise ValueError(f"No *{ext} files found in directory {path}")
        return found
    # A single .cz file (per-cell or catcz).
    if os.path.isfile(path) and path.endswith(ext):
        return [path]
    # A text cell-table (one .cz path per line).
    if os.path.isfile(path):
        out = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.replace("\t", ",").split(",")[0].strip()
                if not token:
                    continue
                cand = _abspath(token)
                if not os.path.isfile(cand):
                    raise ValueError(
                        f"cell-table entry {token!r} is not an existing file")
                out.append(cand)
        if not out:
            raise ValueError(f"cell table {path} is empty")
        return out
    # Fall back: comma-separated string of paths.
    return [_abspath(p) for p in cells.split(",") if p.strip()]


def _expand_cell_specs(paths):
    """Expand ``.cz`` paths into ``(path, suffix)`` per-cell specs.

    A per-cell ``.cz`` (``chunk_dims=['chrom']``) yields ``(path, ())``; a
    ``catcz``'d file (``chunk_dims=['chrom', 'cell_id', ...]``) yields one spec
    per unique non-chromosome key suffix. Mirrors
    :func:`cytozip.dmr._expand_cell_specs` so a single catted file behaves like
    the equivalent set of per-cell files.
    """
    specs = []
    for p in paths:
        r = Reader(p)
        try:
            if len(r.header["chunk_dims"]) <= 1:
                specs.append((p, ()))
            else:
                seen = set()
                for k in r.chunk_key2offset:
                    suffix = k[1:]
                    if suffix not in seen:
                        seen.add(suffix)
                        specs.append((p, suffix))
        finally:
            r.close()
    return specs


def _spec_cell_id(path, suffix):
    """Human-readable cell id for a ``(path, suffix)`` spec."""
    if suffix:
        return ".".join(str(s) for s in suffix)
    return os.path.basename(path)[:-3] if path.endswith(".cz") \
        else os.path.basename(path)


def resolve_cell_ids(cells):
    """Return the ordered list of cell ids for ``cells`` without reading data.

    Only resolves paths and reads each file's header, so it is cheap — the row
    order of the returned list matches the row order of the ``mc`` / ``cov``
    matrices produced by :func:`load_chrom_matrix` / :func:`iter_windows`.
    """
    paths = _resolve_cell_paths(cells)
    specs = _expand_cell_specs(paths)
    return [_spec_cell_id(p, s) for (p, s) in specs]


# ---------------------------------------------------------------------------
# Per-reader dtype + column helpers
# ---------------------------------------------------------------------------
def _reader_np_dtype(reader):
    """Return (and memoize on the reader) the numpy structured dtype of its records."""
    dt = getattr(reader, "_dl_np_dtype", None)
    if dt is None:
        dt = _make_np_dtype(reader.header["formats"], reader.header["columns"])
        try:
            reader._dl_np_dtype = dt
        except Exception:
            pass
    return dt


def _resolve_col(header_cols, col, default_idx):
    """Resolve a column selector (None / int / name) to a column name."""
    if col is None:
        return header_cols[default_idx]
    if isinstance(col, int):
        return header_cols[col]
    if col not in header_cols:
        raise ValueError(f"column {col!r} not in {header_cols}")
    return col


# ---------------------------------------------------------------------------
# Public: whole-chromosome load
# ---------------------------------------------------------------------------
def load_chrom_matrix(reference, cells, chrom, index=None, mc_col=None,
                      cov_col=None, dtype=np.int32, jobs=1):
    """Load one chromosome across all cells into dense matrices.

    One-shot wrapper around :class:`CzWindowLoader` (opens the reference / cells,
    loads the chromosome, then closes). For repeated access across chromosomes
    or epochs, build a :class:`CzWindowLoader` once instead.

    Parameters
    ----------
    reference, cells, index, mc_col, cov_col, dtype, jobs
        See :class:`CzWindowLoader`.
    chrom : str
        Chromosome name (a single chunk key, e.g. ``'chr1'``).

    Returns
    -------
    (cell_ids, pos, mc, cov)
        ``cell_ids`` (list of str), ``pos`` (1-D int64 coordinates) and the
        ``(n_cells, n_sites)`` ``mc`` / ``cov`` matrices of the chosen ``dtype``.
    """
    if isinstance(chrom, (list, tuple)):
        raise ValueError("load_chrom_matrix handles one chromosome; pass a str")
    with CzWindowLoader(reference, cells, index=index, mc_col=mc_col,
                        cov_col=cov_col, dtype=dtype, jobs=jobs) as loader:
        pos, mc, cov = loader.load_chrom(chrom)
        return loader.cell_ids, pos, mc, cov


# ---------------------------------------------------------------------------
# Chromosome-level prefetch
# ---------------------------------------------------------------------------
def _iter_loaded_chroms(chroms, loader, prefetch):
    """Yield ``loader(chrom)`` for each chrom, optionally with 1-ahead prefetch.

    When ``prefetch`` is True, a single background thread loads the *next*
    chromosome's matrix while the caller consumes the current one's windows,
    overlapping decompression with downstream (e.g. GPU) compute. A single
    worker is used deliberately so only one chromosome loads at a time — the
    module's shared :class:`Reader` cache (file handles) is not safe for
    concurrent loads.
    """
    if not prefetch or len(chroms) <= 1:
        for c in chroms:
            yield loader(c)
        return
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        next_fut = ex.submit(loader, chroms[0])
        for i in range(len(chroms)):
            cur = next_fut.result()
            if i + 1 < len(chroms):
                next_fut = ex.submit(loader, chroms[i + 1])
            yield cur
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _windows_from_arrays(chrom, pos, mc, cov, binsize, min_sites):
    """Yield :class:`Window` slices of a loaded chromosome matrix."""
    if pos.shape[0] == 0:
        return
    bin_ids = pos // binsize
    # Boundaries where the bin id changes (pos is ascending & sorted).
    boundaries = np.flatnonzero(np.diff(bin_ids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [bin_ids.shape[0]]))
    for s, e in zip(starts, ends):
        if e - s < min_sites:
            continue
        b = int(bin_ids[s])
        yield Window(chrom=chrom, start=b * binsize,
                     end=b * binsize + binsize,
                     pos=pos[s:e], mc=mc[:, s:e], cov=cov[:, s:e])


# ---------------------------------------------------------------------------
# Public: stateful loader (opens reference / index / cells once)
# ---------------------------------------------------------------------------
class CzWindowLoader:
    """Reusable loader that opens the reference, index and cells **once**.

    Construct it with the reference, the single-cell data and (optionally) a
    context index; the reference / index readers and every per-cell reader are
    opened at construction and the cell order is resolved from headers. Then
    call :meth:`iter_windows` (or :meth:`load_chrom`) repeatedly for any
    chromosome without re-resolving cells or re-opening files — ideal for a
    deep-learning training loop that revisits chromosomes across epochs.

    Parameters
    ----------
    reference : str
        Reference ``.cz`` (``build_ref`` / ``allc2cz`` output) supplying the
        ``pos`` axis and the per-chromosome row count.
    cells : str or list
        The single-cell data, given as any of: a directory of per-cell
        ``*.cz`` files, a single ``catcz``'d ``.cz``, a list / comma-separated
        string of ``.cz`` paths, or a cell-table text file (one ``.cz`` path
        per line).
    index : str or None, default None
        Optional context index ``.cz`` (e.g. ``*.CGN.cz`` / ``*.CHN.cz``)
        restricting the returned rows to CG / CH sites.
    mc_col, cov_col : None, int or str, default None
        Methylated-count / coverage column selectors. ``None`` uses the first /
        last column of the per-cell header; an ``int`` is a 0-based column
        index; a ``str`` is a column name.
    dtype : numpy dtype, default ``np.int32``
        Output dtype of the ``mc`` / ``cov`` matrices. For raw single-cell
        ``mc`` / ``cov`` (values ≤ 255) pass ``np.uint8`` to cut memory 4× vs
        the default ``int32`` (and speed up allocation / gather).
    jobs : int, default 1
        Number of threads used to (a) open the per-cell files at construction
        and (b) decode per-cell chunks concurrently. Both are I/O / C-bound and
        release the GIL, so threads scale well.
    cache_chroms : int, default 1
        Keep this many whole-chromosome ``(pos, mc, cov)`` matrices in an
        in-memory LRU so a chromosome revisited (across windows, calls or
        epochs) is decompressed only once. ``0`` disables it. Each cached
        chromosome costs about ``2 * n_cells * n_sites * itemsize`` bytes —
        size it against RAM (or set ``0`` when memory is tight).

    Example
    -------
    >>> loader = CzWindowLoader(
    ...     reference='~/Ref/hg38/hg38_with_chrL.allc.cz',
    ...     cells='~/Projects/test_cytozip/benchmark/cz/',
    ...     index='~/Ref/hg38/hg38_with_chrL.CGN.cz', jobs=8)
    >>> loader.cell_ids                       # row order (no data read)
    ['cellA', 'cellB', ...]
    >>> for w in loader.iter_windows('chr1', binsize=100_000):
    ...     w.mc, w.cov                        # (n_cells, sites)
    >>> loader.close()
    """

    def __init__(self, reference, cells, index=None, mc_col=None, cov_col=None,
                 dtype=np.int32, jobs=1, cache_chroms=1):
        """Open the reference / index / cells and resolve cell order (see the
        class docstring for the parameters)."""
        self.dtype = dtype
        self.jobs = int(jobs)
        self._specs = _expand_cell_specs(_resolve_cell_paths(cells))
        if not self._specs:
            raise ValueError("no cells resolved from `cells` argument")
        self.cell_ids = [_spec_cell_id(p, s) for (p, s) in self._specs]
        # Instance-owned reader cache (independent of the module-global one)
        # so close() has a well-defined lifetime.
        self._readers = {}
        self._ref = self._reader(os.path.abspath(os.path.expanduser(reference)))
        self._ref.advise_sequential()
        self._ref_dt = _reader_np_dtype(self._ref)
        self._ref_pos_name = self._ref.header["columns"][_find_pos_col(self._ref)]
        self._ix = None
        if index is not None:
            self._ix = self._reader(os.path.abspath(os.path.expanduser(index)))
        # Pre-open every unique per-cell file. Opening is I/O-bound (mmap +
        # header + chunk-index read), so run it on a thread pool — a loader over
        # thousands of cells then builds in seconds instead of minutes. Readers
        # are reused for every chromosome afterwards.
        cell_paths = list(dict.fromkeys(p for p, _ in self._specs))
        if self.jobs > 1 and len(cell_paths) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                for p, r in zip(cell_paths, ex.map(Reader, cell_paths)):
                    self._readers.setdefault(p, r)
        else:
            for p in cell_paths:
                self._reader(p)
        cols = self._reader(self._specs[0][0]).header["columns"]
        self.mc_col = _resolve_col(cols, mc_col, 0)
        self.cov_col = _resolve_col(cols, cov_col, -1)
        self._sel_cache = {}   # chrom dim -> 0-based sel_ids (or None)
        self._ref_cache = {}   # chrom dim -> (pos, n_sites): reference axis reuse
        self._mat_cache = collections.OrderedDict()  # chrom dim -> (pos, mc, cov)
        self._mat_cache_size = int(cache_chroms)

    @property
    def chroms(self):
        """List of chromosome names in the reference (in on-disk order)."""
        return [k[0] for k in self._ref.chunk_key2offset]

    def _reader(self, path):
        """Return the cached :class:`Reader` for ``path``, opening it on first use."""
        r = self._readers.get(path)
        if r is None:
            r = Reader(path)
            self._readers[path] = r
        return r

    def _sel_ids(self, dim):
        """0-based context-index row ids for a chrom, cached; None if no index."""
        if self._ix is None:
            return None
        hit = self._sel_cache.get(dim)
        if hit is not None:
            return hit
        sel = np.empty(0, dtype=np.int64)
        if dim in self._ix.chunk_key2offset:
            ids = self._ix.get_ids_from_index(dim)
            if ids.ndim == 1:
                sel = ids.astype(np.int64, copy=False) - 1
        self._sel_cache[dim] = sel
        return sel

    def _load_one_cell(self, spec, dim, n_sites, sel_ids):
        """Return one cell's ``(mc_row, cov_row)`` for chunk ``dim``, gathered to
        ``sel_ids`` (context filter) when given. A missing or length-mismatched
        chunk yields zero rows."""
        path, suffix = spec
        nsel = n_sites if sel_ids is None else sel_ids.shape[0]
        r = self._reader(path)
        key = (dim[0],) + suffix if suffix else dim
        if key not in r.chunk_key2offset:
            return np.zeros(nsel, self.dtype), np.zeros(nsel, self.dtype)
        raw = r.fetch_chunk_bytes(key)
        if not raw:
            return np.zeros(nsel, self.dtype), np.zeros(nsel, self.dtype)
        arr = np.frombuffer(raw, dtype=_reader_np_dtype(r))
        if arr.shape[0] != n_sites:
            r.release_chunk(key)
            return np.zeros(nsel, self.dtype), np.zeros(nsel, self.dtype)
        mc_full = arr[self.mc_col]
        cov_full = arr[self.cov_col]
        if sel_ids is not None:
            mc_row = mc_full[sel_ids].astype(self.dtype, copy=False)
            cov_row = cov_full[sel_ids].astype(self.dtype, copy=False)
        else:
            mc_row = mc_full.astype(self.dtype, copy=False)
            cov_row = cov_full.astype(self.dtype, copy=False)
        r.release_chunk(key)
        return mc_row, cov_row

    def _chrom_axis(self, dim):
        """Return ``(pos, n_sites, sel_ids)`` for a chrom, caching the (small)
        reference axis so repeated loads never re-decompress the reference chunk.
        """
        sel_ids = self._sel_ids(dim)
        hit = self._ref_cache.get(dim)
        if hit is not None:
            pos, n_sites = hit
            return pos, n_sites, sel_ids
        ref_arr = np.frombuffer(self._ref.fetch_chunk_bytes(dim),
                                dtype=self._ref_dt)
        pos = ref_arr[self._ref_pos_name].astype(np.int64, copy=True)
        n_sites = ref_arr.shape[0]
        self._ref.release_chunk(dim)
        if sel_ids is not None:
            pos = pos[sel_ids]
        self._ref_cache[dim] = (pos, n_sites)
        return pos, n_sites, sel_ids

    def load_chrom(self, chrom):
        """Load one chromosome across all cells → ``(pos, mc, cov)``.

        ``pos`` is 1-D (length ``n_sites`` after any context filter); ``mc`` /
        ``cov`` are ``(n_cells, n_sites)`` in :attr:`cell_ids` row order. With
        ``cache_chroms > 0`` the same chromosome is decompressed once and the
        cached arrays are returned as-is on later calls (treat them read-only).
        """
        dim = (chrom,)
        if dim not in self._ref.chunk_key2offset:
            raise ValueError(f"chromosome {chrom!r} not in reference")
        cached = self._mat_cache.get(dim)
        if cached is not None:
            self._mat_cache.move_to_end(dim)
            return cached
        pos, n_sites, sel_ids = self._chrom_axis(dim)
        n_cells = len(self._specs)
        nsel = pos.shape[0]
        mc = np.zeros((n_cells, nsel), dtype=self.dtype)
        cov = np.zeros((n_cells, nsel), dtype=self.dtype)

        def _fill(i):
            mc[i], cov[i] = self._load_one_cell(
                self._specs[i], dim, n_sites, sel_ids)

        if self.jobs > 1 and n_cells > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                list(ex.map(_fill, range(n_cells)))
        else:
            for i in range(n_cells):
                _fill(i)
        if self._mat_cache_size > 0:
            self._mat_cache[dim] = (pos, mc, cov)
            while len(self._mat_cache) > self._mat_cache_size:
                self._mat_cache.popitem(last=False)
        return pos, mc, cov

    def iter_windows(self, chrom, binsize, min_sites=1, prefetch=False):
        """Yield :class:`Window` tuples for each non-empty ``binsize`` window.

        Parameters
        ----------
        chrom : str or list
            One chromosome name, or a list/tuple of them streamed in turn.
        binsize : int
            Window width in bp; a site's bin id is ``pos // binsize``.
        min_sites : int, default 1
            Skip windows with fewer than this many sites.
        prefetch : bool, default False
            With a list of chromosomes, decompress the next chromosome on a
            background thread while the current one's windows are consumed.
            No effect for a single chromosome.

        Row order of ``mc`` / ``cov`` follows :attr:`cell_ids`.
        """
        binsize = int(binsize)
        if binsize <= 0:
            raise ValueError("binsize must be a positive integer")
        chroms = [chrom] if isinstance(chrom, str) else list(chrom)

        def _load(c):
            pos, mc, cov = self.load_chrom(c)
            return c, pos, mc, cov

        for c, pos, mc, cov in _iter_loaded_chroms(chroms, _load, prefetch):
            yield from _windows_from_arrays(c, pos, mc, cov, binsize, min_sites)

    def close(self):
        """Close every reader opened by this loader."""
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()
        self._ref_cache.clear()
        self._mat_cache.clear()

    def __enter__(self):
        """Return self so the loader can be used as a context manager."""
        return self

    def __exit__(self, *exc):
        """Close the loader on context-manager exit."""
        self.close()


# ---------------------------------------------------------------------------
# Public: iterable dataset wrapper (optional torch)
# ---------------------------------------------------------------------------
class CzWindowDataset:
    """Iterable over :class:`Window` tuples, usable as a torch IterableDataset.

    A thin wrapper around :class:`CzWindowLoader` (which opens the reference /
    index / cells once at construction). Iterating re-streams the requested
    chromosome(s). Set ``to_torch=True`` to yield tensors instead of numpy
    arrays (``torch`` is imported lazily only then). :attr:`cell_ids` gives the
    row order on demand without iterating.

    Example
    -------
    >>> ds = CzWindowDataset(
    ...     reference='~/Ref/hg38/hg38_with_chrL.allc.cz',
    ...     cells='~/Projects/test_cytozip/benchmark/cz/',
    ...     chrom='chr1', binsize=100_000,
    ...     index='~/Ref/hg38/hg38_with_chrL.CGN.cz')
    >>> ds.cell_ids            # row order, no data read
    ['cellA', 'cellB', ...]
    >>> for w in ds:
    ...     # w.mc, w.cov: (n_cells, n_sites_in_window)
    ...     ...
    """

    def __init__(self, reference, cells, chrom, binsize, index=None,
                 mc_col=None, cov_col=None, dtype=np.int32,
                 jobs=1, min_sites=1, prefetch=False, to_torch=False,
                 cache_chroms=1):
        """Build the underlying :class:`CzWindowLoader` and store the windowing
        options (see :class:`CzWindowLoader` and
        :meth:`CzWindowLoader.iter_windows` for the parameters)."""
        self.loader = CzWindowLoader(
            reference, cells, index=index, mc_col=mc_col, cov_col=cov_col,
            dtype=dtype, jobs=jobs, cache_chroms=cache_chroms)
        self.chrom = chrom
        self.binsize = binsize
        self.min_sites = min_sites
        self.prefetch = prefetch
        self.to_torch = to_torch

    @property
    def cell_ids(self):
        """Ordered cell ids (row order of ``mc`` / ``cov``)."""
        return self.loader.cell_ids

    def close(self):
        """Close the underlying loader (releases readers and caches)."""
        self.loader.close()

    def __iter__(self):
        """Stream :class:`Window` tuples for the configured chromosome(s); yields
        torch tensors instead of numpy arrays when ``to_torch`` is set."""
        gen = self.loader.iter_windows(
            self.chrom, self.binsize, min_sites=self.min_sites,
            prefetch=self.prefetch)
        if not self.to_torch:
            yield from gen
            return
        import torch
        for win in gen:
            yield Window(
                chrom=win.chrom, start=win.start, end=win.end,
                pos=torch.from_numpy(np.ascontiguousarray(win.pos)),
                mc=torch.from_numpy(np.ascontiguousarray(win.mc)),
                cov=torch.from_numpy(np.ascontiguousarray(win.cov)))
