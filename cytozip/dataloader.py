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

  :meth:`CzWindowLoader.iter_windows` runs in **low-memory streaming mode**:
  the chromosome is processed in row-segments (at most ``max_sites`` sites)
  and only the ``.cz`` blocks covering each window are decompressed, so the
  whole ``(n_cells, n_sites)`` chromosome matrix is never materialised. Use
  :meth:`CzWindowLoader.load_region` to load a single ``[start, end)`` window,
  or :meth:`CzWindowLoader.load_chrom` for the whole-chromosome matrix.
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

from .cz import (Reader, _make_np_dtype, _find_pos_col,
                 _BLOCK_MAX_LEN, _VO_OFFSET_BITS, np)


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


def _specs_from_reader(path, reader):
    """Cell specs ``[(path, suffix), ...]`` for one already-open Reader.

    A per-cell ``.cz`` (``chunk_dims=['chrom']``) yields one ``(path, ())``
    spec; a ``catcz``'d file (``chunk_dims=['chrom', 'cell_id', ...]``) yields
    one spec per unique non-chromosome key suffix.
    """
    if len(reader.header["chunk_dims"]) <= 1:
        return [(path, ())]
    out, seen = [], set()
    for k in reader.chunk_key2offset:
        suffix = k[1:]
        if suffix not in seen:
            seen.add(suffix)
            out.append((path, suffix))
    return out


def _expand_cell_specs(paths, threads=1):
    """Expand ``.cz`` paths into ``(path, suffix)`` per-cell specs.

    Each unique file is opened once (on a thread pool when ``threads > 1``)
    purely to read its structure, then closed. Mirrors
    :func:`cytozip.dmr._expand_cell_specs` so a single catted file behaves like
    the equivalent set of per-cell files.
    """
    unique = list(dict.fromkeys(paths))
    readers = {}
    if threads and int(threads) > 1 and len(unique) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            for p, r in zip(unique, ex.map(Reader, unique)):
                readers[p] = r
    else:
        for p in unique:
            readers[p] = Reader(p)
    try:
        specs = []
        for p in paths:
            specs.extend(_specs_from_reader(p, readers[p]))
        return specs
    finally:
        for r in readers.values():
            r.close()


def _spec_cell_id(path, suffix):
    """Human-readable cell id for a ``(path, suffix)`` spec."""
    if suffix:
        return ".".join(str(s) for s in suffix)
    return os.path.basename(path)[:-3] if path.endswith(".cz") \
        else os.path.basename(path)


def resolve_cell_ids(cells, threads=1):
    """Return the ordered list of cell ids for ``cells`` without reading data.

    Only resolves paths and reads each file's header, so it is cheap — the row
    order of the returned list matches the row order of the ``mc`` / ``cov``
    matrices produced by :func:`load_chrom_matrix` / :func:`iter_windows`.
    ``threads`` parallelises the (latency-bound) per-file header reads.
    """
    paths = _resolve_cell_paths(cells)
    specs = _expand_cell_specs(paths, threads=threads)
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
                      cov_col=None, dtype=np.int32, threads=1):
    """Load one chromosome across all cells into dense matrices.

    One-shot wrapper around :class:`CzWindowLoader` (opens the reference / cells,
    loads the chromosome, then closes). For repeated access across chromosomes
    or epochs, build a :class:`CzWindowLoader` once instead.

    Parameters
    ----------
    reference, cells, index, mc_col, cov_col, dtype, threads
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
                        cov_col=cov_col, dtype=dtype, threads=threads) as loader:
        pos, mc, cov = loader.load_chrom(chrom)
        return loader.cell_ids, pos, mc, cov


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
    threads : int, default 1
        Number of threads used to (a) open the per-cell files at construction
        and (b) decode per-cell chunks concurrently. Both are I/O / C-bound and
        release the GIL, so threads scale well.

    Example
    -------
    >>> loader = CzWindowLoader(
    ...     reference='~/Ref/hg38/hg38_with_chrL.allc.cz',
    ...     cells='~/Projects/test_cytozip/benchmark/cz/',
    ...     index='~/Ref/hg38/hg38_with_chrL.CGN.cz', threads=8)
    >>> loader.cell_ids                       # row order (no data read)
    ['cellA', 'cellB', ...]
    >>> for w in loader.iter_windows('chr1', binsize=100_000):
    ...     w.mc, w.cov                        # (n_cells, sites)
    >>> loader.close()
    """

    def __init__(self, reference, cells, index=None, mc_col=None, cov_col=None,
                 dtype=np.int32, threads=1):
        """Open the reference / index / cells and resolve cell order (see the
        class docstring for the parameters)."""
        self.dtype = dtype
        self.threads = int(threads)
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
        # Open every unique per-cell file ONCE on a thread pool (opening is
        # I/O / latency-bound: mmap + header + chunk-index read). The same
        # readers are reused both to enumerate cells and for every later read,
        # so files are never opened twice.
        paths = _resolve_cell_paths(cells)
        unique_paths = list(dict.fromkeys(paths))
        if self.threads > 1 and len(unique_paths) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                for p, r in zip(unique_paths, ex.map(Reader, unique_paths)):
                    self._readers.setdefault(p, r)
        else:
            for p in unique_paths:
                self._reader(p)
        # Derive (path, suffix) cell specs from the already-open readers.
        self._specs = []
        for p in paths:
            self._specs.extend(_specs_from_reader(p, self._readers[p]))
        if not self._specs:
            raise ValueError("no cells resolved from `cells` argument")
        self.cell_ids = [_spec_cell_id(p, s) for (p, s) in self._specs]
        cols = self._readers[self._specs[0][0]].header["columns"]
        self.mc_col = _resolve_col(cols, mc_col, 0)
        self.cov_col = _resolve_col(cols, cov_col, -1)
        self._sel_cache = {}   # chrom dim -> 0-based sel_ids (or None)
        self._ref_cache = {}   # chrom dim -> (pos, n_sites): reference axis reuse

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
        n_sites = ref_arr.shape[0]
        pos_col = ref_arr[self._ref_pos_name]
        # Gather the context-filtered positions BEFORE the int64 cast so we
        # never materialise a full-chromosome int64 pos array (the reference
        # holds every C; the CG/CH subset is a small fraction).
        if sel_ids is not None:
            pos = pos_col[sel_ids].astype(np.int64, copy=False)
        else:
            pos = pos_col.astype(np.int64, copy=True)
        self._ref.release_chunk(dim)
        self._ref_cache[dim] = (pos, n_sites)
        return pos, n_sites, sel_ids

    def _read_one_cell_range(self, spec, dim, abs0, abs1, sel_local, width,
                             n_sites):
        """Return one cell's ``(mc_row, cov_row)`` for the reference row range
        ``[abs0, abs1)`` of chunk ``dim``, decompressing **only the blocks that
        cover that range** (not the whole chromosome). When ``sel_local`` is
        given the rows are further gathered to those in-range positions (context
        filter). A missing / length-mismatched chunk yields zero rows.
        """
        path, suffix = spec
        r = self._reader(path)
        key = (dim[0],) + suffix if suffix else dim
        zero = (np.zeros(width, self.dtype), np.zeros(width, self.dtype))
        if key not in r.chunk_key2offset:
            return zero
        np_dt = _reader_np_dtype(r)
        unit = np_dt.itemsize
        if not r._load_chunk(r.chunk_key2offset[key], jump=False):
            return zero
        if r._chunk_data_len // unit != n_sites:
            return zero  # not row-aligned to the reference -> contribute zeros
        if getattr(r, "_delta_cols", ()):
            # Delta blocks are record-aligned (different geometry); the
            # byte-range slice below assumes contiguous packing, so fall back
            # to a whole-chunk decode for the rare delta-encoded cell.
            arr = np.frombuffer(r.fetch_chunk_bytes(key), dtype=np_dt)[abs0:abs1]
        else:
            # Non-delta blocks hold exactly _BLOCK_MAX_LEN decompressed bytes
            # each (except the last), so record r starts at byte r*unit and
            # block b starts at byte b*_BLOCK_MAX_LEN in the decompressed stream.
            byte0 = abs0 * unit
            byte1 = abs1 * unit
            blk0 = byte0 // _BLOCK_MAX_LEN
            blk1 = (byte1 - 1) // _BLOCK_MAX_LEN
            vos = r._chunk_block_1st_record_virtual_offsets
            if blk1 == blk0:
                r._load_block(start_offset=vos[blk0] >> _VO_OFFSET_BITS)
                buf = r._buffer
            else:
                parts = []
                for b in range(blk0, blk1 + 1):
                    r._load_block(start_offset=vos[b] >> _VO_OFFSET_BITS)
                    parts.append(r._buffer)
                buf = b"".join(parts)
            local0 = byte0 - blk0 * _BLOCK_MAX_LEN
            local1 = byte1 - blk0 * _BLOCK_MAX_LEN
            arr = np.frombuffer(buf[local0:local1], dtype=np_dt)
        mc_full = arr[self.mc_col]
        cov_full = arr[self.cov_col]
        if sel_local is not None:
            return (mc_full[sel_local].astype(self.dtype, copy=False),
                    cov_full[sel_local].astype(self.dtype, copy=False))
        return (mc_full.astype(self.dtype, copy=False),
                cov_full.astype(self.dtype, copy=False))

    def _read_all_cells_range(self, dim, abs0, abs1, sel_local, width, n_sites):
        """Read reference row range ``[abs0, abs1)`` (gathered to ``sel_local``
        when given) across all cells into ``(n_cells, width)`` mc / cov."""
        n_cells = len(self._specs)
        mc = np.zeros((n_cells, width), dtype=self.dtype)
        cov = np.zeros((n_cells, width), dtype=self.dtype)

        def _fill(i):
            mc[i], cov[i] = self._read_one_cell_range(
                self._specs[i], dim, abs0, abs1, sel_local, width, n_sites)

        if self.threads > 1 and n_cells > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                list(ex.map(_fill, range(n_cells)))
        else:
            for i in range(n_cells):
                _fill(i)
        return mc, cov

    def _abs_range_for(self, sel_ids, fs, fe):
        """Absolute reference row range ``[abs0, abs1)`` + in-range gather for a
        filtered-index slice ``[fs, fe)``. Returns ``(abs0, abs1, sel_local)``
        (``sel_local`` is ``None`` when no context filter is active)."""
        if sel_ids is None:
            return fs, fe, None
        abs0 = int(sel_ids[fs])
        abs1 = int(sel_ids[fe - 1]) + 1
        sel_local = (sel_ids[fs:fe] - abs0).astype(np.int64, copy=False)
        return abs0, abs1, sel_local

    def load_region(self, chrom, start, end):
        """Load a single genomic window ``[start, end)`` across all cells,
        reading **only the blocks covering that window** — never the whole
        chromosome.

        Returns ``(pos, mc, cov)`` where ``pos`` is 1-D (sites in the window,
        after any context filter) and ``mc`` / ``cov`` are ``(n_cells, n_sites)``
        in :attr:`cell_ids` row order. Peak memory is
        ``~2 * n_cells * n_sites_in_window * itemsize`` instead of the whole
        chromosome.
        """
        dim = (chrom,)
        if dim not in self._ref.chunk_key2offset:
            raise ValueError(f"chromosome {chrom!r} not in reference")
        pos, n_sites, sel_ids = self._chrom_axis(dim)
        lo = int(np.searchsorted(pos, start, side="left"))
        hi = int(np.searchsorted(pos, end, side="left"))
        n_cells = len(self._specs)
        if hi <= lo:
            return (pos[lo:hi], np.zeros((n_cells, 0), self.dtype),
                    np.zeros((n_cells, 0), self.dtype))
        abs0, abs1, sel_local = self._abs_range_for(sel_ids, lo, hi)
        mc, cov = self._read_all_cells_range(
            dim, abs0, abs1, sel_local, hi - lo, n_sites)
        return pos[lo:hi], mc, cov

    def _iter_windows_streaming(self, chrom, binsize, min_sites, max_sites,
                                prefetch=False):
        """Low-memory window stream: process the chromosome in row-segments of
        about ``max_sites`` reference rows, reading only the blocks covering
        each segment (each block decompressed at most once).

        A single bin larger than the cap is *not* split — it becomes its own
        (over-cap) segment, so a window spanning several blocks is always read
        as one contiguous range. Segments are found lazily via ``searchsorted``
        (no per-bin Python loop) so an early ``break`` only reads a segment or
        two. With ``prefetch`` the next segment(s) are read on a background
        thread while the current segment's windows are consumed.
        """
        dim = (chrom,)
        if dim not in self._ref.chunk_key2offset:
            raise ValueError(f"chromosome {chrom!r} not in reference")
        pos, n_sites, sel_ids = self._chrom_axis(dim)
        if pos.shape[0] == 0:
            return
        bin_ids = pos // binsize
        boundaries = np.flatnonzero(np.diff(bin_ids)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [bin_ids.shape[0]]))
        nbins = starts.shape[0]
        # Segment cap in reference rows; auto ≈ one .cz block (aligns reads to a
        # decompression unit) unless the caller overrides via ``max_sites``.
        if max_sites is None:
            unit = _reader_np_dtype(self._reader(self._specs[0][0])).itemsize
            cap = max(1, _BLOCK_MAX_LEN // unit)
        else:
            cap = int(max_sites) if int(max_sites) > 0 else pos.shape[0]
        # Absolute reference-row start/end of each bin (filtered indices map
        # back to absolute rows via sel_ids). Both are ascending -> segment
        # boundaries come from a single searchsorted, not an O(nbins) loop.
        if sel_ids is None:
            bin_start_abs = starts
            bin_end_abs = ends
        else:
            bin_start_abs = sel_ids[starts]
            bin_end_abs = sel_ids[ends - 1] + 1

        def _segments():
            si = 0
            while si < nbins:
                seg_start_abs = int(bin_start_abs[si])
                sj = int(np.searchsorted(bin_end_abs, seg_start_abs + cap,
                                         side="right"))
                if sj <= si:
                    sj = si + 1  # a single over-cap bin stands alone
                seg_fs = int(starts[si])
                seg_fe = int(ends[sj - 1])
                abs0 = seg_start_abs
                abs1 = int(bin_end_abs[sj - 1])
                sel_local = (None if sel_ids is None
                             else (sel_ids[seg_fs:seg_fe] - abs0).astype(
                                 np.int64, copy=False))
                yield (si, sj, seg_fs, seg_fe, abs0, abs1, sel_local)
                si = sj

        def _read(seg):
            _si, _sj, seg_fs, seg_fe, abs0, abs1, sel_local = seg
            return self._read_all_cells_range(
                dim, abs0, abs1, sel_local, seg_fe - seg_fs, n_sites)

        def _emit(seg, mc_seg, cov_seg):
            si, sj, seg_fs, _fe, _a0, _a1, _sl = seg
            for k in range(si, sj):
                fs = int(starts[k])
                fe = int(ends[k])
                if fe - fs < min_sites:
                    continue
                b = int(bin_ids[fs])
                lo = fs - seg_fs
                hi = fe - seg_fs
                yield Window(chrom=chrom, start=b * binsize,
                             end=b * binsize + binsize,
                             pos=pos[fs:fe], mc=mc_seg[:, lo:hi],
                             cov=cov_seg[:, lo:hi])

        # prefetch depth = number of already-read segments buffered ahead.
        depth = (1 if prefetch is True
                 else int(prefetch) if (prefetch and int(prefetch) > 0) else 0)
        if depth == 0:
            for seg in _segments():
                mc_seg, cov_seg = _read(seg)
                yield from _emit(seg, mc_seg, cov_seg)
            return

        # A single background worker reads segments in order into a bounded
        # queue, so reader state stays touched by one thread at a time (safe)
        # while the consumer overlaps window use with the next segment's read.
        import queue as _queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        q = _queue.Queue(maxsize=depth)
        stop = threading.Event()
        err = []
        _STOP = object()

        def _produce():
            try:
                for seg in _segments():
                    if stop.is_set():
                        return
                    try:
                        payload = (seg,) + _read(seg)
                    except BaseException as e:  # surface read errors downstream
                        err.append(e)
                        return
                    while not stop.is_set():
                        try:
                            q.put(payload, timeout=0.25)
                            break
                        except _queue.Full:
                            continue
            finally:
                try:
                    q.put(_STOP, timeout=0.25)
                except _queue.Full:
                    pass

        ex = ThreadPoolExecutor(max_workers=1)
        ex.submit(_produce)
        try:
            while True:
                item = q.get()
                if item is _STOP:
                    break
                seg, mc_seg, cov_seg = item
                yield from _emit(seg, mc_seg, cov_seg)
            if err:
                raise err[0]
        finally:
            # Unblock a producer parked on a full queue, then tear it down.
            stop.set()
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
            ex.shutdown(wait=False)

    def load_chrom(self, chrom):
        """Load one chromosome across all cells → ``(pos, mc, cov)``.

        ``pos`` is 1-D (length ``n_sites`` after any context filter); ``mc`` /
        ``cov`` are ``(n_cells, n_sites)`` in :attr:`cell_ids` row order. This
        materialises the whole chromosome in RAM; for window-by-window training
        use :meth:`iter_windows` (streaming) or :meth:`load_region` instead.
        """
        dim = (chrom,)
        if dim not in self._ref.chunk_key2offset:
            raise ValueError(f"chromosome {chrom!r} not in reference")
        pos, n_sites, sel_ids = self._chrom_axis(dim)
        n_cells = len(self._specs)
        nsel = pos.shape[0]
        mc = np.zeros((n_cells, nsel), dtype=self.dtype)
        cov = np.zeros((n_cells, nsel), dtype=self.dtype)

        def _fill(i):
            mc[i], cov[i] = self._load_one_cell(
                self._specs[i], dim, n_sites, sel_ids)

        if self.threads > 1 and n_cells > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                list(ex.map(_fill, range(n_cells)))
        else:
            for i in range(n_cells):
                _fill(i)
        return pos, mc, cov

    def iter_windows(self, chrom, binsize=50, min_sites=1, max_sites=None,
                     prefetch=False):
        """Yield :class:`Window` tuples for each non-empty ``binsize`` window.

        Windows are produced in **low-memory streaming mode**: the chromosome
        is processed in row-segments, reading only the ``.cz`` blocks covering
        each segment, so the whole-chromosome ``(n_cells, n_sites)`` matrix is
        never materialised and each block is decompressed at most once.

        Parameters
        ----------
        chrom : str or list
            One chromosome name, or a list/tuple of them streamed in turn.
        binsize : int
            Window width in bp; a site's bin id is ``pos // binsize``.
        min_sites : int, default 1
            Skip windows with fewer than this many sites.
        max_sites : int or None, default None
            Internal read-segment size, in reference rows — how many
            *consecutive bins* are grouped into one block-decompress (it is NOT
            the per-window site count; window size is set by ``binsize``).
            Larger segments read more bins per decompress (less overhead, more
            transient memory ``~n_cells * max_sites``). ``None`` (default)
            auto-sizes it to about one ``.cz`` block; pass an int only to tune
            memory / overhead. A single bin larger than this is read whole
            (never split).
        prefetch : bool or int, default False
            Read the next segment(s) on a background thread while the current
            segment's windows are consumed, overlapping decompression with the
            training step. ``True`` prefetches one segment ahead; an int ``N``
            buffers up to ``N`` already-read segments (memory ``~N * n_cells *
            max_sites``). A single background reader is used, so per-cell reader
            access stays serialized / thread-safe.

            When to enable it:

            * **Enable** (``prefetch=True``) in a real training loop where each
              step does meaningful compute (e.g. a GPU forward/backward per
              window batch): the next segment decompresses in the background
              while the model computes, hiding read latency behind compute.
              This is the main win with ``DataLoader(num_workers=0)``, where
              prefetch is the only overlap mechanism.
            * **Leave off** (default) for a pure data scan / preprocessing /
              benchmark with negligible per-window work — there is nothing to
              overlap, so it only adds queue/thread overhead and buffers extra
              segments in RAM.

            Keep the depth small: ``True`` (1 segment) is usually enough; raise
            to ``2`` only if profiling shows reads are not fully hidden and you
            have the RAM. Note: with a single ``catcz``'d multi-cell file (all
            cells share one Reader), ``threads > 1`` reads are not thread-safe
            regardless of this flag — prefetch does not change that.

        Row order of ``mc`` / ``cov`` follows :attr:`cell_ids`.
        """
        binsize = int(binsize)
        if binsize <= 0:
            raise ValueError("binsize must be a positive integer")
        chroms = [chrom] if isinstance(chrom, str) else list(chrom)
        for c in chroms:
            yield from self._iter_windows_streaming(
                c, binsize, min_sites, max_sites, prefetch)

    def close(self):
        """Close every reader opened by this loader."""
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()
        self._ref_cache.clear()

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
                 threads=1, min_sites=1, to_torch=False, max_sites=None,
                 prefetch=False):
        """Build the underlying :class:`CzWindowLoader` and store the windowing
        options (see :class:`CzWindowLoader` and
        :meth:`CzWindowLoader.iter_windows` for the parameters).

        Set ``prefetch=True`` for a real training loop (overlaps the next
        segment's decompression with the model step); leave it off for a plain
        data scan — see :meth:`CzWindowLoader.iter_windows` for the full
        guidance.
        """
        self.loader = CzWindowLoader(
            reference, cells, index=index, mc_col=mc_col, cov_col=cov_col,
            dtype=dtype, threads=threads)
        self.chrom = chrom
        self.binsize = binsize
        self.min_sites = min_sites
        self.to_torch = to_torch
        self.max_sites = max_sites
        self.prefetch = prefetch

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
            max_sites=self.max_sites, prefetch=self.prefetch)
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
