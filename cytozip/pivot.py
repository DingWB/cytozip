#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pivot.py — Pivot per-cell .cz files into wide TSV matrices.

This module hosts the *non-summing* outputs that previously lived in
:func:`cytozip.merge.merge_cz` under ``formats='fraction'`` /
``formats='fisher'`` (and the now-removed ``'2D'``):

  - :func:`pivot_fraction`: write a TSV where each column is one cell's
    per-site methylation fraction (``mc / cov``).
  - :func:`pivot_fisher`: write a TSV with one-vs-rest Fisher exact-test
    odd-ratio + p-value per cell per site. Useful as a pre-computed
    site-level statistics matrix for downstream filtering / browsing.

These operations are *pivots* (N cells × M sites → wide matrix), not
*merges* (N cells × M sites → 1 aggregate). Splitting them out lets
``merge_cz`` stay a single, fast sum path while pivots get their own,
heavier pipeline that benefits from a vectorised Fisher implementation.

The pipeline mirrors ``merge_cz``:

  1. ``catcz`` all input per-cell .cz into ``output.cat.cz`` (adds a
     cell_id chunk_key).
  2. Split each chrom into batches of ``batch_nblock`` blocks. Run a
     pool of :func:`_pivot_worker` processes (largest batches first) that
     emit one ``chrom.{batch}.txt`` per batch.
  3. Stream the batch shards, in order, straight into the final TSV
     (text copied verbatim, no pandas re-parse), optionally prefixing
     chrom/start/pos/strand/context columns from a reference .cz, then
     bgzip + tabix it (via pysam; no ``bgzip``/``tabix`` executables
     needed).

Values are written exactly as formatted by the workers (``'%.3g'``):
e.g. ``0``, ``0.5``, ``1e+03``, and ``nan`` for undefined Fisher odds
ratios. (Before this streaming stitch the shards were re-parsed and
re-written by pandas, which printed ``0.0`` / ``1000.0`` and left
undefined values empty; the numbers are the same.)

Speedups compared to the legacy in-merge implementation:

* ``_fisher_worker_fast`` replaces ``df.apply(..., axis=1)`` (which
  reconstructed a Series per row) with vectorised table construction and
  calls ``fast_fisher.fast_fisher_cython.test1t`` once per *distinct*
  2x2 table (``np.unique``), which for sparse single-cell counts is a
  small fraction of sites x cells.
* The mc/cov accumulator uses ``uint32`` (vs. legacy ``int64``) and
  reads block buffers as a 1D ``frombuffer`` view, avoiding the
  structured-dtype column-split copies the legacy worker performed.

@author: DingWB
"""
import os
import shutil
import struct
import itertools
from loguru import logger

from .cz import (Reader, Writer, _BLOCK_MAX_LEN, _NP_FMT_MAP,
                 _unique_int_rows, _make_np_dtype, _build_record_dtype, np, pd)
from .merge import (_bg_rmtree, _bgzip_tabix, _read_block_batch,
                    _catcz_layout, _batch_nblock)
from ._mem import (WORKER_BASE_BYTES, cap_workers, memory_budget, make_pool,
                   bounded_unordered, fmt_bytes, PerProcessReader)

# Per-worker bytes per (cell, site) of a pivot batch: two int64 columns plus
# the formatted object strings / fisher result columns and the to_csv text.
_PIVOT_BYTES_PER_CELL_SITE = {'fraction': 40, 'fisher': 72}

# Per-process reader of the catcz'd input (see PerProcessReader): the parent
# parses the CZIX index once and workers only ever see their chrom's chunks.
_PIVOT_READERS = PerProcessReader()
_pivot_pool_init = _PIVOT_READERS.init


def _pivot_worker_star(args):
    return _pivot_worker(*args)


# ==========================================================
def _fisher_worker_fast(df):
    """Vectorised one-vs-rest Fisher exact test.

    Replaces the legacy implementation that called
    ``df.apply(cal_fisher_or_p, axis=1)`` for every cell column. That
    pattern reconstructs a pandas ``Series`` per row, dominating cost
    on the >10M-row matrices typical for whole-genome single-cell
    methylation pivots.

    The new path:

    * Promotes ``df`` (mc/cov per cell) to two ``(n_sites, n_cells)``
      ``int64`` ndarrays.
    * Computes row-wise mc/cov sums once.
    * For every (site, cell) derives ``a, b, c, d`` (cell mc, cell
      unmethylated, rest mc, rest unmethylated) by vectorised array
      arithmetic, de-duplicates the tables with ``np.unique`` and calls
      ``fast_fisher_cython.test1t`` / ``odds_ratio`` (and the ``%.3g``
      formatting) once per *distinct* table. Single-cell counts are
      small, so this is typically 10-1000x fewer calls than sites x cells.

    Output is a ``pd.DataFrame`` with columns
    ``[sname.odd_ratio, sname.pval]`` for each cell, with site-level
    odd-ratio / p-value pairs ready for downstream filtering.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from fast_fisher.fast_fisher_cython import test1t, odds_ratio
    except ImportError as e:
        raise ImportError(
            "pivot_fisher requires the optional `fast-fisher` package. "
            "Install it with `pip install fast-fisher` (not available on "
            "conda; conda users: `pip install fast-fisher` inside your env)."
        ) from e

    columns = df.columns.tolist()
    snames = [col[:-3] for col in columns if col.endswith('.mc')]
    if not snames:
        return df.iloc[:, :0]

    # (n_sites, n_cells) int64 matrices, in cell order
    mc_arr = df.loc[:, [s + '.mc' for s in snames]].to_numpy(dtype=np.int64)
    cov_arr = df.loc[:, [s + '.cov' for s in snames]].to_numpy(dtype=np.int64)
    # Guard against malformed input (mc > cov) which would put negative
    # cells into the 2x2 tables.
    np.minimum(mc_arr, cov_arr, out=mc_arr)
    uc_arr = cov_arr - mc_arr
    mc_sum = mc_arr.sum(axis=1)
    uc_sum = uc_arr.sum(axis=1)

    n_sites = mc_arr.shape[0]
    n_cells = len(snames)
    # One-vs-rest 2x2 tables for every (site, cell): a/b = cell mc/uc,
    # c/d = rest-of-cells mc/uc. Single-cell counts are tiny, so the number
    # of *distinct* tables is orders of magnitude smaller than
    # n_sites * n_cells; test + format each distinct table once and
    # scatter the strings back.
    a = mc_arr
    b = uc_arr
    c = mc_sum[:, None] - mc_arr
    d = uc_sum[:, None] - uc_arr
    tables = np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=1)
    uniq, inv = _unique_int_rows(tables)
    inv = inv.reshape(n_sites, n_cells)
    n_u = uniq.shape[0]
    odd_u = np.empty(n_u, dtype=object)
    pv_u = np.empty(n_u, dtype=object)
    for k, (ua, ub, uc, ud) in enumerate(uniq.tolist()):
        odd_u[k] = '%.3g' % odds_ratio(ua, ub, uc, ud)
        pv_u[k] = '%.3g' % test1t(ua, ub, uc, ud)
    out_cols = {}
    for j, sname in enumerate(snames):
        out_cols[sname + '.odd_ratio'] = odd_u[inv[:, j]]
        out_cols[sname + '.pval'] = pv_u[inv[:, j]]
    return pd.DataFrame(out_cols)


# ==========================================================
def _pivot_worker(outfile_cat, outdir, chrom, dims, mode,
                  block_idx_start, batch_nblock):
    """Per-chrom, per-batch worker for pivot pipelines.

    Reads ``batch_nblock`` blocks starting at ``block_idx_start`` from
    every cell sharing this chrom in ``outfile_cat``, then either:

    * ``mode='fraction'``: emits a ``chrom.{idx}.txt`` whose columns are
      the per-cell ``mc/cov`` fraction (formatted ``%.3g``).
    * ``mode='fisher'``: emits a ``chrom.{idx}.txt`` whose columns are
      ``[cell.odd_ratio, cell.pval]`` for each cell (one-vs-rest).
    """
    assert mode in ('fraction', 'fisher')
    outname = os.path.join(outdir, chrom + f'.{block_idx_start}.txt')
    dim_offsets = _PIVOT_READERS.offsets(chrom)
    if dims is None:
        if dim_offsets is None:
            raise ValueError("_pivot_worker: dims=None requires a pool "
                             "initialised by _pivot_pool_init")
        dims = list(dim_offsets)
    # Whole per-chrom block-range scan across all cells -> sequential.
    reader1 = (_PIVOT_READERS.get(outfile_cat, dim_offsets)
               if dim_offsets is not None
               else Reader(outfile_cat, mmap_advise="sequential"))
    in_fmts = reader1.fmts
    in_unit_size = sum(struct.calcsize(c) for c in in_fmts)
    in_dt_struct = _build_record_dtype(in_fmts)
    # Vectorised fast path: when both columns are the same numeric
    # dtype (mc/cov 'BB'/'HH'), view the raw block bytes as (n, 2)
    # native dtype directly — no structured-dtype splitting.
    fast_dtype = None
    if len(in_fmts) == 2 and in_fmts[0] == in_fmts[1] \
            and in_fmts[0] in _NP_FMT_MAP:
        fast_dtype = np.dtype(_NP_FMT_MAP[in_fmts[0]])

    cell_mc = []
    cell_cov = []
    for dim in dims:
        buffer = _read_block_batch(reader1, dim, block_idx_start, batch_nblock,
                                   in_unit_size, who='_pivot_worker')
        if fast_dtype is not None:
            arr = np.frombuffer(buffer, dtype=fast_dtype).reshape(-1, 2)
            cell_mc.append(arr[:, 0].astype(np.int64))
            cell_cov.append(arr[:, 1].astype(np.int64))
        else:
            rec = np.frombuffer(buffer, dtype=in_dt_struct)
            cell_mc.append(np.asarray(rec['f0'], dtype=np.int64))
            cell_cov.append(np.asarray(rec['f1'], dtype=np.int64))
    if dim_offsets is None:
        reader1.close()

    snames = [dim[1] for dim in dims]
    if mode == 'fraction':
        # Build columns directly without DataFrame copy: each column is
        # a vectorised mc/cov with NaN-safe divide, formatted ``%.3g``.
        out_cols = {}
        for sname, mc, cov in zip(snames, cell_mc, cell_cov):
            # Format each distinct (mc, cov) pair once, then scatter.
            pairs = np.stack([mc, cov], axis=1)
            uniq, inv = _unique_int_rows(pairs)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(uniq[:, 1] == 0, 0.0, uniq[:, 0] / uniq[:, 1])
            fmt_u = np.array(['%.3g' % v for v in frac.tolist()],
                             dtype=object)
            out_cols[sname] = fmt_u[inv]
        df = pd.DataFrame(out_cols)
    else:  # fisher
        # Build the wide mc/cov frame once and feed to the vectorised
        # fisher kernel.
        cols = {}
        for sname, mc, cov in zip(snames, cell_mc, cell_cov):
            cols[sname + '.mc'] = mc
            cols[sname + '.cov'] = cov
        df = pd.DataFrame(cols)
        df = _fisher_worker_fast(df)
    df.to_csv(outname, sep='\t', index=False)


# ==========================================================
_COPY_BUF = 4 * 1024 * 1024


def _iter_txt_shards(outdir, chrom, batch_nblock):
    """Yield the per-batch ``{chrom}.{idx}.txt`` shards of one chrom in order."""
    block_idx_start = 0
    path = os.path.join(outdir, f"{chrom}.{block_idx_start}.txt")
    while os.path.exists(path):
        yield path
        block_idx_start += batch_nblock
        path = os.path.join(outdir, f"{chrom}.{block_idx_start}.txt")


def _shard_headers(paths):
    """Header columns (first line) of each shard."""
    heads = []
    for path in paths:
        with open(path) as fh:
            heads.append(fh.readline().rstrip('\n').split('\t'))
    return heads


def _ref_frame(ref_reader, chrom, chrom_col):
    """Reference rows of *chrom* as a DataFrame ``[chrom, start, ref cols]``
    (``start = pos - 1``), built from the raw chunk bytes with numpy.

    Formats exactly like the previous per-record ``Reader.fetch`` route:
    bytes columns are decoded and float32 columns are widened to float64
    (``struct`` unpacks ``'f'`` as a Python double).
    """
    cols = ref_reader.header['columns']
    dt = _make_np_dtype(ref_reader.header['formats'], cols)
    arr = np.frombuffer(ref_reader.fetch_chunk_bytes((chrom,)), dtype=dt)
    data = {}
    for c in cols:
        v = arr[c]
        if v.dtype.kind == 'S':
            v = pd.Series(v).str.decode('utf-8')
        elif v.dtype.kind == 'f':
            v = v.astype(np.float64)
        data[c] = v
    df_ref = pd.DataFrame(data)
    df_ref.insert(0, chrom_col, chrom)
    df_ref.insert(1, 'start', df_ref.iloc[:, 1].astype(np.int64) - 1)
    return df_ref


_STITCH_LINES = 20_000


def _stitch_with_ref(fout, shards, df_ref):
    """Write ``ref_row + '\\t' + shard_row`` lines, streaming each shard in
    blocks of ``_STITCH_LINES`` rows (a shard holds ``sites x n_cells``
    fields, so it is never read whole)."""
    off = 0
    for path in shards:
        with open(path) as fh:
            fh.readline()  # header
            while True:
                body = [ln.rstrip('\n') for ln in
                        itertools.islice(fh, _STITCH_LINES)]
                if not body:
                    break
                n = len(body)
                ref_txt = df_ref.iloc[off:off + n].to_csv(
                    None, sep='\t', header=False, index=False)
                ref_lines = ref_txt.split('\n')
                if len(ref_lines) - 1 != n:
                    raise RuntimeError(
                        f"pivot: {path} has more rows than the "
                        f"{len(ref_lines) - 1} reference rows remaining at "
                        f"offset {off}.")
                fout.write(''.join([r + '\t' + d + '\n'
                                    for r, d in zip(ref_lines, body)]))
                off += n


def _stitch_with_ref_reindex(fout, shards, df_ref, usecols, batch_size):
    """Slow path for shards whose columns differ from the expected layout:
    concat with the reference rows and re-order / fill columns by name."""
    off = 0
    for path in shards:
        for df in pd.read_csv(path, sep='\t', chunksize=batch_size,
                              dtype=str, keep_default_na=False):
            n = df.shape[0]
            out = pd.concat([df_ref.iloc[off:off + n].reset_index(drop=True),
                             df.reset_index(drop=True)], axis=1)
            off += n
            out.reindex(columns=usecols).to_csv(fout, sep='\t', index=False,
                                                header=False)


# ==========================================================
def _pivot(mode, indir=None, cz_paths=None, output=None, prefix=None,
           jobs=12, chroms=None, reference=None,
           keep_cat=False, blocks_per_batch=None, temp=False, bgzip=True,
           batch_size=50000, ext='.cz', max_mem=None):
    """Shared pivot pipeline for fraction / fisher modes.

    Memory: a worker holds ``n_cells x sites_per_batch x ~40-70 B`` (the
    wide per-cell columns of one batch); the batch is shrunk and ``jobs``
    capped so ``jobs x that`` fits ``max_mem`` (default 80% of free RAM /
    cgroup headroom). Workers never parse the catcz'd input's chunk index.
    """
    assert mode in ('fraction', 'fisher')
    if output is None:
        if prefix is None:
            output = f'pivot.{mode}.txt'
        else:
            output = f'{prefix}.{mode}.txt'
    logger.info(output)
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return
    if cz_paths is None:
        cz_paths = [file for file in os.listdir(indir) if file.endswith(ext)]
    if indir is None:
        indir = ''
    reader = Reader(os.path.join(indir, cz_paths[0]), mmap_advise="random")
    header = reader.header
    reader.close()
    outfile_cat = output + '.cat.cz'
    writer = Writer(output=outfile_cat, formats=header['formats'],
                    columns=header['columns'], chunk_dims=header['chunk_dims'],
                    message="catcz")
    writer.catcz(input=[os.path.join(indir, p) for p in cz_paths])

    chrom_col = header['chunk_dims'][0]
    chrom_offsets, chrom_nblocks, chroms, filenames = _catcz_layout(
        outfile_cat, chroms, who=f"pivot_{mode}")
    n_cells = len(filenames)
    if blocks_per_batch is None:
        # For fisher mode the kernel is heavy, so prefer many small
        # batches per chrom for better load balancing across jobs. For
        # fraction the kernel is ~free; default like merge_cz.
        if mode == 'fisher':
            blocks_per_batch = max(1, int(np.ceil(jobs * 4 / max(len(chroms), 1))))
        else:
            blocks_per_batch = max(1, int(np.ceil(jobs * 2 / max(len(chroms), 1))))
    max_nblocks = max((chrom_nblocks[c] for c in chroms), default=1)
    batch_nblock, unit_nblock = _batch_nblock(header['formats'], max_nblocks,
                                              blocks_per_batch)
    unit_size = struct.calcsize(''.join(header['formats']))
    # Memory: a batch of ``batch_nblock`` blocks holds ~batch_nblock *
    # _BLOCK_MAX_LEN / unit_size sites for every cell. Shrink the batch (in
    # whole ``unit_nblock`` alignment periods) and cap ``jobs`` so
    # ``jobs x n_cells x sites x bytes`` fits the budget.
    per_cell_site = _PIVOT_BYTES_PER_CELL_SITE[mode]
    sites_per_block = _BLOCK_MAX_LEN / unit_size
    budget = memory_budget(max_mem, 0.8)
    n_workers = max(1, int(jobs))
    if budget is not None:
        per_batch = lambda nb: WORKER_BASE_BYTES + nb * sites_per_block * n_cells * per_cell_site
        while n_workers > 1 and per_batch(unit_nblock) * n_workers > budget:
            n_workers -= 1
        fit_nb = int((budget / n_workers - WORKER_BASE_BYTES)
                     / (sites_per_block * n_cells * per_cell_site))
        fit_nb = max(unit_nblock, (fit_nb // unit_nblock) * unit_nblock)
        if fit_nb < batch_nblock:
            logger.info(
                f"pivot_{mode}: {n_cells} cells -> batch shrunk from "
                f"{batch_nblock} to {fit_nb} blocks "
                f"(~{fmt_bytes(per_batch(fit_nb))} per worker) to fit the "
                f"{fmt_bytes(budget)} budget with jobs={n_workers}")
            batch_nblock = fit_nb
        if n_workers < int(jobs):
            logger.warning(
                f"pivot_{mode}: jobs={jobs} capped to {n_workers} by the "
                f"memory budget {fmt_bytes(budget)} (pass max_mem= to override)")
    n_workers = cap_workers(n_workers, WORKER_BASE_BYTES + int(
        batch_nblock * sites_per_block * n_cells * per_cell_site),
        max_mem=max_mem, what=f"pivot_{mode} worker")

    jobs_args = []
    outdir = output + '.tmp'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for chrom in chroms:
        if not chrom_offsets.get(chrom):
            continue
        block_idx_start = 0
        while block_idx_start < chrom_nblocks[chrom]:
            n_blk = min(batch_nblock, chrom_nblocks[chrom] - block_idx_start)
            # dims=None: the worker takes the chrom's chunk keys / offsets
            # from the fork-shared pool state.
            jobs_args.append((n_blk, (outfile_cat, outdir, chrom, None, mode,
                                      block_idx_start, batch_nblock)))
            block_idx_start += batch_nblock
    # Largest batches first (LPT); shards are named by (chrom, block index)
    # so completion order does not matter.
    jobs_args.sort(key=lambda x: -x[0])
    if n_workers > 1 and len(jobs_args) > 1:
        with make_pool(n_workers, _pivot_pool_init, (chrom_offsets,)) as ex:
            for _ in bounded_unordered(ex, _pivot_worker_star,
                                       (a for _, a in jobs_args),
                                       n_workers * 2):
                pass
    else:
        _pivot_pool_init(chrom_offsets)
        try:
            for _, args in jobs_args:
                _pivot_worker(*args)
        finally:
            _PIVOT_READERS.init({})
    del chrom_offsets

    # Final stitch: stream every chrom's batch shards straight into the
    # output (text kept verbatim, no pandas re-parse / re-format), optionally
    # prefixing the reference (chrom, start, pos, ...) columns.
    if mode == 'fraction':
        columns = filenames
    else:  # fisher
        columns = []
        for sname in filenames:
            columns.extend([sname + '.odd_ratio', sname + '.pval'])
    ref_reader = None
    if reference is not None:
        reference = os.path.abspath(os.path.expanduser(reference))
        ref_reader = Reader(reference)
    logger.info("Pivoting chromosomes..")
    header_written = False
    with open(output, 'w') as fout:
        for chrom in chroms:
            logger.debug(chrom)
            shards = list(_iter_txt_shards(outdir, chrom, batch_nblock))
            if not shards:
                continue
            if ref_reader is None:
                for path in shards:
                    with open(path) as fh:
                        head = fh.readline()
                        if not header_written:
                            fout.write(head)
                            header_written = True
                        shutil.copyfileobj(fh, fout, _COPY_BUF)
                continue
            df_ref = _ref_frame(ref_reader, chrom, chrom_col)
            usecols = df_ref.columns.tolist() + columns
            if not header_written:
                fout.write('\t'.join(map(str, usecols)) + '\n')
                header_written = True
            if _shard_headers(shards) == [columns] * len(shards):
                _stitch_with_ref(fout, shards, df_ref)
            else:  # column layout differs: align by name via pandas
                _stitch_with_ref_reindex(fout, shards, df_ref, usecols,
                                         batch_size)
    if ref_reader is not None:
        ref_reader.close()

    if not keep_cat:
        os.remove(outfile_cat)
    if not temp:
        _bg_rmtree(outdir)
    if bgzip and not output.endswith(ext):
        # Without a reference the table has no coordinate columns to index.
        _bgzip_tabix(output, index=reference is not None)


def pivot_fraction(indir=None, cz_paths=None, output=None, prefix=None,
                   jobs=12, chroms=None, reference=None,
                   keep_cat=False, blocks_per_batch=None, temp=False,
                   bgzip=True, batch_size=50000, ext='.cz', max_mem=None):
    """Pivot per-cell .cz files into a wide per-cell methylation
    fraction TSV.

    Output columns: optional ``[chrom, start, pos, ...]`` from
    ``reference``, followed by one column per input cell containing
    ``mc/cov`` formatted ``%.3g`` (zero-cov rows become ``0``). With
    ``bgzip=True`` the output is bgzipped, and tabix-indexed when a
    ``reference`` supplies coordinate columns.

    The output is a dense ``sites x cells`` text matrix; ``max_mem`` caps
    the worker count / batch size (see :func:`_pivot`). For large cohorts
    prefer the sparse, streamed
    :func:`cytozip.features.cz_to_anndata_sites` (``.h5ad``).
    """
    return _pivot('fraction', indir=indir, cz_paths=cz_paths,
                  output=output, prefix=prefix, jobs=jobs,
                  chroms=chroms, reference=reference,
                  keep_cat=keep_cat, blocks_per_batch=blocks_per_batch,
                  temp=temp, bgzip=bgzip, batch_size=batch_size, ext=ext,
                  max_mem=max_mem)


def pivot_fisher(indir=None, cz_paths=None, output=None, prefix=None,
                 jobs=12, chroms=None, reference=None,
                 keep_cat=False, blocks_per_batch=None, temp=False,
                 bgzip=True, batch_size=50000, ext='.cz', max_mem=None):
    """Pivot per-cell .cz files into a one-vs-rest Fisher exact-test
    TSV.

    Output columns: optional ``[chrom, start, pos, ...]`` from
    ``reference``, followed by ``[cell.odd_ratio, cell.pval]`` per
    input cell.
    """
    return _pivot('fisher', indir=indir, cz_paths=cz_paths,
                  output=output, prefix=prefix, jobs=jobs,
                  chroms=chroms, reference=reference,
                  keep_cat=keep_cat, blocks_per_batch=blocks_per_batch,
                  temp=temp, bgzip=bgzip, batch_size=batch_size, ext=ext,
                  max_mem=max_mem)


if __name__ == "__main__":
    from cytozip import main
    main()
