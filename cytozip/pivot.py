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
import math
import multiprocessing
from loguru import logger

from .cz import (Reader, Writer, _BLOCK_MAX_LEN, _VO_OFFSET_BITS,
                 _VO_OFFSET_MASK, _NP_FMT_MAP, _unique_int_rows,
                 _make_np_dtype, np, pd)
from .merge import _structured_dtype_for, _bg_rmtree, _bgzip_tabix


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
    # Whole per-chrom block-range scan across all cells -> sequential.
    reader1 = Reader(outfile_cat, mmap_advise="sequential")
    in_fmts = reader1.fmts
    in_unit_size = sum(struct.calcsize(c) for c in in_fmts)
    in_dt_struct = _structured_dtype_for(in_fmts)
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
        reader1._load_chunk(reader1.chunk_key2offset[dim], jump=False)
        vos = reader1._chunk_block_1st_record_virtual_offsets
        # See merge_cz_worker for the alignment invariant: batch
        # boundaries must land on a record boundary because records
        # straddle internal block boundaries (block size 2**18-1 is
        # coprime with most unit_sizes). Validate defensively.
        leading_skip = vos[block_idx_start] & _VO_OFFSET_MASK
        if leading_skip != 0:
            reader1.close()
            raise RuntimeError(
                f"_pivot_worker: batch start at block {block_idx_start} "
                f"is not record-aligned (within_block_offset={leading_skip}, "
                f"unit_size={in_unit_size}).")
        block_start_offset = vos[block_idx_start] >> _VO_OFFSET_BITS
        buf_parts = []
        for _ in range(batch_nblock):
            reader1._load_block(start_offset=block_start_offset)
            buf_parts.append(reader1._buffer)
            block_start_offset = None
        buffer = b''.join(buf_parts)
        if len(buffer) % in_unit_size != 0:
            reader1.close()
            raise RuntimeError(
                f"_pivot_worker: batch byte length {len(buffer)} for chrom "
                f"{chrom!r} dim {dim!r} is not a multiple of unit_size "
                f"{in_unit_size}; record alignment broken.")
        if fast_dtype is not None:
            arr = np.frombuffer(buffer, dtype=fast_dtype).reshape(-1, 2)
            cell_mc.append(arr[:, 0].astype(np.int64))
            cell_cov.append(arr[:, 1].astype(np.int64))
        else:
            rec = np.frombuffer(buffer, dtype=in_dt_struct)
            cell_mc.append(np.asarray(rec['f0'], dtype=np.int64))
            cell_cov.append(np.asarray(rec['f1'], dtype=np.int64))
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


def _stitch_with_ref(fout, shards, df_ref):
    """Write ``ref_row + '\\t' + shard_row`` lines, one shard at a time."""
    off = 0
    for path in shards:
        with open(path) as fh:
            fh.readline()  # header
            body = fh.read().split('\n')
        if body and body[-1] == '':
            body.pop()
        n = len(body)
        ref_txt = df_ref.iloc[off:off + n].to_csv(None, sep='\t',
                                                  header=False, index=False)
        ref_lines = ref_txt.split('\n')
        if len(ref_lines) - 1 != n:
            raise RuntimeError(
                f"pivot: {path} has {n} rows but only {len(ref_lines) - 1} "
                f"reference rows remain at offset {off}.")
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
           batch_size=50000, ext='.cz'):
    """Shared pivot pipeline for fraction / fisher modes."""
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

    reader = Reader(outfile_cat, mmap_advise="random")
    chrom_col = reader.header['chunk_dims'][0]
    chunk_info = reader.chunk_info
    reader.close()

    input_chroms = chunk_info[chrom_col].unique().tolist()
    if chroms is not None:
        chroms = os.path.abspath(os.path.expanduser(chroms))
        df_order = pd.read_csv(chroms, sep='\t', header=None, usecols=[0])
        chroms = [c for c in df_order.iloc[:, 0].tolist() if c in input_chroms]
    else:
        chroms = sorted(input_chroms)
    chrom_nblocks = chunk_info.reset_index().loc[
        :, [chrom_col, 'chunk_nblocks']
    ].drop_duplicates().set_index(chrom_col).chunk_nblocks.to_dict()
    unit_size = struct.calcsize(''.join(header['formats']))
    unit_nblock = int(unit_size / math.gcd(unit_size, _BLOCK_MAX_LEN))
    if blocks_per_batch is None:
        # For fisher mode the kernel is heavy, so prefer many small
        # batches per chrom for better load balancing across jobs. For
        # fraction the kernel is ~free; default like merge_cz.
        if mode == 'fisher':
            blocks_per_batch = max(1, int(np.ceil(jobs * 4 / max(len(chroms), 1))))
        else:
            blocks_per_batch = max(1, int(np.ceil(jobs * 2 / max(len(chroms), 1))))
    nunit_perbatch = int(np.ceil(
        (chunk_info.chunk_nblocks.max() / blocks_per_batch) / unit_nblock))
    batch_nblock = nunit_perbatch * unit_nblock

    pool = multiprocessing.Pool(jobs)
    tasks = []
    jobs_args = []
    outdir = output + '.tmp'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for chrom in chroms:
        dims = chunk_info.loc[chunk_info[chrom_col] == chrom].index.tolist()
        if not dims:
            continue
        block_idx_start = 0
        while block_idx_start < chrom_nblocks[chrom]:
            n_blk = min(batch_nblock, chrom_nblocks[chrom] - block_idx_start)
            jobs_args.append((n_blk, (outfile_cat, outdir, chrom, dims, mode,
                                      block_idx_start, batch_nblock)))
            block_idx_start += batch_nblock
    # Largest batches first (LPT); shards are named by (chrom, block index)
    # so completion order does not matter.
    jobs_args.sort(key=lambda x: -x[0])
    for _, args in jobs_args:
        tasks.append(pool.apply_async(_pivot_worker, args))
    for task in tasks:
        task.get()
    pool.close()
    pool.join()

    # Final stitch: stream every chrom's batch shards straight into the
    # output (text kept verbatim, no pandas re-parse / re-format), optionally
    # prefixing the reference (chrom, start, pos, ...) columns.
    filenames = chunk_info.cell_id.unique().tolist()
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
                   bgzip=True, batch_size=50000, ext='.cz'):
    """Pivot per-cell .cz files into a wide per-cell methylation
    fraction TSV.

    Output columns: optional ``[chrom, start, pos, ...]`` from
    ``reference``, followed by one column per input cell containing
    ``mc/cov`` formatted ``%.3g`` (zero-cov rows become ``0``). With
    ``bgzip=True`` the output is bgzipped, and tabix-indexed when a
    ``reference`` supplies coordinate columns.
    """
    return _pivot('fraction', indir=indir, cz_paths=cz_paths,
                  output=output, prefix=prefix, jobs=jobs,
                  chroms=chroms, reference=reference,
                  keep_cat=keep_cat, blocks_per_batch=blocks_per_batch,
                  temp=temp, bgzip=bgzip, batch_size=batch_size, ext=ext)


def pivot_fisher(indir=None, cz_paths=None, output=None, prefix=None,
                 jobs=12, chroms=None, reference=None,
                 keep_cat=False, blocks_per_batch=None, temp=False,
                 bgzip=True, batch_size=50000, ext='.cz'):
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
                  temp=temp, bgzip=bgzip, batch_size=batch_size, ext=ext)


if __name__ == "__main__":
    from cytozip import main
    main()
