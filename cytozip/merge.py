#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge.py — Parallel merging (sum) of multiple per-cell methylation .cz files.

Pipeline stages provided here:
  - :func:`merge_cz`: main entry point. Sums mc/cov across many per-cell
    .cz files into a single aggregate .cz.
  - :func:`merge_cz_worker`: per-chrom, per-batch worker executed by the
    multiprocessing pool.
  - :func:`merge_cell_type`: convenience wrapper that calls :func:`merge_cz`
    once per cell-type grouping defined by a cell-table TSV.

For *pivot* outputs (per-cell fraction matrix, per-cell Fisher-test
matrix), see :mod:`cytozip.pivot`.

@author: DingWB
"""
import os
import struct
import math
from loguru import logger
import multiprocessing
from .cz import (Reader, Writer,
                 _BLOCK_MAX_LEN, _VO_OFFSET_BITS, _VO_OFFSET_MASK,
                 _chunk_magic, _NP_FMT_MAP, _build_record_dtype, np, pd)


# Per-format-char numpy max value (used to clip sums before packing).
_NP_FMT_MAX = {
    'B': 0xFF, 'H': 0xFFFF, 'I': 0xFFFFFFFF, 'L': 0xFFFFFFFF,
    'Q': 0xFFFFFFFFFFFFFFFF, 'b': 0x7F, 'h': 0x7FFF,
    'i': 0x7FFFFFFF, 'l': 0x7FFFFFFF, 'q': 0x7FFFFFFFFFFFFFFF,
}


def _structured_dtype_for(fmts):
    """Build a numpy structured dtype tiling one record (numeric only)."""
    return np.dtype([(f'f{i}', _NP_FMT_MAP[c]) for i, c in enumerate(fmts)])


def _is_merged_cz(path):
    """Return True if ``path`` is a single .cz that is *already* the
    output of a multi-cell ``catcz`` (i.e., its ``chunk_dims`` length is
    >= 2, e.g. ``['chrom', 'cell_id']``).

    Used by :func:`merge_cz` and :func:`cytozip.features.cz_to_anndata`
    to detect whether the user passed a per-cell directory or a single
    pre-catcz'd file as input.
    """
    if not os.path.isfile(path):
        return False
    try:
        r = Reader(path)
        try:
            return len(r.header['chunk_dims']) >= 2
        finally:
            r.close()
    except Exception:
        return False


def _resolve_cz_input(input, ext='.cz'):
    """Resolve the unified ``input`` argument of :func:`merge_cz` /
    :func:`cytozip.features.cz_to_anndata`-style entry points.

    Accepts:
      * a directory path (string) → list all ``*<ext>`` files inside.
      * a single ``.cz`` file path (string) → either a per-cell file
        or a pre-catcz'd file (auto-detected).
      * a list / tuple of ``.cz`` file paths.
      * a comma-separated string of ``.cz`` paths (CLI convenience).

    Returns
    -------
    cz_paths_abs : list of str
        Absolute paths to the per-cell ``.cz`` files. When the input is
        already a pre-catcz'd ``.cz``, this is a single-element list
        pointing at it.
    merged_path : str or None
        Absolute path to the pre-catcz'd ``.cz`` if detected, else
        ``None``. When non-None, the caller can skip the ``catcz``
        step entirely.
    """
    if input is None:
        raise ValueError("merge_cz: 'input' is required")
    if isinstance(input, str) and ',' in input \
            and not os.path.exists(os.path.expanduser(input)):
        input = [s for s in input.split(',') if s]
    if isinstance(input, (list, tuple)):
        paths = [os.path.abspath(os.path.expanduser(p)) for p in input]
    elif isinstance(input, str):
        p = os.path.abspath(os.path.expanduser(input))
        if os.path.isdir(p):
            paths = sorted(os.path.join(p, f) for f in os.listdir(p)
                           if f.endswith(ext))
        elif os.path.isfile(p):
            paths = [p]
        else:
            raise FileNotFoundError(f"merge_cz: input {input!r} not found")
    else:
        raise TypeError(
            f"merge_cz: 'input' must be str or list, got {type(input).__name__}")
    if not paths:
        raise ValueError(
            f"merge_cz: no '*{ext}' files resolved from input={input!r}")
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"merge_cz: input file not found: {p}")
    merged_path = paths[0] if (len(paths) == 1 and _is_merged_cz(paths[0])) else None
    return paths, merged_path


def _iter_shard_paths(outdir, chrom, batch_nblock):
    """Yield ordered shard paths produced by the worker pool.

    Workers write ``{outdir}/{chrom}.{block_idx_start}.cz`` where
    ``block_idx_start`` advances by ``batch_nblock``. Yields paths in
    ascending order while the next file exists; stops at the first
    missing index.
    """
    block_idx_start = 0
    while True:
        p = os.path.join(outdir, f"{chrom}.{block_idx_start}.cz")
        if not os.path.exists(p):
            return
        yield p
        block_idx_start += batch_nblock


def _bg_rmtree(path):
    """Recursively delete ``path`` in a detached background process.

    Removing hundreds of small per-shard files via ``shutil.rmtree`` /
    ``rm -rf`` synchronously costs ~10-20 s on networked filesystems
    and adds nothing to the produced output. The double-fork pattern
    detaches the deleter so the calling Python process returns
    immediately; the intermediate child is reaped here so no zombie
    is left behind.
    """
    logger.info(f"Removing temp dir {path} (in background)")
    if os.fork() == 0:
        # Child: become a session leader so our death does not signal
        # the grandchild.
        try:
            os.setsid()
        except OSError:
            pass
        if os.fork() == 0:
            # Grandchild: do the actual delete and exit.
            try:
                os.system(f"rm -rf {path}")
            finally:
                os._exit(0)
        os._exit(0)
    else:
        # Parent: reap the (immediately exiting) child so it does not
        # become a zombie. The grandchild keeps running detached.
        os.wait()


# ==========================================================
def merge_cz_worker(outfile_cat, outdir, chrom, dims, formats,
                    block_idx_start, batch_nblock, batch_size=5000,
                    level=6, agg='sum'):
    """Worker function for parallel merge of per-cell .cz data.

    Reads a batch of blocks (``batch_nblock`` blocks starting at
    ``block_idx_start``) for every cell/sample sharing the same chrom
    in ``outfile_cat`` and aggregates their values column-wise,
    writing the result as a per-chrom .cz shard
    ``chrom.{block_idx_start}.cz`` in ``outdir``.

    Aggregation
    -----------
    ``agg`` controls how each column is reduced across the N input
    cells/samples (each contributes the same row count per batch).

    * ``'sum'`` (default) — element-wise sum. The fast path for
      BS-seq mc/cov style data: when both columns share an integer
      format that fits in the chosen accumulator dtype, the worker
      uses a 2-column ``np.frombuffer`` view + 1-D ``+=`` and the
      output ``formats`` set the per-column overflow clip.
    * ``'mean'`` — element-wise arithmetic mean. Suitable for array
      beta / M-value style float columns. Output ``formats`` should
      be float (``'f'`` / ``'d'``).
    * list/tuple of strings — per-column aggregation, e.g.
      ``['sum', 'sum']`` (explicit BS-seq) or ``['mean', 'mean']``.

    For non-summing pivot outputs (fraction matrix, Fisher matrix), see
    :mod:`cytozip.pivot`.
    """
    outname = os.path.join(outdir, chrom + f'.{block_idx_start}.cz')
    reader1 = Reader(outfile_cat)
    in_fmts = reader1.fmts
    in_unit_size = sum(struct.calcsize(c) for c in in_fmts)
    n_cols = len(in_fmts)
    # Normalise ``agg`` into a per-column list of strings.
    if isinstance(agg, str):
        agg_list = [agg] * n_cols
    else:
        agg_list = list(agg)
        if len(agg_list) != n_cols:
            raise ValueError(
                f"agg list length {len(agg_list)} != n_cols {n_cols}")
    for a in agg_list:
        if a not in ('sum', 'mean'):
            raise ValueError(
                f"unsupported agg={a!r}; expected 'sum' or 'mean'")
    all_sum = all(a == 'sum' for a in agg_list)
    # ---- Pick decode strategy (chosen once before the per-cell loop):
    # ``fast_dtype != None``: both columns share a homogeneous numeric
    # dtype (the typical 'BB' / 'HH' mc/cov layout). View raw block
    # bytes as a (n, 2) numpy array and accumulate columns directly
    # into 1D sums — eliminates the per-cell (n, 2) int64 allocation
    # and the structured-dtype column-split copies. Only used when
    # ``agg='sum'`` for every column; otherwise we take the general
    # structured-dtype path.
    fast_dtype = None
    accum_dtype = np.int64
    if all_sum and n_cols == 2 and in_fmts[0] == in_fmts[1] \
            and in_fmts[0] in _NP_FMT_MAP:
        fast_dtype = np.dtype(_NP_FMT_MAP[in_fmts[0]])
        # uint32 accumulator is enough for any realistic cell count when
        # the per-record input is <= 16 bits (cov rarely exceeds a few
        # hundred): max sum = 65535 * 65535 = ~4.3e9, exactly fitting
        # uint32. Halves the bandwidth of the inner ``+=`` step versus
        # int64.
        if fast_dtype.itemsize <= 2:
            accum_dtype = np.uint32
    in_dt_struct = None if fast_dtype is not None else _structured_dtype_for(in_fmts)

    if fast_dtype is not None:
        data_mc = None
        data_cov = None
    else:
        # General path: one float64 accumulator per column. Covers
        # 'mean' and arbitrary n_cols / heterogeneous dtypes uniformly.
        # Float64 is wide enough to hold sums of any reasonable input
        # without precision loss.
        accum_cols = [None] * n_cols
    n_cells_used = 0
    for dim in dims:  # each dim is a per-cell .cz chunk for this chrom
        reader1._load_chunk(reader1.chunk_key2offset[dim], jump=False)
        vos = reader1._chunk_block_1st_record_virtual_offsets
        # Records may straddle block boundaries because the catcz writer
        # uses _block_size = _BLOCK_MAX_LEN (which is *not* a multiple of
        # unit_size for typical mc/cov record sizes). The
        # ``batch_nblock = nunit_perbatch * unit_nblock`` choice in
        # ``merge_cz`` guarantees that batch boundaries (multiples of
        # ``unit_nblock``) always land on a record boundary, so the
        # decompressed bytes for ``[block_idx_start, block_idx_start +
        # batch_nblock)`` start with a complete record
        # (``within_block_offset == 0``) and end on a record boundary.
        # Validate that invariant defensively so a future change that
        # breaks it raises clearly here instead of silently corrupting
        # output.
        leading_skip = vos[block_idx_start] & _VO_OFFSET_MASK
        if leading_skip != 0:
            reader1.close()
            raise RuntimeError(
                f"merge_cz_worker: batch start at block {block_idx_start} "
                f"is not record-aligned (within_block_offset={leading_skip}, "
                f"unit_size={in_unit_size}). Check that batch_nblock is a "
                f"multiple of unit_nblock.")
        # Decompress ``batch_nblock`` blocks and concatenate so records
        # straddling internal block boundaries are reassembled before
        # decode.
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
                f"merge_cz_worker: batch byte length {len(buffer)} for chrom "
                f"{chrom!r} dim {dim!r} is not a multiple of unit_size "
                f"{in_unit_size}; record alignment broken.")
        # Decode this cell's batch and accumulate. The fast path
        # views the raw bytes as a (n, 2) homogeneous array and uses
        # 1-D ``+=`` directly. The general path goes through a
        # structured dtype and folds each column into a float64
        # accumulator.
        if fast_dtype is not None:
            arr = np.frombuffer(buffer, dtype=fast_dtype).reshape(-1, 2)
            mc_view = arr[:, 0]
            cov_view = arr[:, 1]
            if data_mc is None:
                data_mc = np.array(mc_view, dtype=accum_dtype, copy=True)
                data_cov = np.array(cov_view, dtype=accum_dtype, copy=True)
            else:
                data_mc += mc_view
                data_cov += cov_view
        else:
            rec = np.frombuffer(buffer, dtype=in_dt_struct)
            for i in range(n_cols):
                col_view = rec[f'f{i}']
                if accum_cols[i] is None:
                    accum_cols[i] = col_view.astype(np.float64, copy=True)
                else:
                    accum_cols[i] += col_view
        n_cells_used += 1

    writer1 = Writer(outname, formats=formats,
                     columns=reader1.header['columns'],
                     chunk_dims=reader1.header['chunk_dims'][:1],
                     message=outfile_cat, level=level)
    out_fmts = ''.join(writer1.formats)
    out_dt = _structured_dtype_for(out_fmts)
    if fast_dtype is not None:
        n = data_mc.shape[0]
        # Fused 1-pass clip: replaces ``data.max()`` scan + conditional
        # ``np.clip`` (up to 2 passes + an extra allocation when overflow
        # happens) with a single in-place ``np.minimum`` per column.
        # Skipped entirely when the accumulator dtype's range already fits
        # inside the output dtype's max (e.g. uint32 accumulator clipped to
        # uint32 output).
        max0, max1 = _NP_FMT_MAX[out_fmts[0]], _NP_FMT_MAX[out_fmts[1]]
        accum_max = np.iinfo(data_mc.dtype).max
        if max0 < accum_max:
            np.minimum(data_mc, max0, out=data_mc)
        if max1 < accum_max:
            np.minimum(data_cov, max1, out=data_cov)
        # Build the entire output buffer in one shot, then hand it to the
        # writer in ``batch_size`` slices. Building once avoids per-batch
        # structured-array allocations.
        out_arr = np.empty(n, dtype=out_dt)
        out_arr['f0'] = data_mc
        out_arr['f1'] = data_cov
    else:
        # General path: finalise per-column according to ``agg_list``,
        # then clip/cast to output dtype.
        n = accum_cols[0].shape[0]
        out_arr = np.empty(n, dtype=out_dt)
        for i, (col, op, ofmt) in enumerate(zip(accum_cols, agg_list, out_fmts)):
            if op == 'mean' and n_cells_used > 1:
                col = col / n_cells_used
            # Clip integer outputs to format max; float outputs pass through.
            if ofmt in _NP_FMT_MAX:
                np.minimum(col, _NP_FMT_MAX[ofmt], out=col)
            out_arr[f'f{i}'] = col
    out_bytes = out_arr.tobytes()
    rec_size = out_dt.itemsize
    chunk_step = batch_size * rec_size
    for s in range(0, len(out_bytes), chunk_step):
        writer1.write_chunk(out_bytes[s:s + chunk_step], [chrom])
    writer1.close()
    reader1.close()


def merge_cz(input=None, class_table=None,
             output=None, prefix=None, jobs=12, formats=['H', 'H'],
             chroms=None, reference=None,
             keep_cat=False, blocks_per_batch=None, temp=False, bgzip=True,
             batch_size=50000, ext='.cz', level=6, agg='sum'):
    """
    Merge multiple per-cell .cz files into one summed .cz. Example:

    cytozip merge_cz -i ./ -O major_type.cz -j 96 \
                          -P ~/Ref/mm10/mm10_ucsc_with_chrL.main.chrom.sizes.txt

    The single ``input`` argument accepts any of:

    1. A directory of per-cell .cz files
       (e.g. ``input='/path/to/dir'``). All ``*<ext>`` files inside are
       picked up, then concatenated via ``catcz`` (a ``cell_id``
       chunk_key is added) before parallel summing.
    2. A list of per-cell .cz file paths
       (e.g. ``input=['a.cz', 'b.cz', ...]``). Same as (1) but with an
       explicit selection. Also accepts a comma-separated string from
       the CLI.
    3. A single pre-catcz'd .cz file
       (e.g. ``input='all_cells.cz'``). The file must already be the
       output of a ``catcz`` (header ``chunk_dims`` length >= 2, e.g.
       ``['chrom', 'cell_id']``). The catcz step is skipped, and the
       user-supplied file is reused as-is and never deleted by
       ``keep_cat``.

    For per-cell pivot outputs (fraction matrix, Fisher one-vs-rest
    matrix), use ``czip pivot_fraction`` / ``czip pivot_fisher`` — see
    :mod:`cytozip.pivot`.

    Parameters
    ----------
    input : str or list of str
        Directory of per-cell ``.cz`` files, single ``.cz`` path
        (per-cell or pre-catcz'd), or list of ``.cz`` paths. See above.
    class_table : path
        If given, multiple outputs will be generated based on the
        snames and class from this class_table; each output has a
        suffix of class name in this table. ``input`` must then be a
        directory.
    output : path
        Output ``.cz`` (or ``.txt`` legacy) path. Defaults to
        ``'merged.cz'`` (or ``f'{prefix}.cz'``).
    prefix : str
        Output filename prefix when ``output`` is None.
    jobs : int
        Number of parallel worker processes.
    formats : list of str
        Per-column struct formats for the *output* .cz (e.g.
        ``['H', 'H']`` for uint16 mc/cov). The legacy ``'fraction'`` /
        ``'fisher'`` / ``'2D'`` modes were moved to
        :mod:`cytozip.pivot` and now raise.
    chroms : path
        Chrom-size file. If provided, output chunks are written in
        the order of this file's first column (rather than sorted).
    reference : path
        Unused for sum mode (kept for API compatibility).
    keep_cat : bool
        Keep the intermediate ``output + '.cat.cz'`` file. Has no
        effect when ``input`` is a pre-catcz'd file (which is always
        kept).
    blocks_per_batch : int
        Number of batches the LARGEST chrom is split into. ``None``
        (default) = ``jobs``. Smaller chroms get 1 batch each via the
        single-shard rename fast-path. Multi-shard chroms are merged
        via raw compressed-block splice (no decompress + re-deflate),
        so oversharding has near-zero overhead.
    temp : bool
        If True, keep the per-shard tmp directory.
    bgzip : bool
        If True (default) and the output filename does not end with
        ``ext``, bgzip + tabix the output.
    batch_size : int
        Worker batch row count when packing the merged record buffer
        into the output writer. Has no effect on the result, only on
        peak memory of the worker.
    ext : str
        Input file extension (default ``'.cz'``).
    level : int
        DEFLATE compression level for output blocks (default 6).
        Drop to 1 for ~2x faster writes at ~12% larger output.
    agg : str or list of str
        How to aggregate each column across input cells/samples.
        ``'sum'`` (default) — element-wise sum, suitable for BS-seq
        mc/cov. ``'mean'`` — element-wise mean across cells, suitable
        for methylation-array beta. May also be a per-column list,
        e.g. ``['mean', 'mean']``. With non-default ``agg`` the
        output ``formats`` should match: typically ``['f','f']``
        (float32) for ``'mean'``.

    Returns
    -------
    None
        Writes ``output`` (and optionally ``output + '.gz'`` if
        ``bgzip=True``).
    """
    if isinstance(formats, str):
        if formats in ('fraction', 'fisher', '2D'):
            raise ValueError(
                f"merge_cz formats={formats!r} was moved to cytozip.pivot."
                " Use cytozip.pivot.pivot_fraction() / pivot_fisher() or"
                " the `czip pivot_fraction` / `czip pivot_fisher` CLI"
                " subcommands instead."
            )
        raise ValueError(
            f"merge_cz expects a list of struct formats, got {formats!r}."
        )
    if class_table is not None:
        # ``class_table`` mode requires a directory of per-cell .cz files
        # so that snames can be matched against the directory listing.
        if not (isinstance(input, str)
                and os.path.isdir(os.path.expanduser(input))):
            raise ValueError(
                "merge_cz: class_table mode requires 'input' to be a "
                "directory of per-cell .cz files."
            )
        indir = os.path.abspath(os.path.expanduser(input))
        df_class = pd.read_csv(class_table, sep='\t', header=None,
                               names=['sname', 'cell_class'])
        snames = [file.replace(ext, '') for file in os.listdir(indir)]
        df_class = df_class.loc[df_class.sname.isin(snames)]
        class_groups = df_class.groupby('cell_class').sname.apply(
            lambda x: x.tolist()).to_dict()
        for key in class_groups:
            logger.info(key)
            cz_paths = [os.path.join(indir, sname + ext)
                        for sname in class_groups[key]]
            merge_cz(input=cz_paths, class_table=None,
                     output=None, prefix=f"{prefix}.{key}", jobs=jobs,
                     formats=formats, chroms=chroms,
                     reference=reference, keep_cat=keep_cat,
                     blocks_per_batch=blocks_per_batch, temp=temp, bgzip=bgzip,
                     batch_size=batch_size, ext=ext, level=level)
        return None
    if output is None:
        output = 'merged.cz' if prefix is None else f'{prefix}.cz'
    logger.info(output)
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return
    # Resolve the unified ``input`` argument into an absolute path list
    # plus an optional pre-catcz'd path.
    cz_paths_abs, merged_path = _resolve_cz_input(input, ext=ext)
    user_supplied_cat = False
    if merged_path is not None:
        # User passed an already-catcz'd file; reuse it directly.
        outfile_cat = merged_path
        user_supplied_cat = True
        reader = Reader(outfile_cat)
        # Synthesize the per-cell-shape header used by downstream
        # writers: keep formats/columns/etc., but trim ``chunk_dims``
        # back to just the chrom axis (first dim) so per-chrom shards
        # and the final output are written as single-key chunks.
        header = dict(reader.header)
        header['chunk_dims'] = list(reader.header['chunk_dims'])[:1]
        reader.close()
        logger.info(f"Detected pre-catcz'd input {outfile_cat}; "
                    f"skipping catcz step.")
    else:
        reader = Reader(cz_paths_abs[0])
        header = reader.header
        reader.close()
        outfile_cat = output + '.cat.cz'
        # cat all .cz files into one .cz file, add a chunk_key to chunk (filename)
        writer = Writer(output=outfile_cat, formats=header['formats'],
                        columns=header['columns'],
                        chunk_dims=header['chunk_dims'],
                        message="catcz")
        writer.catcz(input=cz_paths_abs)

    reader = Reader(outfile_cat)
    chrom_col = reader.header['chunk_dims'][0]
    chunk_info = reader.chunk_info
    reader.close()

    # get chromosomes order
    input_chroms = chunk_info[chrom_col].unique().tolist()
    if not chroms is None:
        chroms = os.path.abspath(os.path.expanduser(chroms))
        df = pd.read_csv(chroms, sep='\t', header=None, usecols=[0])
        chroms = [chrom for chrom in df.iloc[:, 0].tolist() if chrom in input_chroms]
    else:
        chroms = sorted(input_chroms)
    chrom_nblocks = chunk_info.reset_index().loc[:, [chrom_col, 'chunk_nblocks']
                    ].drop_duplicates().set_index(chrom_col).chunk_nblocks.to_dict()
    # how many blocks can be multiplied by self.unit_size
    in_unit_size = sum(struct.calcsize(c) for c in header['formats'])
    unit_nblock = int(in_unit_size / (math.gcd(in_unit_size, _BLOCK_MAX_LEN)))
    # Auto-pick ``blocks_per_batch``. Now that the per-chrom shard merge
    # is a raw byte-copy splice (no decompress + re-deflate), oversharding
    # has near-zero overhead, so we aim to give the *largest* chrom enough
    # shards to keep all workers busy. Small chroms get 1 shard each via
    # the rename fast-path.  ``blocks_per_batch`` here is the number of
    # batches the LARGEST chrom is split into.
    if blocks_per_batch is None:
        blocks_per_batch = max(1, jobs)
    nunit_perbatch = int(np.ceil((chunk_info.chunk_nblocks.max() / blocks_per_batch
                                  ) / unit_nblock))
    batch_nblock = nunit_perbatch * unit_nblock  # how many block for each batch
    pool = multiprocessing.Pool(jobs)
    tasks = []
    outdir = output + '.tmp'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for chrom in chroms:
        dims = chunk_info.loc[chunk_info[chrom_col] == chrom].index.tolist()
        if len(dims) == 0:
            continue
        block_idx_start = 0
        while block_idx_start < chrom_nblocks[chrom]:
            task = pool.apply_async(merge_cz_worker,
                                   (outfile_cat, outdir, chrom, dims, formats,
                                    block_idx_start, batch_nblock, 5000, level,
                                    agg))
            tasks.append(task)
            block_idx_start += batch_nblock
    for task in tasks:
        task.get()
    pool.close()
    pool.join()

    # First, merge per-batch shards into one .cz per chrom.
    # When >1 shard exists for a chrom we splice their compressed blocks
    # raw — no decompress / re-deflate. This relies on three properties of
    # the shards produced by ``merge_cz_worker``:
    #   1. Each shard is a single (chrom,) chunk with sort_col=None.
    #   2. Each shard's chunk_data_len is record-aligned (we feed
    #      ``write_chunk`` with record-aligned slices), so the
    #      ``within_block_offset`` bits stored in each block's virtual
    #      offset remain valid when shard payloads are concatenated.
    #   3. Blocks are independently DEFLATE-compressed (no cross-block
    #      dictionary), so raw byte-copy is sound.
    _COPY_BUF = 4 * 1024 * 1024
    for chrom in chroms:
        outname = os.path.join(outdir, f"{chrom}.cz")
        # Fast path: if only a single batch shard exists for this chrom
        # (the typical case when blocks_per_batch covers the whole
        # chrom), the shard is already a complete .cz file with the
        # exact (chrom,) chunk we want — just rename it.
        single = os.path.join(outdir, f"{chrom}.0.cz")
        second = os.path.join(outdir, f"{chrom}.{batch_nblock}.cz")
        if os.path.exists(single) and not os.path.exists(second):
            os.rename(single, outname)
            continue
        shard_paths = list(_iter_shard_paths(outdir, chrom, batch_nblock))

        writer = Writer(output=outname, formats=formats,
                        columns=header['columns'],
                        chunk_dims=header['chunk_dims'],
                        message=outfile_cat, level=level)
        # Open the chunk manually so we can splice raw block bytes.
        writer._chunk_start_offset = writer._handle.tell()
        writer._handle.write(_chunk_magic)
        writer._handle.write(struct.pack("<Q", 0))  # chunk_size placeholder
        writer._chunk_data_len = 0
        writer._block_1st_record_virtual_offsets = []
        writer._block_first_coords = []
        writer._chunk_dims = [chrom]
        for shard_path in shard_paths:
            reader = Reader(shard_path)
            reader._load_chunk(reader.header['header_size'], jump=False)
            shard_payload_start = reader._chunk_start_offset + 10
            # chunk_size = 10 (magic+size field) + payload (blocks);
            # the chunk tail lives AFTER the chunk_size bytes, so payload
            # size is just chunk_size - 10.
            payload_size = reader._chunk_size - 10
            # Translate per-block virtual offsets to the merged file.
            cur_phys = writer._handle.tell()
            delta_phys = cur_phys - shard_payload_start
            vos_app = writer._block_1st_record_virtual_offsets.append
            for vo in reader._chunk_block_1st_record_virtual_offsets:
                vos_app(((((vo >> _VO_OFFSET_BITS) + delta_phys)) << _VO_OFFSET_BITS) | (vo & _VO_OFFSET_MASK))
            # Raw copy of the compressed-block payload region.
            reader._handle.seek(shard_payload_start)
            remaining = payload_size
            while remaining > 0:
                buf = reader._handle.read(min(remaining, _COPY_BUF))
                if not buf:
                    break
                writer._handle.write(buf)
                remaining -= len(buf)
            writer._chunk_data_len += reader._chunk_data_len
            reader.close()
        # write chunk tail
        writer.close()

    # Second, concatenate per-chrom .cz into the final output.
    writer = Writer(output=output, formats=formats,
                    columns=header['columns'], chunk_dims=header['chunk_dims'],
                    message=(os.path.abspath(os.path.expanduser(reference))
                             if reference else "merged"))
    writer.catcz(input=[f"{outdir}/{chrom}.cz" for chrom in chroms],
                 key_added=None)
    if not keep_cat and not user_supplied_cat:
        os.remove(outfile_cat)
    if not temp:
        # Detached cleanup: deleting hundreds of small per-shard .cz
        # files (~17s for 9-cell × 67-chrom on a network FS) adds
        # nothing to the output, so we fork-and-forget it.
        _bg_rmtree(outdir)
    if bgzip and not output.endswith(ext):
        cmd = f"bgzip {output} && tabix -S 1 -s 1 -b 2 -e 3 -f {output}.gz"
        logger.info(f"Run bgzip, CMD: {cmd}")
        os.system(cmd)


# ==========================================================
# Multi-reference merge: pool per-cell .cz mapped to *different*
# (donor-specific) reference .cz files into a single pseudobulk aligned
# to a common target reference. Unlike ``merge_cz`` (which assumes every
# cell is row-aligned to one shared reference), here each cell declares its
# own reference in a table, so cells are aligned by genomic coordinate
# (chrom, pos) via scatter-add rather than by row index.
# ==========================================================
def _cell_id_from_path(path, ext):
    """Derive a cell id from a .cz path by stripping ``ext`` (or ``.cz``)."""
    name = os.path.basename(path)
    for suf in (ext, '.cz'):
        if suf and name.endswith(suf):
            return name[:-len(suf)]
    return name


def _resolve_cz_table(cz_table, ext='.cz', require=('path', 'reference')):
    """Normalize the per-cell table into a DataFrame.

    Accepts a pandas DataFrame or a path to a TSV **with a header row**. Each
    row describes one cell; recognised columns:

      * ``path``      (required) — per-cell reference-less ``.cz`` path.
      * ``reference`` (required) — that cell's reference ``.cz`` (cells from
        the same donor simply repeat the same reference path).
      * ``index``     (optional) — that cell's 1-D context index ``.cz``
        (used by :func:`cytozip.features.cz_to_anndata_multiref`).
      * ``cell_id``   (optional) — cell label; default = ``basename(path)``
        with ``ext`` (or ``.cz``) stripped.
      * ``cell_type`` (optional) — used by :func:`merge_cell_type_multiref`.

    ``path`` / ``reference`` / ``index`` are expanded (``~``) and made
    absolute; a missing ``cell_id`` is derived from ``path``.
    """
    if isinstance(cz_table, pd.DataFrame):
        df = cz_table.copy()
    elif isinstance(cz_table, str):
        df = pd.read_csv(os.path.abspath(os.path.expanduser(cz_table)),
                         sep='\t')
    else:
        raise TypeError(
            f"cz_table must be a DataFrame or a TSV path, got "
            f"{type(cz_table).__name__}")
    for col in require:
        if col not in df.columns:
            raise ValueError(
                f"cz_table is missing required column {col!r}; "
                f"got columns {list(df.columns)}")

    def _abspath(x):
        return os.path.abspath(os.path.expanduser(str(x)))

    df['path'] = df['path'].map(_abspath)
    if 'reference' in df.columns:
        df['reference'] = df['reference'].map(_abspath)
    if 'index' in df.columns:
        # Blank / NaN means "no per-cell index" -> keep as None.
        df['index'] = df['index'].map(
            lambda x: _abspath(x) if pd.notna(x) and str(x) != '' else None)
    if 'cell_id' not in df.columns:
        df['cell_id'] = df['path'].map(lambda p: _cell_id_from_path(p, ext))
    else:
        df['cell_id'] = df['cell_id'].astype(str)
    return df


def _ctx_key(ctx_s3, policy):
    """Turn trinucleotide contexts (contiguous ``S3`` array) into the
    comparison key for a given ``context_policy``:

    * ``'strict'``   — the exact 3-mer bytes (returned as-is).
    * ``'category'`` — CG vs CH only: ``0`` when the 2nd base is ``G`` (CpG),
      else ``1`` (any CpH); collapses CHG/CHH together.
    """
    if policy == 'strict':
        return ctx_s3
    # 'category': CG (2nd base G) vs CH.
    b = ctx_s3.view('S1').reshape(-1, 3)
    return (b[:, 1] != b'G').astype(np.int8)


def _ref_pos_ctx(reader, ck, policy):
    """Decode only ``pos`` (and, unless ``policy == 'ignore'``, the context
    comparison key) of a reference chunk directly from its raw bytes.

    Returns ``(pos_uint64_contiguous, ctx_key_or_None, n)`` or
    ``(None, None, 0)`` when the chunk is absent/empty. Avoids
    ``chunk2numpy``'s full-record copy and the per-row unicode decode of the
    strand/context string columns.
    """
    if ck not in reader.chunk_key2offset:
        return None, None, 0
    cols = reader.header['columns']
    dt = _build_record_dtype(reader.header['formats'])
    raw = reader.fetch_chunk_bytes(ck)
    if not raw:
        return None, None, 0
    arr = np.frombuffer(raw, dtype=dt)
    pos = np.ascontiguousarray(arr[f'f{cols.index("pos")}'], dtype=np.uint64)
    key = None
    if policy != 'ignore':
        ctx = np.ascontiguousarray(arr[f'f{cols.index("context")}']).view('S3')
        key = _ctx_key(ctx, policy)
    return pos, key, pos.shape[0]


# Shared read-only state for the per-chrom pool workers, populated once per
# worker via ``_merge_mr_init`` so the (potentially large) ``ref_to_cells``
# map is pickled once per worker instead of once per chromosome task.
_MERGE_MR_STATE = {}


def _merge_mr_init(target_reference, ref_to_cells, formats, columns,
                   context_policy, level, outdir):
    """Pool initializer: stash the shared merge state in the worker."""
    _MERGE_MR_STATE.clear()
    _MERGE_MR_STATE.update(
        target_reference=target_reference, ref_to_cells=ref_to_cells,
        formats=list(formats), columns=columns, context_policy=context_policy,
        level=level, outdir=outdir)


def _merge_multiref_chrom_worker(chrom):
    """Aggregate one chromosome across references into a pseudobulk shard.

    For the given chrom, sums mc/cov of every cell onto the target reference
    axis. Cells are grouped by their own reference; each group is first summed
    row-aligned to that reference, then scattered onto the target axis by
    matching genomic position (and, when ``context_policy='match'``, the
    methylation context class). Writes ``{outdir}/{chrom}.cz`` (reference-less
    mc/cov, aligned to the target reference) or returns ``None`` if the chrom
    is absent from target.
    """
    st = _MERGE_MR_STATE
    target_reference = st['target_reference']
    ref_to_cells = st['ref_to_cells']
    formats = st['formats']
    columns = st['columns']
    context_policy = st['context_policy']
    level = st['level']
    outdir = st['outdir']
    ck = (chrom,)

    target = Reader(target_reference)
    try:
        target_pos, target_key, n = _ref_pos_ctx(target, ck, context_policy)
    finally:
        target.close()
    if n == 0:
        return None

    mc_sum = np.zeros(n, dtype=np.int64)
    cov_sum = np.zeros(n, dtype=np.int64)

    for ref_path, cell_paths in ref_to_cells.items():
        dref = Reader(ref_path)
        try:
            donor_pos, donor_key, m = _ref_pos_ctx(dref, ck, context_policy)
        finally:
            dref.close()
        if m == 0:
            continue
        # reference row -> target row by coordinate (and optional context key).
        idx = np.searchsorted(target_pos, donor_pos, side='left')
        in_range = idx < n
        idx_clip = np.where(in_range, idx, 0)
        valid = in_range & (target_pos[idx_clip] == donor_pos)
        if context_policy != 'ignore':
            valid &= target_key[idx_clip] == donor_key
        donor_rows = np.nonzero(valid)[0]
        if donor_rows.size == 0:
            continue
        tgt_idx = idx[donor_rows]
        # Sum these cells row-aligned to their shared reference, on the valid
        # subset only (keeps memory ~ intersection size). frombuffer gives a
        # zero-copy view; the fancy-index gather produces the owned subset,
        # and ``+=`` upcasts the uint16/uint8 counts without a full temp.
        donor_mc = np.zeros(donor_rows.size, dtype=np.int64)
        donor_cov = np.zeros(donor_rows.size, dtype=np.int64)
        for cp in cell_paths:
            cr = Reader(cp)
            try:
                if ck not in cr.chunk_key2offset:
                    continue  # cell has no data on this chrom
                cell_dt = _build_record_dtype(cr.header['formats'])
                raw = cr.fetch_chunk_bytes(ck)
            finally:
                cr.close()
            if not raw:
                continue
            cell_arr = np.frombuffer(raw, dtype=cell_dt)
            if cell_arr.shape[0] != m:
                raise ValueError(
                    f"cell {cp} chrom {chrom}: {cell_arr.shape[0]} rows but its "
                    f"reference {ref_path} has {m}; cell is not aligned to its "
                    f"reference.")
            donor_mc += cell_arr['f0'][donor_rows]
            donor_cov += cell_arr['f1'][donor_rows]
        # Positions are unique within a chrom on both axes, so each target row
        # is hit at most once per reference -> plain indexed += is correct.
        mc_sum[tgt_idx] += donor_mc
        cov_sum[tgt_idx] += donor_cov

    out_fmts = ''.join(formats)
    out_dt = _structured_dtype_for(out_fmts)
    np.minimum(mc_sum, _NP_FMT_MAX[out_fmts[0]], out=mc_sum)
    np.minimum(cov_sum, _NP_FMT_MAX[out_fmts[1]], out=cov_sum)
    out_arr = np.empty(n, dtype=out_dt)
    out_arr['f0'] = mc_sum
    out_arr['f1'] = cov_sum
    out_bytes = out_arr.tobytes()

    outname = os.path.join(outdir, f"{chrom}.cz")
    writer = Writer(outname, formats=formats, columns=columns,
                    chunk_dims=['chrom'], message=target_reference, level=level)
    rec_size = out_dt.itemsize
    step = 50000 * rec_size
    for s in range(0, len(out_bytes), step):
        writer.write_chunk(out_bytes[s:s + step], [chrom])
    writer.close()
    return outname


def merge_cz_multiref(cz_table=None, target_reference=None, output=None,
                      context_policy='strict', formats=['H', 'H'], jobs=12,
                      chroms=None, ext='.cz', level=6, temp=False):
    """Merge per-cell .cz files with *different* references into one
    pseudobulk .cz aligned to a common target reference.

    Unlike :func:`merge_cz`, cells here are NOT all row-aligned to a single
    reference: each cell was mapped to a donor-specific genome and thus
    row-aligns to its *own* reference ``.cz`` (declared per row in
    ``cz_table``; different C positions, possibly different context at the
    same coordinate). Cells are pooled by genomic coordinate: for each chrom,
    every cell's mc/cov is scattered onto the ``target_reference`` axis via
    its own reference, matching on position (and, with
    ``context_policy='match'``, on context).

    Mismatch handling
    -----------------
    * A position present in a cell's reference but absent from the target
      (e.g. a donor-specific SNP creating a C) is dropped.
    * A target position absent from a cell's reference (mutated away in that
      donor) simply receives no coverage from that cell.
    * ``context_policy`` controls how the per-position context is matched
      between a cell's reference and the target (both store the actual
      trinucleotide, e.g. ``CGA`` / ``CAG``):

      - ``'strict'`` (default): require the exact 3-mer to be identical. A
        downstream SNP that changes the +2 base (``CGA`` -> ``CGG``, still a
        CpG) counts as a mismatch and is dropped.
      - ``'category'``: only distinguish CG vs CH (2nd base ``G`` or not), so
        the CpG above stays pooled while a true CG<->CH change is separated.
      - ``'ignore'``: pool purely by coordinate, regardless of context.

    Parameters
    ----------
    cz_table : DataFrame or path
        Per-cell table (DataFrame or headered TSV) with at least ``path``
        (per-cell reference-less ``.cz``) and ``reference`` (that cell's
        reference ``.cz``) columns. See :func:`_resolve_cz_table` for the
        full column set. Cells sharing a ``reference`` value are grouped so
        the reference→target mapping is computed once per reference.
    target_reference : path
        Common reference ``.cz`` (e.g. a standard-genome allC.cz) defining
        the output coordinate axis. The output is reference-less and
        row-aligned to this file.
    output : path
        Output pseudobulk ``.cz`` path.
    context_policy : {'strict', 'category', 'ignore'}
        See *Mismatch handling* above. Default ``'strict'``.
    formats : list of str
        Output per-column struct formats (default ``['H', 'H']``).
    jobs : int
        Number of parallel worker processes (across chromosomes).
    chroms : path, optional
        Chrom-size / whitelist file; output chunks follow its first-column
        order. Default: target reference order.
    ext : str
        Per-cell filename extension used to derive ``cell_id`` when the
        table has no ``cell_id`` column (default ``'.cz'``).
    level : int
        DEFLATE level for output blocks (default 6).
    temp : bool
        Keep the per-chrom temp shard directory (default False).
    """
    if context_policy not in ('strict', 'category', 'ignore'):
        raise ValueError(
            f"context_policy must be 'strict', 'category' or 'ignore', "
            f"got {context_policy!r}")
    if target_reference is None:
        raise ValueError("merge_cz_multiref requires target_reference")
    if cz_table is None:
        raise ValueError("merge_cz_multiref requires cz_table")
    if output is None:
        raise ValueError("merge_cz_multiref requires output")
    target_reference = os.path.abspath(os.path.expanduser(target_reference))
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return

    df = _resolve_cz_table(cz_table, ext=ext, require=('path', 'reference'))
    for p in df['path']:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"merge_cz_multiref: cell not found: {p}")

    # Group cells by their reference so each reference->target mapping is
    # computed once and reused across all cells sharing it.
    ref_to_cells = {ref: sub['path'].tolist()
                    for ref, sub in df.groupby('reference')}

    # Column names come from a cell; a reference-less mc/cov cell has these.
    r0 = Reader(df['path'].iloc[0])
    columns = list(r0.header['columns'])
    r0.close()

    # Chromosome order: target reference order, optionally filtered/ordered
    # by a user chrom file.
    tref = Reader(target_reference)
    target_chroms = [k[0] for k in tref.chunk_key2offset.keys()]
    tref.close()
    if chroms is not None:
        chroms = os.path.abspath(os.path.expanduser(chroms))
        cdf = pd.read_csv(chroms, sep='\t', header=None, usecols=[0])
        order = cdf.iloc[:, 0].astype(str).tolist()
        chrom_list = [c for c in order if c in set(target_chroms)]
    else:
        chrom_list = target_chroms

    outdir = output + '.tmp'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    init_args = (target_reference, ref_to_cells, list(formats), columns,
                 context_policy, level, outdir)
    written = []
    if jobs and int(jobs) > 1 and len(chrom_list) > 1:
        with multiprocessing.Pool(int(jobs), initializer=_merge_mr_init,
                                  initargs=init_args) as pool:
            for res in pool.imap_unordered(_merge_multiref_chrom_worker,
                                           chrom_list):
                if res is not None:
                    written.append(res)
    else:
        _merge_mr_init(*init_args)
        for chrom in chrom_list:
            res = _merge_multiref_chrom_worker(chrom)
            if res is not None:
                written.append(res)

    # Concatenate per-chrom shards in target order into the final output.
    shard_paths = [os.path.join(outdir, f"{c}.cz") for c in chrom_list
                   if os.path.exists(os.path.join(outdir, f"{c}.cz"))]
    writer = Writer(output=output, formats=formats, columns=columns,
                    chunk_dims=['chrom'], message=target_reference, level=level)
    writer.catcz(input=shard_paths, key_added=None)
    if not temp:
        _bg_rmtree(outdir)
    logger.info(f"Done: {output}")


def merge_cell_type(indir=None, cell_table=None, outdir=None,
                    jobs=64, chroms=None, ext='.CGN.merged.cz'):
    """Merge per-cell .cz files into per-cell-type aggregates.

    Reads a TSV ``cell_table`` with columns (cell, cell_type), groups
    cells by type, and calls :func:`merge_cz` once per group.
    """
    indir = os.path.abspath(os.path.expanduser(indir))
    outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    chroms = os.path.abspath(os.path.expanduser(chroms))
    df_ct = pd.read_csv(cell_table, sep='\t', header=None, names=['cell', 'ct'])
    for ct in df_ct.ct.unique():
        output = os.path.join(outdir, ct + '.cz')
        if os.path.exists(output):
            logger.info(f"{output} existed.")
            continue
        logger.info(ct)
        snames = df_ct.loc[df_ct.ct == ct, 'cell'].tolist()
        cz_paths = [os.path.join(indir, sname + ext) for sname in snames]
        merge_cz(input=cz_paths, bgzip=False,
                 output=output, jobs=jobs, chroms=chroms)


def merge_cell_type_multiref(cz_table=None, outdir=None,
                             target_reference=None, context_policy='strict',
                             jobs=64, chroms=None, ext='.cz',
                             formats=['H', 'H'], level=6):
    """Per-cell-type pseudobulk merge across *different* references.

    The multi-reference analogue of :func:`merge_cell_type`: groups the cells
    in ``cz_table`` by their ``cell_type`` column and calls
    :func:`merge_cz_multiref` once per group so each cell is pooled by genomic
    coordinate onto the common ``target_reference`` (see
    :func:`merge_cz_multiref` for the mismatch handling and ``context_policy``
    semantics).

    Parameters
    ----------
    cz_table : DataFrame or path
        Per-cell table (DataFrame or headered TSV) with columns ``path``,
        ``reference`` and ``cell_type`` (plus optional ``cell_id`` /
        ``index``). See :func:`_resolve_cz_table`.
    outdir : path
        Output directory; one ``{cell_type}.cz`` pseudobulk per group.
    target_reference : path
        Common reference ``.cz`` defining the output axis.
    context_policy : {'match', 'ignore'}
        Context-mismatch policy, forwarded to :func:`merge_cz_multiref`.
    jobs, chroms, ext, formats, level
        Forwarded to :func:`merge_cz_multiref`.
    """
    if outdir is None:
        raise ValueError("merge_cell_type_multiref requires outdir")
    outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df = _resolve_cz_table(cz_table, ext=ext,
                           require=('path', 'reference', 'cell_type'))
    for ct in df['cell_type'].unique():
        output = os.path.join(outdir, str(ct) + '.cz')
        if os.path.exists(output):
            logger.info(f"{output} existed.")
            continue
        logger.info(ct)
        sub = df.loc[df['cell_type'] == ct]
        merge_cz_multiref(
            cz_table=sub, target_reference=target_reference, output=output,
            context_policy=context_policy, formats=formats,
            jobs=jobs, chroms=chroms, ext=ext, level=level)


if __name__ == "__main__":
    from cytozip import main
    main()
