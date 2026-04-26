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
                 _chunk_magic, _NP_FMT_MAP, np, pd)


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
        # uses _block_size = _BLOCK_MAX_LEN = 65535 (which is *not* a
        # multiple of unit_size for typical mc/cov record sizes). The
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
             chrom_order=None, reference=None,
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
    chrom_order : path
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
                     formats=formats, chrom_order=chrom_order,
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
    if not chrom_order is None:
        chrom_order = os.path.abspath(os.path.expanduser(chrom_order))
        df = pd.read_csv(chrom_order, sep='\t', header=None, usecols=[0])
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
                    message="merged")
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


def merge_cell_type(indir=None, cell_table=None, outdir=None,
                    jobs=64, chrom_order=None, ext='.CGN.merged.cz'):
    """Merge per-cell .cz files into per-cell-type aggregates.

    Reads a TSV ``cell_table`` with columns (cell, cell_type), groups
    cells by type, and calls :func:`merge_cz` once per group.
    """
    indir = os.path.abspath(os.path.expanduser(indir))
    outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    chrom_order = os.path.abspath(os.path.expanduser(chrom_order))
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
                 output=output, jobs=jobs, chrom_order=chrom_order)


if __name__ == "__main__":
    from cytozip import main
    main()
