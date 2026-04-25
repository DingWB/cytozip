#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge.py — Parallel merging of multiple per-cell methylation .cz files.

Pipeline stages provided here:
  - :func:`merge_cz`: main entry point. Merges many per-cell .cz files into
    a single aggregate (summed mc/cov .cz, fraction .txt, 2D mc,cov matrix,
    or Fisher-test one-vs-rest p-value matrix).
  - :func:`merge_cz_worker`: per-chrom, per-batch worker executed by the
    multiprocessing pool.
  - :func:`_fisher_worker`: per-chunk Fisher's exact test (one-vs-rest)
    helper, used when ``formats='fisher'``.
  - :func:`catchr`: chromosome-level txt batch concatenator.
  - :func:`merge_cell_type`: convenience wrapper that calls :func:`merge_cz`
    once per cell-type grouping defined by a cell-table TSV.

These operations are methylation-specific (mc/cov column semantics,
Fisher-test mode) and therefore live in their own module rather than
in the generic :mod:`cytozip.cz` format layer.

@author: DingWB
"""
import os
import struct
import math
from loguru import logger
import multiprocessing
from .cz import (Reader, Writer, get_dtfuncs,
                 _BLOCK_MAX_LEN, _chunk_magic, _NP_FMT_MAP, np, pd)


# Per-format-char numpy max value (used to clip sums before packing).
_NP_FMT_MAX = {
    'B': 0xFF, 'H': 0xFFFF, 'I': 0xFFFFFFFF, 'L': 0xFFFFFFFF,
    'Q': 0xFFFFFFFFFFFFFFFF, 'b': 0x7F, 'h': 0x7FFF,
    'i': 0x7FFFFFFF, 'l': 0x7FFFFFFF, 'q': 0x7FFFFFFFFFFFFFFF,
}


def _structured_dtype_for(fmts):
    """Build a numpy structured dtype tiling one record (numeric only)."""
    return np.dtype([(f'f{i}', _NP_FMT_MAP[c]) for i, c in enumerate(fmts)])


# ==========================================================
def _fisher_worker(df):
    """Perform one-versus-rest Fisher's exact test on each CpG site.

    For each sample, tests whether its methylation level differs
    significantly from the rest of the samples combined.  Computes
    odds ratio and p-value per site per sample.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from fast_fisher import fast_fisher_exact, odds_ratio
    # one verse rest
    columns = df.columns.tolist()
    snames = [col[:-3] for col in columns if col.endswith('.mc')]
    df['mc_sum'] = df.loc[:, [name for name in columns if name.endswith('.mc')]].sum(axis=1)
    df['cov_sum'] = df.loc[:, [name for name in columns if name.endswith('.cov')]].sum(axis=1)
    df['uc_sum'] = df.cov_sum - df.mc_sum

    def cal_fisher_or_p(x):
        uc = int(x[f"{sname}.cov"] - x[f"{sname}.mc"])
        a, b, c, d = int(x[f"{sname}.mc"]), uc, int(x.mc_sum - x[f"{sname}.mc"]), int(x.uc_sum - uc)
        or_val = odds_ratio(a, b, c, d)
        p_val = fast_fisher_exact(a, b, c, d)
        return tuple(['%.3g' % or_val, '%.3g' % p_val])

    for sname in snames:
        df[sname] = df.apply(cal_fisher_or_p, axis=1)
        df[f"{sname}.odd_ratio"] = df[sname].apply(lambda x: x[0])
        df[f"{sname}.pval"] = df[sname].apply(lambda x: x[1])
        df.drop([f"{sname}.cov", f"{sname}.mc", sname], axis=1, inplace=True)
    usecols = []
    for sname in snames:
        usecols.extend([f"{sname}.odd_ratio", f"{sname}.pval"])
    return df.reindex(columns=usecols)


# ==========================================================
def merge_cz_worker(outfile_cat, outdir, chrom, dims, formats,
                    block_idx_start, batch_nblock, batch_size=5000):
    """Worker function for parallel merge of per-cell .cz data.

    Reads a batch of blocks (batch_nblock blocks starting at
    block_idx_start) for every cell/sample sharing the same chrom,
    then either:

    - Sums mc/cov values across cells (when formats is a list like ['H','H'])
    - Computes fraction (mc/cov) per cell ('fraction' mode)
    - Produces a 2D matrix of mc,cov per cell ('2D' mode)
    - Runs Fisher's exact test per cell ('fisher' mode)

    Results are written to per-chrom temporary files.
    """
    if formats in ['fraction', '2D', 'fisher']:
        ext = 'txt'
    else:
        ext = 'cz'
    outname = os.path.join(outdir, chrom + f'.{block_idx_start}.{ext}')
    reader1 = Reader(outfile_cat)
    data = None
    for dim in dims:  # each dim is a file of the same chrom
        r = reader1._load_chunk(reader1.chunk_key2offset[dim], jump=False)
        block_start_offset = reader1._chunk_block_1st_record_virtual_offsets[
                                 block_idx_start] >> 16
        buf_parts = []
        for i in range(batch_nblock):
            reader1._load_block(start_offset=block_start_offset)
            buf_parts.append(reader1._buffer)
            block_start_offset = None
        buffer = b''.join(buf_parts)
        # Vectorised parse of (mc, cov) — replaces the per-record Python
        # ``iter_unpack`` loop, which was the dominant cost in the worker.
        # ``reader1.fmts`` is e.g. "HH" / "BB" for the input mc_cov layout.
        in_fmts = reader1.fmts
        in_dt = _structured_dtype_for(in_fmts)
        rec = np.frombuffer(buffer, dtype=in_dt)
        # Materialise as a 2-column int64 matrix so accumulation cannot
        # overflow when summing thousands of cells.
        values = np.empty((rec.size, 2), dtype=np.int64)
        values[:, 0] = rec['f0']
        values[:, 1] = rec['f1']
        if formats == 'fraction':
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(values[:, 1] == 0, 0.0,
                                values[:, 0] / values[:, 1])
            values = np.array(['%.3g' % v for v in frac]).reshape(-1, 1)
        if data is None:
            data = values.copy()
        else:
            if formats in ["fraction", "2D", 'fisher']:
                data = np.hstack((data, values))
            else:
                data += values
    if formats in ['fraction', '2D', 'fisher']:
        snames = [dim[1] for dim in dims]
        if formats == 'fraction':
            columns = snames
        else:
            columns = []
            for sname in snames:
                columns.extend([sname + '.mc', sname + '.cov'])
        df = pd.DataFrame(data, columns=columns)
        if formats == 'fisher':
            df = _fisher_worker(df)
        df.to_csv(outname, sep='\t', index=False)
        return

    writer1 = Writer(outname, formats=formats,
                     columns=reader1.header['columns'],
                     chunk_dims=reader1.header['chunk_dims'][:1],
                     message=outfile_cat)
    # Vectorised pack: clip the int64 sums to the output dtype range,
    # build a structured array, and emit one ``tobytes`` blob per
    # ``batch_size`` records — replaces the per-record ``struct.pack``
    # loop which dominated the writer cost on large merges.
    out_fmts = ''.join(writer1.formats)
    out_dt = _structured_dtype_for(out_fmts)
    n = data.shape[0]
    # Clip per output column to its dtype max (matches old per-record
    # behaviour silently truncating overflow via struct).
    col0 = np.clip(data[:, 0], 0, _NP_FMT_MAX[out_fmts[0]])
    col1 = np.clip(data[:, 1], 0, _NP_FMT_MAX[out_fmts[1]])
    out_arr = np.empty(n, dtype=out_dt)
    out_arr['f0'] = col0
    out_arr['f1'] = col1
    for s in range(0, n, batch_size):
        writer1.write_chunk(out_arr[s:s + batch_size].tobytes(), [chrom])
    writer1.close()
    reader1.close()
    return


def catchr(outdir, chrom, ext, batch_nblock, batch_size):
    """Concatenate per-batch txt shards for one chromosome."""
    outname = os.path.join(outdir, f"{chrom}.{ext}")
    block_idx_start = 0
    infile = os.path.join(outdir, chrom + f'.{block_idx_start}.{ext}')
    while os.path.exists(infile):
        for df in pd.read_csv(infile, sep='\t', batch_size=batch_size):
            if not os.path.exists(outname):
                df.to_csv(outname, sep='\t', index=False, header=True)
            else:
                df.to_csv(outname, sep='\t', index=False, header=False, mode='a')
        block_idx_start += batch_nblock
        infile = os.path.join(outdir, chrom + f'.{block_idx_start}.{ext}')
    return


def merge_cz(indir=None, cz_paths=None, class_table=None,
             output=None, prefix=None, threads=12, formats=['H', 'H'],
             chrom_order=None, reference=None,
             keep_cat=False, blocks_per_batch=10, temp=False, bgzip=True,
             batch_size=50000, ext='.cz'):
    """
    Merge multiple .cz files. For example:
    cytozip merge_cz -i ./ -o major_type.2D.txt -n 96 -f 2D \
                          -P ~/Ref/mm10/mm10_ucsc_with_chrL.main.chrom.sizes.txt \
                          -r ~/Ref/mm10/annotations/mm10_with_chrL.allCG.forward.cz

    Parameters
    ----------
    indir :path
        If cz_paths is not provided, indir will be used to get cz_paths.
    cz_paths :paths
    class_table: path
        If class_table is given, multiple output will be generated based on the
        snames and class from this class_table, each output will have a suffix of
        class name in this table.
    output : path
    threads :int
    formats : str of list
        Could be fraction, 2D, fisher or list of formats.
        if formats is a list, then mc and cov will be summed up and write to .cz file.
        otherwise, if formats=='fraction', summed mc divided by summed cov
        will be calculated and written to .txt file. If formats=='2D', mc and cov
        will be kept and write to .txt matrix file.
    chrom_order : path
        path to chrom size file.
    reference : path
        path to reference .cz file, only need if fraction="fraction" or "2D".
    keep_cat : bool
    blocks_per_batch :int
    temp : bool
    bgzip : bool
    batch_size : int

    Returns
    -------

    """
    if not class_table is None:
        df_class = pd.read_csv(class_table, sep='\t', header=None,
                               names=['sname', 'cell_class'])
        snames = [file.replace(ext, '') for file in os.listdir(indir)]
        df_class = df_class.loc[df_class.sname.isin(snames)]
        class_groups = df_class.groupby('cell_class').sname.apply(
            lambda x: x.tolist()).to_dict()
        for key in class_groups:
            logger.info(key)
            cz_paths = [sname + ext for sname in class_groups[key]]
            merge_cz(indir, cz_paths, class_table=None,
                     output=None, prefix=f"{prefix}.{key}", threads=threads,
                     formats=formats, chrom_order=chrom_order,
                     reference=reference, keep_cat=keep_cat,
                     blocks_per_batch=blocks_per_batch, temp=temp, bgzip=bgzip,
                     batch_size=batch_size, ext=ext)
        return None
    if output is None:
        if prefix is None:
            output = 'merged.cz' if formats not in ['fraction', '2D', 'fisher'] else 'merged.txt'
        else:
            output = f'{prefix}.cz' if formats not in ['fraction', '2D', 'fisher'] else f'{prefix}.txt'
    logger.info(output)
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return
    if cz_paths is None:
        cz_paths = [file for file in os.listdir(indir) if file.endswith(ext)]
    reader = Reader(os.path.join(indir, cz_paths[0]))
    header = reader.header
    reader.close()
    outfile_cat = output + '.cat.cz'
    # cat all .cz files into one .cz file, add a chunk_key to chunk (filename)
    writer = Writer(output=outfile_cat, formats=header['formats'],
                    columns=header['columns'], chunk_dims=header['chunk_dims'],
                    message="catcz")
    writer.catcz(input=[os.path.join(indir, cz_path) for cz_path in cz_paths],
                 add_key=True)

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
    unit_nblock = int(writer._unit_size / (math.gcd(writer._unit_size, _BLOCK_MAX_LEN)))
    nunit_perbatch = int(np.ceil((chunk_info.chunk_nblocks.max() / blocks_per_batch
                                  ) / unit_nblock))
    batch_nblock = nunit_perbatch * unit_nblock  # how many block for each batch
    pool = multiprocessing.Pool(threads)
    jobs = []
    outdir = output + '.tmp'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for chrom in chroms:
        dims = chunk_info.loc[chunk_info[chrom_col] == chrom].index.tolist()
        if len(dims) == 0:
            continue
        block_idx_start = 0
        while block_idx_start < chrom_nblocks[chrom]:
            job = pool.apply_async(merge_cz_worker,
                                   (outfile_cat, outdir, chrom, dims, formats,
                                    block_idx_start, batch_nblock))
            jobs.append(job)
            block_idx_start += batch_nblock
    for job in jobs:
        r = job.get()
    pool.close()
    pool.join()

    # First, merge different batch for each chrom
    if formats in ['fraction', '2D', 'fisher']:
        out_ext = 'txt'
    else:
        out_ext = 'cz'
    if out_ext == 'cz':
        for chrom in chroms:
            # merge batches into chrom (chunk)
            outname = os.path.join(outdir, f"{chrom}.{out_ext}")
            writer = Writer(output=outname, formats=formats,
                            columns=header['columns'], chunk_dims=header['chunk_dims'],
                            message=outfile_cat)
            writer._chunk_start_offset = writer._handle.tell()
            writer._handle.write(_chunk_magic)
            # chunk total size place holder: 0
            writer._handle.write(struct.pack("<Q", 0))  # 8 bytes; including this chunk_size
            writer._chunk_data_len = 0
            writer._block_1st_record_virtual_offsets = []
            writer._chunk_dims = [chrom]
            block_idx_start = 0
            infile = os.path.join(outdir, chrom + f'.{block_idx_start}.{out_ext}')
            while os.path.exists(infile):
                reader = Reader(infile)
                reader._load_chunk(reader.header['header_size'])
                block_start_offset = reader._chunk_start_offset + 10
                writer._buffer = bytearray()
                for i in range(reader._chunk_nblocks):
                    reader._load_block(start_offset=block_start_offset)
                    if len(writer._buffer) + len(reader._buffer) < _BLOCK_MAX_LEN:
                        writer._buffer.extend(reader._buffer)
                    else:
                        writer._buffer.extend(reader._buffer)
                        while len(writer._buffer) >= _BLOCK_MAX_LEN:
                            writer._write_block(bytes(writer._buffer[:_BLOCK_MAX_LEN]))
                            del writer._buffer[:_BLOCK_MAX_LEN]
                    block_start_offset = None
                reader.close()
                block_idx_start += batch_nblock
                infile = os.path.join(outdir, chrom + f'.{block_idx_start}.{out_ext}')
            # write chunk tail
            writer.close()
    else:  # txt
        pool = multiprocessing.Pool(threads)
        jobs = []
        for chrom in chroms:
            job = pool.apply_async(catchr,
                                   (outdir, chrom, out_ext, batch_nblock, batch_size))
            jobs.append(job)
        for job in jobs:
            r = job.get()
        pool.close()
        pool.join()

    # Second, merge chromosomes to output
    if out_ext == 'cz':  # merge chroms into final output
        writer = Writer(output=output, formats=formats,
                        columns=header['columns'], chunk_dims=header['chunk_dims'],
                        message="merged")
        writer.catcz(input=[f"{outdir}/{chrom}.cz" for chrom in chroms])
    else:  # txt
        filenames = chunk_info.filename.unique().tolist()
        if formats == 'fraction':
            columns = filenames
        elif formats == '2D':
            columns = []
            for sname in filenames:
                columns.extend([sname + '.mc', sname + '.cov'])
        else:  # fisher
            columns = []
            for sname in filenames:
                columns.extend([sname + '.odd_ratio', sname + '.pval'])
        if not reference is None:
            reference = os.path.abspath(os.path.expanduser(reference))
            reader = Reader(reference)
        logger.info("Merging chromosomes..")
        for chrom in chroms:
            logger.debug(chrom)
            infile = os.path.join(outdir, f"{chrom}.{out_ext}")
            if not reference is None:
                df_ref = pd.DataFrame([
                    record for record in reader.fetch(tuple([chrom]))
                ], columns=reader.header['columns'])
                # insert a column 'start'
                df_ref.insert(0, chrom_col, chrom)
                df_ref.insert(1, 'start', df_ref.iloc[:, 1].map(int) - 1)
                usecols = df_ref.columns.tolist() + columns
            for df in pd.read_csv(infile, sep='\t', batch_size=batch_size):
                if not reference is None:
                    df = pd.concat([df_ref.iloc[:batch_size].reset_index(drop=True),
                                    df.reset_index(drop=True)], axis=1)
                    df_ref = df_ref.iloc[batch_size:]
                if not os.path.exists(output):
                    df.reindex(columns=usecols).to_csv(output, sep='\t', index=False, header=True)
                else:
                    df.reindex(columns=usecols).to_csv(output, sep='\t', index=False,
                                                       header=False, mode='a')
        if not reference is None:
            reader.close()
    if not keep_cat:
        os.remove(outfile_cat)
    if not temp:
        logger.info(f"Removing temp dir {outdir}")
        os.system(f"rm -rf {outdir}")
    if bgzip and not output.endswith(ext):
        cmd = f"bgzip {output} && tabix -S 1 -s 1 -b 2 -e 3 -f {output}.gz"
        logger.info(f"Run bgzip, CMD: {cmd}")
        os.system(cmd)


def merge_cell_type(indir=None, cell_table=None, outdir=None,
                    threads=64, chrom_order=None, ext='.CGN.merged.cz'):
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
        merge_cz(indir=indir, cz_paths=cz_paths, bgzip=False,
                 output=output, threads=threads, chrom_order=chrom_order)


if __name__ == "__main__":
    from cytozip import main
    main()
