#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dmr.py — Methylation peak calling and differentially-methylated-region (DMR)
analysis tools that read from the cytozip .cz format.

Functions:
  - :func:`call_peaks`: call peaks from a methylation .cz file using MACS3
    by turning unmethylated (cov - mc) or methylated counts into pseudo-reads.
  - :func:`to_bedgraph`: dump a per-site methylation signal as a bedGraph.
  - :func:`combp`: run combined-pvalues (comb-p) on a Fisher-test matrix
    produced by ``merge_cz -f fisher`` to call DMRs.
  - :func:`annot_dmr`: annotate a merged DMR table with hypo/hyper sample
    assignments and delta-beta values.
  - :func:`__split_mat`: internal helper used by :func:`combp` to split a
    Fisher-test matrix into per-sample, per-chrom bed files.

These pipelines are methylation-specific downstream analyses (MACS3 /
comb-p integration) and therefore live in their own module rather than in
the generic :mod:`cytozip.cz` format layer.

@author: DingWB
"""
import os
import multiprocessing
from .cz import (Reader, _STRUCT_TO_NP_DTYPE, np, pd)


def __split_mat(infile, chrom, snames, outdir, n_ref):
    import pysam
    tbi = pysam.TabixFile(infile)
    records = tbi.fetch(reference=chrom)
    N = n_ref + len(snames) * 2
    fout_dict = {}
    for sname in snames:
        fout_dict[sname] = open(os.path.join(outdir, f"{sname}.{chrom}.bed"), 'w')
        fout_dict[sname].write("chrom\tstart\tend\tstrand\tpval\todd_ratio\n")
    for line in records:
        values = line.replace('\n', '').split('\t')
        if len(values) < N:
            print(infile, chrom)
            raise ValueError("Number of fields is wrong.")
        ch, beg, end, strand = values[:4]
        beg = int(beg)
        end = int(end)
        for i, sname in enumerate(snames):
            or_value = values[n_ref + i * 2]
            try:
                OR = float(or_value)
            except (ValueError, TypeError):
                OR = 1
            if OR >= 1:  # hyper methylation
                pval = 1
            else:
                pval = values[n_ref + i * 2 + 1]
            fout_dict[sname].write(f"{chrom}\t{beg}\t{end}\t{strand}\t{pval}\t{or_value}\n")
    for sname in snames:
        fout_dict[sname].close()
    tbi.close()


# ==========================================================================
# Peak calling from methylation data
# ==========================================================================
def _make_np_dtype(formats, columns):
    """Map .cz struct format chars to a numpy structured dtype."""
    dt = []
    for fmt, col in zip(formats, columns):
        np_dt = _STRUCT_TO_NP_DTYPE.get(fmt[-1])
        if np_dt is None:
            n = int(fmt[:-1]) if len(fmt) > 1 else 1
            dt.append((col, f'S{n}'))
        else:
            dt.append((col, np_dt))
    return np.dtype(dt)


def call_peaks(input=None, reference=None, output=None, name='peaks',
               signal='unmeth', index=None, genome_size='mm',
               fragment_size=300, qvalue=0.05, broad=False,
               min_cov=1, keep_bed=False, macs3_args='',
               mc_col=None, cov_col=None):
    """Call peaks from a methylation .cz file using MACS3.

    Treats unmethylated counts (cov - mc) at each cytosine site as the
    signal (analogous to ATAC-seq read counts).  For each site with
    unmethylated count *u*, generates *u* pseudo-reads (BED intervals)
    of length ``fragment_size`` centered on the site.  These are then
    fed to ``macs3 callpeak --nomodel``.

    This is useful for identifying regions of low methylation
    (e.g., open chromatin in NOMe-seq, or regulatory elements in WGBS).

    Parameters
    ----------
    input : str
        Input .cz file with mc/cov columns.
    reference : str
        Reference .cz file with genomic coordinates (pos, strand, context).
    output : str or None
        Output directory for MACS3 results.  Defaults to
        ``<input_stem>_peaks/``.
    name : str
        Name prefix for MACS3 output files.
    signal : str
        ``'unmeth'`` uses (cov - mc) as signal;
        ``'meth'`` uses mc as signal.
    index : str or None
        Path to index file for context filtering (e.g., CpG-only index
        from ``index_context``).
    genome_size : str or int
        Genome size for MACS3.  Use ``'hs'`` for human (~2.7e9),
        ``'mm'`` for mouse (~1.87e9), or an integer.
    fragment_size : int
        Length of each pseudo-read (default 300 bp).
    qvalue : float
        MACS3 q-value cutoff (default 0.05).
    broad : bool
        If True, call broad peaks (``--broad``).
    min_cov : int
        Minimum coverage to include a site (default 1).
    keep_bed : bool
        If True, keep the intermediate pseudo-reads BED file.
    macs3_args : str
        Additional arguments passed to ``macs3 callpeak``.
    mc_col : int or str or None
        Column index (0-based) or name for the methylation count.
        Defaults to the first data column (index 0, typically ``'mc'``).
    cov_col : int or str or None
        Column index (0-based) or name for the coverage count.
        Defaults to the last data column (index -1, typically ``'cov'``).

    Returns
    -------
    str
        Path to the output directory containing MACS3 results.

    Examples
    --------
    ::

        # CpG-only peak calling
        czip call_peaks -I cell.cz -r mm10.allc.cz -s mm10.CGN.index \\
             -g mm -n cell_unmeth

        # Python API
        import cytozip as czip
        czip.call_peaks(input='cell.cz', reference='mm10.allc.cz',
                        index='mm10.CGN.index', genome_size='mm')

    """
    import subprocess

    # ---- Step 1: Open the data .cz (mc/cov) and the reference .cz (pos/strand/context) ----
    cz_path = os.path.abspath(os.path.expanduser(input))
    ref_path = os.path.abspath(os.path.expanduser(reference))

    reader = Reader(cz_path)       # per-cell methylation data (mc, cov)
    ref_reader = Reader(ref_path)  # shared genomic coordinates (pos, strand, context)

    if output is None:
        output = os.path.splitext(cz_path)[0] + '_peaks'
    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(output, exist_ok=True)

    # Optional index for context filtering (e.g., CpG only)
    index_reader = None
    if index is not None:
        index_path = os.path.abspath(os.path.expanduser(index))
        index_reader = Reader(index_path)

    # ---- Step 2: Build numpy structured dtypes for zero-copy binary decoding ----
    data_dtype = _make_np_dtype(reader.header['formats'],
                                reader.header['columns'])
    ref_dtype = _make_np_dtype(ref_reader.header['formats'],
                               ref_reader.header['columns'])

    # ---- Step 3: Generate pseudo-reads BED from methylation signal ----
    half = fragment_size // 2
    # Resolve mc/cov column names from user params or header defaults
    _cols = reader.header['columns']
    if mc_col is None:
        mc_col = _cols[0]
    elif isinstance(mc_col, int):
        mc_col = _cols[mc_col]
    if cov_col is None:
        cov_col = _cols[-1]
    elif isinstance(cov_col, int):
        cov_col = _cols[cov_col]

    bed_path = os.path.join(output, f'{name}.pseudo_reads.bed')

    total_reads = 0
    with open(bed_path, 'w') as fh:
        for dim in reader.dim2chunk_start:
            if dim not in ref_reader.dim2chunk_start:
                continue
            chrom = dim[0]

            raw = reader.fetch_chunk_bytes(dim)
            if not raw:
                continue
            data_arr = np.frombuffer(raw, dtype=data_dtype)

            ref_raw = ref_reader.fetch_chunk_bytes(dim)
            if not ref_raw:
                continue
            ref_arr = np.frombuffer(ref_raw, dtype=ref_dtype)

            if index_reader is not None and dim in index_reader.dim2chunk_start:
                ids = index_reader.get_ids_from_index(dim)
                if len(ids.shape) == 1:
                    data_arr = data_arr[ids]
                    ref_arr = ref_arr[ids]

            pos = ref_arr['pos'].astype(np.int64)
            mc = data_arr[mc_col].astype(np.int32)
            cov = data_arr[cov_col].astype(np.int32)

            mask = cov >= min_cov
            pos = pos[mask]
            mc = mc[mask]
            cov = cov[mask]

            if signal == 'unmeth':
                sig = cov - mc
            elif signal == 'meth':
                sig = mc.copy()
            else:
                raise ValueError(f"Unknown signal type: {signal!r}")

            pos_mask = sig > 0
            pos = pos[pos_mask]
            sig = sig[pos_mask]

            if len(pos) == 0:
                continue

            expanded = np.repeat(pos, sig)
            starts = np.maximum(0, expanded - half)
            ends = expanded + half
            total_reads += len(starts)

            bed_df = pd.DataFrame({
                'chrom': chrom,
                'start': starts,
                'end': ends,
            })
            bed_df.to_csv(fh, sep='\t', header=False, index=False)

            print(f"  {chrom}: {len(pos)} sites, "
                  f"{int(sig.sum())} pseudo-reads")

    reader.close()
    ref_reader.close()
    if index_reader:
        index_reader.close()

    print(f"Total pseudo-reads: {total_reads}")

    # ---- Step 4: Sort BED and run MACS3 peak calling ----
    sorted_bed = bed_path.replace('.bed', '.sorted.bed')
    print("Sorting BED file...")
    subprocess.run(
        ['sort', '-k1,1', '-k2,2n', bed_path, '-o', sorted_bed],
        check=True,
    )

    cmd = [
        'macs3', 'callpeak',
        '-t', sorted_bed,
        '-f', 'BED',
        '--outdir', output,
        '-n', name,
        '-g', str(genome_size),
        '--nomodel',
        '--extsize', str(fragment_size),
        '-q', str(qvalue),
    ]
    if broad:
        cmd.append('--broad')
    if macs3_args:
        cmd.extend(macs3_args.split())

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    os.remove(bed_path)
    if not keep_bed:
        os.remove(sorted_bed)
    else:
        print(f"Pseudo-reads BED kept at: {sorted_bed}")

    print(f"Peak results saved to: {output}")
    return output


def to_bedgraph(input=None, reference=None, output=None,
                signal='unmeth', index=None, min_cov=1,
                mc_col=None, cov_col=None):
    """Export methylation signal from a .cz file as a bedGraph.

    For each cytosine site, writes one bedGraph entry with the chosen
    signal value (unmethylated count or methylation count).
    The output can be loaded into a genome browser or used with
    ``macs3 bdgpeakcall`` for simple threshold-based peak calling.

    Parameters
    ----------
    input : str
        Input .cz file with mc/cov columns.
    reference : str
        Reference .cz file with genomic coordinates.
    output : str or None
        Output bedGraph path.  Defaults to ``<input_stem>.bedgraph``.
    signal : str
        ``'unmeth'`` writes (cov - mc); ``'meth'`` writes mc;
        ``'frac_unmeth'`` writes (cov - mc) / cov.
    index : str or None
        Path to index file for context filtering.
    min_cov : int
        Minimum coverage to include a site.
    mc_col : int or str or None
        Column index (0-based) or name for the methylation count.
        Defaults to the first data column (index 0, typically ``'mc'``).
    cov_col : int or str or None
        Column index (0-based) or name for the coverage count.
        Defaults to the last data column (index -1, typically ``'cov'``).

    Returns
    -------
    str
        Path to the output bedGraph file.
    """
    cz_path = os.path.abspath(os.path.expanduser(input))
    ref_path = os.path.abspath(os.path.expanduser(reference))

    reader = Reader(cz_path)
    ref_reader = Reader(ref_path)

    if output is None:
        output = os.path.splitext(cz_path)[0] + '.bedgraph'
    output = os.path.abspath(os.path.expanduser(output))

    index_reader = None
    if index is not None:
        index_path = os.path.abspath(os.path.expanduser(index))
        index_reader = Reader(index_path)

    data_dtype = _make_np_dtype(reader.header['formats'],
                                reader.header['columns'])
    ref_dtype = _make_np_dtype(ref_reader.header['formats'],
                               ref_reader.header['columns'])

    _cols = reader.header['columns']
    if mc_col is None:
        mc_col = _cols[0]
    elif isinstance(mc_col, int):
        mc_col = _cols[mc_col]
    if cov_col is None:
        cov_col = _cols[-1]
    elif isinstance(cov_col, int):
        cov_col = _cols[cov_col]

    with open(output, 'w') as fh:
        for dim in reader.dim2chunk_start:
            if dim not in ref_reader.dim2chunk_start:
                continue
            chrom = dim[0]

            raw = reader.fetch_chunk_bytes(dim)
            if not raw:
                continue
            data_arr = np.frombuffer(raw, dtype=data_dtype)

            ref_raw = ref_reader.fetch_chunk_bytes(dim)
            if not ref_raw:
                continue
            ref_arr = np.frombuffer(ref_raw, dtype=ref_dtype)

            if index_reader is not None and dim in index_reader.dim2chunk_start:
                ids = index_reader.get_ids_from_index(dim)
                if len(ids.shape) == 1:
                    data_arr = data_arr[ids]
                    ref_arr = ref_arr[ids]

            pos = ref_arr['pos'].astype(np.int64)
            mc = data_arr[mc_col].astype(np.float64)
            cov = data_arr[cov_col].astype(np.float64)

            mask = cov >= min_cov
            pos = pos[mask]
            mc = mc[mask]
            cov = cov[mask]

            if signal == 'unmeth':
                values = cov - mc
            elif signal == 'meth':
                values = mc
            elif signal == 'frac_unmeth':
                values = (cov - mc) / cov
            else:
                raise ValueError(f"Unknown signal type: {signal!r}")

            keep = values > 0
            pos = pos[keep]
            values = values[keep]

            if len(pos) == 0:
                continue

            df = pd.DataFrame({
                'chrom': chrom,
                'start': pos - 1,  # bedGraph is 0-based
                'end': pos,
                'value': values,
            })
            df.to_csv(fh, sep='\t', header=False, index=False)

    reader.close()
    ref_reader.close()
    if index_reader:
        index_reader.close()

    print(f"bedGraph written to: {output}")
    return output


def combp(input, outdir="cpv", threads=24, dist=300, temp=True, bed=False):
    """
    Run comb-p on a fisher result matrix (generated by `merge_cz -f fisher`),
    /usr/bin/time -f "%e\t%M\t%P" cytozip combp -i major_type.fisher.txt.gz -n 64
    Run one samples (all chromosomes), 8308.18(2.3h) 65053112(62G)        321%

    Parameters
    ----------
    input : path
        path to result from merge_cz -f fisher.
    outdir : path
    threads : int
    dist: int
        max distance between two site to be included in one DMR.
    temp : bool
        whether to keep temp dir
    bed : bool
        whether to keep bed directory

    Returns
    -------

    """
    try:
        from cpv.pipeline import pipeline as cpv_pipeline
    except ImportError:
        print("Please install cpv using: pip install git+https://github.com/DingWB/combined-pvalues")

    infile = os.path.abspath(os.path.expanduser(input))
    outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    columns = pd.read_csv(infile, sep='\t', nrows=1).columns.tolist()
    snames = [col[:-5] for col in columns[4:] if col.endswith('.pval')]
    import pysam
    tbi = pysam.TabixFile(infile)
    chroms = sorted(tbi.contigs)
    tbi.close()

    bed_dir = os.path.join(outdir, 'bed')
    if not os.path.exists(bed_dir):
        os.mkdir(bed_dir)
        pool = multiprocessing.Pool(threads)
        jobs = []
        print("Splitting matrix into different samples and chroms.")
        for chrom in chroms:
            job = pool.apply_async(__split_mat,
                                   (infile, chrom, snames, bed_dir, 5))
            jobs.append(job)
        for job in jobs:
            r = job.get()
        pool.close()
        pool.join()
    else:
        print("bed directory existed, skip split matrix into bed files.")

    tmpdir = os.path.join(outdir, "tmp")
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    pool = multiprocessing.Pool(threads)
    jobs = []
    print("Running cpv..")
    acf_dist = int(round(dist / 3, -1))
    for chrom in chroms:
        for sname in snames:
            bed_file = os.path.join(bed_dir, f"{sname}.{chrom}.bed")
            prefix = os.path.join(tmpdir, f"{sname}.{chrom}")
            output = os.path.join(tmpdir, f"{sname}.{chrom}.regions-p.bed.gz")
            if os.path.exists(output):
                continue
            job = pool.apply_async(cpv_pipeline,
                                   (4, None, dist, acf_dist, prefix,
                                    0.05, 0.05, "refGene", [bed_file], True, 1, None, False,
                                    None, True))
            jobs.append(job)
    for job in jobs:
        r = job.get()
    pool.close()
    pool.join()

    print("Merging cpv results..")
    for sname in snames:
        output = os.path.join(outdir, f"{sname}.bed")
        for chrom in chroms:
            infile = os.path.join(tmpdir, f"{sname}.{chrom}.regions-p.bed.gz")
            if not os.path.exists(infile):
                continue
            df = pd.read_csv(infile, sep='\t')
            df = df.loc[df.z_sidak_p <= 0.05]
            if df.shape[0] == 0:
                continue
            if not os.path.exists(output):
                df.to_csv(output, sep='\t', index=False, header=True)
            else:
                df.to_csv(output, sep='\t', index=False, header=False, mode='a')
    merged_dmr_path = os.path.join(outdir, 'merged_dmr.txt')
    data = None
    for sname in snames:
        infile = os.path.join(outdir, f"{sname}.bed")
        df = pd.read_csv(infile, sep='\t')
        df.drop(['min_p', 'z_p', 'z_sidak_p'], inplace=True, axis=1)
        df['sname'] = sname
        if data is None:
            data = df.copy()
        else:
            data = pd.concat([data, df], ignore_index=True)
    data.to_csv(merged_dmr_path, sep='\t', index=False)
    if not bed:
        os.system(f"rm -rf {bed_dir}")
    if not temp:
        os.system(f"rm -rf {tmpdir}")


def annot_dmr(input="merged_dmr.txt", matrix="merged_dmr.cell_class.beta.txt",
              output='dmr.annotated.txt', delta_cutoff=None):
    """
    Annotate DMR result from cpv.

    Parameters
    ----------
    input : path
        Merged dmr from cytozip combp.
    matrix : path
        result of agg_beta using dmr and output of merge_cz (fraction) as input.
    output : path
        annotated dmr, containing hypomethylated sname, delta
    Returns
    -------

    """
    data = pd.read_csv(os.path.expanduser(matrix), sep='\t', index_col=[0, 1, 2])
    df_rows = data.index.to_frame()
    cols = data.columns.tolist()
    values_arr = data.values
    df_rows['Hypo'] = [cols[i] for i in np.argmin(values_arr, axis=1)]
    df_rows['Hyper'] = [cols[i] for i in np.argmax(values_arr, axis=1)]
    df_rows['Max'] = np.max(values_arr, axis=1)
    df_rows['Min'] = np.min(values_arr, axis=1)
    df_rows['delta_beta'] = df_rows.Max - df_rows.Min
    df_dmr = pd.read_csv(os.path.expanduser(input), sep='\t')
    cols = df_dmr.columns.tolist()
    n_cpg = df_dmr.iloc[:, :4].drop_duplicates().set_index(cols[:3])[cols[3]].to_dict()
    dmr_sample_dict = df_dmr.loc[:, cols[:3] + ['sname']].drop_duplicates().groupby(
        cols[:3]).sname.agg(lambda x: x.tolist())
    df_rows['n_dms'] = df_rows.index.to_series().map(n_cpg)
    df_rows['sname'] = df_rows.index.to_series().map(dmr_sample_dict)
    df_rows['sname'] = df_rows['sname'].apply(lambda x: ','.join(x))
    if not delta_cutoff is None:
        df_rows = df_rows.loc[df_rows.delta_beta >= delta_cutoff]
    df_rows.to_csv(os.path.expanduser(output),
                   sep='\t', index=False)


if __name__ == "__main__":
    from cytozip import main
    main()
