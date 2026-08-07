#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
peaks.py — ATAC-style peak calling from methylation .cz files via MACS3.

CpG hypomethylation marks open chromatin / regulatory elements, so the
per-site unmethylated count ``umc = cov - mc`` is used as an ATAC-like read
signal and fed to MACS3. Two routes are provided:

  - :func:`call_peaks`: expand each site into ``umc`` pseudo-reads and run
    ``macs3 callpeak`` (optionally with a coverage-based control track).
  - :func:`call_peaks_bdg`: build piecewise-constant pileup bedGraphs (memory
    ``O(n_sites)``) and drive MACS3's bedGraph back-end
    (``bdgopt`` -> ``bdgcmp`` -> ``bdgpeakcall``); the coverage pileup is used
    as the control lambda so peaks reflect local unmethylation enrichment.
  - :func:`to_bedgraph`: dump a per-site methylation signal as a bedGraph.

Both peak callers run on a single-track (pseudobulk) ``.cz``; merge same-type
single cells with ``merge_cz`` first. These are methylation-specific downstream
analyses (MACS3 integration) and therefore live outside the generic
:mod:`cytozip.cz` format layer.

@author: DingWB
"""
import os
from loguru import logger
from .cz import Reader, _make_np_dtype, np, pd


# ==========================================================================
# Peak calling from methylation data
# ==========================================================================
def _write_pseudo_reads(fh, chrom, pos, sig, half):
    """Expand per-site counts into pseudo-read BED intervals and write them.

    Each site at ``pos`` with count ``sig`` emits ``sig`` intervals of length
    ``2*half`` centred on the site. Returns the number of reads written.
    """
    sig = np.asarray(sig)
    keep = sig > 0
    pos = np.asarray(pos)[keep]
    sig = sig[keep]
    if pos.size == 0:
        return 0
    expanded = np.repeat(pos, sig)
    starts = np.maximum(0, expanded - half)
    ends = expanded + half
    pd.DataFrame({'chrom': chrom, 'start': starts, 'end': ends}).to_csv(
        fh, sep='\t', header=False, index=False)
    return int(starts.size)


def _concat_files(paths, dest):
    """Concatenate ``paths`` into ``dest`` in order, deleting each piece."""
    import shutil
    with open(dest, 'wb') as out:
        for p in paths:
            if p and os.path.exists(p):
                with open(p, 'rb') as fh:
                    shutil.copyfileobj(fh, out)
                os.remove(p)
    return dest


def _pseudo_reads_chrom_worker(dim, cz_path, ref_path, index_path,
                               mc_col, cov_col, min_cov, half, control,
                               output, name):
    """Worker: write one chromosome's pseudo-read BED piece(s).

    Opens its own readers (process-safe), materialises the treat (and optional
    control) pseudo-reads for ``dim`` into per-chromosome temp BEDs, and
    returns ``(chrom, bed_path, n_reads, ctrl_path, n_ctrl)`` with ``*_path``
    set to ``None`` when empty. Used by :func:`call_peaks` when ``jobs > 1``.
    """
    reader = Reader(cz_path)
    ref_reader = Reader(ref_path)
    index_reader = Reader(index_path) if index_path else None
    try:
        chrom = dim[0]
        if dim not in ref_reader.chunk_key2offset:
            return chrom, None, 0, None, 0
        data_dtype = _make_np_dtype(reader.header['formats'],
                                    reader.header['columns'])
        ref_dtype = _make_np_dtype(ref_reader.header['formats'],
                                   ref_reader.header['columns'])
        raw = reader.fetch_chunk_bytes(dim)
        if not raw:
            return chrom, None, 0, None, 0
        data_arr = np.frombuffer(raw, dtype=data_dtype)
        ref_raw = ref_reader.fetch_chunk_bytes(dim)
        if not ref_raw:
            return chrom, None, 0, None, 0
        ref_arr = np.frombuffer(ref_raw, dtype=ref_dtype)
        if index_reader is not None and dim in index_reader.chunk_key2offset:
            ids = index_reader.get_ids_from_index(dim)
            if len(ids.shape) == 1:
                data_arr = data_arr[ids]
                ref_arr = ref_arr[ids]
        pos = ref_arr['pos'].astype(np.int64)
        mc = data_arr[mc_col].astype(np.int32)
        cov = data_arr[cov_col].astype(np.int32)
        mask = cov >= min_cov
        pos, mc, cov = pos[mask], mc[mask], cov[mask]
        sig = cov - mc  # unmethylated count = ATAC-like signal
        bed_path = os.path.join(output, f'{name}.{chrom}.pseudo.bed')
        with open(bed_path, 'w') as fh:
            n = _write_pseudo_reads(fh, chrom, pos, sig, half)
        if n == 0:
            os.remove(bed_path)
            bed_path = None
        ctrl_path = None
        nc = 0
        if control is not None:
            csig = cov.copy() if control == 'cov' else mc.copy()
            ctrl_path = os.path.join(output, f'{name}.{chrom}.control.bed')
            with open(ctrl_path, 'w') as cfh:
                nc = _write_pseudo_reads(cfh, chrom, pos, csig, half)
            if nc == 0:
                os.remove(ctrl_path)
                ctrl_path = None
        return chrom, bed_path, n, ctrl_path, nc
    finally:
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()


def _gen_pseudo_reads(reader, ref_reader, index_reader, cz_path, ref_path,
                      index, data_dtype, ref_dtype, mc_col, cov_col, min_cov,
                      half, control, bed_path, ctrl_bed_path, output, name,
                      jobs):
    """Materialise pseudo-reads into ``bed_path`` (+ optional control BED).

    Serial (``jobs<=1``) walks chromosomes on the already-open readers;
    parallel (``jobs>1``) closes them and fans per-chromosome generation out to
    a process pool, then concatenates the per-chrom BED pieces. All readers are
    closed before returning. Returns ``(total_reads, total_ctrl)``.
    """
    dims = list(reader.chunk_key2offset)
    if jobs and jobs > 1:
        idx_path = (os.path.abspath(os.path.expanduser(index))
                    if index else None)
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        wargs = dict(cz_path=cz_path, ref_path=ref_path, index_path=idx_path,
                     mc_col=mc_col, cov_col=cov_col, min_cov=min_cov,
                     half=half, control=control, output=output, name=name)
        results = {}
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_pseudo_reads_chrom_worker, dim, **wargs): dim
                    for dim in dims}
            for fut in as_completed(futs):
                res = fut.result()
                results[res[0]] = res
        total_reads = total_ctrl = 0
        bed_pieces, ctrl_pieces = [], []
        for dim in dims:
            res = results.get(dim[0])
            if not res:
                continue
            _, bp, n, cp, nc = res
            total_reads += n
            total_ctrl += nc
            if bp:
                bed_pieces.append(bp)
            if cp:
                ctrl_pieces.append(cp)
        _concat_files(bed_pieces, bed_path)
        if ctrl_bed_path is not None:
            _concat_files(ctrl_pieces, ctrl_bed_path)
    else:
        total_reads = total_ctrl = 0
        ctrl_fh = open(ctrl_bed_path, 'w') if ctrl_bed_path else None
        try:
            with open(bed_path, 'w') as fh:
                for dim in dims:
                    if dim not in ref_reader.chunk_key2offset:
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
                    if index_reader is not None and dim in index_reader.chunk_key2offset:
                        ids = index_reader.get_ids_from_index(dim)
                        if len(ids.shape) == 1:
                            data_arr = data_arr[ids]
                            ref_arr = ref_arr[ids]
                    pos = ref_arr['pos'].astype(np.int64)
                    mc = data_arr[mc_col].astype(np.int32)
                    cov = data_arr[cov_col].astype(np.int32)
                    mask = cov >= min_cov
                    pos, mc, cov = pos[mask], mc[mask], cov[mask]
                    sig = cov - mc  # unmethylated count = ATAC-like signal
                    total_reads += _write_pseudo_reads(fh, chrom, pos, sig, half)
                    if ctrl_fh is not None:
                        csig = cov.copy() if control == 'cov' else mc.copy()
                        total_ctrl += _write_pseudo_reads(
                            ctrl_fh, chrom, pos, csig, half)
                    logger.debug(f"  {chrom}: {len(pos)} sites")
                    reader.release_chunk(dim)
                    ref_reader.release_chunk(dim)
        finally:
            if ctrl_fh is not None:
                ctrl_fh.close()
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()
    logger.info(f"Total pseudo-reads: {total_reads}"
                + (f"; control reads: {total_ctrl}" if control else ""))
    return total_reads, total_ctrl


def call_peaks(input=None, reference=None, output=None, name='peaks',
               control=None, index=None, genome_size='mm',
               fragment_size=300, qvalue=0.05, broad=False,
               min_cov=1, keep_bed=False, macs3_args='',
               mc_col=None, cov_col=None, jobs=1):
    """Call peaks from a methylation .cz file using MACS3 (pseudo-read route).

    Peak calling on a .cz always treats the **unmethylated count**
    ``umc = cov - mc`` at each cytosine as the ATAC-like read signal: CpG
    hypomethylation marks open chromatin / regulatory elements, so only
    unmethylation (not ``mc``) is a meaningful peak signal here.

    Mechanism (differs from :func:`call_peaks_bdg`): this function
    **materialises pseudo-reads**. For each site with ``umc = u`` it writes
    ``u`` BED intervals of length ``fragment_size`` centred on the site, then
    runs ``macs3 callpeak --nomodel`` on that BED exactly like an ATAC-seq
    experiment. MACS3's full model (local lambda, fragment pileup, q-value
    FDR) is therefore used unchanged. The cost is that the BED holds
    ``sum(umc)`` reads, so runtime and disk grow with sequencing depth and
    it can explode on deep pseudobulks — use :func:`call_peaks_bdg` there.

    Useful for identifying regions of low methylation (e.g. open chromatin
    in NOMe-seq, or regulatory elements in WGBS).

    Because ``cov - mc`` is confounded by sequencing depth and cytosine
    density, pass ``control='cov'`` to also emit a coverage-based control
    track (``macs3 -c``): peaks then reflect a genuine **local enrichment of
    unmethylation** (``umc/cov`` above the global rate) rather than merely
    deeply covered / CpG-dense regions. Strongly recommended.

    See Also
    --------
    call_peaks_bdg : memory-efficient equivalent that never materialises
        reads; builds pileup bedGraphs directly and scores them with MACS3's
        bedGraph back-end (``bdgcmp`` Poisson test + ``bdgpeakcall``). Prefer
        it for deep pseudobulks; results are comparable but not identical
        because the scoring model differs (see its docstring).

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
    control : str or None
        Control (input) track fed to ``macs3 -c`` to correct for the
        coverage / cytosine-density bias in the ``umc`` signal. ``'cov'`` uses
        total coverage (recommended), ``'mc'`` uses the methylated count,
        ``None`` (default) calls peaks against MACS3's genome background only
        (the original, less specific behaviour).
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
        0-based column index or column name (from the .cz
        ``header['columns']``) for the methylation count.
        Defaults to the first data column (index 0, typically ``'mc'``).
    cov_col : int or str or None
        0-based column index or column name (from the .cz
        ``header['columns']``) for the coverage count.
        Defaults to the last data column (index -1, typically ``'cov'``).
    jobs : int
        Worker processes for pseudo-read generation and ``sort`` (default 1).
        Generation is fanned out one chromosome per process and ``sort`` gets
        ``--parallel jobs``. The ``macs3 callpeak`` step itself is
        single-threaded and is not sped up by this.

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

    jobs = int(jobs) if jobs else 1

    # ---- Step 1: Open the data .cz (mc/cov) and the reference .cz (pos/strand/context) ----
    cz_path = os.path.abspath(os.path.expanduser(input))
    ref_path = os.path.abspath(os.path.expanduser(reference))

    reader = Reader(cz_path)       # per-cell methylation data (mc, cov)
    ref_reader = Reader(ref_path)  # shared genomic coordinates (pos, strand, context)
    # Sequential whole-file walks ahead; tell the kernel to evict
    # already-read pages so the multi-GB ref doesn't pin our RSS.
    reader.advise_sequential()
    ref_reader.advise_sequential()

    # call_peaks produces a single aggregate peak set from one track. A
    # catcz'd multi-cell file is N tracks, not one; pooling sparse single
    # cells into a peak call is not meaningful. Ask the user to build a
    # pseudobulk track first (merge_cz sums cells into a single-dim .cz).
    if len(reader.header['chunk_dims']) > 1:
        reader.close()
        ref_reader.close()
        raise ValueError(
            "call_peaks expects a single-track .cz (chunk_dims=['chrom']); "
            f"got a catcz'd multi-cell file (chunk_dims="
            f"{reader.header['chunk_dims']}). Build a pseudobulk track first "
            "with `merge_cz`, then call peaks on that.")

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

    if control is not None and control not in ('cov', 'mc'):
        raise ValueError(
            f"control must be 'cov', 'mc', or None; got {control!r}")
    ctrl_bed_path = (os.path.join(output, f'{name}.control_reads.bed')
                     if control is not None else None)

    sorted_bed = bed_path.replace('.bed', '.sorted.bed')
    sorted_ctrl = (ctrl_bed_path.replace('.bed', '.sorted.bed')
                   if ctrl_bed_path is not None else None)

    # Reuse existing sorted BED(s) instead of regenerating them.
    if os.path.exists(sorted_bed) and (sorted_ctrl is None
                                       or os.path.exists(sorted_ctrl)):
        logger.info(
            f"Sorted BED already exists: {sorted_bed}"
            + (f", {sorted_ctrl}" if sorted_ctrl else "")
            + "; skip pseudo-read generation and sorting.")
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()
    else:
        _gen_pseudo_reads(
            reader, ref_reader, index_reader, cz_path, ref_path, index,
            data_dtype, ref_dtype, mc_col, cov_col, min_cov, half, control,
            bed_path, ctrl_bed_path, output, name, jobs)

        # ---- Step 4: Sort BED (sort uses jobs threads) ----
        logger.info("Sorting BED file...")
        sort_cmd = ['sort', '-k1,1', '-k2,2n']
        if jobs and jobs > 1:
            sort_cmd += ['--parallel', str(jobs)]
        sort_cmd += [bed_path, '-o', sorted_bed]
        logger.info(f"Running: {' '.join(sort_cmd)}")
        subprocess.run(sort_cmd, check=True)
        if sorted_ctrl is not None:
            sort_ctrl_cmd = ['sort', '-k1,1', '-k2,2n']
            if jobs and jobs > 1:
                sort_ctrl_cmd += ['--parallel', str(jobs)]
            sort_ctrl_cmd += [ctrl_bed_path, '-o', sorted_ctrl]
            logger.info(f"Running: {' '.join(sort_ctrl_cmd)}")
            subprocess.run(sort_ctrl_cmd, check=True)

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
    if sorted_ctrl is not None:
        cmd.extend(['-c', sorted_ctrl])
    if broad:
        cmd.append('--broad')
    if macs3_args:
        cmd.extend(macs3_args.split())

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if os.path.exists(bed_path):
        os.remove(bed_path)
    if ctrl_bed_path is not None and os.path.exists(ctrl_bed_path):
        os.remove(ctrl_bed_path)
    if not keep_bed:
        if os.path.exists(sorted_bed):
            os.remove(sorted_bed)
        if sorted_ctrl is not None and os.path.exists(sorted_ctrl):
            os.remove(sorted_ctrl)
    else:
        logger.info(f"Pseudo-reads BED kept at: {sorted_bed}")
        if sorted_ctrl is not None:
            logger.info(f"Control pseudo-reads BED kept at: {sorted_ctrl}")

    logger.info(f"Peak results saved to: {output}")
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
        0-based column index or column name (from the .cz
        ``header['columns']``) for the methylation count.
        Defaults to the first data column (index 0, typically ``'mc'``).
    cov_col : int or str or None
        0-based column index or column name (from the .cz
        ``header['columns']``) for the coverage count.
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
    # Sequential whole-file walk; release pages as we go.
    reader.advise_sequential()
    ref_reader.advise_sequential()

    # to_bedgraph emits a single track (unique sorted positions). A catcz'd
    # multi-cell file holds many cells per chromosome, so pooling them would
    # produce duplicated positions. Build a pseudobulk track first.
    if len(reader.header['chunk_dims']) > 1:
        reader.close()
        ref_reader.close()
        raise ValueError(
            "to_bedgraph expects a single-track .cz (chunk_dims=['chrom']); "
            f"got a catcz'd multi-cell file (chunk_dims="
            f"{reader.header['chunk_dims']}). Build a pseudobulk track first "
            "with `merge_cz`, then export the bedGraph.")

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
        for dim in reader.chunk_key2offset:
            if dim not in ref_reader.chunk_key2offset:
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

            if index_reader is not None and dim in index_reader.chunk_key2offset:
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

            # Release this chunk's pages on both readers.
            reader.release_chunk(dim)
            ref_reader.release_chunk(dim)

    reader.close()
    ref_reader.close()
    if index_reader:
        index_reader.close()

    logger.info(f"bedGraph written to: {output}")
    return output


def _pileup_from_sites(pos, sig, half):
    """Piecewise-constant coverage pileup from extended point signals.

    Each site at ``pos`` contributes ``sig`` over ``[pos-half, pos+half)``;
    overlapping contributions are summed via a difference array (memory is
    ``O(n_sites)``, never ``O(sum(sig))``). Returns ``(starts, ends, values)``
    for the nonzero bedGraph segments, or ``None`` when empty.
    """
    pos = np.asarray(pos, dtype=np.int64)
    sig = np.asarray(sig, dtype=np.float64)
    keep = sig != 0
    pos, sig = pos[keep], sig[keep]
    if pos.size == 0:
        return None
    starts = np.maximum(0, pos - half)
    ends = pos + half
    idx = np.concatenate([starts, ends])
    delta = np.concatenate([sig, -sig])
    order = np.argsort(idx, kind='mergesort')
    idx, delta = idx[order], delta[order]
    uniq, inv = np.unique(idx, return_inverse=True)
    agg = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(agg, inv, delta)
    cum = np.cumsum(agg)
    seg_start, seg_end, seg_val = uniq[:-1], uniq[1:], cum[:-1]
    nz = seg_val > 0
    if not nz.any():
        return None
    return seg_start[nz], seg_end[nz], seg_val[nz]


def _pileup_control(pos, sig, half, floor=1.0):
    """Continuous coverage pileup for the control lambda track.

    Same difference-array pileup as :func:`_pileup_from_sites`, but returns a
    **gap-free** track over ``[0, max(pos)+half]`` with every segment floored
    at ``floor`` (> 0). ``macs3 bdgcmp -m ppois`` evaluates the Poisson score
    at every base and requires the lambda to be strictly positive there; a
    sparse (zero-gap) control makes ``ppois`` abort with
    ``AssertionError: Lambda must > 0``. Returns ``(starts, ends, values)`` or
    ``None`` when empty.
    """
    pos = np.asarray(pos, dtype=np.int64)
    sig = np.asarray(sig, dtype=np.float64)
    keep = sig != 0
    pos, sig = pos[keep], sig[keep]
    if pos.size == 0:
        return None
    starts = np.maximum(0, pos - half)
    ends = pos + half
    idx = np.concatenate([starts, ends])
    delta = np.concatenate([sig, -sig])
    order = np.argsort(idx, kind='mergesort')
    idx, delta = idx[order], delta[order]
    uniq, inv = np.unique(idx, return_inverse=True)
    agg = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(agg, inv, delta)
    cum = np.cumsum(agg)
    seg_start, seg_end, seg_val = uniq[:-1], uniq[1:], cum[:-1]
    # Prepend the leading [0, uniq[0]) gap so the track starts at base 0.
    if uniq[0] > 0:
        seg_start = np.concatenate([[0], seg_start])
        seg_end = np.concatenate([[uniq[0]], seg_end])
        seg_val = np.concatenate([[0.0], seg_val])
    # Floor internal gaps (cum == 0) and the leading gap so lambda is never 0.
    seg_val = np.maximum(seg_val, floor)
    return seg_start, seg_end, seg_val


def _joint_pileup(pos, sig, csig, half):
    """Unified piecewise-constant pileup of treatment and control signals.

    Each site at ``pos`` spreads its ``sig`` (treatment) and ``csig`` (control)
    counts over ``[pos-half, pos+half)``; overlapping contributions are summed
    with difference arrays over a single shared breakpoint set, so treatment
    and control are returned already aligned on the *same* segmentation (memory
    ``O(n_sites)``). Returns ``(seg_start, seg_end, sig_val, cov_val)`` or
    ``None`` when empty.
    """
    pos = np.asarray(pos, dtype=np.int64)
    sig = np.asarray(sig, dtype=np.float64)
    csig = np.asarray(csig, dtype=np.float64)
    if pos.size == 0:
        return None
    starts = np.maximum(0, pos - half)
    ends = pos + half
    pts = np.concatenate([starts, ends])
    d_sig = np.concatenate([sig, -sig])
    d_cov = np.concatenate([csig, -csig])
    order = np.argsort(pts, kind='mergesort')
    pts = pts[order]
    d_sig = d_sig[order]
    d_cov = d_cov[order]
    uniq, inv = np.unique(pts, return_inverse=True)
    a_sig = np.zeros(uniq.size, dtype=np.float64)
    a_cov = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(a_sig, inv, d_sig)
    np.add.at(a_cov, inv, d_cov)
    cum_sig = np.cumsum(a_sig)[:-1]
    cum_cov = np.cumsum(a_cov)[:-1]
    return uniq[:-1], uniq[1:], cum_sig, cum_cov


def _peak_score(treat, lam, method):
    """Per-segment enrichment score of ``treat`` pileup against ``lam``.

    ``'ppois'`` returns the MACS3-equivalent ``-log10 P(X >= treat)`` for
    ``X ~ Poisson(lam)`` (upper tail), computed vectorised via scipy's
    ``poisson.logsf`` (``gammaincc`` under the hood: ``O(1)`` per element
    regardless of pileup depth — this is what replaces the slow per-segment
    ``macs3 bdgcmp``). ``'FE'`` returns the linear fold enrichment
    ``treat / lam``.
    """
    if method == 'ppois':
        from scipy.stats import poisson
        # P(X >= k) = P(X > k-1) = sf(k-1); treat is integer-valued.
        sc = -poisson.logsf(treat - 1, lam) / np.log(10.0)
        sc[~np.isfinite(sc)] = 3100.0  # cap extreme significance
        return sc
    if method in ('FE', 'fe'):
        return treat / np.maximum(lam, 1e-9)
    raise ValueError(
        "native call_peaks_bdg supports method 'ppois' or 'FE'; "
        f"got {method!r}")


def _peaks_from_scores(chrom, seg_s, seg_e, score, treat, lam,
                       cutoff, min_len, max_gap):
    """Threshold scored segments into peaks (per chromosome).

    Segments with ``score >= cutoff`` are merged across coordinate gaps
    ``<= max_gap``; runs shorter than ``min_len`` are dropped. For each peak
    the summit is the max-score segment, from which fold enrichment and the
    summit offset are recorded. Returns a DataFrame (columns ``chrom, start,
    end, summit_score, fe, offset``) or ``None`` when no peak passes.
    """
    on = score >= cutoff
    if not on.any():
        return None
    s = seg_s[on]
    e = seg_e[on]
    sc = score[on]
    tv = treat[on]
    lv = lam[on]
    # New peak wherever the gap to the previous on-segment exceeds max_gap.
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    if s.size > 1:
        new[1:] = (s[1:] - e[:-1]) > max_gap
    grp = np.cumsum(new) - 1
    df = pd.DataFrame({'grp': grp, 's': s, 'e': e, 'sc': sc,
                       'tv': tv, 'lv': lv, 'mid': (s + e) // 2})
    g = df.groupby('grp', sort=False)
    start = g['s'].min().to_numpy()
    end = g['e'].max().to_numpy()
    summit = df.loc[g['sc'].idxmax().to_numpy()]
    ssc = summit['sc'].to_numpy()
    smid = summit['mid'].to_numpy()
    stv = summit['tv'].to_numpy()
    slv = summit['lv'].to_numpy()
    keep = (end - start) >= min_len
    if not keep.any():
        return None
    start, end = start[keep], end[keep]
    ssc, smid = ssc[keep], smid[keep]
    stv, slv = stv[keep], slv[keep]
    return pd.DataFrame({
        'chrom': chrom,
        'start': start.astype(np.int64),
        'end': end.astype(np.int64),
        'summit_score': ssc,
        'fe': stv / np.maximum(slv, 1e-9),
        'offset': np.maximum(0, smid - start).astype(np.int64),
    })


def _local_rate(seg_s, seg_e, tval, cval, r_global, llocal, binsize=1000):
    """Per-segment local background unmeth rate over a +/-``llocal`` window.

    Methylation analogue of MACS3's dynamic local lambda. The umc (``tval``)
    and coverage (``cval``) pileups are integrated into ``binsize`` bins, a
    moving window of width ~``llocal`` gives local ``sum(umc)/sum(cov)``, and
    the result is floored at the genome-wide rate ``r_global`` (background is
    never below genome-wide). Scoring a region against this instead of the
    global rate suppresses peaks that merely sit inside broadly hypomethylated
    domains. ``seg_*`` are the *contiguous* joint-pileup segments, so their
    boundaries form a strictly increasing breakpoint set usable by ``interp``.
    """
    L = int(seg_e[-1])
    if L <= 0 or llocal <= 0:
        return np.full(seg_s.shape, r_global, dtype=np.float64)
    n = L // binsize + 1
    # Cumulative bp-weighted integrals at segment breakpoints, then sampled at
    # bin edges (piecewise-linear within a segment) -> per-bin integrals.
    xb = np.empty(seg_s.size + 1, dtype=np.int64)
    xb[0] = seg_s[0]
    xb[1:] = seg_e
    seglen = (seg_e - seg_s).astype(np.float64)
    Fu = np.concatenate([[0.0], np.cumsum(tval * seglen)])
    Fc = np.concatenate([[0.0], np.cumsum(cval * seglen)])
    edges = np.arange(0, (n + 1) * binsize, binsize, dtype=np.int64)
    u_bin = np.diff(np.interp(edges, xb, Fu))
    c_bin = np.diff(np.interp(edges, xb, Fc))
    w = max(1, int(round(llocal / binsize)))
    cu = np.concatenate([[0.0], np.cumsum(u_bin)])
    cc = np.concatenate([[0.0], np.cumsum(c_bin)])
    nb = u_bin.size
    ar = np.arange(nb)
    lo = np.clip(ar - w // 2, 0, nb)
    hi = np.clip(ar + w // 2 + 1, 0, nb)
    u_win = cu[hi] - cu[lo]
    c_win = cc[hi] - cc[lo]
    rate_bin = np.where(c_win > 0, u_win / np.maximum(c_win, 1e-9), r_global)
    rate_bin = np.maximum(rate_bin, r_global)
    bidx = np.clip(((seg_s + seg_e) // 2) // binsize, 0, nb - 1)
    return rate_bin[bidx]


def _blacklist_mask(df, bl_path):
    """Boolean per peak-row: overlaps any blacklist interval (per chrom)."""
    bl = pd.read_csv(bl_path, sep='\t', header=None, comment='#',
                     usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
    df = df.reset_index(drop=True)
    mask = np.zeros(len(df), dtype=bool)
    bg = {c: (g['start'].to_numpy(), g['end'].to_numpy())
          for c, g in bl.groupby('chrom', sort=False)}
    for chrom, ga in df.groupby('chrom', sort=False):
        gb = bg.get(chrom)
        if gb is None:
            continue
        bs, be = gb
        o = np.argsort(bs, kind='mergesort')
        bs, be = bs[o], be[o]
        cend = np.maximum.accumulate(be)
        k = np.searchsorted(bs, ga['end'].to_numpy(), side='left')
        hit = k > 0
        idx = np.clip(k - 1, 0, len(bs) - 1)
        res = np.zeros(len(ga), dtype=bool)
        res[hit] = cend[idx[hit]] > ga['start'].to_numpy()[hit]
        mask[ga.index.to_numpy()] = res
    return mask


def _bdg_chrom_worker(dim, cz_path, ref_path, index_path, mc_col, cov_col,
                      min_cov, half, control, r, method, cutoff, min_len,
                      max_gap, keep_bdg, output, name, llocal=10000):
    """Worker: pileup, score and call peaks for one chromosome.

    Opens its own readers (process-safe), builds the joint umc/coverage pileup
    for ``dim``, scores each segment against the global-rate lambda and
    thresholds it into peaks. Returns ``(chrom, frame_or_None,
    treat_bdg_or_None, score_bdg_or_None)``; the bdg pieces are written only
    when ``keep_bdg``. Used by :func:`call_peaks_bdg`.
    """
    reader = Reader(cz_path)
    ref_reader = Reader(ref_path)
    index_reader = Reader(index_path) if index_path else None
    try:
        chrom = dim[0]
        data_dtype = _make_np_dtype(reader.header['formats'],
                                    reader.header['columns'])
        ref_dtype = _make_np_dtype(ref_reader.header['formats'],
                                   ref_reader.header['columns'])
        raw = reader.fetch_chunk_bytes(dim)
        if not raw:
            return chrom, None, None, None
        data_arr = np.frombuffer(raw, dtype=data_dtype)
        ref_raw = ref_reader.fetch_chunk_bytes(dim)
        if not ref_raw:
            return chrom, None, None, None
        ref_arr = np.frombuffer(ref_raw, dtype=ref_dtype)
        if index_reader is not None and dim in index_reader.chunk_key2offset:
            ids = index_reader.get_ids_from_index(dim)
            if len(ids.shape) == 1:
                data_arr = data_arr[ids]
                ref_arr = ref_arr[ids]
        pos = ref_arr['pos'].astype(np.int64)
        mc = data_arr[mc_col].astype(np.int64)
        cov = data_arr[cov_col].astype(np.int64)
        mask = cov >= min_cov
        pos, mc, cov = pos[mask], mc[mask], cov[mask]
        sig = cov - mc  # unmethylated count = ATAC-like signal
        csig = cov if control == 'cov' else mc
        jp = _joint_pileup(pos, sig, csig, half)
        if jp is None:
            return chrom, None, None, None
        seg_s, seg_e, tval, cval = jp
        # Local background rate on the full (contiguous) pileup, then subset.
        rate_full = _local_rate(seg_s, seg_e, tval, cval, r, llocal)
        keep = tval > 0  # peaks can only live where umc pileup > 0
        if not keep.any():
            return chrom, None, None, None
        seg_s, seg_e = seg_s[keep], seg_e[keep]
        tval, cval = tval[keep], cval[keep]
        lam = np.maximum(cval, 1.0) * rate_full[keep]  # expected umc pileup
        score = _peak_score(tval, lam, method)
        treat_path = score_path = None
        if keep_bdg:
            treat_path = os.path.join(output, f'{name}.{chrom}.treat.bdg')
            score_path = os.path.join(output, f'{name}.{chrom}.{method}.bdg')
            pd.DataFrame({'c': chrom, 's': seg_s, 'e': seg_e,
                          'v': tval.astype(np.int64)}).to_csv(
                treat_path, sep='\t', header=False, index=False)
            pd.DataFrame({'c': chrom, 's': seg_s, 'e': seg_e,
                          'v': np.round(score, 5)}).to_csv(
                score_path, sep='\t', header=False, index=False)
        frame = _peaks_from_scores(chrom, seg_s, seg_e, score, tval, lam,
                                   cutoff, min_len, max_gap)
        return chrom, frame, treat_path, score_path
    finally:
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()


def call_peaks_bdg(input=None, reference=None, output=None, name='peaks',
                   control='cov', index=None,
                   ext=300, method='ppois', cutoff=2,
                   min_len=None, max_gap=None, min_cov=1, llocal=10000,
                   blacklist=None, keep_bdg=False, mc_col=None, cov_col=None,
                   jobs=1):
    """Call peaks from a methylation .cz (native vectorised Poisson).

    Like :func:`call_peaks`, the signal is always the **unmethylated count**
    ``umc = cov - mc`` (CpG hypomethylation = open chromatin); ``mc`` is never
    used as signal.

    Mechanism (differs from :func:`call_peaks`): instead of materialising
    ``sum(umc)`` pseudo-reads (which explodes on deep pseudobulks), each
    site's count is spread over ``ext`` bp and overlapping contributions
    summed with a difference array, giving a piecewise-constant pileup with
    memory ``O(n_sites)`` regardless of depth. The Poisson score and peak
    calling are then done **directly in numpy/scipy per chromosome** — no
    multi-GB bedGraphs are written and no ``macs3`` subprocess is spawned.
    (Earlier versions drove ``macs3 bdgopt``/``bdgcmp``/``bdgpeakcall``; on
    deep pseudobulks ``bdgcmp -m ppois`` scored ~10^8 segments one at a time
    and took hours. The vectorised ``poisson.logsf`` here is ``O(1)`` per
    segment and finishes in minutes.)

    Scoring model. The **treatment** track is the ``umc`` pileup. The
    **control (lambda)** track is the coverage pileup scaled by a background
    unmethylation rate. By default (``llocal > 0``) that rate is estimated
    **locally** in a +/-``llocal`` window (``sum(umc)/sum(cov)``), floored at
    the genome-wide rate ``r = sum(umc)/sum(cov)`` — the methylation analogue
    of MACS3's dynamic local lambda. This is important: CpG umc has a small
    dynamic range (open chromatin is rarely >2x the global unmeth rate), so a
    single *global* lambda over-calls badly on deep pseudobulks — a mere
    1.2-1.5x enrichment yields ``-log10 p`` in the hundreds, and vast
    stretches of broadly hypomethylated (e.g. intergenic) domains pass. The
    local lambda rejects those by comparing each region to its own
    neighbourhood. Set ``llocal=0`` to fall back to the old global-only
    lambda. For ``method='ppois'`` each segment is scored by
    ``-log10 P(X >= umc_pileup | Poisson(lambda))`` (upper tail);
    ``bdgpeakcall``-style thresholding then merges segments with
    ``score >= cutoff`` across gaps ``<= max_gap`` and drops runs shorter than
    ``min_len``. Peaks are thus regions of genuine **local unmethylation
    enrichment**, corrected for coverage / cytosine-density bias.

    .. note::
       The default ``cutoff=50`` and ``llocal=10000`` were tuned by
       benchmarking against matched ATAC-seq peaks (precision/recall/Jaccard
       vs open chromatin); they roughly maximise agreement on deep snm3C
       pseudobulks. Because the fold-enrichment (``signalValue``) rarely
       exceeds ~2x, ``method='FE'`` cutoffs must instead be small (e.g.
       1.2-1.4). For ``ppois`` the ``-log10 p`` is depth-inflated, so raise
       ``cutoff`` (e.g. 60-100) for stricter, higher-precision peaks or lower
       it (e.g. 20-30) for higher recall; pairing with a ``blacklist`` further
       improves reliability.

    How this differs from :func:`call_peaks` in practice:

    - **Memory/scaling**: ``O(n_sites)`` here vs ``O(sum(umc))`` reads there;
      this route is the one to use for deep pseudobulks.
    - **Statistical model**: an explicit single-lambda Poisson test against a
      global-rate expectation (this function) vs MACS3's native ``callpeak``
      pipeline with its dynamic local lambda and q-value FDR
      (:func:`call_peaks`). The two give comparable but not bit-identical
      peak sets.
    - **Control is mandatory** here (the expected-signal lambda is the whole
      point), whereas :func:`call_peaks` can optionally run without one.

    Parameters
    ----------
    input, reference, output, name, index, min_cov, mc_col, cov_col :
        As in :func:`call_peaks` (``output`` defaults to
        ``<input_stem>_peaks/``).
    control : str
        Coverage-bias control track: ``'cov'`` (default) or ``'mc'``.
    ext : int
        Extension (bp) each site's count is spread over (default 300).
    method : str
        Score method: ``'ppois'`` (default, ``-log10`` Poisson p-value) or
        ``'FE'`` (linear fold enrichment ``treat/lambda``).
    cutoff : float
        Score cutoff. For ``ppois`` this is ``-log10(p)`` (default 50, tuned
        against ATAC); for ``FE`` it is the minimum fold enrichment.
    min_len : int or None
        Minimum peak length; ``None`` -> ``ext``.
    max_gap : int or None
        Maximum gap to merge nearby peaks; ``None`` -> ``ext // 2``.
    llocal : int
        Half-width (bp) of the local-background window for the dynamic lambda
        (default 10000). ``0`` disables it and uses the genome-wide rate only
        (the old behaviour, which over-calls on deep data).
    blacklist : str or None
        Optional BED(.gz) of blacklist regions (e.g. ENCODE); peaks
        overlapping any interval are dropped and the survivors renumbered.
    keep_bdg : bool
        Also write the treatment pileup and score bedGraphs (for genome-
        browser inspection); default False.
    jobs : int
        Worker processes for the per-chromosome pileup/score/peak-call pass
        (default 1). Chromosomes are independent and CPU-bound, so this scales
        near-linearly; each worker holds one chromosome's arrays in memory, so
        raise ``jobs`` only as far as RAM allows (the largest chromosomes are
        the heaviest). Pass 1 (global rate) stays serial and cheap.

    Returns
    -------
    str
        Path to the output ``<name>_peaks.narrowPeak`` file.
    """
    if control not in ('cov', 'mc'):
        raise ValueError(f"control must be 'cov' or 'mc'; got {control!r}")

    jobs = int(jobs) if jobs else 1

    cz_path = os.path.abspath(os.path.expanduser(input))
    ref_path = os.path.abspath(os.path.expanduser(reference))
    reader = Reader(cz_path)
    ref_reader = Reader(ref_path)
    reader.advise_sequential()
    ref_reader.advise_sequential()

    if len(reader.header['chunk_dims']) > 1:
        reader.close()
        ref_reader.close()
        raise ValueError(
            "call_peaks_bdg expects a single-track .cz (chunk_dims=['chrom']); "
            f"got a catcz'd multi-cell file (chunk_dims="
            f"{reader.header['chunk_dims']}). Build a pseudobulk track first "
            "with `merge_cz`, then call peaks on that.")

    if output is None:
        output = os.path.splitext(cz_path)[0] + '_peaks'
    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(output, exist_ok=True)

    index_reader = None
    if index is not None:
        index_reader = Reader(os.path.abspath(os.path.expanduser(index)))

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

    half = ext // 2
    if min_len is None:
        min_len = ext
    if max_gap is None:
        max_gap = ext // 2
    narrowpeak = os.path.join(output, f'{name}_peaks.narrowPeak')

    # ---- Pass 1: global unmethylation rate r = sum(umc) / sum(cov) ----
    # r scales the coverage pileup into the expected-signal (lambda) track.
    total_sig = 0.0
    total_cov = 0.0
    for dim in reader.chunk_key2offset:
        if dim not in ref_reader.chunk_key2offset:
            continue
        raw = reader.fetch_chunk_bytes(dim)
        if not raw:
            continue
        data_arr = np.frombuffer(raw, dtype=data_dtype)
        if index_reader is not None and dim in index_reader.chunk_key2offset:
            ids = index_reader.get_ids_from_index(dim)
            if len(ids.shape) == 1:
                data_arr = data_arr[ids]
        mc = data_arr[mc_col].astype(np.int64)
        cov = data_arr[cov_col].astype(np.int64)
        mask = cov >= min_cov
        mc, cov = mc[mask], cov[mask]
        total_sig += float((cov - mc).sum())
        total_cov += float((cov if control == 'cov' else mc).sum())
        reader.release_chunk(dim)

    if total_sig <= 0 or total_cov <= 0:
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()
        raise ValueError(
            "no signal after filtering; check context index / min_cov.")
    r = total_sig / total_cov
    logger.info(f"global unmeth rate r = {r:.4g} (lambda = cov pileup * r)")

    # ---- Pass 2: per-chromosome pileup -> Poisson score -> peak calls ----
    # Each chromosome is independent (pileup + vectorised scipy.poisson.logsf),
    # so with jobs > 1 they run in a process pool; the parent readers are
    # closed first and every worker opens its own. The multi-GB bedGraphs and
    # the slow ``macs3 bdgcmp`` pass are gone either way.
    dims = [d for d in reader.chunk_key2offset
            if d in ref_reader.chunk_key2offset]
    idx_path = os.path.abspath(os.path.expanduser(index)) if index else None
    reader.close()
    ref_reader.close()
    if index_reader:
        index_reader.close()

    wargs = dict(cz_path=cz_path, ref_path=ref_path, index_path=idx_path,
                 mc_col=mc_col, cov_col=cov_col, min_cov=min_cov, half=half,
                 control=control, r=r, method=method, cutoff=cutoff,
                 min_len=min_len, max_gap=max_gap, keep_bdg=keep_bdg,
                 output=output, name=name, llocal=llocal)

    results = {}
    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_bdg_chrom_worker, dim, **wargs): dim
                    for dim in dims}
            for fut in as_completed(futs):
                chrom, frame, tp, sp = fut.result()
                results[chrom] = (frame, tp, sp)
    else:
        for dim in dims:
            chrom, frame, tp, sp = _bdg_chrom_worker(dim, **wargs)
            results[chrom] = (frame, tp, sp)
            logger.debug(f"  {chrom}: done")

    peak_frames = []
    treat_pieces, score_pieces = [], []
    for dim in dims:
        frame, tp, sp = results.get(dim[0], (None, None, None))
        if frame is not None and len(frame):
            peak_frames.append(frame)
        if tp:
            treat_pieces.append(tp)
        if sp:
            score_pieces.append(sp)

    if keep_bdg:
        _concat_files(treat_pieces,
                      os.path.join(output, f'{name}.treat.bdg'))
        _concat_files(score_pieces,
                      os.path.join(output, f'{name}.{method}.bdg'))

    # ---- Assemble the narrowPeak (BED6+4): clean header, real name/score ----
    cols = ['chrom', 'start', 'end', 'name', 'score', 'strand',
            'signalValue', 'pValue', 'qValue', 'peak']
    if peak_frames:
        allpk = pd.concat(peak_frames, ignore_index=True)
        n = len(allpk)
        out = pd.DataFrame({
            'chrom': allpk['chrom'].to_numpy(),
            'start': allpk['start'].to_numpy(),
            'end': allpk['end'].to_numpy(),
            'name': [f'{name}_peak_{i}' for i in range(1, n + 1)],
            # UCSC display score, 0-1000; the real value is in pValue below.
            'score': np.minimum(
                (allpk['summit_score'].to_numpy() * 10).astype(np.int64), 1000),
            'strand': '.',
            'signalValue': np.round(allpk['fe'].to_numpy(), 5),
            'pValue': (np.round(allpk['summit_score'].to_numpy(), 5)
                       if method == 'ppois' else -1), # -log10 p-value
            'qValue': -1,  # genome-wide BH not computed on this route
            'peak': allpk['offset'].to_numpy(),
        })
    else:
        out = pd.DataFrame(columns=cols)
    if blacklist and len(out):
        bl_path = os.path.abspath(os.path.expanduser(blacklist))
        m = _blacklist_mask(out, bl_path)
        logger.info(f"blacklist: dropping {int(m.sum())} / {len(out)} peaks")
        out = out[~m].reset_index(drop=True)
        out['name'] = [f'{name}_peak_{i}' for i in range(1, len(out) + 1)]
    # No track line: a bare narrowPeak loads cleanly everywhere.
    out.to_csv(narrowpeak, sep='\t', header=True, index=False, columns=cols)
    logger.info(f"{len(out)} peaks written to: {narrowpeak}")
    return narrowpeak

