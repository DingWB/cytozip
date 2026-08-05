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


def call_peaks(input=None, reference=None, output=None, name='peaks',
               control=None, index=None, genome_size='mm',
               fragment_size=300, qvalue=0.05, broad=False,
               min_cov=1, keep_bed=False, macs3_args='',
               mc_col=None, cov_col=None):
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
        total_reads = 0
        total_ctrl = 0
        ctrl_fh = open(ctrl_bed_path, 'w') if ctrl_bed_path else None
        try:
            with open(bed_path, 'w') as fh:
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
                    mc = data_arr[mc_col].astype(np.int32)
                    cov = data_arr[cov_col].astype(np.int32)

                    mask = cov >= min_cov
                    pos = pos[mask]
                    mc = mc[mask]
                    cov = cov[mask]

                    sig = cov - mc  # unmethylated count = ATAC-like signal

                    total_reads += _write_pseudo_reads(fh, chrom, pos, sig, half)

                    if ctrl_fh is not None:
                        csig = cov.copy() if control == 'cov' else mc.copy()
                        total_ctrl += _write_pseudo_reads(
                            ctrl_fh, chrom, pos, csig, half)

                    logger.debug(f"  {chrom}: {len(pos)} sites")

                    # Release this chunk's pages on both readers.
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

        # ---- Step 4: Sort BED ----
        logger.info("Sorting BED file...")
        sort_cmd = ['sort', '-k1,1', '-k2,2n', bed_path, '-o', sorted_bed]
        logger.info(f"Running: {' '.join(sort_cmd)}")
        subprocess.run(sort_cmd, check=True)
        if sorted_ctrl is not None:
            sort_ctrl_cmd = ['sort', '-k1,1', '-k2,2n', ctrl_bed_path, '-o', sorted_ctrl]
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


def call_peaks_bdg(input=None, reference=None, output=None, name='peaks',
                   control='cov', index=None,
                   ext=300, method='ppois', cutoff=2.0,
                   min_len=None, max_gap=None, min_cov=1,
                   keep_bdg=False, mc_col=None, cov_col=None):
    """Call peaks from a methylation .cz via MACS3 bedGraph back-end.

    Like :func:`call_peaks`, the signal is always the **unmethylated count**
    ``umc = cov - mc`` (CpG hypomethylation = open chromatin); ``mc`` is never
    used as signal.

    Mechanism (differs from :func:`call_peaks`): instead of materialising
    ``sum(umc)`` pseudo-reads (which explodes on deep pseudobulks), this
    builds piecewise-constant **pileup bedGraphs** analytically — each site's
    count is spread over ``ext`` bp and overlapping contributions summed with
    a difference array, so memory is ``O(n_sites)`` regardless of depth — and
    drives MACS3's bedGraph back-end (``bdgcmp`` + ``bdgpeakcall``) rather than
    ``callpeak``.

    Scoring model. The **treatment** track is the ``umc`` pileup. The
    **control (lambda)** track is the coverage pileup scaled by the global
    unmethylation rate ``r = sum(umc) / sum(cov)`` (``control='cov'``) — i.e.
    the *expected* unmethylated pileup if methylation were spatially uniform.
    ``bdgcmp -m ppois`` then scores every position by the Poisson p-value of
    the observed ``umc`` pileup against that local expectation, and
    ``bdgpeakcall`` thresholds the score. Peaks are thus regions of genuine
    **local unmethylation enrichment**, corrected for coverage / cytosine-
    density bias.

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
        Coverage-bias control track: ``'cov'`` (default) or ``'mc'``. The
        bedGraph route always uses a control (that is its purpose).
    ext : int
        Extension (bp) each site's count is spread over (default 300).
    method : str
        ``macs3 bdgcmp`` score method (``'ppois'`` default, or ``'qpois'``,
        ``'FE'``, ``'logLR'``, ``'subtract'`` ...).
    cutoff : float
        ``bdgpeakcall`` score cutoff. For ``ppois`` this is ``-log10(p)``
        (default 2.0 -> p < 0.01); for ``qpois`` it is ``-log10(q)``.
    min_len : int or None
        Minimum peak length (``bdgpeakcall -l``); ``None`` -> ``ext``.
    max_gap : int or None
        Maximum gap to merge nearby peaks (``bdgpeakcall -g``); ``None`` ->
        ``ext // 2``.
    keep_bdg : bool
        Keep the intermediate bedGraph tracks (default False).

    Returns
    -------
    str
        Path to the output ``<name>_peaks.narrowPeak`` file.
    """
    import subprocess

    if control not in ('cov', 'mc'):
        raise ValueError(f"control must be 'cov' or 'mc'; got {control!r}")

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
    treat_bdg = os.path.join(output, f'{name}.treat.bdg')
    ctrl_bdg = os.path.join(output, f'{name}.control.bdg')

    def _write_seg(fh, chrom, seg):
        if seg is None:
            return
        s, e, v = seg
        pd.DataFrame({'chrom': chrom, 'start': s, 'end': e,
                      'value': v.astype(np.int64)}).to_csv(
            fh, sep='\t', header=False, index=False)

    lambda_bdg = os.path.join(output, f'{name}.control_lambda.bdg')
    score_bdg = os.path.join(output, f'{name}.{method}.bdg')
    narrowpeak = os.path.join(output, f'{name}_peaks.narrowPeak')
    if min_len is None:
        min_len = ext
    if max_gap is None:
        max_gap = ext // 2

    # Reuse existing pileup / lambda bedGraphs instead of regenerating them.
    r = None
    if (os.path.exists(treat_bdg) and os.path.exists(ctrl_bdg)
            and os.path.exists(lambda_bdg)):
        logger.info(
            f"bedGraph tracks already exist: {treat_bdg}, {ctrl_bdg}, "
            f"{lambda_bdg}; skip pileup generation.")
        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()
    else:
        total_sig = 0.0
        total_cov = 0.0
        with open(treat_bdg, 'w') as tf, open(ctrl_bdg, 'w') as cf:
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
                mc = data_arr[mc_col].astype(np.int64)
                cov = data_arr[cov_col].astype(np.int64)
                mask = cov >= min_cov
                pos, mc, cov = pos[mask], mc[mask], cov[mask]
                sig = cov - mc  # unmethylated count = ATAC-like signal
                csig = cov if control == 'cov' else mc
                total_sig += float(sig.sum())
                total_cov += float(csig.sum())
                _write_seg(tf, chrom, _pileup_from_sites(pos, sig, half))
                _write_seg(cf, chrom, _pileup_from_sites(pos, csig, half))
                reader.release_chunk(dim)
                ref_reader.release_chunk(dim)

        reader.close()
        ref_reader.close()
        if index_reader:
            index_reader.close()

        if total_sig <= 0 or total_cov <= 0:
            raise ValueError(
                "no signal after filtering; check context index / min_cov.")
        r = total_sig / total_cov  # global unmeth rate -> control lambda

    # Scale the coverage pileup to the expected-signal lambda (cov * r).
    if os.path.exists(lambda_bdg):
        logger.info(f"{lambda_bdg} already exists; skip bdgopt scaling.")
    else:
        logger.info(f"scaling control by r={r:.4g} (global unmeth rate)")
        bdgopt_cmd = ['macs3', 'bdgopt', '-i', ctrl_bdg, '-m', 'multiply',
                      '-p', f'{r:.8g}', '-o', lambda_bdg]
        logger.info(f"Running: {' '.join(bdgopt_cmd)}")
        subprocess.run(bdgopt_cmd, check=True)
    bdgcmp_cmd = ['macs3', 'bdgcmp', '-t', treat_bdg, '-c', lambda_bdg,
                  '-m', method, '-o', score_bdg]
    logger.info(f"Running: {' '.join(bdgcmp_cmd)}")
    subprocess.run(bdgcmp_cmd, check=True)
    bdgpeakcall_cmd = ['macs3', 'bdgpeakcall', '-i', score_bdg,
                       '-c', str(cutoff), '-l', str(min_len), '-g', str(max_gap),
                       '-o', narrowpeak]
    logger.info(f"Running: {' '.join(bdgpeakcall_cmd)}")
    subprocess.run(bdgpeakcall_cmd, check=True)

    if not keep_bdg:
        for f in (treat_bdg, ctrl_bdg, lambda_bdg, score_bdg):
            if os.path.exists(f):
                os.remove(f)
    logger.info(f"Peaks written to: {narrowpeak}")
    return narrowpeak
