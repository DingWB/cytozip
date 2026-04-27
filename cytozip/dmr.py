#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dmr.py — Methylation peak calling and differentially-methylated-region (DMR)
analysis tools that read from the cytozip .cz format.

Functions:
  - :func:`call_dmr`: call differentially methylated regions between two
    groups of single-cell ``.cz`` files using a permutation root-mean-square
    (RMS) test (Methylpy / ALLCools style) at each cytosine site, then
    merging adjacent significant sites (DMS) into DMRs.
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
from loguru import logger
from .cz import (Reader, _STRUCT_TO_NP_DTYPE, _make_np_dtype, np, pd)


# =====================================================================
# DMR calling — permutation root-mean-square (RMS) test
#
# Re-implements ALLCools' ``call_dms``/``call_dmr`` (which itself follows
# Methylpy's chi-square style RMS test) on top of the cytozip ``.cz``
# format.  The hot kernel lives in the Cython extension
# :mod:`cytozip.dmr_accel` (built by ``setup.py``); it releases the GIL
# and parallelizes per-site work with OpenMP ``prange``.  The outer
# loop over chunks is therefore left single-threaded so OpenMP can
# fully use all cores.
#
# References
# ----------
# * Schultz et al. (2015), Methylpy.
# * He et al. (2020), ALLCools (`ALLCools/dmr/rms_test.py`,
#   `ALLCools/dmr/call_dms.py`, `ALLCools/dmr/call_dmr.py`).
# =====================================================================
from .dmr_accel import rms_run_sites as _rms_run_sites


# ----- per-worker reader cache --------------------------------------
# When call_dmr submits many chunks to the same worker process via
# ProcessPoolExecutor, opening / closing every Reader on each task
# is wasteful (it re-mmaps the file and re-parses the header).  Cache
# them at module level keyed by absolute path; the cache is local to
# each worker process so it stays consistent with copy-on-write fork
# or fresh spawn.
_READER_CACHE: dict = {}


def _get_cached_reader(path):
    p = os.path.abspath(os.path.expanduser(path))
    r = _READER_CACHE.get(p)
    if r is None:
        r = Reader(p)
        r.advise_sequential()
        _READER_CACHE[p] = r
    return r


def _reader_np_dtype(reader):
    """Return (and cache) the numpy structured dtype for ``reader``.

    The dtype is fully determined by the reader's header and never
    changes once the file is opened, so we attach it as an attribute
    on the Reader object and reuse it across chunks.
    """
    dt = getattr(reader, '_np_dtype_cached', None)
    if dt is None:
        dt = _make_np_dtype(reader.header['formats'],
                            reader.header['columns'])
        try:
            reader._np_dtype_cached = dt
        except Exception:
            pass
    return dt


def _close_cached_readers():
    for r in _READER_CACHE.values():
        try:
            r.close()
        except Exception:
            pass
    _READER_CACHE.clear()


def _load_chunk_matrix(readers, dim, mc_col, cov_col, n_sites):
    """Load mc and cov as ``(n_cells, n_sites)`` int32 arrays for one chunk.

    Cells whose chunk is missing or has a different length contribute zeros.
    """
    n = len(readers)
    mc = np.zeros((n, n_sites), dtype=np.int32)
    cov = np.zeros((n, n_sites), dtype=np.int32)
    for i, r in enumerate(readers):
        if dim not in r.chunk_key2offset:
            continue
        raw = r.fetch_chunk_bytes(dim)
        if not raw:
            continue
        dt = _reader_np_dtype(r)
        arr = np.frombuffer(raw, dtype=dt)
        if arr.shape[0] != n_sites:
            r.release_chunk(dim)
            continue
        mc[i] = arr[mc_col].astype(np.int32, copy=False)
        cov[i] = arr[cov_col].astype(np.int32, copy=False)
        r.release_chunk(dim)
    return mc, cov


def _resolve_paths(arg):
    """Accept comma-separated string, list, or path to a text file (one per line)."""
    if isinstance(arg, (list, tuple)):
        items = list(arg)
    elif isinstance(arg, str) and os.path.isfile(arg) and not arg.endswith('.cz'):
        with open(arg) as fh:
            items = [ln.strip() for ln in fh if ln.strip()
                     and not ln.lstrip().startswith('#')]
    elif isinstance(arg, str):
        items = [s.strip() for s in arg.split(',') if s.strip()]
    else:
        raise TypeError(f"Cannot interpret group argument: {arg!r}")
    return [os.path.abspath(os.path.expanduser(p)) for p in items]


def _fisher_1v1_sites(mc_all, cov_all, keep):
    """Per-site Fisher's exact (two-sided) for the n_a=n_b=1 case.

    Returns ``(p_arr, delta_arr)`` matching ``rms_run_sites``' API.
    Uses ``scipy.stats.fisher_exact`` and a small Python loop; intended
    only for the degenerate 1-vs-1 layout where the permutation test
    has very few distinct permutations.
    """
    from scipy.stats import fisher_exact
    n = int(keep.size)
    p = np.ones(n, dtype=np.float64)
    da = np.zeros(n, dtype=np.float64)
    if n == 0:
        return p, da
    m_a = mc_all[0, keep]; c_a = cov_all[0, keep]
    m_b = mc_all[1, keep]; c_b = cov_all[1, keep]
    u_a = c_a - m_a
    u_b = c_b - m_b
    for i in range(n):
        ca = int(c_a[i]); cb = int(c_b[i])
        if ca <= 0 or cb <= 0:
            continue
        try:
            _, pv = fisher_exact(
                [[int(m_a[i]), int(u_a[i])],
                 [int(m_b[i]), int(u_b[i])]],
                alternative='two-sided')
        except Exception:
            pv = 1.0
        p[i] = float(pv)
        da[i] = (float(m_a[i]) / float(ca)
                 - float(m_b[i]) / float(cb))
    return p, da


def _process_chunk_for_dmr(args):
    """Process one reference chunk: returns DMS records (per-site).

    Uses a per-process reader cache so repeated submissions to the same
    worker reuse already-opened mmaps + headers instead of paying the
    open/close cost on every chunk.
    """
    (group_a_paths, group_b_paths, ref_path, index_path,
     dim, mc_col, cov_col,
     min_cov, min_samples_per_group,
     n_permute, min_pvalue, max_row_count, max_total_count,
     n_threads, delta_prefilter, prefilter_cutoff,
     use_fisher_1v1) = args
    ref = _get_cached_reader(ref_path)
    if dim not in ref.chunk_key2offset:
        return dim, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    ref_raw = ref.fetch_chunk_bytes(dim)
    if not ref_raw:
        return dim, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    ref_dt = _reader_np_dtype(ref)
    ref_arr = np.frombuffer(ref_raw, dtype=ref_dt)
    pos = ref_arr['pos'].astype(np.int64)

    # optional context index restriction (e.g., CGN sites)
    sel_ids = None
    if index_path is not None:
        ix = _get_cached_reader(index_path)
        if dim in ix.chunk_key2offset:
            ids = ix.get_ids_from_index(dim)
            if ids.ndim == 1:
                sel_ids = ids

    n_sites = ref_arr.shape[0]
    a_readers = [_get_cached_reader(p) for p in group_a_paths]
    b_readers = [_get_cached_reader(p) for p in group_b_paths]
    mc_a, cov_a = _load_chunk_matrix(a_readers, dim, mc_col, cov_col, n_sites)
    mc_b, cov_b = _load_chunk_matrix(b_readers, dim, mc_col, cov_col, n_sites)
    ref.release_chunk(dim)

    if sel_ids is not None:
        mc_a = mc_a[:, sel_ids]
        cov_a = cov_a[:, sel_ids]
        mc_b = mc_b[:, sel_ids]
        cov_b = cov_b[:, sel_ids]
        pos = pos[sel_ids]

    # filter sites: enough samples per group with min_cov
    pass_a = (cov_a >= min_cov).sum(axis=0) >= min_samples_per_group
    pass_b = (cov_b >= min_cov).sum(axis=0) >= min_samples_per_group
    keep = np.where(pass_a & pass_b)[0]
    if keep.size == 0:
        return dim, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    # Optional delta pre-filter: skip sites whose unpermuted
    # |mean(frac_a) - mean(frac_b)| is already below the post-test
    # cutoff.  Computed on cells with cov>0 (matches the kernel's
    # `out_da` definition) so this is an exact, no-loss optimization
    # whenever ``prefilter_cutoff <= frac_delta_cutoff``.
    if delta_prefilter and float(prefilter_cutoff) > 0.0 and keep.size:
        with np.errstate(invalid='ignore', divide='ignore'):
            mc_ak = mc_a[:, keep].astype(np.float64)
            cov_ak = cov_a[:, keep].astype(np.float64)
            mc_bk = mc_b[:, keep].astype(np.float64)
            cov_bk = cov_b[:, keep].astype(np.float64)
            frac_a = np.where(cov_ak > 0, mc_ak / np.maximum(cov_ak, 1), np.nan)
            frac_b = np.where(cov_bk > 0, mc_bk / np.maximum(cov_bk, 1), np.nan)
            mean_a = np.nanmean(frac_a, axis=0)
            mean_b = np.nanmean(frac_b, axis=0)
            delta = np.abs(mean_a - mean_b)
        keep = keep[np.nan_to_num(delta, nan=0.0) >= float(prefilter_cutoff)]
        if keep.size == 0:
            return dim, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    # Single allocation + cast in one step (avoids the temporary
    # int32 vstack array followed by astype copy).
    mc_all = np.concatenate([mc_a, mc_b], axis=0,
                            dtype=np.int64, casting='unsafe')
    cov_all = np.concatenate([cov_a, cov_b], axis=0,
                             dtype=np.int64, casting='unsafe')
    # ``keep`` already comes from ``np.where`` -> int64; reuse directly.
    if (use_fisher_1v1
            and mc_a.shape[0] == 1 and mc_b.shape[0] == 1):
        p_arr, da_arr = _fisher_1v1_sites(mc_all, cov_all, keep)
    else:
        p_arr, da_arr = _rms_run_sites(
            mc_all, cov_all, keep,
            group_a_n=mc_a.shape[0],
            n_permute=int(n_permute), min_pvalue=float(min_pvalue),
            max_row_count=int(max_row_count),
            max_total_count=int(max_total_count),
            n_threads=int(n_threads),
        )
    return dim, pos[keep], p_arr, da_arr


def _merge_dms_to_dmr(dms_df, max_dist, min_dms):
    """Group adjacent DMS within ``max_dist`` and same sign of delta."""
    if dms_df.shape[0] == 0:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'n_dms',
                                     'p_min', 'frac_delta_mean', 'state'])
    dms_df = dms_df.sort_values(['chrom', 'pos']).reset_index(drop=True)
    rows = []
    cur_chr = None
    cur_pos = []
    cur_p = []
    cur_d = []

    def _flush():
        if len(cur_pos) >= min_dms:
            n = len(cur_pos)
            mean_d = float(np.mean(cur_d))
            state = 1 if mean_d > 0 else (-1 if mean_d < 0 else 0)
            rows.append((cur_chr,
                         int(min(cur_pos)) - 1,           # 0-based BED start
                         int(max(cur_pos)),               # half-open end
                         n,
                         float(min(cur_p)),
                         mean_d,
                         state))

    sign = 0
    for chrom, pos, p, d in zip(dms_df['chrom'], dms_df['pos'],
                                dms_df['p'], dms_df['delta']):
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if (cur_chr != chrom
                or (cur_pos and pos - cur_pos[-1] > max_dist)
                or (sign != 0 and s != 0 and s != sign)):
            _flush()
            cur_pos, cur_p, cur_d = [], [], []
            sign = 0
        cur_chr = chrom
        cur_pos.append(pos)
        cur_p.append(p)
        cur_d.append(d)
        if sign == 0:
            sign = s
    _flush()
    return pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'n_dms',
                                       'p_min', 'frac_delta_mean', 'state'])


def call_dmr(group_a, group_b, reference, output,
             group_names=('A', 'B'),
             p_value_cutoff=0.001, frac_delta_cutoff=0.2,
             min_cov=1, min_samples_per_group=1,
             max_dist=250, min_dms=1,
             n_permute=3000, min_pvalue=0.01,
             max_row_count=50, max_total_count=3000,
             mc_col=None, cov_col=None,
             index=None, dms_output=None,
             chroms=None, jobs=1,
             delta_prefilter=True,
             use_fisher_1v1=True):
    """Call **CG** DMRs between two groups of methylation ``.cz`` files.

    This function is tuned for symmetric CpG methylation: site-level
    testing (no binning), absolute frac-delta cutoff (CG signal is
    bimodal and high-amplitude), short merge distance (~250 bp), and
    no global-rate normalization (CG global levels are similar across
    cell types).  For non-CG (mCH / mCA / mCT) use :func:`call_dmr_ch`
    instead, which adds bin-level pooling and global-rate
    normalization to handle CH's low fraction and large per-cell
    global differences.

    Pseudobulk vs single-cell input
    -------------------------------
    ``group_a`` / ``group_b`` accept **either** single-cell ``.cz`` files
    **or** pseudobulk ``.cz`` files (e.g., per-cluster summed mc/cov
    produced by ``merge_cz``).  **Pseudobulk is strongly recommended**
    for cell-type-vs-cell-type comparisons:

    * Single-cell coverage is sparse (CG cov is mostly 0/1), so the
      per-site permutation test has limited power.
    * Pseudobulk pools mc/cov within a group so per-site cov reaches
      tens-to-hundreds, giving stable p-values; this matches
      ALLCools' / Methylpy's "one ALLC = one sample" convention.
    * Runtime scales with the number of input ``.cz`` files, so going
      from N cells to K clusters is a large speed-up.

    The recommended workflow is to first build **multiple replicate
    pseudobulks per cluster** (split by donor / batch, or random
    splits if no biological replicates) using ``merge_cz``, then pass
    those replicate pseudobulks as ``group_a`` / ``group_b``.  Direct
    single-cell input is still supported and useful when no
    replicate structure exists (e.g., comparing sub-states inside one
    fine cluster), but expect lower power and longer runtime.

    Re-implements ALLCools' permutation root-mean-square (RMS) test
    (see :func:`_rms_pvalue`) on the cytozip ``.cz`` format and merges
    adjacent significant sites (DMS) into DMRs.

    Algorithm
    ---------
    1.  For every chunk in ``reference``, load each cell's ``mc``/``cov``
        column into an ``(n_cells, n_sites)`` int32 matrix.
    2.  Keep sites where both groups have at least
        ``min_samples_per_group`` cells with coverage ``>= min_cov``.
    3.  Build, per site, a contingency table with one row per usable cell
        and columns ``(mc, unmethylated)``, then compute the permutation
        RMS p-value (Cython + OpenMP, with early stopping at
        ``min_pvalue``).
    4.  Sites with ``p < p_value_cutoff`` and
        ``|mean(frac_A) - mean(frac_B)| > frac_delta_cutoff`` are DMS.
    5.  Adjacent DMS on the same chromosome with the same sign and gap
        ``<= max_dist`` are merged into a DMR; DMRs with at least
        ``min_dms`` constituent sites are kept.

    Parallelism
    -----------
    ``jobs`` is the **total** number of CPU cores to use.  It is split
    automatically into two layers:

    * ``processes = min(jobs, n_chunks)`` worker **processes**, each
      running one reference chunk via ``ProcessPoolExecutor``
      (``spawn``).
    * ``threads = ceil(jobs / processes)`` OpenMP threads inside each
      worker's per-site Cython ``prange``.

    So total CPU usage is ``processes * threads ≈ jobs``.  When
    ``jobs <= n_chunks`` the split is purely process-level (one chunk
    per process, no OpenMP); when there are fewer chunks than cores the
    spare cores are spent on per-site OpenMP inside each chunk.

    Parameters
    ----------
    group_a, group_b : list of str | str
        Group A / B paths.  Accepts a python list of paths,
        a comma-separated string, or a text file (one path per line).
    reference : str
        Reference ``.cz`` (genomic coordinates: pos / strand / context).
    output : str
        Output DMR TSV path (chrom, start, end, n_dms, p_min,
        frac_delta_mean, state).  ``start`` is 0-based, ``end``
        half-open (BED-style).
    group_names : tuple of str
        Labels (only used for the column header / log messages).
    p_value_cutoff : float
        Per-site p-value cutoff for calling a DMS.
    frac_delta_cutoff : float
        Minimum |mean_frac_A - mean_frac_B| to call a DMS.
    min_cov : int
        Minimum per-cell coverage at a site for that cell to contribute.
    min_samples_per_group : int
        Minimum cells per group passing ``min_cov`` at a site.
    max_dist : int
        Maximum gap (bp) between adjacent DMS to merge into the same DMR.
    min_dms : int
        Minimum number of DMS per DMR.
    n_permute : int
        Number of permutations for the RMS test.
    min_pvalue : float
        Permutation early-stopping threshold (matches ALLCools default).
    max_row_count, max_total_count : int
        Per-row / total cap on counts (downsampling) before the test;
        matches ALLCools defaults of 50 / 3000.
    mc_col, cov_col : int or str or None
        Column index (0-based) or name of the methylation / coverage
        column.  Defaults: first / last data column.
    index : str or None
        Optional context-index ``.cz`` (e.g. CGN-only) to restrict to.
    dms_output : str or None
        If given, also write the per-site DMS table to this TSV.
    chroms : list of str or None
        Restrict to these chromosomes (matched against the chunk-key's
        leading dimension).
    jobs : int
        Total CPU cores to use.  Automatically split into worker
        processes (across chunks) and OpenMP threads (per-site inside
        each chunk) so that ``processes * threads ≈ jobs``.

    Returns
    -------
    str
        Path to the output DMR TSV.

    Examples
    --------
    ::

        # Python API
        import cytozip as czip
        czip.call_dmr(
            group_a=['cell1.cz', 'cell2.cz'],
            group_b=['cell3.cz', 'cell4.cz'],
            reference='mm10.allc.cz',
            output='dmr.tsv',
            index='mm10.CGN.index',
        )

        # CLI
        # czip call_dmr -a a.txt -b b.txt -r mm10.allc.cz \\
        #               -s mm10.CGN.index -o dmr.tsv -j 8

    """
    a_paths = _resolve_paths(group_a)
    b_paths = _resolve_paths(group_b)
    if not a_paths or not b_paths:
        raise ValueError("group_a and group_b must each contain >=1 .cz file")
    ref_path = os.path.abspath(os.path.expanduser(reference))
    index_path = (os.path.abspath(os.path.expanduser(index))
                  if index else None)

    # Resolve mc / cov column names from the first cell's header
    probe = Reader(a_paths[0])
    cols = probe.header['columns']
    if mc_col is None:
        mc_col = cols[0]
    elif isinstance(mc_col, int):
        mc_col = cols[mc_col]
    if cov_col is None:
        cov_col = cols[-1]
    elif isinstance(cov_col, int):
        cov_col = cols[cov_col]

    # Iterate over chunks defined by the reference
    ref = Reader(ref_path)
    chunk_keys = list(ref.chunk_key2offset)
    ref.close()
    probe.close()
    if chroms is not None:
        chroms = set(chroms)
        chunk_keys = [k for k in chunk_keys if k[0] in chroms]
    if not chunk_keys:
        raise ValueError("No matching chunks in reference.")

    # Split total `jobs` cores into (processes, threads) so that
    # processes * threads ~= jobs.  Each chunk goes to one process;
    # spare cores become OpenMP threads inside that chunk's prange.
    n_chunks = len(chunk_keys)
    total = max(1, int(jobs))
    n_proc = min(total, n_chunks)
    n_thr = max(1, total // max(1, n_proc))

    logger.info(f"call_dmr: {len(a_paths)} cells in {group_names[0]} "
                f"vs {len(b_paths)} cells in {group_names[1]} "
                f"over {n_chunks} chunks "
                f"(jobs={total} = {n_proc} proc x {n_thr} threads)")

    tasks = [
        (a_paths, b_paths, ref_path, index_path, dim, mc_col, cov_col,
         int(min_cov), int(min_samples_per_group),
         int(n_permute), float(min_pvalue),
         int(max_row_count), int(max_total_count),
         int(n_thr),
         bool(delta_prefilter), float(frac_delta_cutoff),
         bool(use_fisher_1v1))
        for dim in chunk_keys
    ]

    dms_chrom = []
    dms_pos = []
    dms_p = []
    dms_delta = []

    def _collect(result):
        dim, pos, p_arr, da_arr = result
        if pos.size == 0:
            return
        chrom = dim[0]
        sig = (p_arr < p_value_cutoff) & (np.abs(da_arr) > frac_delta_cutoff)
        if not sig.any():
            return
        dms_pos.append(pos[sig])
        dms_p.append(p_arr[sig])
        dms_delta.append(da_arr[sig])
        dms_chrom.append(np.full(int(sig.sum()), chrom, dtype=object))

    n_done = 0
    if n_proc > 1 and len(tasks) > 1:
        # Multi-process across chunks; each worker runs its own OpenMP
        # ``prange`` with ``n_thr`` threads.  Use 'spawn' to avoid
        # inheriting parent fds / loggers and to play nice with OpenMP.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as _mp
        ctx = _mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_proc,
                                 mp_context=ctx) as ex:
            futures = [ex.submit(_process_chunk_for_dmr, t) for t in tasks]
            for fut in as_completed(futures):
                _collect(fut.result())
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(f"  processed {n_done}/{len(tasks)} chunks")
    else:
        # Single-process: let the Cython OpenMP kernel use all `n_thr`
        # cores per chunk without process-level over-subscription.
        try:
            for t in tasks:
                _collect(_process_chunk_for_dmr(t))
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(f"  processed {n_done}/{len(tasks)} chunks")
        finally:
            # Drop cached readers from the parent process so call_dmr
            # doesn't leak open mmaps across calls.
            _close_cached_readers()

    if dms_pos:
        dms_df = pd.DataFrame({
            'chrom': np.concatenate(dms_chrom),
            'pos': np.concatenate(dms_pos),
            'p': np.concatenate(dms_p),
            'delta': np.concatenate(dms_delta),
        })
    else:
        dms_df = pd.DataFrame(columns=['chrom', 'pos', 'p', 'delta'])

    if dms_output is not None:
        dms_df.sort_values(['chrom', 'pos']).to_csv(
            os.path.abspath(os.path.expanduser(dms_output)),
            sep='\t', index=False)
        logger.info(f"DMS table ({dms_df.shape[0]} sites) written to "
                    f"{dms_output}")

    dmr_df = _merge_dms_to_dmr(dms_df, max_dist=int(max_dist),
                               min_dms=int(min_dms))
    output = os.path.abspath(os.path.expanduser(output))
    dmr_df.to_csv(output, sep='\t', index=False)
    logger.info(f"call_dmr: {dms_df.shape[0]} DMS -> "
                f"{dmr_df.shape[0]} DMRs written to {output}")
    return output


# ==========================================================================
# CH (mCH / mCA / mCT) DMR calling
#
# CH methylation differs from CG in ways that require an adapted pipeline:
#   * Per-site coverage is sparse and the methylation fraction is very
#     low (~0.5-5% in neurons, <1% elsewhere) -> single-site permutation
#     tests have almost no power.  We pool counts within fixed-size bins
#     (default 5 kb) before testing.
#   * Per-cell global mCH varies by 2-3x between cell types (e.g.
#     excitatory vs. inhibitory neurons).  Without normalization the
#     RMS test would flag almost every bin as differential simply
#     because of the global shift.  We optionally rescale each cell's
#     mc counts to a common global rate before testing, so the test
#     reflects bin-specific deviations.
#   * Significant signals are smoother and broader, so the merge
#     distance and the minimum number of significant bins per DMR are
#     larger by default.
#   * Strand- / sub-context-specific signals (mCA dominates in
#     mammalian neurons; mCT/mCC are weaker and structurally
#     different).  Pass an ``index`` .cz that restricts to one
#     sub-context (e.g., CAN-only) to analyze it in isolation.
# ==========================================================================
def _aggregate_bins(mc, cov, pos, bin_size):
    """Sum ``mc`` and ``cov`` per cell within fixed-size genomic bins.

    Parameters
    ----------
    mc, cov : (n_cells, n_sites) int arrays
    pos     : (n_sites,) int64 array of 1-based site positions
    bin_size: int, bin width in bp

    Returns
    -------
    bin_starts : (n_bins,) int64, 1-based start of each bin (= ``floor(pos/bin)*bin + 1``)
    mc_bin     : (n_cells, n_bins) int64
    cov_bin    : (n_cells, n_bins) int64
    """
    if pos.size == 0:
        empty = np.zeros((mc.shape[0], 0), dtype=np.int64)
        return np.empty(0, dtype=np.int64), empty, empty
    # 0-based bin index for each site, contiguous since pos is sorted
    bin_idx = (pos - 1) // int(bin_size)
    # boundaries in the sorted site array
    boundaries = np.flatnonzero(np.diff(bin_idx)) + 1
    starts = np.concatenate(([0], boundaries))
    n_bins = starts.size
    mc64 = mc.astype(np.int64, copy=False)
    cov64 = cov.astype(np.int64, copy=False)
    mc_bin = np.add.reduceat(mc64, starts, axis=1)
    cov_bin = np.add.reduceat(cov64, starts, axis=1)
    bin_starts = bin_idx[starts] * int(bin_size) + 1
    return bin_starts.astype(np.int64), mc_bin, cov_bin


def _scan_global_ch(args):
    """Worker: accumulate per-cell mc/cov totals over one chunk."""
    paths, ref_path, index_path, dim, mc_col, cov_col = args
    ref = _get_cached_reader(ref_path)
    if dim not in ref.chunk_key2offset:
        n = len(paths)
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    ref_raw = ref.fetch_chunk_bytes(dim)
    if not ref_raw:
        n = len(paths)
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    ref_dt = _reader_np_dtype(ref)
    n_sites = np.frombuffer(ref_raw, dtype=ref_dt).shape[0]
    sel_ids = None
    if index_path is not None:
        ix = _get_cached_reader(index_path)
        if dim in ix.chunk_key2offset:
            ids = ix.get_ids_from_index(dim)
            if ids.ndim == 1:
                sel_ids = ids
    readers = [_get_cached_reader(p) for p in paths]
    mc, cov = _load_chunk_matrix(readers, dim, mc_col, cov_col, n_sites)
    ref.release_chunk(dim)
    if sel_ids is not None:
        mc = mc[:, sel_ids]
        cov = cov[:, sel_ids]
    return mc.sum(axis=1, dtype=np.int64), cov.sum(axis=1, dtype=np.int64)


def _compute_global_ch_rates(paths, ref_path, index_path, chunk_keys,
                             mc_col, cov_col, n_proc):
    """Compute per-cell global mCH rate (mc_total / cov_total) over the
    selected chunks (and optional context index).

    Returns
    -------
    rates : (n_cells,) float64
    mc_tot, cov_tot : (n_cells,) int64
    """
    n = len(paths)
    mc_tot = np.zeros(n, dtype=np.int64)
    cov_tot = np.zeros(n, dtype=np.int64)
    tasks = [(paths, ref_path, index_path, dim, mc_col, cov_col)
             for dim in chunk_keys]
    if n_proc > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as _mp
        ctx = _mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            futures = [ex.submit(_scan_global_ch, t) for t in tasks]
            for fut in as_completed(futures):
                mc_part, cov_part = fut.result()
                mc_tot += mc_part
                cov_tot += cov_part
    else:
        try:
            for t in tasks:
                mc_part, cov_part = _scan_global_ch(t)
                mc_tot += mc_part
                cov_tot += cov_part
        finally:
            _close_cached_readers()
    rates = np.where(cov_tot > 0,
                     mc_tot.astype(np.float64) / np.maximum(cov_tot, 1),
                     0.0)
    return rates, mc_tot, cov_tot


def _process_chunk_for_dmr_ch(args):
    """Process one reference chunk for CH DMR: bin-aggregate, optional
    per-cell rescaling to a common global rate, then RMS test on the
    bin-level counts.

    Returns
    -------
    dim, bin_starts, p_arr, delta_arr, rate_a_arr, rate_b_arr
    """
    (group_a_paths, group_b_paths, ref_path, index_path,
     dim, mc_col, cov_col, bin_size,
     scale_a, scale_b,
     min_cov, min_samples_per_group,
     n_permute, min_pvalue, max_row_count, max_total_count,
     n_threads, delta_prefilter,
     prefilter_log2fc, prefilter_abs_delta) = args
    empty = (dim, np.empty(0, dtype=np.int64),
             np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    ref = _get_cached_reader(ref_path)
    if dim not in ref.chunk_key2offset:
        return empty
    ref_raw = ref.fetch_chunk_bytes(dim)
    if not ref_raw:
        return empty
    ref_dt = _reader_np_dtype(ref)
    ref_arr = np.frombuffer(ref_raw, dtype=ref_dt)
    pos = ref_arr['pos'].astype(np.int64)

    sel_ids = None
    if index_path is not None:
        ix = _get_cached_reader(index_path)
        if dim in ix.chunk_key2offset:
            ids = ix.get_ids_from_index(dim)
            if ids.ndim == 1:
                sel_ids = ids

    n_sites = ref_arr.shape[0]
    a_readers = [_get_cached_reader(p) for p in group_a_paths]
    b_readers = [_get_cached_reader(p) for p in group_b_paths]
    mc_a, cov_a = _load_chunk_matrix(a_readers, dim, mc_col, cov_col, n_sites)
    mc_b, cov_b = _load_chunk_matrix(b_readers, dim, mc_col, cov_col, n_sites)
    ref.release_chunk(dim)

    if sel_ids is not None:
        mc_a = mc_a[:, sel_ids]
        cov_a = cov_a[:, sel_ids]
        mc_b = mc_b[:, sel_ids]
        cov_b = cov_b[:, sel_ids]
        pos = pos[sel_ids]

    if pos.size == 0:
        return empty

    # Aggregate to bins
    bin_starts, mc_a_bin, cov_a_bin = _aggregate_bins(mc_a, cov_a, pos, bin_size)
    _, mc_b_bin, cov_b_bin = _aggregate_bins(mc_b, cov_b, pos, bin_size)
    n_bins = bin_starts.size
    if n_bins == 0:
        return empty

    # Optional per-cell rescaling to a common global rate
    if scale_a is not None:
        s = np.asarray(scale_a, dtype=np.float64)[:, None]
        mc_a_bin = np.minimum(np.rint(mc_a_bin * s).astype(np.int64), cov_a_bin)
    if scale_b is not None:
        s = np.asarray(scale_b, dtype=np.float64)[:, None]
        mc_b_bin = np.minimum(np.rint(mc_b_bin * s).astype(np.int64), cov_b_bin)

    # Per-bin mean rates (count-weighted across cells in each group)
    rate_a = (mc_a_bin.sum(axis=0).astype(np.float64)
              / np.maximum(cov_a_bin.sum(axis=0), 1))
    rate_b = (mc_b_bin.sum(axis=0).astype(np.float64)
              / np.maximum(cov_b_bin.sum(axis=0), 1))

    # Coverage filter: enough cells per group with bin cov >= min_cov
    pass_a = (cov_a_bin >= min_cov).sum(axis=0) >= min_samples_per_group
    pass_b = (cov_b_bin >= min_cov).sum(axis=0) >= min_samples_per_group
    keep = np.where(pass_a & pass_b)[0]
    if keep.size == 0:
        return empty

    # Optional delta pre-filter using bin-level rate_a / rate_b that
    # we already computed.  This drops bins that cannot satisfy the
    # post-test |log2fc| / |delta-rate| cutoffs without ever running
    # the permutation kernel.
    if delta_prefilter and keep.size:
        with np.errstate(invalid='ignore', divide='ignore'):
            ra = rate_a[keep]
            rb = rate_b[keep]
            abs_d = np.abs(ra - rb)
            log2fc = np.where((ra > 0) & (rb > 0),
                              np.log2(np.maximum(ra, 1e-12)
                                      / np.maximum(rb, 1e-12)),
                              0.0)
        mask = np.ones(keep.size, dtype=bool)
        if float(prefilter_abs_delta) > 0.0:
            mask &= (abs_d >= float(prefilter_abs_delta))
        if float(prefilter_log2fc) > 0.0:
            mask &= (np.abs(log2fc) >= float(prefilter_log2fc))
        keep = keep[mask]
        if keep.size == 0:
            return empty

    mc_all = np.concatenate([mc_a_bin, mc_b_bin], axis=0,
                            dtype=np.int64, casting='unsafe')
    cov_all = np.concatenate([cov_a_bin, cov_b_bin], axis=0,
                             dtype=np.int64, casting='unsafe')
    p_arr, da_arr = _rms_run_sites(
        mc_all, cov_all, keep,
        group_a_n=mc_a_bin.shape[0],
        n_permute=int(n_permute), min_pvalue=float(min_pvalue),
        max_row_count=int(max_row_count),
        max_total_count=int(max_total_count),
        n_threads=int(n_threads),
    )
    return (dim, bin_starts[keep], p_arr, da_arr,
            rate_a[keep], rate_b[keep])


def call_dmr_ch(group_a, group_b, reference, output,
                bin_size=5000,
                context='CHN',
                group_names=('A', 'B'),
                p_value_cutoff=0.001,
                log2fc_cutoff=1.0,
                abs_delta_cutoff=0.005,
                normalize=True,
                global_a=None, global_b=None,
                min_cov=3, min_samples_per_group=2,
                max_dist=2000, min_dms=2,
                n_permute=10000, min_pvalue=0.001,
                max_row_count=200, max_total_count=10000,
                mc_col=None, cov_col=None,
                index=None, dms_output=None,
                chroms=None, jobs=1,
                delta_prefilter=True):
    """Call **non-CG (CH / CA / CT)** DMRs between two groups of
    single-cell methylation ``.cz`` files.

    CH methylation has very low fractions (~0.5-5%) and large global
    differences between cell types, so the CG-tuned :func:`call_dmr`
    is unsuitable.  This function adapts the same RMS-permutation
    kernel by:

    1.  **Bin aggregation** — pooling per-cell mc/cov counts into
        fixed-size genomic bins (default 5 kb) so that bin-level
        coverage is high enough for the permutation test.
    2.  **Global-rate normalization** (default ON) — pre-rescaling
        each cell's mc counts so all cells share a common reference
        global mCH rate.  This removes the dominant "cell type with
        higher global mCH" effect that would otherwise saturate the
        test.
    3.  **log2-fold-change filter** — DMS ("differentially methylated
        bins") must satisfy ``p < p_value_cutoff`` AND
        ``|log2(rate_A / rate_B)| >= log2fc_cutoff`` AND
        ``|rate_A - rate_B| >= abs_delta_cutoff`` (rates are computed
        on the rescaled counts).
    4.  **Context restriction** — pass ``index=mm10.CAN.index`` to
        analyze only a sub-context (e.g., CA).  Strongly recommended
        because mCA carries the dominant neuronal CH signal whereas
        mCT/mCC have different biology.
    5.  **Wider merge distance** (``max_dist=2000``) and minimum DMS
        per DMR (``min_dms=2``) since CH signal is broader/smoother.

    Pseudobulk vs single-cell input
    -------------------------------
    ``group_a`` / ``group_b`` accept **either** single-cell ``.cz``
    files **or** pseudobulk ``.cz`` files.  For CH **pseudobulk input
    is even more strongly recommended than for CG**: per-cell, per-site
    CH coverage is essentially zero, so even after 5 kb binning the
    signal in single-cell input is dominated by sampling noise.  Pool
    cells of the same cluster (preferably into multiple replicate
    pseudobulks split by donor / batch) with ``merge_cz`` first, then
    pass those replicates as ``group_a`` / ``group_b``.  Single-cell
    input still works (the binning step here acts as an in-memory
    pseudobulk) and is appropriate when no replicate structure is
    available, but expect much lower power and slower runtime.

    Parameters
    ----------
    group_a, group_b : list of str | str
        Group A / B paths.  Same parsing as :func:`call_dmr`.
    reference, output : str
        Reference ``.cz`` and output DMR TSV path.
    bin_size : int
        Bin width in bp for pooling per-cell counts (default 5000).
    context : str
        Tag used in log messages only (e.g. ``'CHN'``, ``'CAN'``).
    p_value_cutoff : float
        Per-bin RMS p-value cutoff.
    log2fc_cutoff : float
        Minimum ``|log2(rate_a / rate_b)|`` after normalization.
    abs_delta_cutoff : float
        Minimum ``|rate_a - rate_b|`` after normalization.  Acts as
        a low-rate floor so tiny ratios with both rates ~0 are
        rejected.
    normalize : bool
        If True, pre-compute per-cell global mCH rates and rescale mc
        counts so all cells share a common reference rate.
    global_a, global_b : array-like or None
        Pre-computed per-cell global mCH rates (one entry per cell in
        the corresponding group).  If provided, the global pre-pass
        is skipped.
    min_cov : int
        Minimum per-cell coverage in a bin for that cell to count
        toward ``min_samples_per_group``.
    max_dist : int
        Maximum gap (bp) between adjacent significant bins for merging.
    min_dms : int
        Minimum number of significant bins per DMR.
    n_permute, min_pvalue, max_row_count, max_total_count : numeric
        RMS-test knobs.  Defaults are stricter than CG's (10000
        permutations, 0.001 stop) because CH signals tend to have
        smaller effect sizes.
    mc_col, cov_col, index, dms_output, chroms, jobs : see :func:`call_dmr`.

    Returns
    -------
    str
        Path to the output DMR TSV.

    Examples
    --------
    ::

        import cytozip as czip
        czip.call_dmr_ch(
            group_a='excitatory.txt', group_b='inhibitory.txt',
            reference='mm10.allc.cz',
            index='mm10.CAN.index',     # mCA only
            output='ch_dmr.tsv',
            bin_size=5000,
            jobs=16,
        )
    """
    a_paths = _resolve_paths(group_a)
    b_paths = _resolve_paths(group_b)
    if not a_paths or not b_paths:
        raise ValueError("group_a and group_b must each contain >=1 .cz file")
    ref_path = os.path.abspath(os.path.expanduser(reference))
    index_path = (os.path.abspath(os.path.expanduser(index))
                  if index else None)

    probe = Reader(a_paths[0])
    cols = probe.header['columns']
    if mc_col is None:
        mc_col = cols[0]
    elif isinstance(mc_col, int):
        mc_col = cols[mc_col]
    if cov_col is None:
        cov_col = cols[-1]
    elif isinstance(cov_col, int):
        cov_col = cols[cov_col]
    probe.close()

    ref = Reader(ref_path)
    chunk_keys = list(ref.chunk_key2offset)
    ref.close()
    if chroms is not None:
        chroms = set(chroms)
        chunk_keys = [k for k in chunk_keys if k[0] in chroms]
    if not chunk_keys:
        raise ValueError("No matching chunks in reference.")

    n_chunks = len(chunk_keys)
    total = max(1, int(jobs))
    n_proc = min(total, n_chunks)
    n_thr = max(1, total // max(1, n_proc))

    logger.info(
        f"call_dmr_ch[{context}]: {len(a_paths)} cells in {group_names[0]} "
        f"vs {len(b_paths)} cells in {group_names[1]} "
        f"over {n_chunks} chunks, bin_size={bin_size} "
        f"(jobs={total} = {n_proc} proc x {n_thr} threads)")

    # ----- Optional global mCH rate pre-pass -------------------------
    scale_a = scale_b = None
    if normalize:
        if global_a is None:
            logger.info("call_dmr_ch: scanning group A for per-cell global rates")
            ga, _, _ = _compute_global_ch_rates(
                a_paths, ref_path, index_path, chunk_keys,
                mc_col, cov_col, n_proc)
        else:
            ga = np.asarray(global_a, dtype=np.float64)
            if ga.shape[0] != len(a_paths):
                raise ValueError("len(global_a) must match len(group_a)")
        if global_b is None:
            logger.info("call_dmr_ch: scanning group B for per-cell global rates")
            gb, _, _ = _compute_global_ch_rates(
                b_paths, ref_path, index_path, chunk_keys,
                mc_col, cov_col, n_proc)
        else:
            gb = np.asarray(global_b, dtype=np.float64)
            if gb.shape[0] != len(b_paths):
                raise ValueError("len(global_b) must match len(group_b)")

        all_rates = np.concatenate([ga, gb])
        valid = all_rates[all_rates > 0]
        if valid.size == 0:
            logger.warning("call_dmr_ch: all per-cell global rates are zero; "
                           "skipping normalization")
        else:
            target = float(np.median(valid))
            # avoid divide-by-zero / huge blow-up for empty cells
            ga_safe = np.where(ga > 0, ga, target)
            gb_safe = np.where(gb > 0, gb, target)
            scale_a = target / ga_safe
            scale_b = target / gb_safe
            logger.info(f"call_dmr_ch: normalized to global rate "
                        f"target={target:.4f}; A scale range "
                        f"[{scale_a.min():.2f},{scale_a.max():.2f}], "
                        f"B scale range "
                        f"[{scale_b.min():.2f},{scale_b.max():.2f}]")

    tasks = [
        (a_paths, b_paths, ref_path, index_path, dim, mc_col, cov_col,
         int(bin_size),
         scale_a, scale_b,
         int(min_cov), int(min_samples_per_group),
         int(n_permute), float(min_pvalue),
         int(max_row_count), int(max_total_count),
         int(n_thr),
         bool(delta_prefilter),
         float(log2fc_cutoff), float(abs_delta_cutoff))
        for dim in chunk_keys
    ]

    dms_chrom = []
    dms_pos = []
    dms_p = []
    dms_delta = []
    dms_rate_a = []
    dms_rate_b = []

    def _collect(result):
        dim, pos, p_arr, da_arr, ra, rb = result
        if pos.size == 0:
            return
        chrom = dim[0]
        # Normalized rate filter: log2fc + abs delta + p-value
        with np.errstate(divide='ignore', invalid='ignore'):
            log2fc = np.where((ra > 0) & (rb > 0),
                              np.log2(np.maximum(ra, 1e-12)
                                      / np.maximum(rb, 1e-12)),
                              0.0)
        sig = ((p_arr < p_value_cutoff)
               & (np.abs(log2fc) >= log2fc_cutoff)
               & (np.abs(ra - rb) >= abs_delta_cutoff))
        if not sig.any():
            return
        dms_pos.append(pos[sig])
        dms_p.append(p_arr[sig])
        dms_delta.append((ra - rb)[sig])
        dms_rate_a.append(ra[sig])
        dms_rate_b.append(rb[sig])
        dms_chrom.append(np.full(int(sig.sum()), chrom, dtype=object))

    n_done = 0
    if n_proc > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as _mp
        ctx = _mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            futures = [ex.submit(_process_chunk_for_dmr_ch, t) for t in tasks]
            for fut in as_completed(futures):
                _collect(fut.result())
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(f"  processed {n_done}/{len(tasks)} chunks")
    else:
        try:
            for t in tasks:
                _collect(_process_chunk_for_dmr_ch(t))
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(f"  processed {n_done}/{len(tasks)} chunks")
        finally:
            _close_cached_readers()

    if dms_pos:
        dms_df = pd.DataFrame({
            'chrom': np.concatenate(dms_chrom),
            'pos': np.concatenate(dms_pos),
            'p': np.concatenate(dms_p),
            'delta': np.concatenate(dms_delta),
            'rate_a': np.concatenate(dms_rate_a),
            'rate_b': np.concatenate(dms_rate_b),
        })
    else:
        dms_df = pd.DataFrame(columns=['chrom', 'pos', 'p', 'delta',
                                       'rate_a', 'rate_b'])

    if dms_output is not None:
        dms_df.sort_values(['chrom', 'pos']).to_csv(
            os.path.abspath(os.path.expanduser(dms_output)),
            sep='\t', index=False)
        logger.info(f"DMS bins ({dms_df.shape[0]}) written to {dms_output}")

    # Sites here are bin-start coordinates (gap between adjacent bins
    # is exactly ``bin_size``), so ``max_dist`` must be >= bin_size to
    # ever bridge two neighbouring significant bins.  If the caller
    # left the default 2000 with a larger bin_size (e.g. 5000), bump
    # to ``bin_size`` and warn so the merge step actually runs.
    eff_max_dist = int(max_dist)
    if eff_max_dist < int(bin_size):
        logger.warning(
            f"call_dmr_ch: max_dist={max_dist} < bin_size={bin_size}; "
            f"adjacent significant bins (gap == bin_size) would never "
            f"be merged. Bumping max_dist -> {int(bin_size)}.")
        eff_max_dist = int(bin_size)
    dmr_df = _merge_dms_to_dmr(dms_df[['chrom', 'pos', 'p', 'delta']],
                               max_dist=eff_max_dist,
                               min_dms=int(min_dms))
    # Bin-end is bin_start + bin_size - 1 (inclusive); _merge_dms_to_dmr
    # uses ``max(pos)`` as half-open end, which under-counts the last
    # bin's width by ``bin_size - 1`` -- adjust here so the BED end
    # spans the full last bin.
    if dmr_df.shape[0] > 0:
        dmr_df['end'] = dmr_df['end'] + int(bin_size) - 1
    output = os.path.abspath(os.path.expanduser(output))
    dmr_df.to_csv(output, sep='\t', index=False)
    logger.info(f"call_dmr_ch[{context}]: {dms_df.shape[0]} DMS bins -> "
                f"{dmr_df.shape[0]} DMRs written to {output}")
    return output


# ==========================================================================
# One-vs-rest DMR wrapper (optionally stratified by a sample-class table)
# ==========================================================================
def _list_pseudobulk_cz(indir, ext):
    indir = os.path.abspath(os.path.expanduser(indir))
    if not os.path.isdir(indir):
        raise NotADirectoryError(indir)
    files = sorted(f for f in os.listdir(indir) if f.endswith(ext))
    snames = [f[:-len(ext)] for f in files]
    paths = [os.path.join(indir, f) for f in files]
    return snames, paths


def _load_class_table(class_table, snames):
    """Resolve sname -> class.  Accepts a TSV path (no header, two
    columns ``sname<TAB>class``), a DataFrame with those two columns
    (any order), or a dict ``{sname: class}``.

    Returns
    -------
    dict mapping each input sname (that has an entry) to its class label.
    """
    if class_table is None:
        return None
    if isinstance(class_table, dict):
        mapping = dict(class_table)
    elif isinstance(class_table, pd.DataFrame):
        df = class_table.copy()
        df.columns = ['sname', 'cell_class'] + list(df.columns[2:])
        mapping = dict(zip(df['sname'].astype(str),
                           df['cell_class'].astype(str)))
    elif isinstance(class_table, str):
        path = os.path.abspath(os.path.expanduser(class_table))
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['sname', 'cell_class'])
        mapping = dict(zip(df['sname'].astype(str),
                           df['cell_class'].astype(str)))
    else:
        raise TypeError(f"class_table must be None / str / DataFrame / dict, "
                        f"got {type(class_table)}")
    return {s: mapping[s] for s in snames if s in mapping}


def call_dmr_one_vs_rest(indir, reference, outdir,
                         ext='.cz',
                         method='cg',
                         class_table=None,
                         min_class_size=2,
                         samples=None,
                         overwrite=False,
                         jobs=1,
                         index=None,
                         dms_output_dir=None,
                         auto_merge=True,
                         merge_kwargs=None,
                         **dmr_kwargs):
    """Batch one-vs-rest DMR calling over a folder of pseudobulk ``.cz``.

    For each pseudobulk sample in ``indir`` (or each sample within a
    class group when ``class_table`` is given), runs DMR of that
    sample vs. **all other samples in the same group**.  The
    underlying caller is :func:`call_dmr` (``method='cg'``) or
    :func:`call_dmr_ch` (``method='ch'``); extra parameters are
    forwarded via ``**dmr_kwargs``.

    Parameters
    ----------
    indir : str
        Directory containing pseudobulk ``.cz`` files.
    reference : str
        Reference ``.cz`` (genomic coordinates).
    outdir : str
        Root output directory.  Layout:

        * Global one-vs-rest (``class_table=None``):
          ``outdir/<sname>.dmr.tsv``
        * Stratified (``class_table`` provided):
          ``outdir/<class>/<sname>.dmr.tsv``
    ext : str
        File suffix used to discover pseudobulks (default ``.cz``).
        ``sname = filename[:-len(ext)]``.
    method : {'cg', 'ch'}
        Which DMR caller to use.
    class_table : None | str | pandas.DataFrame | dict
        If given, restrict each one-vs-rest comparison to **within
        the same class**.  Accepts a 2-column TSV (no header:
        ``sname<TAB>class``), a DataFrame with the same columns, or
        a dict ``{sname: class}``.  Classes with fewer than
        ``min_class_size`` members are skipped.
    min_class_size : int
        Skip classes with fewer than this many samples (only used when
        ``class_table`` is given).
    samples : list of str or None
        Optional explicit subset / ordering of sample names to use.
        Snames not present in ``indir`` are dropped with a warning.
    overwrite : bool
        If False, skip comparisons whose output TSV already exists
        (resume-friendly).
    jobs : int
        Total CPU cores forwarded to each ``call_dmr`` /
        ``call_dmr_ch`` invocation.
    index : str or None
        Optional context-index ``.cz`` (e.g. CGN/CAN-only) forwarded
        to the underlying caller.
    dms_output_dir : str or None
        If given, also write per-sample DMS TSVs here, with the same
        ``<class>/<sname>.dms.tsv`` layout.
    **dmr_kwargs : dict
        Forwarded to :func:`call_dmr` or :func:`call_dmr_ch` (e.g.
        ``frac_delta_cutoff``, ``log2fc_cutoff``, ``n_permute``,
        ``bin_size``, ``min_cov``, ...).  ``group_a``, ``group_b``,
        ``reference``, ``output``, ``index``, ``dms_output``,
        ``jobs``, ``chroms``, ``global_a``, ``global_b`` are managed
        by this wrapper and must not be supplied here.

    Notes
    -----
    For ``method='ch'`` and ``normalize=True`` (the
    :func:`call_dmr_ch` default), the per-cell global mCH rates are
    computed **once** at the wrapper level over the full sample set
    and reused across every one-vs-rest comparison via the
    ``global_a`` / ``global_b`` parameters.  Without this, each
    comparison would re-scan all input files for global rates.

    Returns
    -------
    dict
        ``{sname: output_tsv}`` for all comparisons that produced a
        result (including those skipped because the output already
        existed).

    Examples
    --------
    ::

        # Global CG one-vs-rest
        czip.call_dmr_one_vs_rest(
            indir='pseudobulk/Subclass/cz',
            reference='mm10.allc.cz',
            outdir='DMR/Subclass',
            ext='.CGN.cz',
            method='cg',
            index='mm10.CGN.index',
            jobs=16,
        )

        # CH one-vs-rest, stratified by CellClass
        czip.call_dmr_one_vs_rest(
            indir='pseudobulk/Subclass/cz',
            class_table='Subclass2CellClass.tsv',
            reference='mm10.allc.cz',
            outdir='DMR/Subclass-CH',
            ext='.CHN.cz', method='ch',
            index='mm10.CHN.index',
            bin_size=5000, jobs=32,
        )
    """
    if method not in ('cg', 'ch'):
        raise ValueError(f"method must be 'cg' or 'ch', got {method!r}")
    forbidden = {'group_a', 'group_b', 'reference', 'output',
                 'dms_output', 'jobs', 'index',
                 'global_a', 'global_b'}
    bad = forbidden & set(dmr_kwargs)
    if bad:
        raise ValueError(f"these dmr_kwargs are managed by the wrapper "
                         f"and must not be set: {sorted(bad)}")

    snames_all, paths_all = _list_pseudobulk_cz(indir, ext)
    if samples is not None:
        s2p = dict(zip(snames_all, paths_all))
        missing = [s for s in samples if s not in s2p]
        if missing:
            logger.warning(f"call_dmr_one_vs_rest: {len(missing)} requested "
                           f"samples not found in {indir}: "
                           f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        snames_all = [s for s in samples if s in s2p]
        paths_all = [s2p[s] for s in snames_all]
    if not snames_all:
        raise ValueError(f"No '{ext}' files found in {indir}")

    outdir = os.path.abspath(os.path.expanduser(outdir))
    os.makedirs(outdir, exist_ok=True)
    if dms_output_dir is not None:
        dms_output_dir = os.path.abspath(os.path.expanduser(dms_output_dir))
        os.makedirs(dms_output_dir, exist_ok=True)

    # ----- Group samples by class (or one global group) -------------
    if class_table is None:
        groups = {None: list(zip(snames_all, paths_all))}
    else:
        cls = _load_class_table(class_table, snames_all)
        groups = {}
        for s, p in zip(snames_all, paths_all):
            c = cls.get(s)
            if c is None:
                continue
            groups.setdefault(c, []).append((s, p))
        # filter undersized classes
        skipped = [c for c, lst in groups.items() if len(lst) < min_class_size]
        for c in skipped:
            logger.info(f"call_dmr_one_vs_rest: skip class {c!r} "
                        f"({len(groups[c])} < {min_class_size})")
            del groups[c]
        if not groups:
            raise ValueError(f"No class has >= {min_class_size} members")

    # ----- Optional CH global mCH rate pre-pass at wrapper level ----
    global_rates = None
    if method == 'ch' and dmr_kwargs.get('normalize', True):
        # Compute per-sample global mCH rate ONCE over all samples.
        # Each pseudobulk is a single 'cell' for our purposes.
        ref_path_abs = os.path.abspath(os.path.expanduser(reference))
        # Resolve mc/cov column from the first cz's header
        probe = Reader(paths_all[0])
        cols = probe.header['columns']
        mc_col_resolved = dmr_kwargs.get('mc_col')
        cov_col_resolved = dmr_kwargs.get('cov_col')
        if mc_col_resolved is None:
            mc_col_resolved = cols[0]
        elif isinstance(mc_col_resolved, int):
            mc_col_resolved = cols[mc_col_resolved]
        if cov_col_resolved is None:
            cov_col_resolved = cols[-1]
        elif isinstance(cov_col_resolved, int):
            cov_col_resolved = cols[cov_col_resolved]
        probe.close()

        ref_r = Reader(ref_path_abs)
        chunk_keys = list(ref_r.chunk_key2offset)
        ref_r.close()
        chroms = dmr_kwargs.get('chroms')
        if chroms is not None:
            chroms_set = set(chroms)
            chunk_keys = [k for k in chunk_keys if k[0] in chroms_set]
        idx_path_abs = (os.path.abspath(os.path.expanduser(index))
                        if index else None)
        n_proc = max(1, min(int(jobs), max(1, len(chunk_keys))))
        logger.info(f"call_dmr_one_vs_rest[CH]: pre-scanning global mCH "
                    f"rates for {len(paths_all)} pseudobulks")
        rates, _, _ = _compute_global_ch_rates(
            paths_all, ref_path_abs, idx_path_abs, chunk_keys,
            mc_col_resolved, cov_col_resolved, n_proc)
        global_rates = dict(zip(snames_all, rates.tolist()))
        logger.info(f"  global rate range "
                    f"[{rates.min():.4f}, {rates.max():.4f}], "
                    f"median={np.median(rates):.4f}")
        # Drop readers opened by the global-rate pre-scan so the
        # main per-comparison loop doesn't keep N mmaps alive at once.
        _close_cached_readers()

    # ----- Run one-vs-rest within each group ------------------------
    caller = call_dmr if method == 'cg' else call_dmr_ch
    results = {}
    n_total = sum(len(lst) for lst in groups.values())
    n_done = 0
    for cls_label, members in groups.items():
        if cls_label is None:
            sub_outdir = outdir
            sub_dms_dir = dms_output_dir
        else:
            sub_outdir = os.path.join(outdir, str(cls_label).replace(' ', '_'))
            os.makedirs(sub_outdir, exist_ok=True)
            sub_dms_dir = (os.path.join(dms_output_dir,
                                        str(cls_label).replace(' ', '_'))
                           if dms_output_dir else None)
            if sub_dms_dir:
                os.makedirs(sub_dms_dir, exist_ok=True)

        for sname, spath in members:
            n_done += 1
            out_tsv = os.path.join(sub_outdir, f"{sname}.dmr.tsv")
            if (not overwrite) and os.path.exists(out_tsv):
                logger.info(f"[{n_done}/{n_total}] skip existing {out_tsv}")
                results[(cls_label, sname)] = out_tsv
                continue
            rest_paths = [p for s, p in members if s != sname]
            if not rest_paths:
                logger.info(f"[{n_done}/{n_total}] {sname}: no other "
                            f"members in class {cls_label!r}, skipping")
                continue

            call_kwargs = dict(dmr_kwargs)
            call_kwargs.update(
                group_a=[spath],
                group_b=rest_paths,
                reference=reference,
                output=out_tsv,
                index=index,
                jobs=jobs,
                group_names=(sname, 'rest'),
            )
            if sub_dms_dir is not None:
                call_kwargs['dms_output'] = os.path.join(
                    sub_dms_dir, f"{sname}.dms.tsv")
            if method == 'ch' and global_rates is not None:
                call_kwargs['global_a'] = [global_rates[sname]]
                call_kwargs['global_b'] = [global_rates[s]
                                           for s, _ in members
                                           if s != sname]

            logger.info(f"[{n_done}/{n_total}] "
                        f"{cls_label or 'global'}: {sname} vs rest "
                        f"({len(rest_paths)} samples)")
            caller(**call_kwargs)
            results[(cls_label, sname)] = out_tsv

    # Auto-merge per-sample TSVs into a single long-format table with
    # derived metrics (delta_meth/delta_rate, direction, log2fc, ...).
    if auto_merge:
        mk = dict(merge_kwargs) if merge_kwargs else {}
        # Strip Nones so merge_dmr_results' own defaults apply.
        mk = {k: v for k, v in mk.items() if v is not None}
        out_fmt = mk.get('output_format', 'tsv') or 'tsv'
        ext_out = '.parquet' if out_fmt == 'parquet' else '.tsv'
        merged_path = os.path.join(outdir, f'merged_dmr{ext_out}')
        try:
            merge_dmr_results(outdir, method=method,
                              output=merged_path,
                              class_table=class_table,
                              **mk)
        except Exception as e:
            logger.warning(f"call_dmr_one_vs_rest: merge step failed: {e}")

    return results


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR (q-values).  NaNs propagate."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    valid = ~np.isnan(p)
    q = np.full(n, np.nan, dtype=np.float64)
    if not valid.any():
        return q
    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    qv_sorted = pv[order] * m / ranks
    # enforce monotonicity from the largest p downwards
    qv_sorted = np.minimum.accumulate(qv_sorted[::-1])[::-1]
    qv_sorted = np.clip(qv_sorted, 0.0, 1.0)
    qv = np.empty(m, dtype=np.float64)
    qv[order] = qv_sorted
    q[valid] = qv
    return q


def merge_dmr_results(outdir, method='cg',
                      output=None,
                      class_table=None,
                      pattern='.dmr.tsv',
                      add_metrics=True,
                      add_fdr=True,
                      output_format=None):
    """Merge per-sample DMR TSVs produced by :func:`call_dmr_one_vs_rest`
    into a single long-format table, attaching the sample name (and
    class label, when stratified) plus per-DMR metrics.

    Output columns
    --------------
    Always: ``chrom, start, end, n_dms, p_min, frac_delta_mean, state,
    sname, class, length, region_id``.

    For ``method='cg'``:
        * ``delta_meth`` (= ``frac_delta_mean``, alias for clarity)
        * ``direction`` (``'hypo'`` if delta<0 else ``'hyper'``)

    For ``method='ch'``:
        * ``delta_rate`` (= ``frac_delta_mean``; rate_a - rate_b)
        * ``log2fc``  (sign-preserving log2 fold change inferred from
          ``state`` and absolute delta; precise per-bin values live in
          the ``--dms_output`` files when those were written)
        * ``direction`` (``'hypo'`` / ``'hyper'``)

    Parameters
    ----------
    outdir : str
        Directory containing per-sample DMR TSVs (the directory passed
        as ``outdir`` to :func:`call_dmr_one_vs_rest`).  Stratified
        layouts (one subdirectory per class) are auto-detected.
    method : {'cg', 'ch'}
        Determines which extra metrics columns are added.
    output : str or None
        If given, also write the merged TSV to this path.
    class_table : same as :func:`call_dmr_one_vs_rest`
        Optional override.  When omitted, the ``class`` column is
        filled from the directory layout if present.
    pattern : str
        File suffix of per-sample DMR files (default ``.dmr.tsv``).
    add_metrics : bool
        If False, only stitch the rows together without computing
        derived metric columns.

    Returns
    -------
    pandas.DataFrame
        Long-format merged table.
    """
    outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.isdir(outdir):
        raise NotADirectoryError(outdir)

    rows = []
    # Walk one or two levels: <outdir>/<sname>.dmr.tsv (global) or
    # <outdir>/<class>/<sname>.dmr.tsv (stratified).
    for entry in sorted(os.listdir(outdir)):
        full = os.path.join(outdir, entry)
        if os.path.isdir(full):
            cls = entry
            for f in sorted(os.listdir(full)):
                if f.endswith(pattern):
                    rows.append((cls, f[:-len(pattern)],
                                 os.path.join(full, f)))
        elif entry.endswith(pattern):
            rows.append((None, entry[:-len(pattern)], full))

    if not rows:
        return pd.DataFrame()

    # Optional class override
    if class_table is not None:
        snames = [s for _, s, _ in rows]
        cls_map = _load_class_table(class_table, snames)
    else:
        cls_map = None

    parts = []
    for cls, sname, path in rows:
        try:
            df = pd.read_csv(path, sep='\t')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if df.shape[0] == 0:
            continue
        df['sname'] = sname
        df['class'] = (cls_map.get(sname) if cls_map is not None else cls)
        parts.append(df)

    if not parts:
        cols = ['chrom', 'start', 'end', 'n_dms', 'p_min',
                'frac_delta_mean', 'state', 'sname', 'class']
        merged = pd.DataFrame(columns=cols)
    else:
        merged = pd.concat(parts, ignore_index=True)
        merged['length'] = (merged['end'].astype(int)
                            - merged['start'].astype(int))
        merged['region_id'] = (merged['chrom'].astype(str) + ':'
                               + merged['start'].astype(str) + '-'
                               + merged['end'].astype(str))

        if add_metrics and 'frac_delta_mean' in merged:
            delta = merged['frac_delta_mean'].astype(float)
            merged['direction'] = np.where(delta < 0, 'hypo', 'hyper')
            if method == 'cg':
                merged['delta_meth'] = delta
            else:
                merged['delta_rate'] = delta
                # Approximate log2fc on the magnitude: ratio = 1 + |delta|
                # cannot be recovered without per-bin rates; use NaN unless
                # rate_a / rate_b are present (e.g. when merging dms_output).
                if {'rate_a', 'rate_b'}.issubset(merged.columns):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ra = merged['rate_a'].astype(float).clip(lower=1e-12)
                        rb = merged['rate_b'].astype(float).clip(lower=1e-12)
                        merged['log2fc'] = np.log2(ra / rb)
                else:
                    merged['log2fc'] = np.nan

        # Per-(sname, class) BH-FDR on p_min, written as ``q_min``.
        if add_fdr and 'p_min' in merged.columns and len(merged):
            q = np.full(len(merged), np.nan, dtype=np.float64)
            grp_keys = ['sname']
            if 'class' in merged.columns:
                grp_keys = ['class', 'sname']
            p_arr = merged['p_min'].to_numpy()
            # ``groupby.indices`` already returns positional int arrays,
            # avoiding the per-group ``np.asarray(list(...))`` round-trip.
            for idx in merged.groupby(grp_keys, dropna=False).indices.values():
                q[idx] = _bh_fdr(p_arr[idx])
            merged['q_min'] = q

    if output is not None:
        output = os.path.abspath(os.path.expanduser(output))
        fmt = output_format
        if fmt is None:
            fmt = 'parquet' if output.endswith(('.parquet', '.pq')) else 'tsv'
        if fmt == 'parquet':
            merged.to_parquet(output, index=False)
        else:
            merged.to_csv(output, sep='\t', index=False)
        logger.info(f"merge_dmr_results: {len(merged)} DMRs across "
                    f"{merged['sname'].nunique() if len(merged) else 0} "
                    f"samples written to {output}")
    return merged


def consensus_dmr(merged, slop=0, min_samples=1, by_direction=True,
                  output=None):
    """Collapse a long-format merged DMR table into consensus regions.

    Performs a single-pass interval merge per chromosome (optionally
    per ``direction``) on (chrom, start, end) intervals extended by
    ``slop`` bp on each side; reports per-region recurrence statistics
    across ``sname`` (and ``class`` when present).

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`merge_dmr_results`.
    slop : int
        Extra bp to add on each side before merging adjacent or nearby
        DMRs (default 0; e.g. 200 to bridge small gaps).
    min_samples : int
        Drop consensus regions supported by fewer than this many
        distinct snames.
    by_direction : bool
        If True (default) and a ``direction`` column is present, merge
        hypo / hyper DMRs separately.
    output : str or None
        If given, write the consensus table to this TSV / parquet path.

    Returns
    -------
    pandas.DataFrame with columns
        ``chrom, start, end, n_samples, n_classes, snames, classes,
        direction, mean_delta, min_p_min``.
    """
    if merged is None or len(merged) == 0:
        return pd.DataFrame(columns=['chrom', 'start', 'end',
                                     'n_samples', 'n_classes',
                                     'snames', 'classes', 'direction',
                                     'mean_delta', 'min_p_min'])
    df = merged.copy()
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    has_dir = by_direction and 'direction' in df.columns
    delta_col = ('delta_meth' if 'delta_meth' in df.columns
                 else ('delta_rate' if 'delta_rate' in df.columns
                       else 'frac_delta_mean'))
    rows = []
    group_keys = ['chrom']
    if has_dir:
        group_keys.append('direction')
    for keys, sub in df.groupby(group_keys, sort=True):
        sub = sub.sort_values('start').reset_index(drop=True)
        starts = (sub['start'].to_numpy() - int(slop)).clip(min=0)
        ends = sub['end'].to_numpy() + int(slop)
        cur_s, cur_e = starts[0], ends[0]
        cur_idx = [0]
        for i in range(1, len(sub)):
            if starts[i] <= cur_e:
                cur_e = max(cur_e, ends[i])
                cur_idx.append(i)
            else:
                rows.append((keys, sub.iloc[cur_idx], cur_s, cur_e))
                cur_s, cur_e = starts[i], ends[i]
                cur_idx = [i]
        rows.append((keys, sub.iloc[cur_idx], cur_s, cur_e))

    out = []
    for keys, members, s, e in rows:
        if isinstance(keys, tuple):
            chrom = keys[0]
            direction = keys[1] if has_dir else ''
        else:
            chrom = keys
            direction = ''
        snames = sorted(members['sname'].astype(str).unique().tolist())
        classes = (sorted(members['class'].dropna().astype(str).unique().tolist())
                   if 'class' in members.columns else [])
        if len(snames) < int(min_samples):
            continue
        # restore real BED extent (undo slop)
        true_s = max(0, int(members['start'].min()))
        true_e = int(members['end'].max())
        out.append({
            'chrom': chrom, 'start': true_s, 'end': true_e,
            'n_samples': len(snames), 'n_classes': len(classes),
            'snames': ','.join(snames),
            'classes': ','.join(classes),
            'direction': direction,
            'mean_delta': float(members[delta_col].astype(float).mean())
                            if delta_col in members.columns else float('nan'),
            'min_p_min': float(members['p_min'].astype(float).min())
                            if 'p_min' in members.columns else float('nan'),
        })
    cons = pd.DataFrame(out)
    if not has_dir and 'direction' in cons.columns:
        cons = cons.drop(columns=['direction'])
    cons = cons.sort_values(['chrom', 'start']).reset_index(drop=True)
    if output is not None:
        output = os.path.abspath(os.path.expanduser(output))
        if output.endswith(('.parquet', '.pq')):
            cons.to_parquet(output, index=False)
        else:
            cons.to_csv(output, sep='\t', index=False)
        logger.info(f"consensus_dmr: {len(cons)} consensus regions "
                    f"written to {output}")
    return cons


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
            logger.debug(f"{infile} {chrom}")
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
    # Sequential whole-file walks ahead; tell the kernel to evict
    # already-read pages so the multi-GB ref doesn't pin our RSS.
    reader.advise_sequential()
    ref_reader.advise_sequential()

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

            logger.debug(f"  {chrom}: {len(pos)} sites, "
                         f"{int(sig.sum())} pseudo-reads")

            # Release this chunk's pages on both readers.
            reader.release_chunk(dim)
            ref_reader.release_chunk(dim)

    reader.close()
    ref_reader.close()
    if index_reader:
        index_reader.close()

    logger.info(f"Total pseudo-reads: {total_reads}")

    # ---- Step 4: Sort BED and run MACS3 peak calling ----
    sorted_bed = bed_path.replace('.bed', '.sorted.bed')
    logger.info("Sorting BED file...")
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

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    os.remove(bed_path)
    if not keep_bed:
        os.remove(sorted_bed)
    else:
        logger.info(f"Pseudo-reads BED kept at: {sorted_bed}")

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


def combp(input, outdir="cpv", jobs=24, dist=300, temp=True, bed=False):
    """
    Run comb-p on a fisher result matrix (generated by `merge_cz -f fisher`),
    /usr/bin/time -f "%e\t%M\t%P" cytozip combp -i major_type.fisher.txt.gz -n 64
    Run one samples (all chromosomes), 8308.18(2.3h) 65053112(62G)        321%

    Parameters
    ----------
    input : path
        path to result from merge_cz -f fisher.
    outdir : path
    jobs : int
        number of parallel processes (CPUs).
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
        logger.info("Please install cpv using: pip install git+https://github.com/DingWB/combined-pvalues")

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
        pool = multiprocessing.Pool(jobs)
        tasks = []
        logger.info("Splitting matrix into different samples and chroms.")
        for chrom in chroms:
            task = pool.apply_async(__split_mat,
                                   (infile, chrom, snames, bed_dir, 5))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()
    else:
        logger.info("bed directory existed, skip split matrix into bed files.")

    tmpdir = os.path.join(outdir, "tmp")
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    pool = multiprocessing.Pool(jobs)
    tasks = []
    logger.info("Running cpv..")
    acf_dist = int(round(dist / 3, -1))
    for chrom in chroms:
        for sname in snames:
            bed_file = os.path.join(bed_dir, f"{sname}.{chrom}.bed")
            prefix = os.path.join(tmpdir, f"{sname}.{chrom}")
            output = os.path.join(tmpdir, f"{sname}.{chrom}.regions-p.bed.gz")
            if os.path.exists(output):
                continue
            task = pool.apply_async(cpv_pipeline,
                                   (4, None, dist, acf_dist, prefix,
                                    0.05, 0.05, "refGene", [bed_file], True, 1, None, False,
                                    None, True))
            tasks.append(task)
    for task in tasks:
        task.get()
    pool.close()
    pool.join()

    logger.info("Merging cpv results..")
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
