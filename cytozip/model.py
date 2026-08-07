#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py — Site-likelihood cell-type classifier for shallow snm3C-seq.

Given a single-cell ``.cz`` file (per-cytosine ``mc``/``cov`` over *all*
cytosines, CpG and CpH together) and a set of cell-type-level pseudobulk
``.cz`` files (deep-sequencing reference), predict which cell type the query
cell most likely belongs to.

Method (naive-Bayes / methylation-frequency deconvolution)
----------------------------------------------------------
For every discriminative cytosine ``c`` and candidate type ``t`` the reference
methylation frequency is estimated with a Beta-shrinkage estimator::

    theta[c,t] = (m[c,t] + alpha0) / (n[c,t] + alpha0 + beta0)

where ``m`` = methylated count, ``n`` = coverage. The frequency is kept
*continuous* (never binarized) because the discriminative signal lives in the
intermediate frequencies.

The aggregated per-site Bernoulli log-likelihood of a query cell (with ``mc``
methylated and ``cov`` total observations at each site) under type ``t`` is::

    loglik[t] = sum_c ( mc[c] * log(theta[c,t])
                        + (cov[c] - mc[c]) * log(1 - theta[c,t]) )

CpG and CpH have very different backgrounds, so the two contexts are modelled
as independent channels (each with its own frequencies and shrinkage prior)
and summed in log-space with weights ``lambda_cg`` / ``lambda_ch``::

    logpost[t] = log(pi[t]) + lambda_cg * loglik_cg[t] + lambda_ch * loglik_ch[t]

The CpG / CpH split is derived from the ``context`` column (2nd base ``G`` ->
CpG, ``A``/``C``/``T`` -> CpH) so a single ``.cz`` file carrying all cytosines
is enough — you do not pre-split CG / CH into separate files. A softmax over
types yields calibrated probabilities that support abstention.

All input ``.cz`` files must be aligned to the same reference axis (one row per
reference cytosine, in reference order), so positions align by row index across
the query and all references. The cell-type ``.cz`` normally store only
``mc``/``cov`` (no ``pos``/``context``), so the per-row ``context`` is taken
from a separate ``reference=`` ``.cz`` (a ``build_ref`` output with
``pos, strand, context``); if the files happen to carry their own ``context``
column, ``reference`` can be omitted.

@author: DingWB
"""
import os
import json
import glob
import shutil
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from .cz import Reader, np, pd


# Map a struct format char -> numpy dtype so mc/cov are kept in their native
# (small) integer dtype instead of being upcast to float64 on read. Upcasting
# a whole-genome all-C axis to float64 is a major, avoidable memory cost.
_STRUCT_NP = {
    'B': np.uint8, 'H': np.uint16, 'I': np.uint32, 'L': np.uint32,
    'Q': np.uint64, 'b': np.int8, 'h': np.int16, 'i': np.int32,
    'l': np.int32, 'q': np.int64, 'f': np.float32, 'd': np.float64,
}


def _np_dtype_for(fmt):
    """numpy dtype for a struct format string (e.g. ``'B'`` -> uint8)."""
    return np.dtype(_STRUCT_NP.get(str(fmt)[-1], np.float64))


def _abspath(path):
    """Absolutize a local path; pass remote URLs (http/s3/gs/...) through.

    ``os.path.abspath`` would corrupt a URL (prepend cwd, collapse ``//``), so
    any ``scheme://`` path except ``file://`` is returned unchanged for
    ``cz.Reader`` to open over HTTP range / fsspec.
    """
    if isinstance(path, str) and '://' in path and not path.startswith('file://'):
        return path
    return os.path.abspath(os.path.expanduser(path))


# ==========================================================
def _resolve_n_jobs(n_jobs):
    """Normalize ``n_jobs`` to a positive worker count.

    ``None``/``1``/``0`` -> serial (1 worker). A negative value (e.g. ``-1``)
    -> all available CPUs. Any positive integer is used as-is, capped at the
    CPU count. Reading/decompressing ``.cz`` (zlib + Cython) and the numpy
    scoring both release the GIL, so threads give real multi-core speedup
    without pickling the model.
    """
    cpu = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        return cpu
    if n_jobs == 0:
        return 1
    return min(n_jobs, cpu)


# ==========================================================
def estimate_theta(mc, cov, alpha0=1.0, beta0=1.0, dtype=np.float32):
    """Beta-shrinkage estimate of the per-site methylation frequency.

    Parameters
    ----------
    mc, cov : np.ndarray
        Methylated-count and coverage arrays (same shape).
    alpha0, beta0 : float
        Beta-prior pseudocounts. The shrinkage prior mean is
        ``alpha0 / (alpha0 + beta0)``; both must be > 0 so the estimate
        never hits exactly 0 or 1 (which would make ``log(theta)`` or
        ``log(1 - theta)`` diverge).
    dtype : np.dtype
        Output floating dtype (default ``float32``).

    Returns
    -------
    np.ndarray
        ``theta`` in the open interval (0, 1), same shape as inputs.
    """
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("alpha0 and beta0 must both be > 0 for shrinkage.")
    mc = np.asarray(mc, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    theta = (mc + alpha0) / (cov + alpha0 + beta0)
    return theta.astype(dtype)


# ==========================================================
def estimate_beta_prior(mc, cov, min_cov=2, df_correction=True, eps=1e-6):
    """Empirical-Bayes Beta prior ``(alpha0, beta0)`` via Beta-Binomial MoM.

    Estimates the shrinkage prior ``Beta(alpha0, beta0)`` for
    :func:`estimate_theta` from the *global* ``mc`` / ``cov`` of a reference,
    correcting for the finite-coverage (binomial) sampling noise that a plain
    Beta method-of-moments (e.g. ALLCools ``calculate_posterior_mc_frac``)
    ignores. This should be estimated on a DEEP pseudobulk reference and, since
    CpG / CpH have different backgrounds, run separately per context.

    Model: site ``i`` has ``cov[i]`` reads and ``mc[i]`` methylated, with a
    latent rate ``p_i ~ Beta(alpha, beta)`` and ``mc[i] | p_i ~
    Binomial(cov[i], p_i)``. Parameterize by mean ``mu = alpha/(alpha+beta)``,
    concentration ``kappa = alpha+beta`` and intra-class correlation
    ``rho = 1/(kappa+1)``. The marginal moments are::

        E[mc_i]   = cov_i * mu
        Var[mc_i] = cov_i * mu*(1-mu) * (1 + (cov_i-1)*rho)

    Method of moments (handles unequal coverage):

    * ``mu_hat = sum(mc) / sum(cov)``  (first moment = global rate)
    * Pearson over-dispersion statistic
      ``X = sum( (mc_i - cov_i*mu)^2 / (cov_i*mu*(1-mu)) )`` with
      ``E[X] = N + rho * sum(cov_i - 1)`` (``N`` = #sites), giving
      ``rho_hat = (X - N) / sum(cov_i - 1)`` (``N -> N-1`` if
      ``df_correction``; Tarone 1979).
    * ``kappa = (1-rho)/rho``; ``alpha0 = mu*kappa``; ``beta0 = (1-mu)*kappa``.

    Parameters
    ----------
    mc, cov : np.ndarray
        Methylated-count and coverage arrays (same shape) for one context.
    min_cov : int
        Ignore sites with ``cov < min_cov`` (default 2; ``cov=1`` contributes
        nothing to ``sum(cov-1)`` and is noisy).
    df_correction : bool
        Use ``N-1`` instead of ``N`` in ``rho_hat`` (Tarone d.f. correction).
    eps : float
        Clamp for ``mu`` in ``(eps, 1-eps)`` and ``rho`` in ``(eps, 1-eps)``
        so ``alpha0``/``beta0`` stay finite and > 0.

    Returns
    -------
    (alpha0, beta0) : tuple of float
        Beta-prior pseudocounts, both > 0.
    """
    mc = np.asarray(mc, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64).ravel()
    keep = cov >= max(min_cov, 1)
    mc, cov = mc[keep], cov[keep]
    n_sites = cov.size
    if n_sites == 0 or cov.sum() == 0:
        raise ValueError("no covered sites (cov >= min_cov) to estimate prior.")

    mu = mc.sum() / cov.sum()                      # first moment -> Beta mean
    mu = min(max(mu, eps), 1.0 - eps)

    denom_var = mu * (1.0 - mu)
    # Pearson-type over-dispersion statistic (per-site binomial variance scaled)
    X = np.sum((mc - cov * mu) ** 2 / (cov * denom_var))
    sum_cov_minus_1 = np.sum(cov - 1.0)            # = sum(cov) - n_sites
    if sum_cov_minus_1 <= 0:
        # every kept site has cov == 1 -> no info on dispersion; fall back to
        # a weak prior with the estimated mean.
        kappa = 2.0
        return float(mu * kappa), float((1.0 - mu) * kappa)

    n_eff = (n_sites - 1) if df_correction else n_sites
    rho = (X - n_eff) / sum_cov_minus_1            # intra-class correlation
    rho = min(max(rho, eps), 1.0 - eps)            # clamp to (0, 1)

    kappa = (1.0 - rho) / rho                       # = alpha + beta
    return float(mu * kappa), float((1.0 - mu) * kappa)


def _beta_prior_from_moments(S1, S2, S3, N, df_correction=True, eps=1e-6):
    """Beta-Binomial MoM prior from streamed sufficient statistics.

    Numerically identical to :func:`estimate_beta_prior` but consumes only
    scalar accumulators, so the caller never has to materialise (let alone
    pool across cell types) the per-site ``mc``/``cov`` arrays::

        S1 = sum(mc)   S2 = sum(cov)   S3 = sum(mc**2 / cov)   N = #sites

    all taken over sites with ``cov >= min_cov`` (the same filter
    :func:`estimate_beta_prior` applies). The Pearson over-dispersion
    statistic expands to ``X = (S3 - 2*mu*S1 + mu**2*S2) / (mu*(1-mu))`` and
    ``sum(cov - 1) = S2 - N``.
    """
    if N == 0 or S2 == 0:
        raise ValueError("no covered sites (cov >= min_cov) to estimate prior.")
    mu = S1 / S2
    mu = min(max(mu, eps), 1.0 - eps)
    denom_var = mu * (1.0 - mu)
    X = (S3 - 2.0 * mu * S1 + mu * mu * S2) / denom_var
    sum_cov_minus_1 = S2 - N
    if sum_cov_minus_1 <= 0:
        kappa = 2.0
        return float(mu * kappa), float((1.0 - mu) * kappa)
    n_eff = (N - 1) if df_correction else N
    rho = (X - n_eff) / sum_cov_minus_1
    rho = min(max(rho, eps), 1.0 - eps)
    kappa = (1.0 - rho) / rho
    return float(mu * kappa), float((1.0 - mu) * kappa)


# ==========================================================
def _read_mc_cov(path, mc_col='mc', cov_col='cov', chunk_keys=None,
                 chunk_lens=None):
    """Read (mc, cov) as 1-D arrays concatenated over ``chunk_keys``.

    Parameters
    ----------
    path : str
        Path to a ``.cz`` file with ``mc_col`` / ``cov_col`` columns.
    mc_col, cov_col : str
        Column names (in the .cz ``header['columns']``) holding the
        methylated count and coverage.
    chunk_keys : list of tuple or None
        Chunk keys (e.g. ``[('chr1',), ('chr2',), ...]``) in the desired
        concatenation order. ``None`` uses ``sorted(chunk_key2offset)``.
    chunk_lens : dict {chunk_key: int} or None
        Per-chunk row counts from the reference axis. When given, a chunk
        that is *absent* from this ``.cz`` (e.g. ``chrL`` present in the
        reference but empty in a single cell) is zero-filled to that length
        instead of raising, so all files stay aligned to the reference axis.

    Returns
    -------
    (chunk_keys, mc, cov) : (list, np.ndarray float64, np.ndarray float64)
        The keys actually used and the concatenated arrays.
    """
    path = _abspath(path)
    r = Reader(path)
    try:
        cols = list(r.header['columns'])
        if mc_col not in cols or cov_col not in cols:
            raise ValueError(
                f"{path}: expected columns {mc_col!r} and {cov_col!r}, "
                f"got {cols!r}")
        mi, ci = cols.index(mc_col), cols.index(cov_col)
        mc_dt = _np_dtype_for(r.header['formats'][mi])
        cov_dt = _np_dtype_for(r.header['formats'][ci])
        if chunk_keys is None:
            chunk_keys = sorted(r.chunk_key2offset.keys())
        mc_parts, cov_parts = [], []
        for k in chunk_keys:
            if k not in r.chunk_key2offset:
                # Missing chunk: zero-fill to the reference length if known,
                # otherwise it is a genuine misalignment -> error. Zero-fill in
                # the native dtype so the concatenation stays compact.
                if chunk_lens is not None and k in chunk_lens:
                    nrow = int(chunk_lens[k])
                    mc_parts.append(np.zeros(nrow, dtype=mc_dt))
                    cov_parts.append(np.zeros(nrow, dtype=cov_dt))
                    continue
                raise ValueError(f"{path}: missing chunk {k!r}")
            arr = r.chunk2numpy(k)
            mc_parts.append(np.asarray(arr[f'f{mi}']))
            cov_parts.append(np.asarray(arr[f'f{ci}']))
    finally:
        r.close()
    # Keep the native (small-int) dtype — do NOT upcast to float64. The theta /
    # prior math converts only the small masked/selected subsets it touches.
    if mc_parts:
        mc = np.concatenate(mc_parts)
        cov = np.concatenate(cov_parts)
    else:
        mc = np.zeros(0, dtype=mc_dt)
        cov = np.zeros(0, dtype=cov_dt)
    return chunk_keys, mc, cov


def _read_context(path, context_col='context', chunk_keys=None):
    """Read the per-row ``context`` bytes concatenated over ``chunk_keys``.

    Parameters
    ----------
    path : str
        Path to a ``.cz`` file carrying a ``context_col`` column (e.g. a
        ``build_ref`` reference with ``pos, strand, context``, or any file
        that stores its own context).
    context_col : str
        Name of the context column (in the .cz ``header['columns']``).
    chunk_keys : list of tuple or None
        Chunk order; ``None`` uses ``sorted(chunk_key2offset)``.

    Returns
    -------
    (chunk_keys, ctx, chunk_lens) : (list, np.ndarray of ``S{n}`` bytes, dict)
        The keys used, the concatenated fixed-width byte strings (e.g.
        ``b'CGN'`` / ``b'CAC'``), and a ``{chunk_key: row_count}`` map that
        defines the reference axis length of each chunk (used to zero-fill
        chunks missing from a query/pseudobulk).
    """
    path = _abspath(path)
    r = Reader(path)
    try:
        cols = list(r.header['columns'])
        if context_col not in cols:
            raise ValueError(
                f"{path}: no {context_col!r} column (got {cols!r}); pass "
                f"context= a reference .cz that has one.")
        ci = cols.index(context_col)
        width = struct.calcsize(r.header['formats'][ci])
        if chunk_keys is None:
            chunk_keys = sorted(r.chunk_key2offset.keys())
        parts = []
        chunk_lens = {}
        for k in chunk_keys:
            if k not in r.chunk_key2offset:
                raise ValueError(f"{path}: missing chunk {k!r}")
            arr = r.chunk2numpy(k)
            fld = np.asarray(arr[f'f{ci}'])
            part = np.frombuffer(fld.tobytes(), dtype=f'S{width}')
            chunk_lens[k] = int(part.size)
            parts.append(part)
    finally:
        r.close()
    ctx = np.concatenate(parts) if parts else np.zeros(0, dtype='S1')
    return chunk_keys, ctx, chunk_lens


def _read_mc_cov_by_chunk(path, mc_col='mc', cov_col='cov', needed_keys=None,
                          chunk_lens=None):
    """Return ``{chunk_key: (mc, cov)}`` for the needed chunks present in a file.

    Unlike :func:`_read_mc_cov` (which concatenates the whole reference axis
    and zero-fills gaps), this reads only the requested chunks and simply
    **skips** any that are absent from the file — the per-chunk scorer treats
    a missing chunk as a zero contribution, so no alignment/zero-fill is
    needed.

    Parameters
    ----------
    path : str
        Path to a ``.cz`` file with ``mc_col`` / ``cov_col`` columns.
    mc_col, cov_col : str
        Methylated-count and coverage column names (in the .cz
        ``header['columns']``).
    needed_keys : iterable of tuple or None
        Chunk keys the model actually needs (those carrying selected sites).
        ``None`` reads every chunk in the file.
    chunk_lens : dict {chunk_key: int} or None
        Reference row count per chunk; when given, a present chunk whose row
        count disagrees raises (guards against within-chrom misalignment).

    Returns
    -------
    dict {chunk_key: (np.ndarray float64, np.ndarray float64)}
    """
    path = _abspath(path)
    r = Reader(path)
    out = {}
    try:
        cols = list(r.header['columns'])
        if mc_col not in cols or cov_col not in cols:
            raise ValueError(
                f"{path}: expected columns {mc_col!r} and {cov_col!r}, "
                f"got {cols!r}")
        mi, ci = cols.index(mc_col), cols.index(cov_col)
        dims = r.header.get('chunk_dims', []) or []
        if len(dims) > 1:
            raise ValueError(
                f"{path}: multi-cell (cat) .cz with chunk_dims={list(dims)}; "
                f"predict()/predict_batch() handle single-cell files only. Pass "
                f"it to predict_cell_type(query=...) (auto-detected) or "
                f"CellTypeClassifier.predict_multicell().")
        keys = (list(r.chunk_key2offset.keys()) if needed_keys is None
                else list(needed_keys))
        for k in keys:
            if k not in r.chunk_key2offset:
                continue  # absent chunk -> skip (zero contribution)
            arr = r.chunk2numpy(k)
            mc = np.asarray(arr[f'f{mi}'])
            cov = np.asarray(arr[f'f{ci}'])
            if (chunk_lens is not None and k in chunk_lens
                    and mc.size != int(chunk_lens[k])):
                raise ValueError(
                    f"{path}: chunk {k!r} has {mc.size} rows but the "
                    f"reference axis has {int(chunk_lens[k])}; the query must "
                    f"be aligned to the same reference axis within each chrom.")
            out[k] = (mc, cov)
    finally:
        r.close()
    return out


def _context_masks(ctx):
    """Split a context byte array into (CpG mask, CpH mask).

    A cytosine is CpG when the 2nd base is ``G`` (e.g. ``CGN``); it is CpH
    when the 2nd base is ``A``/``C``/``T`` (e.g. ``CAC``, ``CTG``). Anything
    else (e.g. ``CNN``) belongs to neither mask and is ignored.

    Parameters
    ----------
    ctx : np.ndarray of ``S{n}`` bytes

    Returns
    -------
    (cg_mask, ch_mask) : (np.ndarray bool, np.ndarray bool)
    """
    n = ctx.size
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    width = ctx.dtype.itemsize
    if width < 2:
        raise ValueError("context strings must be at least 2 bases wide.")
    second = np.frombuffer(ctx.tobytes(), dtype='S1').reshape(n, width)[:, 1]
    second = np.char.upper(second)
    cg = second == b'G'
    ch = np.isin(second, [b'A', b'C', b'T'])
    return cg, ch


def _select_from_score(score, top=None, min_range=0.0):
    """Select sites from a per-site discriminative ``score`` array.

    ``score`` is the across-type frequency range at each site. Selection is
    identical to :func:`_select_discriminative` but takes the pre-computed
    range so the caller need not hold the full ``(n_sites, n_types)`` theta
    matrix in memory. Returns sorted 1-D site indices.

    Sites are kept only when ``score > min_range`` (strict). With the default
    ``min_range=0`` this drops every site whose ``score`` is exactly 0, i.e.
    where all cell types share the same theta: such a site adds an identical
    constant to every type's log-posterior, which cancels in the softmax /
    argmax, so it is useless for classification (dropping it is lossless).
    """
    if score.size == 0:
        return np.zeros(0, dtype=np.int64)
    keep = np.where(score > min_range)[0]
    if top is None:
        return keep
    order = keep[np.argsort(score[keep])[::-1]]
    if isinstance(top, float):
        if not 0.0 < top <= 1.0:
            raise ValueError("float `top` must be in (0, 1].")
        k = max(1, int(round(top * order.size)))
    else:
        k = int(top)
    return np.sort(order[:k])


def _select_discriminative(theta, top=None, min_range=0.0):
    """Select discriminative sites from a ``(n_sites, n_types)`` theta matrix.

    The per-site score is the across-type frequency range
    ``theta.max(axis=1) - theta.min(axis=1)``: sites where types disagree
    most carry the most classification signal, while constant-methylation
    sites (range ~0) are uninformative and dropped.

    Parameters
    ----------
    theta : np.ndarray, shape (n_sites, n_types)
    top : None, int, or float
        If ``None``, keep every site with score > ``min_range``. If an
        ``int``, keep the top-``top`` sites by score. If a ``float`` in
        (0, 1], keep the top fraction of sites.
    min_range : float
        Sites are kept only when their across-type range is strictly greater
        than this (default 0.0, which drops the non-discriminative
        score-0 sites).

    Returns
    -------
    np.ndarray
        Sorted 1-D array of selected site indices (into the channel subset).
    """
    if theta.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    score = theta.max(axis=1) - theta.min(axis=1)
    return _select_from_score(score, top=top, min_range=min_range)


# ==========================================================
def _solve_fractions(theta, beta, weight=None, sum_to_one=True,
                     allow_unknown=False):
    """Reference-based deconvolution: bulk ``beta`` -> cell-type fractions.

    Models the bulk methylation level at each site as a (coverage-weighted)
    linear mixture of the reference cell types::

        beta[c] ~= sum_t f_t * theta[c, t]

    and solves for the mixing fractions ``f`` under a non-negativity constraint
    (Houseman / CIBERSORT-style constrained least squares). Coverage weighting
    turns the fit into a weighted least squares so deeply covered sites count
    more (``weight = cov``); an all-ones weight recovers ordinary least
    squares (used for methylation-array beta values, which have no coverage).

    Parameters
    ----------
    theta : np.ndarray, shape (n_sites, n_types)
        Reference per-site methylation frequency for each cell type (the
        design matrix), in the open interval (0, 1).
    beta : np.ndarray, shape (n_sites,)
        Observed bulk methylation level per site (``mc / cov`` for WGBS, or the
        array beta value).
    weight : np.ndarray or None
        Per-site non-negative weight (``cov`` for WGBS). ``None`` -> equal
        weights (ordinary least squares).
    sum_to_one : bool
        If True (default), constrain ``sum_t f_t == 1`` (a full simplex).
        If False, only ``f_t >= 0`` is enforced (plain NNLS); fractions then
        need not sum to 1 (useful as a diagnostic).
    allow_unknown : bool
        Only meaningful with ``sum_to_one=True``. If True, relax the equality
        to ``sum_t f_t <= 1`` so an unmodelled ("unknown") compartment can
        absorb the remainder ``1 - sum_t f_t`` (returned separately by the
        caller). Default False.

    Returns
    -------
    (fractions, r2) : (np.ndarray, float)
        ``fractions`` is the (n_types,) non-negative fraction vector; ``r2`` is
        the (weighted) coefficient of determination of the fit.
    """
    from scipy.optimize import nnls, minimize

    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("theta must be a 2-D (n_sites, n_types) matrix.")
    n_sites, n_types = theta.shape
    if beta.shape[0] != n_sites:
        raise ValueError("theta and beta must share the site axis length.")
    if n_sites == 0:
        raise ValueError("no sites to deconvolve.")
    if weight is None:
        A, b = theta, beta
    else:
        sw = np.sqrt(np.clip(np.asarray(weight, dtype=np.float64), 0.0, None))
        A = theta * sw[:, None]
        b = beta * sw

    if n_types == 1:
        frac = np.array([1.0] if sum_to_one else
                        [max(0.0, (A[:, 0] @ b) / (A[:, 0] @ A[:, 0]))])
    elif not sum_to_one:
        frac, _ = nnls(A, b)
    else:
        # Constrained QP: min ||A f - b||^2 s.t. f >= 0 and either
        # sum(f) == 1 (allow_unknown=False) or sum(f) <= 1 (allow_unknown=True).
        AtA = A.T @ A
        Atb = A.T @ b

        def _obj(f):
            return float(f @ (AtA @ f) - 2.0 * (Atb @ f))

        def _grad(f):
            return 2.0 * (AtA @ f - Atb)

        if allow_unknown:
            cons = [{'type': 'ineq', 'fun': lambda f: 1.0 - np.sum(f),
                     'jac': lambda f: -np.ones(n_types)}]
        else:
            cons = [{'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0,
                     'jac': lambda f: np.ones(n_types)}]
        f0, _ = nnls(A, b)
        s0 = f0.sum()
        f0 = f0 / s0 if s0 > 0 else np.full(n_types, 1.0 / n_types)
        res = minimize(_obj, f0, jac=_grad, method='SLSQP',
                       bounds=[(0.0, None)] * n_types, constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-10})
        frac = np.clip(res.x, 0.0, None)
        if not allow_unknown and frac.sum() > 0:
            frac = frac / frac.sum()

    resid = A @ frac - b
    ss_res = float(resid @ resid)
    bw = b
    ss_tot = float(((bw - bw.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return frac.astype(np.float64), r2


# ==========================================================
class CellTypeClassifier:
    """Site-likelihood (naive-Bayes) cell-type classifier for methylation .cz.

    Purpose
    -------
    Assign a cell type to a **shallow / low-coverage** single cell (few reads
    per cytosine) by comparing its methylation pattern against a panel of
    **deep** cell-type pseudobulk references. It answers: *of the candidate
    cell types, which one most likely produced the methylation observed in
    this cell?*

    How it works
    ------------
    1. ``fit`` reads each cell-type pseudobulk ``.cz`` and estimates, for
       every discriminative cytosine, the reference methylation frequency
       ``theta[c, t]`` of type ``t`` (Beta-shrinkage of ``mc``/``cov``).
    2. ``predict`` scores a query cell as the aggregated per-cytosine
       Bernoulli log-likelihood of its ``mc``/``cov`` under each type, in two
       independent channels (CpG and CpH), then a softmax over types gives
       calibrated probabilities (supporting abstention).

    Each input ``.cz`` holds *all* cytosines (CpG + CpH) as ``mc``/``cov``;
    the CpG vs CpH split comes from the ``context`` column, read from a
    ``build_ref`` ``reference`` ``.cz`` (the cell-type files usually store
    only ``mc``/``cov``), so no pre-splitting into separate CG / CH files is
    needed.

    Parameters
    ----------
    lambda_cg, lambda_ch : float
        Log-space channel weights (default 1 each). CpH sites are far more
        numerous and often dominate implicitly; lower ``lambda_ch`` to
        rebalance.
    mc_col, cov_col, context_col : str
        Column names (in the .cz ``header['columns']``) for methylated
        count, coverage, and sequence context in the ``.cz`` files.

    Attributes
    ----------
    cell_types : list of str
        The candidate cell-type labels, in the column order used by every
        ``(n_types,)`` score / probability vector. Set by :meth:`fit` from the
        keys of ``pseudobulks``.
    cell_counts : dict {cell_type: int} or None
        How many reference cells were pooled into each cell-type pseudobulk
        (i.e. the abundance of each type in the reference atlas). It is
        **not** used to fit the frequencies; it only feeds the optional
        *abundance prior* at predict time: when ``prior_alpha > 0`` the
        classifier favours more abundant types via ``pi_t ∝ cell_counts[t] **
        prior_alpha`` (see :meth:`_log_prior`). ``None`` (default) -> a uniform
        prior over types (every type equally likely a priori). Set by
        :meth:`fit` from its ``cell_counts=`` argument.
    alpha0_cg, beta0_cg, alpha0_ch, beta0_ch : float or None
        The Beta-shrinkage pseudocounts actually used per channel (either the
        user override or the value auto-estimated in :meth:`fit`).
    """

    def __init__(self, lambda_cg=1.0, lambda_ch=1.0,
                 mc_col='mc', cov_col='cov', context_col='context'):
        self.lambda_cg = float(lambda_cg)
        self.lambda_ch = float(lambda_ch)
        self.mc_col = mc_col
        self.cov_col = cov_col
        self.context_col = context_col

        # Beta-prior pseudocounts per channel. Set by fit() (either the user
        # override or the auto-estimated value) and persisted by save().
        self.alpha0_cg = None
        self.beta0_cg = None
        self.alpha0_ch = None
        self.beta0_ch = None

        self.cell_types = None
        self.cell_counts = None
        self._chunk_keys = None
        self._chunk_lens = None
        self._chunk_span = None
        self._n_full = None
        # On-disk memory-mapped model store. ``fit`` spills the per-channel
        # log-theta / log1m-theta / sites arrays to ``.npy`` files under
        # ``_store_dir`` (a temp dir it owns), so neither fit nor predict has
        # to hold the full (n_sites x n_types) table in RAM — the OS pages in
        # only the chunk being scored. ``save`` copies the store to a chosen
        # directory; ``load`` mmaps it (read-only, not owned).
        self._store_dir = None
        self._owns_store = False
        # Per-channel fitted state (None if that context has no sites):
        #   alpha0/beta0 : the shrinkage prior actually used
        #   n_sites      : total selected discriminative sites in the channel
        #   chunks       : {chunk_key: (lo, hi)} row range into the channel's
        #                  concatenated (memory-mapped) arrays
        #   sites        : (n_sel,) int64 local row indices (mmap)
        #   log_theta / log1m_theta : (n_sel, n_types) float32 (mmap)
        # Scoring sums each present chunk's contribution, so a chunk missing
        # from a query is simply skipped (zero contribution).
        self._cg = None
        self._ch = None

    # ----------------------------------------------------------------------
    def _cleanup_store(self):
        """Release memmaps and delete the temp store if this object owns it."""
        for chan in (self._cg, self._ch):
            if not chan:
                continue
            for key in ('log_theta', 'log1m_theta', 'sites'):
                arr = chan.get(key)
                mm = getattr(arr, '_mmap', None)
                if mm is not None:
                    try:
                        mm.close()
                    except Exception:
                        pass
        if getattr(self, '_owns_store', False) and self._store_dir \
                and os.path.isdir(self._store_dir):
            shutil.rmtree(self._store_dir, ignore_errors=True)
        self._store_dir = None
        self._owns_store = False

    def close(self):
        """Free memory-mapped arrays and remove the owned temp store, if any.

        Called automatically at garbage-collection; safe to call explicitly.
        After ``close`` the model can no longer score (its arrays are gone).
        """
        self._cleanup_store()
        self._cg = None
        self._ch = None

    def __del__(self):
        try:
            self._cleanup_store()
        except Exception:
            pass

    def fit(self, pseudobulks, reference=None, cell_counts=None,
            top_cg=None, top_ch=None, min_range_cg=0.0, min_range_ch=0.0,
            top_per_class=None,
            alpha0_cg=None, beta0_cg=None, alpha0_ch=None, beta0_ch=None,
            prior_min_cov=2, contexts='cg+ch', n_jobs=None, outdir=None):
        """Estimate per-type methylation frequencies from cell-type ``.cz`` files.

        Parameters
        ----------
        pseudobulks : dict {cell_type: path} or str
            Per-type pseudobulk ``.cz`` files, each holding *all* cytosines
            (CpG + CpH) as ``mc``/``cov`` aligned to a common reference axis.
            Either a ``{cell_type: path}`` dict or a **directory** of such
            ``.cz`` files (each file's stem is the cell-type name). These
            typically store **only** ``mc``/``cov`` (no ``pos`` / ``context``),
            in which case ``reference`` is required.
        reference : str or None
            Path to the ``build_ref`` reference ``.cz`` (with ``pos, strand,
            context``) supplying the per-row ``context`` used to split CpG
            vs CpH. ``None`` (default) only works when the ``pseudobulks``
            files themselves carry a ``context_col`` column; otherwise pass
            the reference here.
        cell_counts : dict {cell_type: int}, optional
            Number of reference cells pooled into each cell-type pseudobulk
            (the abundance of each type in the reference atlas). Stored on the
            model and used **only** at predict time to build the optional
            abundance prior (``pi_t ∝ cell_counts[t] ** prior_alpha`` when
            ``prior_alpha > 0``); it does not affect the fitted frequencies.
            ``None`` (default) -> uniform prior only.
        top_cg, top_ch : None, int, or float, optional
            Discriminative-site selection for each channel. ``None`` keeps
            all sites with range > ``min_range_*``; an int keeps the top-N
            sites; a float in (0, 1] keeps the top fraction.
        min_range_cg, min_range_ch : float, optional
            Keep a site only when its across-type frequency range is strictly
            greater than this (default 0.0, which already drops the
            non-discriminative score-0 sites — where every cell type shares
            the same theta — since those cancel in the softmax and cannot
            affect the prediction).
        top_per_class : int or None, optional
            Balanced, per-cell-type marker selection. When set (e.g. ``2000``),
            **replaces** the global ``top_cg`` / ``top_ch`` ranking with a
            one-vs-rest scheme: for every cell type, keep the ``top_per_class``
            sites where that type is the *uniquely* highest-methylated (by the
            gap ``max - 2nd-highest`` across types) plus ``top_per_class`` where
            it is the *uniquely* lowest (gap ``2nd-lowest - min``), in each
            channel, then take the union. This guarantees each type — including
            **subtly different, dominant types** (e.g. D1 vs D2 vs hybrid MSN) —
            contributes its own discriminative markers instead of being starved
            by a global top-fraction that is dominated by coarse
            (neuron-vs-glia) contrasts. ``None`` (default) uses the global
            ``top_cg`` / ``top_ch`` selection.
        alpha0_cg, beta0_cg : float or None, optional
            Beta-prior pseudocounts for the CpG channel. ``None`` (default)
            -> estimate them empirically from the pooled reference counts of
            that context via :func:`estimate_beta_prior` (Beta-Binomial method
            of moments). Pass explicit floats to override the auto-estimate
            (both must be given to take effect).
        alpha0_ch, beta0_ch : float or None, optional
            Beta-prior pseudocounts for the CpH channel. ``None`` (default) ->
            auto-estimated like the CpG channel (its own separate prior).
        prior_min_cov : int, optional
            Minimum coverage for a reference site to enter the empirical prior
            estimation (passed to :func:`estimate_beta_prior`; default 2).
        contexts : str, optional
            Which cytosine context channel(s) to actually fit: ``'cg'``,
            ``'ch'``, or ``'both'`` / ``'cg+ch'`` (default, fit both). Fitting
            a single context skips the other channel entirely (no sites read,
            selected, or stored for it), so the model is smaller and faster to
            build. A model fit on one context can only be scored on that
            context.
        n_jobs : int or None, optional
            Threads used to read/decode the pseudobulk ``.cz`` files in
            parallel across cell types (reads/decompression release the GIL).
            ``None``/``1`` -> serial (lowest memory); ``-1`` -> all CPUs.
            Fit streams **chunk by chunk**, so peak memory grows only with the
            number of concurrent workers (each holds one chunk), not with the
            number of cell types.
        outdir : str or None, optional
            Directory for the memory-mapped model store (the ``log_theta``
            etc. ``.npy`` files are written directly here). ``None`` (default)
            uses a system temp dir (``$TMPDIR`` / ``/tmp``), which on HPC
            nodes is often small or RAM-backed (tmpfs) and can crash with
            **"Bus error" (SIGBUS)** when the model is large. Point this at a
            **scratch / large-disk** path with enough free space (roughly
            ``n_selected_sites x n_types x 8 bytes``). When set, the store is
            persistent (not auto-deleted).

        Returns
        -------
        self
        """
        # Reuse a previously saved model when ``outdir`` already holds a
        # complete store (meta.json + per-channel .npy): skip the expensive
        # fit entirely and load it (memory-mapped) into this instance.
        if outdir is not None:
            store = os.path.abspath(os.path.expanduser(outdir))
            if _model_store_complete(store):
                logger.info(
                    f"found existing model at {store}; skipping fit and "
                    f"loading it")
                loaded = CellTypeClassifier.load(store)
                self.__dict__.update(loaded.__dict__)
                # Detach the throwaway loader so its __del__ does not close the
                # memmaps this instance now shares.
                loaded._cg = loaded._ch = None
                loaded._store_dir = None
                return self
        pseudobulks = self._resolve_pseudobulks(pseudobulks)
        use_ctx = self._resolve_contexts(contexts)
        self.alpha0_cg = None if alpha0_cg is None else float(alpha0_cg)
        self.beta0_cg = None if beta0_cg is None else float(beta0_cg)
        self.alpha0_ch = None if alpha0_ch is None else float(alpha0_ch)
        self.beta0_ch = None if beta0_ch is None else float(beta0_ch)
        prior_min_cov = int(prior_min_cov)
        cell_types = list(pseudobulks)
        paths = {t: _abspath(pseudobulks[t])
                 for t in cell_types}

        # ---- Canonical axis + CpG/CpH masks from the context source ------
        # Cell-type files are usually mc/cov-only, so read context from the
        # build_ref `reference` when provided; otherwise fall back to the
        # first pseudobulk (only works if it carries `context`).
        ctx_source = reference if reference is not None else pseudobulks[cell_types[0]]
        chunk_keys, ctx, chunk_lens = _read_context(
            ctx_source, self.context_col, None)
        cg_mask, ch_mask = _context_masks(ctx)
        del ctx  # free the (n_full x 3-byte) context array early
        # Restrict to the requested context channel(s): dropping a mask to
        # all-False makes n_cg / n_ch below 0, so every downstream pass skips
        # that channel (nothing read, selected, or stored for it).
        if 'cg' not in use_ctx:
            cg_mask = np.zeros_like(cg_mask)
        if 'ch' not in use_ctx:
            ch_mask = np.zeros_like(ch_mask)
        n_full = int(sum(int(chunk_lens[k]) for k in chunk_keys))

        # Per-chunk row span on the full axis, plus per-chunk (offset, count)
        # into each context's compact site axis. These let us stream one chunk
        # at a time yet still address a global CpG/CpH position array.
        chunk_spans = []
        chunk_span = {}
        cg_off, ch_off = {}, {}
        _off = _cg = _ch = 0
        for k in chunk_keys:
            n_k = int(chunk_lens[k])
            s, e = _off, _off + n_k
            chunk_spans.append((k, s, e))
            chunk_span[k] = (s, e)
            ncg = int(cg_mask[s:e].sum())
            nch = int(ch_mask[s:e].sum())
            cg_off[k] = (_cg, ncg)
            ch_off[k] = (_ch, nch)
            _cg += ncg
            _ch += nch
            _off = e
        n_cg, n_ch = _cg, _ch

        # ---- Streaming helpers: one pseudobulk open at a time, one chunk at
        # a time. Reads/decompression release the GIL, so cell types are
        # processed by a thread pool (``n_jobs`` workers). Peak memory is
        # O(n_full) shared arrays + ``n_workers`` in-flight chunks, still
        # independent of the number of cell types.
        def _open(t):
            r = Reader(paths[t])
            cols = list(r.header['columns'])
            if self.mc_col not in cols or self.cov_col not in cols:
                r.close()
                raise ValueError(
                    f"{paths[t]}: expected columns {self.mc_col!r} and "
                    f"{self.cov_col!r}, got {cols!r}")
            return r, cols.index(self.mc_col), cols.index(self.cov_col)

        def _chunk(r, k, mi, ci):
            # (mc, cov) native arrays for chunk k, or None when the file lacks
            # that chunk (treated as an all-zero, i.e. prior-mean, chunk).
            if k not in r.chunk_key2offset:
                return None
            arr = r.chunk2numpy(k)
            mc = np.asarray(arr[f'f{mi}'])
            cov = np.asarray(arr[f'f{ci}'])
            n_k = chunk_span[k][1] - chunk_span[k][0]
            if mc.size != n_k:
                raise ValueError(
                    f"chunk {k!r} has {mc.size} rows but the reference axis "
                    f"has {n_k}; every cell-type file must be aligned to the "
                    f"same reference axis within each chrom.")
            return mc, cov

        n_workers = _resolve_n_jobs(n_jobs)
        _lock = threading.Lock()

        def _map(fn, items, desc=None):
            """Run ``fn`` over ``items`` with a thread pool (or serially).

            When ``desc`` is given, emit INFO progress as each item finishes so
            the otherwise-silent heavy passes visibly advance (a big model with
            many cell types can spend minutes here between two log lines).
            """
            n = len(items)
            run = fn
            if desc is not None:
                done = [0]
                plock = threading.Lock()

                def run(x, _fn=fn):
                    out = _fn(x)
                    with plock:
                        done[0] += 1
                        d = done[0]
                    logger.info(f"{desc}: {d}/{n} cell types done")
                    return out
            if n_workers > 1 and n > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return list(ex.map(run, items))
            return [run(x) for x in items]

        # ---- PASS A: per-context Beta prior via streamed scalar moments ----
        # Parallel across cell types: each task accumulates thread-local
        # scalars (one chunk in memory at a time), combined at the end.
        need_cg = n_cg > 0 and (self.alpha0_cg is None or self.beta0_cg is None)
        need_ch = n_ch > 0 and (self.alpha0_ch is None or self.beta0_ch is None)
        if need_cg or need_ch:
            thr = max(prior_min_cov, 1)

            def _accum(acc, mc_k, cov_k, m):
                if not m.any():
                    return
                mc = mc_k[m].astype(np.float64)
                cov = cov_k[m].astype(np.float64)
                keep = cov >= thr
                mc, cov = mc[keep], cov[keep]
                if cov.size:
                    acc[0] += float(mc.sum())
                    acc[1] += float(cov.sum())
                    acc[2] += float(np.sum(mc * mc / cov))
                    acc[3] += int(cov.size)

            def _prior_one(t):
                loc_cg = [0.0, 0.0, 0.0, 0]
                loc_ch = [0.0, 0.0, 0.0, 0]
                r, mi, ci = _open(t)
                try:
                    for k, s, e in chunk_spans:
                        got = _chunk(r, k, mi, ci)
                        if got is None:
                            continue
                        mc_k, cov_k = got
                        if need_cg:
                            _accum(loc_cg, mc_k, cov_k, cg_mask[s:e])
                        if need_ch:
                            _accum(loc_ch, mc_k, cov_k, ch_mask[s:e])
                finally:
                    r.close()
                return loc_cg, loc_ch

            acc_cg = [0.0, 0.0, 0.0, 0]
            acc_ch = [0.0, 0.0, 0.0, 0]
            for loc_cg, loc_ch in _map(_prior_one, cell_types):
                for i in range(4):
                    acc_cg[i] += loc_cg[i]
                    acc_ch[i] += loc_ch[i]
            if need_cg:
                self.alpha0_cg, self.beta0_cg = _beta_prior_from_moments(*acc_cg)
            if need_ch:
                self.alpha0_ch, self.beta0_ch = _beta_prior_from_moments(*acc_ch)

        a_cg, b_cg = self.alpha0_cg, self.beta0_cg
        a_ch, b_ch = self.alpha0_ch, self.beta0_ch
        mean_cg = (a_cg / (a_cg + b_cg)) if n_cg > 0 else 0.0
        mean_ch = (a_ch / (a_ch + b_ch)) if n_ch > 0 else 0.0
        if n_cg > 0:
            logger.info(
                f"CG: Beta prior alpha0={a_cg:.4g}, beta0={b_cg:.4g} "
                f"(mean={mean_cg:.3f}, kappa={a_cg + b_cg:.4g})")
        if n_ch > 0:
            logger.info(
                f"CH: Beta prior alpha0={a_ch:.4g}, beta0={b_ch:.4g} "
                f"(mean={mean_ch:.3f}, kappa={a_ch + b_ch:.4g})")

        # ---- PASS B: across-type theta range via running max/min ----------
        # Parallel across cell types; only the per-chunk max/min update touches
        # the shared arrays, so it is guarded by a lock (the costly decompress
        # + estimate_theta run concurrently outside it).
        per_class = top_per_class is not None
        kpc = int(top_per_class) if per_class else 0
        # For per-class (one-vs-rest) selection we also track, per site, which
        # type is the extreme high/low (argmax/argmin) and the runner-up value
        # (2nd max / 2nd min) so the OvR gap of the winning type is available.
        amax_cg = amin_cg = tmax2_cg = tmin2_cg = None
        amax_ch = amin_ch = tmax2_ch = tmin2_ch = None
        tmax_cg = tmin_cg = tmax_ch = tmin_ch = None
        if n_cg > 0:
            tmax_cg = np.full(n_cg, -np.inf, dtype=np.float32)
            tmin_cg = np.full(n_cg, np.inf, dtype=np.float32)
            if per_class:
                tmax2_cg = np.full(n_cg, -np.inf, dtype=np.float32)
                tmin2_cg = np.full(n_cg, np.inf, dtype=np.float32)
                amax_cg = np.full(n_cg, -1, dtype=np.int32)
                amin_cg = np.full(n_cg, -1, dtype=np.int32)
        if n_ch > 0:
            tmax_ch = np.full(n_ch, -np.inf, dtype=np.float32)
            tmin_ch = np.full(n_ch, np.inf, dtype=np.float32)
            if per_class:
                tmax2_ch = np.full(n_ch, -np.inf, dtype=np.float32)
                tmin2_ch = np.full(n_ch, np.inf, dtype=np.float32)
                amax_ch = np.full(n_ch, -1, dtype=np.int32)
                amin_ch = np.full(n_ch, -1, dtype=np.int32)

        logger.info(
            f"PASS B: scoring cross-type theta range over {len(cell_types)} "
            f"cell types ({n_cg:,} CG + {n_ch:,} CH sites)"
            + (f", per-class top {kpc}" if per_class else "") + "...")

        def _update_extreme(seg, th, j, tmax, tmin, tmax2, tmin2, amax, amin):
            # Running max/min plus 2nd-max/2nd-min and their argmax/argmin type,
            # correct under any type-processing order (each type seen once/site).
            hi = tmax[seg]
            gt1 = th > hi
            gt2 = (~gt1) & (th > tmax2[seg])
            tmax2[seg] = np.where(gt1, hi, np.where(gt2, th, tmax2[seg]))
            amax[seg] = np.where(gt1, j, amax[seg])
            tmax[seg] = np.where(gt1, th, hi)
            lo = tmin[seg]
            lt1 = th < lo
            lt2 = (~lt1) & (th < tmin2[seg])
            tmin2[seg] = np.where(lt1, lo, np.where(lt2, th, tmin2[seg]))
            amin[seg] = np.where(lt1, j, amin[seg])
            tmin[seg] = np.where(lt1, th, lo)

        def _range_one(t):
            j = cell_types.index(t)
            r, mi, ci = _open(t)
            try:
                for k, s, e in chunk_spans:
                    got = _chunk(r, k, mi, ci)
                    present = got is not None
                    if present:
                        mc_k, cov_k = got
                    if n_cg > 0:
                        off, nk = cg_off[k]
                        if nk:
                            if present:
                                m = cg_mask[s:e]
                                th = estimate_theta(
                                    mc_k[m], cov_k[m], a_cg, b_cg)
                            else:
                                th = np.full(nk, mean_cg, dtype=np.float32)
                            seg = slice(off, off + nk)
                            with _lock:
                                if per_class:
                                    _update_extreme(
                                        seg, th, j, tmax_cg, tmin_cg,
                                        tmax2_cg, tmin2_cg, amax_cg, amin_cg)
                                else:
                                    np.maximum(tmax_cg[seg], th, out=tmax_cg[seg])
                                    np.minimum(tmin_cg[seg], th, out=tmin_cg[seg])
                    if n_ch > 0:
                        off, nk = ch_off[k]
                        if nk:
                            if present:
                                m = ch_mask[s:e]
                                th = estimate_theta(
                                    mc_k[m], cov_k[m], a_ch, b_ch)
                            else:
                                th = np.full(nk, mean_ch, dtype=np.float32)
                            seg = slice(off, off + nk)
                            with _lock:
                                if per_class:
                                    _update_extreme(
                                        seg, th, j, tmax_ch, tmin_ch,
                                        tmax2_ch, tmin2_ch, amax_ch, amin_ch)
                                else:
                                    np.maximum(tmax_ch[seg], th, out=tmax_ch[seg])
                                    np.minimum(tmin_ch[seg], th, out=tmin_ch[seg])
            finally:
                r.close()

        _map(_range_one, cell_types, desc="PASS B (theta range)")
        logger.info("PASS B done; selecting discriminative sites...")

        def _select_per_class(tmax, tmin, tmax2, tmin2, amax, amin, min_range):
            # Union of, per type, the top-kpc sites where it is uniquely highest
            # (gap = max - 2nd-max) and the top-kpc where it is uniquely lowest
            # (gap = 2nd-min - min), among sites whose overall range passes
            # ``min_range``.
            valid = (tmax - tmin) > min_range
            gap_hi = np.where(np.isfinite(tmax2), tmax - tmax2, tmax - tmin)
            gap_lo = np.where(np.isfinite(tmin2), tmin2 - tmin, tmax - tmin)
            keep = set()
            for j in range(len(cell_types)):
                ih = np.where(valid & (amax == j))[0]
                if ih.size:
                    order = ih[np.argsort(gap_hi[ih])[::-1][:kpc]]
                    keep.update(order.tolist())
                il = np.where(valid & (amin == j))[0]
                if il.size:
                    order = il[np.argsort(gap_lo[il])[::-1][:kpc]]
                    keep.update(order.tolist())
            return np.array(sorted(keep), dtype=np.int64)

        sites_cg = sites_ch = None
        if n_cg > 0:
            if per_class:
                sites_cg = _select_per_class(
                    tmax_cg, tmin_cg, tmax2_cg, tmin2_cg,
                    amax_cg, amin_cg, min_range_cg)
                del tmax_cg, tmin_cg, tmax2_cg, tmin2_cg, amax_cg, amin_cg
            else:
                score = tmax_cg - tmin_cg
                del tmax_cg, tmin_cg
                sites_cg = _select_from_score(
                    score, top=top_cg, min_range=min_range_cg)
                del score
        if n_ch > 0:
            if per_class:
                sites_ch = _select_per_class(
                    tmax_ch, tmin_ch, tmax2_ch, tmin2_ch,
                    amax_ch, amin_ch, min_range_ch)
                del tmax_ch, tmin_ch, tmax2_ch, tmin2_ch, amax_ch, amin_ch
            else:
                score = tmax_ch - tmin_ch
                del tmax_ch, tmin_ch
                sites_ch = _select_from_score(
                    score, top=top_ch, min_range=min_range_ch)
                del score

        # Map selected compact-axis indices back to per-chunk local rows (into
        # the chunk's mc/cov) and output-row ranges.
        def _chunk_sel(sites, off_map, mask):
            sel = {}
            for k, s, e in chunk_spans:
                off, nk = off_map[k]
                if nk == 0:
                    continue
                lo = int(np.searchsorted(sites, off, side='left'))
                hi = int(np.searchsorted(sites, off + nk, side='left'))
                if hi <= lo:
                    continue
                rows = np.flatnonzero(mask[s:e])[sites[lo:hi] - off]
                sel[k] = (lo, hi, rows.astype(np.int32))
            return sel

        cg_sel = _chunk_sel(sites_cg, cg_off, cg_mask) if n_cg > 0 else {}
        ch_sel = _chunk_sel(sites_ch, ch_off, ch_mask) if n_ch > 0 else {}
        n_sel_cg = int(sites_cg.size) if n_cg > 0 else 0
        n_sel_ch = int(sites_ch.size) if n_ch > 0 else 0
        n_types = len(cell_types)

        # ---- PASS C: gather log-theta at the selected sites, spilled to a
        # memory-mapped on-disk store so the full (n_sites x n_types) table
        # never has to live in RAM (neither at fit nor at predict time).
        # The store MUST live on a real filesystem with enough free space;
        # the default ``tempfile`` location ($TMPDIR / /tmp) is often small or
        # RAM-backed (tmpfs), which overflows to a SIGBUS ("Bus error") when
        # the mmap grows. Pass ``outdir=`` a path on scratch/large disk (or
        # use ``predict_cell_type(outdir=...)``, which routes it here).
        self._cleanup_store()
        if outdir is not None:
            store = os.path.abspath(os.path.expanduser(outdir))
            os.makedirs(store, exist_ok=True)
            self._store_dir = store
            self._owns_store = False  # user-managed, not auto-deleted
        else:
            store = tempfile.mkdtemp(prefix='cztclf_')
            self._store_dir = store
            self._owns_store = True

        def _alloc(name, n_sel):
            lt = np.lib.format.open_memmap(
                os.path.join(store, f'{name}_log_theta.npy'),
                mode='w+', dtype=np.float32, shape=(n_sel, n_types))
            l1 = np.lib.format.open_memmap(
                os.path.join(store, f'{name}_log1m_theta.npy'),
                mode='w+', dtype=np.float32, shape=(n_sel, n_types))
            si = np.lib.format.open_memmap(
                os.path.join(store, f'{name}_sites.npy'),
                mode='w+', dtype=np.int32, shape=(n_sel,))
            return lt, l1, si

        lt_cg = l1_cg = si_cg = None
        lt_ch = l1_ch = si_ch = None
        if n_cg > 0:
            lt_cg, l1_cg, si_cg = _alloc('cg', n_sel_cg)
            for k, (lo, hi, rows) in cg_sel.items():
                si_cg[lo:hi] = rows
        if n_ch > 0:
            lt_ch, l1_ch, si_ch = _alloc('ch', n_sel_ch)
            for k, (lo, hi, rows) in ch_sel.items():
                si_ch[lo:hi] = rows

        # Fill each cell type's column independently: tasks write disjoint
        # columns of the shared memmaps, so no lock is needed.
        def _gather_one(jt):
            j, t = jt
            r, mi, ci = _open(t)
            try:
                for k, s, e in chunk_spans:
                    in_cg = k in cg_sel
                    in_ch = k in ch_sel
                    if not in_cg and not in_ch:
                        continue
                    got = _chunk(r, k, mi, ci)
                    present = got is not None
                    if present:
                        mc_k, cov_k = got
                    if in_cg:
                        lo, hi, rows = cg_sel[k]
                        if present:
                            th = estimate_theta(
                                mc_k[rows], cov_k[rows], a_cg, b_cg)
                        else:
                            th = np.full(rows.size, mean_cg, dtype=np.float32)
                        lt_cg[lo:hi, j] = np.log(th)
                        l1_cg[lo:hi, j] = np.log1p(-th)
                    if in_ch:
                        lo, hi, rows = ch_sel[k]
                        if present:
                            th = estimate_theta(
                                mc_k[rows], cov_k[rows], a_ch, b_ch)
                        else:
                            th = np.full(rows.size, mean_ch, dtype=np.float32)
                        lt_ch[lo:hi, j] = np.log(th)
                        l1_ch[lo:hi, j] = np.log1p(-th)
            finally:
                r.close()

        logger.info(
            f"PASS C: gathering log-theta at {n_sel_cg + n_sel_ch:,} selected "
            f"sites x {n_types} cell types...")
        _map(_gather_one, list(enumerate(cell_types)),
             desc="PASS C (gather log-theta)")
        logger.info("PASS C done; flushing model store to disk...")
        for arr in (lt_cg, l1_cg, si_cg, lt_ch, l1_ch, si_ch):
            if arr is not None:
                arr.flush()

        def _assemble(sel, lt, l1, si, a, b, n_sel):
            chunks = {k: (int(lo), int(hi)) for k, (lo, hi, _) in sel.items()}
            return {'alpha0': float(a), 'beta0': float(b),
                    'n_sites': int(n_sel), 'chunks': chunks,
                    'sites': si, 'log_theta': lt, 'log1m_theta': l1}

        self._cg = (_assemble(cg_sel, lt_cg, l1_cg, si_cg, a_cg, b_cg, n_sel_cg)
                    if n_cg > 0 else None)
        self._ch = (_assemble(ch_sel, lt_ch, l1_ch, si_ch, a_ch, b_ch, n_sel_ch)
                    if n_ch > 0 else None)
        if self._cg is None and self._ch is None:
            raise ValueError("no CpG or CpH sites found; check context_col.")
        if self._cg is not None:
            logger.info(
                f"CG: {n_cg:,} context sites -> {n_sel_cg:,} discriminative "
                f"sites across {n_types} cell types in {len(cg_sel)} chunks")
        if self._ch is not None:
            logger.info(
                f"CH: {n_ch:,} context sites -> {n_sel_ch:,} discriminative "
                f"sites across {n_types} cell types in {len(ch_sel)} chunks")

        # Warn when the model would store a huge log-theta matrix (defaults
        # keep every site). Selecting sites (top_*/min_range_*) is the lever.
        n_sel_total = n_sel_cg + n_sel_ch
        approx_gb = n_sel_total * n_types * 4 * 2 / 1e9
        if approx_gb > 2:
            logger.warning(
                f"model keeps {n_sel_total:,} sites x {n_types} types "
                f"(~{approx_gb:.1f} GB of log-theta). Pass top_cg/top_ch (e.g. "
                f"top_cg=20000) or min_range_cg/min_range_ch (>0) to select "
                f"fewer discriminative sites and cut memory drastically.")

        self.cell_types = cell_types
        self.cell_counts = cell_counts
        self._chunk_keys = chunk_keys
        self._chunk_lens = chunk_lens
        self._chunk_span = chunk_span
        self._n_full = n_full
        return self

    # ----------------------------------------------------------------------
    def _log_prior(self, prior_alpha):
        """Return the (n_types,) log-prior vector.

        Uniform when ``cell_counts`` is unknown; otherwise the tempered
        abundance prior ``pi_t ∝ N_t ** prior_alpha`` (``prior_alpha=0`` ->
        uniform, ``1`` -> pure abundance).
        """
        T = len(self.cell_types)
        if self.cell_counts is None or prior_alpha == 0:
            return np.full(T, -np.log(T), dtype=np.float64)
        N = np.array([float(self.cell_counts[t]) for t in self.cell_types])
        w = np.power(N, float(prior_alpha))
        pi = w / w.sum()
        return np.log(pi)

    @staticmethod
    def _score(chan, query_chunks, downsample=None, rng=None):
        """Aggregated Bernoulli log-likelihood for one channel: (n_types,).

        Sums each chunk's contribution; a chunk that is absent from
        ``query_chunks`` (missing in the query ``.cz``) is simply skipped,
        which is exactly a zero contribution — so no flat-axis alignment or
        zero-fill is needed.

        When ``downsample`` is an int, only that many of the query's covered
        (``cov > 0``) sites (drawn uniformly at random across all chunks via
        ``rng``) are kept; the rest are zeroed so they add nothing. If the
        query has at most ``downsample`` covered sites, all are used.
        """
        lt = chan['log_theta']
        l1 = chan['log1m_theta']
        sarr = chan['sites']
        # Materialise per-chunk (mc, cov) at the model's selected sites first,
        # so optional downsampling can pick a random subset *across* chunks.
        parts = []  # (lo, hi, mc, cov)
        for k, (lo, hi) in chan['chunks'].items():
            arrs = query_chunks.get(k)
            if arrs is None:
                continue  # chunk missing from this query -> zero contribution
            mc_k, cov_k = arrs
            s = np.asarray(sarr[lo:hi])  # memmap slice -> small owned index array
            parts.append((lo, hi, mc_k[s], cov_k[s]))
        if not parts:
            return None
        if downsample is not None:
            # Randomly keep only ``downsample`` of the query's covered sites in
            # this channel; zero out the rest (zero mc & cov -> no likelihood
            # contribution). The random indices span all chunks jointly.
            cov_masks = [cov > 0 for (_, _, _, cov) in parts]
            n_cov = int(sum(int(m.sum()) for m in cov_masks))
            if 0 < int(downsample) < n_cov:
                sel = np.zeros(n_cov, dtype=bool)
                sel[rng.choice(n_cov, size=int(downsample), replace=False)] = True
                off = 0
                new_parts = []
                for (lo, hi, mc, cov), m in zip(parts, cov_masks):
                    c = int(m.sum())
                    keep = np.zeros(mc.shape[0], dtype=bool)
                    if c:
                        keep[np.flatnonzero(m)[sel[off:off + c]]] = True
                    off += c
                    mc = np.where(keep, mc, 0)
                    cov = np.where(keep, cov, 0)
                    new_parts.append((lo, hi, mc, cov))
                parts = new_parts
        # Only covered sites (cov > 0) contribute: an uncovered site has
        # mc == 0 and cov == 0, so mc*log_theta + (cov-mc)*log1m_theta == 0.
        # Gathering just those rows turns the dense GEMV over *every* selected
        # site into a tiny one over the (usually <1%) covered sites, reading
        # only those rows from the memmapped log-theta tables.
        acc = None
        for lo, hi, mc, cov in parts:
            nz = np.flatnonzero(cov)
            if nz.size == 0:
                continue  # nothing covered in this chunk -> zero contribution
            gidx = lo + nz
            m = mc[nz]
            umc = cov[nz] - m
            part = m @ lt[gidx] + umc @ l1[gidx]
            acc = part if acc is None else acc + part
        return acc

    def _query_chunks(self, query):
        """Return ``{chunk_key: (mc, cov)}`` for the chunks the model needs.

        Reads only the chunks carrying selected sites (union over the CpG and
        CpH channels), skipping any absent from the query. Accepts a ``.cz``
        path or a preloaded full-axis ``(mc, cov)`` array pair (sliced by the
        stored chunk spans).
        """
        needed = set()
        for chan in (self._cg, self._ch):
            if chan is not None:
                needed.update(chan['chunks'].keys())
        if isinstance(query, (tuple, list)):
            mc = np.asarray(query[0], np.float64)
            cov = np.asarray(query[1], np.float64)
            if mc.size != self._n_full:
                raise ValueError(
                    f"query has {mc.size} sites but the model was fit on "
                    f"{self._n_full}; preloaded (mc, cov) arrays must span the "
                    f"full reference axis.")
            out = {}
            for k in needed:
                s, e = self._chunk_span[k]
                out[k] = (mc[s:e], cov[s:e])
            return out
        return _read_mc_cov_by_chunk(
            query, self.mc_col, self.cov_col, needed, self._chunk_lens)

    def log_posterior(self, query, prior_alpha=0.0,
                      max_query_cg=None, max_query_ch=None, contexts='cg+ch'):
        """Unnormalized log-posterior per cell type: pandas Series.

        Parameters
        ----------
        query : str or (mc, cov) tuple
            The query cell's all-cytosine ``.cz`` (or preloaded full-axis
            ``(mc, cov)`` arrays).
        prior_alpha : float
            Temperature for the abundance prior (see :meth:`_log_prior`).
        max_query_cg, max_query_ch : int or None
            If given, randomly keep at most this many of the query cell's
            *covered* (``cov > 0``) cytosines in the CpG / CpH channel when
            scoring; the rest contribute nothing. ``None`` (default) uses
            every covered site (no downsampling). When the query has fewer
            covered sites than the cap, all of them are used. A fresh random
            draw is made per call.
        contexts : str, optional
            Which cytosine context(s) to score on: ``'cg'``, ``'ch'``, or
            ``'both'`` / ``'cg+ch'`` (default, use both channels). Selecting a
            single context ignores the other channel entirely.
        """
        if self.cell_types is None:
            raise RuntimeError("call fit() before predicting.")
        query_chunks = self._query_chunks(query)
        return self._log_posterior_from_chunks(
            query_chunks, prior_alpha, max_query_cg, max_query_ch, contexts)

    def _log_posterior_from_chunks(self, query_chunks, prior_alpha=0.0,
                                   max_query_cg=None, max_query_ch=None,
                                   contexts='cg+ch'):
        """Log-posterior from a pre-built ``{chunk_key: (mc, cov)}`` map."""
        use = self._resolve_contexts(contexts)
        if 'cg' in use and self._cg is None:
            raise ValueError(
                "contexts requested 'cg' but the model has no CpG channel "
                "(none fitted); pass contexts='ch'.")
        if 'ch' in use and self._ch is None:
            raise ValueError(
                "contexts requested 'ch' but the model has no CpH channel "
                "(none fitted); pass contexts='cg'.")
        max_query_cg = None if max_query_cg is None else int(max_query_cg)
        max_query_ch = None if max_query_ch is None else int(max_query_ch)
        rng = (np.random.default_rng()
               if (max_query_cg is not None
                   or max_query_ch is not None) else None)
        logpost = self._log_prior(prior_alpha)
        if 'cg' in use and self._cg is not None:
            part = self._score(self._cg, query_chunks,
                               downsample=max_query_cg, rng=rng)
            if part is not None:
                logpost = logpost + self.lambda_cg * part
        if 'ch' in use and self._ch is not None:
            part = self._score(self._ch, query_chunks,
                               downsample=max_query_ch, rng=rng)
            if part is not None:
                logpost = logpost + self.lambda_ch * part
        return pd.Series(logpost, index=self.cell_types)

    @staticmethod
    def _softmax(logpost):
        """Numerically stable softmax of an array -> array."""
        vals = np.asarray(logpost, dtype=np.float64)
        vals = vals - vals.max()
        p = np.exp(vals)
        p /= p.sum()
        return p

    def predict_proba(self, query, prior_alpha=0.0,
                      max_query_cg=None, max_query_ch=None, contexts='cg+ch'):
        """Softmax posterior probabilities per cell type: pandas Series."""
        logpost = self.log_posterior(
            query, prior_alpha,
            max_query_cg=max_query_cg, max_query_ch=max_query_ch,
            contexts=contexts)
        return pd.Series(self._softmax(logpost.values), index=self.cell_types)

    @staticmethod
    def _resolve_pseudobulks(pseudobulks):
        """Resolve ``pseudobulks`` into a ``{cell_type: path}`` dict.

        Accepts either a ``{cell_type: path}`` dict (returned as a shallow
        copy) or a directory containing per-cell-type ``.cz`` files (each
        file's stem becomes the cell-type name).
        """
        if isinstance(pseudobulks, dict):
            if not pseudobulks:
                raise ValueError("pseudobulks dict is empty.")
            return dict(pseudobulks)
        if isinstance(pseudobulks, str):
            d = os.path.abspath(os.path.expanduser(pseudobulks))
            if os.path.isdir(d):
                files = sorted(glob.glob(os.path.join(d, '*.cz')))
                if not files:
                    raise ValueError(
                        f"no .cz files found in pseudobulk dir {d!r}")
                return {_cz_stem(f): f for f in files}
        raise ValueError(
            "pseudobulks must be a {cell_type: path} dict or a directory of "
            f".cz files; got {type(pseudobulks)!r}.")

    @staticmethod
    def _resolve_queries(query):
        """Resolve ``query`` into ``(kind, resolved)``.

        ``kind`` is ``'single'`` (with a path or ``(mc, cov)`` arrays),
        ``'multicell'`` (with a cat ``.cz`` path), or ``'batch'`` (with a
        ``{cell_id: path}`` dict). Accepts:

        * a preloaded ``(mc, cov)`` array pair -> single;
        * a single ``.cz`` file path -> single;
        * a **concatenated (cat) multi-cell ``.cz``** (chunk keys carry a cell
          dimension) -> ``'multicell'``;
        * a directory of ``.cz`` files -> batch (file stem = cell id);
        * a table (``pandas.DataFrame`` or delimited file) whose first two
          columns are ``[cell_id, cz_path]`` -> batch;
        * a ``{cell_id: path}`` dict or a list of paths -> batch.
        """
        if query is None:
            raise ValueError("query is required.")
        # preloaded (mc, cov) arrays -> single
        if (isinstance(query, (tuple, list)) and len(query) == 2
                and all(isinstance(x, np.ndarray) for x in query)):
            return 'single', query
        if isinstance(query, pd.DataFrame):
            if query.shape[1] < 2:
                raise ValueError(
                    "query DataFrame needs >= 2 columns [cell_id, cz_path].")
            return 'batch', {str(cid): str(p) for cid, p
                             in zip(query.iloc[:, 0], query.iloc[:, 1])}
        if isinstance(query, dict):
            return 'batch', dict(query)
        if isinstance(query, (list, tuple)):
            return 'batch', _as_id_map(list(query))
        if isinstance(query, str):
            q = _abspath(query)
            if os.path.isdir(q):
                files = sorted(glob.glob(os.path.join(q, '*.cz')))
                if not files:
                    raise ValueError(f"no .cz files found in query dir {q!r}")
                return 'batch', {_cz_stem(f): f for f in files}
            if q.endswith('.cz'):
                return ('multicell', q) if _cz_is_multicell(q) else ('single', q)
            return 'batch', _read_query_table(q)  # any other file -> table
        raise ValueError(f"unsupported query type: {type(query)!r}.")

    def predict(self, query, prior_alpha=0.0, abstain_threshold=None,
                max_query_cg=None, max_query_ch=None, n_jobs=None,
                contexts='cg+ch'):
        """Predict cell type(s) for any query, dispatching on its form.

        Mirrors :func:`predict_cell_type`'s ``query`` handling: the shape of
        the return value depends on what ``query`` is.

        Parameters
        ----------
        query : str, (mc, cov) tuple, list, dict, or pandas.DataFrame
            The cell(s) to predict, in any of these forms:

            * a single all-cytosine ``.cz`` file (or a preloaded ``(mc, cov)``
              array pair) -> a single prediction ``dict``;
            * a **concatenated (cat) ``.cz``** holding many cells (auto-
              detected) -> ``(labels, proba)`` like :meth:`predict_multicell`;
            * a **directory** of ``.cz`` files (each file's stem is the cell
              id), a ``{cell_id: path}`` dict, a list of ``.cz`` paths, or a
              **table** (``pandas.DataFrame`` or delimited file whose first
              two columns are ``[cell_id, cz_path]``) -> ``(labels, proba)``
              like :meth:`predict_batch`.
        prior_alpha : float
            Abundance-prior temperature (default 0 -> uniform prior).
        abstain_threshold : float or None
            If given and the top probability is below it, the label becomes
            ``'unassigned'`` (abstention).
        max_query_cg, max_query_ch : int or None
            If given, randomly keep at most this many of the query cell's
            covered (``cov > 0``) CpG / CpH cytosines when scoring. ``None``
            (default) uses every covered site. See :meth:`log_posterior`.
        n_jobs : int or None, optional
            Threads used to score cells in parallel for batch / multi-cell
            queries (ignored for a single cell). ``None``/``1`` -> serial.
        contexts : str, optional
            Which cytosine context(s) to score on: ``'cg'``, ``'ch'``, or
            ``'both'`` / ``'cg+ch'`` (default, use both channels).

        Returns
        -------
        dict or (pandas.DataFrame, pandas.DataFrame)
            A single prediction ``dict`` (``label``, ``confidence``, ``proba``,
            ``log_posterior``) for a single-cell query, else ``(labels,
            proba)`` DataFrames for a batch / multi-cell query.
        """
        kind, q = self._resolve_queries(query)
        if kind == 'single':
            return self._predict_single(
                q, prior_alpha=prior_alpha, abstain_threshold=abstain_threshold,
                max_query_cg=max_query_cg, max_query_ch=max_query_ch,
                contexts=contexts)
        if kind == 'multicell':
            return self.predict_multicell(
                q, prior_alpha=prior_alpha, abstain_threshold=abstain_threshold,
                max_query_cg=max_query_cg, max_query_ch=max_query_ch,
                n_jobs=n_jobs, contexts=contexts)
        return self.predict_batch(
            q, prior_alpha=prior_alpha, abstain_threshold=abstain_threshold,
            max_query_cg=max_query_cg, max_query_ch=max_query_ch, n_jobs=n_jobs,
            contexts=contexts)

    def _predict_single(self, query, prior_alpha=0.0, abstain_threshold=None,
                        max_query_cg=None, max_query_ch=None,
                        contexts='cg+ch'):
        """Predict the most likely cell type for one query cell.

        Parameters
        ----------
        query : str or (mc, cov) tuple
            The query cell's all-cytosine ``.cz`` (or preloaded arrays).
        prior_alpha : float
            Abundance-prior temperature (default 0 -> uniform prior).
        abstain_threshold : float or None
            If given and the top probability is below it, the label becomes
            ``'unassigned'`` (abstention).
        max_query_cg, max_query_ch : int or None
            If given, randomly keep at most this many of the query cell's
            covered (``cov > 0``) CpG / CpH cytosines when scoring. ``None``
            (default) uses every covered site. See :meth:`log_posterior`.

        Returns
        -------
        dict
            ``{'label', 'confidence', 'proba' (Series), 'log_posterior'
            (Series)}``.
        """
        logpost = self.log_posterior(
            query, prior_alpha,
            max_query_cg=max_query_cg, max_query_ch=max_query_ch,
            contexts=contexts)
        proba = pd.Series(self._softmax(logpost.values), index=self.cell_types)
        top = proba.idxmax()
        conf = float(proba.max())
        label = top
        if abstain_threshold is not None and conf < abstain_threshold:
            label = 'unassigned'
        return {'label': label, 'confidence': conf,
                'proba': proba, 'log_posterior': logpost}

    def predict_batch(self, queries, prior_alpha=0.0, abstain_threshold=None,
                      max_query_cg=None, max_query_ch=None, n_jobs=None,
                      contexts='cg+ch'):
        """Predict many cells at once.

        Parameters
        ----------
        queries : dict {cell_id: path} or list of paths
            One all-cytosine ``.cz`` per query cell.
        prior_alpha, abstain_threshold, max_query_cg, max_query_ch, contexts :
            see :meth:`predict`.
        n_jobs : int or None, optional
            Threads used to score cells in parallel. ``None``/``1`` -> serial;
            ``-1`` -> all CPUs (see :func:`_resolve_n_jobs`). Each cell is
            independent and its ``.cz`` read/decompression releases the GIL,
            so this scales across cores. Output order matches ``queries``.

        Returns
        -------
        (labels, proba) : (pandas.DataFrame, pandas.DataFrame)
            ``labels`` has columns ``['label', 'confidence']`` indexed by
            cell id; ``proba`` is a cell x cell_type probability matrix.
        """
        qmap = _as_id_map(queries)
        items = list(qmap.items())

        def _predict_one(item):
            cid, path = item
            res = self._predict_single(path, prior_alpha=prior_alpha,
                                       abstain_threshold=abstain_threshold,
                                       max_query_cg=max_query_cg,
                                       max_query_ch=max_query_ch,
                                       contexts=contexts)
            return cid, res['label'], res['confidence'], res['proba'].rename(cid)

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_predict_one, items))
        else:
            results = [_predict_one(it) for it in items]

        rows = [(cid, label, conf) for cid, label, conf, _ in results]
        proba_rows = [pr for _, _, _, pr in results]
        labels = pd.DataFrame(
            [(l, c) for _, l, c in rows],
            index=[i for i, _, _ in rows], columns=['label', 'confidence'])
        proba = pd.DataFrame(proba_rows)
        return labels, proba

    def predict_multicell(self, path, prior_alpha=0.0, abstain_threshold=None,
                          max_query_cg=None, max_query_ch=None, n_jobs=None,
                          contexts='cg+ch'):
        """Predict every cell packed in one concatenated (cat) ``.cz`` file.

        A ``catcz`` output stacks many single-cell ``.cz`` into one file,
        appending a cell dimension so chunk keys become e.g.
        ``('chr1', 'cell1')`` instead of ``('chr1',)``. This reads each
        cell's ``mc``/``cov`` at the model's selected sites and scores it
        independently, returning ``(labels, proba)`` like
        :meth:`predict_batch` (indexed by the per-cell key).

        Parameters
        ----------
        path : str
            A multi-cell (cat) ``.cz`` aligned to the same reference axis the
            model was fit on.
        prior_alpha, abstain_threshold, max_query_cg, max_query_ch, contexts :
            see :meth:`predict`.
        n_jobs : int or None, optional
            Threads used to score cells in parallel (each worker opens its own
            reader; ``.cz`` decode releases the GIL). ``None``/``1`` -> serial.

        Returns
        -------
        (labels, proba) : (pandas.DataFrame, pandas.DataFrame)
        """
        if self.cell_types is None:
            raise RuntimeError("call fit() before predicting.")
        path = _abspath(path)
        needed = set()
        for chan in (self._cg, self._ch):
            if chan is not None:
                needed.update(chan['chunks'].keys())
        # Inspect the layout once: column indices, chunk dims, and all keys.
        r = Reader(path)
        try:
            cols = list(r.header['columns'])
            if self.mc_col not in cols or self.cov_col not in cols:
                raise ValueError(
                    f"{path}: expected columns {self.mc_col!r} and "
                    f"{self.cov_col!r}, got {cols!r}")
            mc_i, cov_i = cols.index(self.mc_col), cols.index(self.cov_col)
            dims = list(r.header.get('chunk_dims', []) or [])
            keys = list(r.chunk_key2offset.keys())
        finally:
            r.close()
        if len(dims) <= 1:
            raise ValueError(
                f"{path}: not a multi-cell (cat) .cz (chunk_dims={dims}); use "
                f"predict()/predict_batch() for single-cell files.")
        # The chromosome dim is the one whose values best match the model's
        # chromosomes; the remaining dim(s) identify the cell.
        model_chroms = {k[0] for k in needed}
        chrom_idx, best_ov = 0, -1
        for i in range(len(dims)):
            ov = len({key[i] for key in keys} & model_chroms)
            if ov > best_ov:
                chrom_idx, best_ov = i, ov
        cell_idxs = [j for j in range(len(dims)) if j != chrom_idx]
        cells = {}  # cell_id -> {chrom_1tuple: full_chunk_key}
        for key in keys:
            chrom = (key[chrom_idx],)
            if chrom not in needed:
                continue
            cell = (key[cell_idxs[0]] if len(cell_idxs) == 1
                    else tuple(key[j] for j in cell_idxs))
            cells.setdefault(cell, {})[chrom] = key
        if not cells:
            raise ValueError(
                f"{path}: no chunks match the model's chromosomes; is it "
                f"aligned to the same reference axis as training?")
        items = list(cells.items())

        def _predict_one(item):
            cell, keymap = item
            rr = Reader(path)
            try:
                query_chunks = {}
                for chrom, full_key in keymap.items():
                    arr = rr.chunk2numpy(full_key)
                    mc = np.asarray(arr[f'f{mc_i}'])
                    cov = np.asarray(arr[f'f{cov_i}'])
                    n_ref = self._chunk_lens.get(chrom)
                    if n_ref is not None and mc.size != int(n_ref):
                        raise ValueError(
                            f"{path}: chunk {chrom!r} of cell {cell!r} has "
                            f"{mc.size} rows but the reference axis has "
                            f"{int(n_ref)}; query must share the training axis.")
                    query_chunks[chrom] = (mc, cov)
            finally:
                rr.close()
            logpost = self._log_posterior_from_chunks(
                query_chunks, prior_alpha, max_query_cg, max_query_ch,
                contexts)
            proba = self._softmax(logpost.values)
            top = int(np.argmax(proba))
            conf = float(proba[top])
            label = self.cell_types[top]
            if abstain_threshold is not None and conf < abstain_threshold:
                label = 'unassigned'
            cid = ('_'.join(str(v) for v in cell)
                   if isinstance(cell, tuple) else str(cell))
            return cid, label, conf, pd.Series(
                proba, index=self.cell_types).rename(cid)

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_predict_one, items))
        else:
            results = [_predict_one(it) for it in items]

        rows = [(cid, label, conf) for cid, label, conf, _ in results]
        proba_rows = [pr for _, _, _, pr in results]
        labels = pd.DataFrame(
            [(l, c) for _, l, c in rows],
            index=[i for i, _, _ in rows], columns=['label', 'confidence'])
        proba = pd.DataFrame(proba_rows)
        return labels, proba

    # ----------------------------------------------------------------------
    def site_importance(self, reference=None, top=None, include_theta=True):
        """Rank the model's kept cytosines by how discriminative they are.

        Importance is the across-cell-type methylation-frequency range
        ``theta.max - theta.min`` — the very signal :meth:`fit` selected sites
        on. A large value means the site is differentially methylated across
        cell types (biologically interpretable); ``top_type`` / ``min_type``
        name the most / least methylated cell type there. Returns a pandas
        DataFrame with one row per kept site, sorted by importance.

        Parameters
        ----------
        reference : str or None
            A ``build_ref`` reference ``.cz`` (with ``pos``/``strand``/
            ``context``) to annotate each site with its genomic coordinate.
            ``None`` -> only ``chrom`` + the reference-row index are given.
        top : int or None
            Keep only the top-N most important sites (across both channels).
            ``None`` -> all kept sites.
        include_theta : bool
            Add one column per cell type holding that type's methylation
            frequency ``theta``. For large models pass ``top=`` to bound the
            table size.

        Returns
        -------
        pandas.DataFrame
            Columns: ``context`` (CG/CH), ``chrom``, ``ref_row``,
            ``importance``, ``top_type``, ``min_type`` (+ ``pos``/``strand``/
            ``ref_context`` if ``reference`` given, + one column per cell type
            if ``include_theta``).
        """
        if self.cell_types is None:
            raise RuntimeError("call fit()/load() before site_importance().")
        types = list(self.cell_types)
        types_arr = np.asarray(types, dtype=object)
        n_total = sum(c['n_sites'] for c in (self._cg, self._ch)
                      if c is not None)
        if include_theta and top is None and n_total * len(types) > 5_000_000:
            logger.warning(
                f"site_importance is materialising {n_total:,} sites x "
                f"{len(types)} theta columns; pass top= to bound memory.")
        ref_reader = Reader(_abspath(reference)) if reference is not None else None
        parts = []
        try:
            for cname, chan in (('CG', self._cg), ('CH', self._ch)):
                if chan is None:
                    continue
                lt = chan['log_theta']
                sarr = chan['sites']
                for k, (lo, hi) in chan['chunks'].items():
                    theta = np.exp(np.asarray(lt[lo:hi], dtype=np.float64))
                    imp = theta.max(axis=1) - theta.min(axis=1)
                    rows = np.asarray(sarr[lo:hi]).astype(np.int64)
                    cols = {
                        'context': cname,
                        'chrom': k[0],
                        'ref_row': rows,
                        'importance': imp.astype(np.float32),
                        'top_type': types_arr[theta.argmax(axis=1)],
                        'min_type': types_arr[theta.argmin(axis=1)],
                    }
                    if ref_reader is not None and k in ref_reader.chunk_key2offset:
                        arr = ref_reader.chunk2numpy(k, reformat=True)
                        for src, dst in (('pos', 'pos'), ('strand', 'strand'),
                                         ('context', 'ref_context')):
                            if src in arr:
                                cols[dst] = np.asarray(arr[src])[rows]
                    df = pd.DataFrame(cols)
                    if include_theta:
                        df = pd.concat(
                            [df, pd.DataFrame(theta.astype(np.float32),
                                              columns=types, index=df.index)],
                            axis=1)
                    parts.append(df)
        finally:
            if ref_reader is not None:
                ref_reader.close()
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values('importance', ascending=False,
                              ignore_index=True, kind='stable')
        if top is not None:
            out = out.head(int(top)).reset_index(drop=True)
        return out

    # ----------------------------------------------------------------------
    @staticmethod
    def _resolve_contexts(contexts):
        """Normalize ``contexts`` into an ordered list of ``'cg'``/``'ch'``.

        Accepts a string (``'cg'``, ``'ch'``, ``'both'``, ``'cg+ch'``,
        ``'cg,ch'``) or an iterable of such names (case-insensitive).
        """
        if isinstance(contexts, str):
            s = contexts.lower().strip()
            if s == 'both':
                s = 'cg,ch'
            parts = s.replace('+', ',').replace(' ', ',')
            contexts = [p for p in parts.split(',') if p]
        contexts = [str(c).lower() for c in contexts]
        seen, ordered = set(), []
        for c in contexts:
            if c not in ('cg', 'ch'):
                raise ValueError(
                    f"contexts must be from {{'cg', 'ch'}}, got {c!r}.")
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        if not ordered:
            raise ValueError("contexts is empty; pass 'cg', 'ch', or 'cg+ch'.")
        return ordered

    def _deconv_design(self, query_chunks, contexts, weight_by_cov, min_cov):
        """Assemble the deconvolution design ``(theta, beta, weight)``.

        Gathers, over the requested ``contexts`` (``'cg'``/``'ch'``), the
        reference frequencies ``theta`` at the model's selected sites together
        with the bulk query's per-site methylation level ``beta = mc / cov``
        and weight (``cov`` when ``weight_by_cov`` else 1). Only sites the bulk
        actually covers (``cov >= min_cov``) are kept; a chunk absent from the
        query is skipped.
        """
        chan_map = {'cg': self._cg, 'ch': self._ch}
        thetas, betas, weights = [], [], []
        for name in contexts:
            chan = chan_map[name]
            if chan is None:
                continue
            lt = chan['log_theta']
            sarr = chan['sites']
            for k, (lo, hi) in chan['chunks'].items():
                arrs = query_chunks.get(k)
                if arrs is None:
                    continue
                mc_k, cov_k = arrs
                s = np.asarray(sarr[lo:hi])
                cov = np.asarray(cov_k[s], dtype=np.float64)
                obs = cov >= float(min_cov)
                if not obs.any():
                    continue
                mc = np.asarray(mc_k[s], dtype=np.float64)[obs]
                cov = cov[obs]
                theta = np.exp(np.asarray(lt[lo:hi], dtype=np.float64)[obs])
                thetas.append(theta)
                betas.append(np.clip(mc / cov, 0.0, 1.0))
                weights.append(cov if weight_by_cov
                               else np.ones(cov.shape[0], dtype=np.float64))
        if not thetas:
            raise ValueError(
                "no covered sites overlap the model's markers; is the query "
                "aligned to the same reference axis and covered at the "
                "selected sites?")
        return (np.concatenate(thetas, axis=0), np.concatenate(betas),
                np.concatenate(weights))

    def _deconv_query_chunks(self, query):
        """``{chunk_key: (mc, cov)}`` for a bulk query, incl. array beta input.

        A methylation array carries only a per-site beta value (no coverage);
        pass it as a full-axis 1-D ``np.ndarray`` (or a length-1 ``(beta,)``
        tuple) and it is treated as unit coverage so ``beta = mc / cov`` equals
        the array value. Otherwise ``query`` is a ``.cz`` path or an
        ``(mc, cov)`` array pair, handled by :meth:`_query_chunks`.
        """
        is_beta = isinstance(query, np.ndarray) or (
            isinstance(query, (tuple, list)) and len(query) == 1
            and isinstance(query[0], np.ndarray))
        if is_beta:
            beta = np.asarray(query[0] if isinstance(query, (tuple, list))
                              else query, dtype=np.float64)
            return self._query_chunks((beta, np.ones_like(beta)))
        return self._query_chunks(query)

    def deconvolve(self, query, contexts='cg', weight_by_cov=True,
                   sum_to_one=True, allow_unknown=False, min_cov=1):
        """Deconvolve one bulk sample into cell-type fractions.

        Treats the bulk methylation profile as a mixture of the fitted
        reference cell types and solves for their fractions via constrained
        least squares (:func:`_solve_fractions`).

        Parameters
        ----------
        query : str, np.ndarray, or (mc, cov) tuple
            The bulk sample: a WGBS ``.cz`` path (or preloaded full-axis
            ``(mc, cov)`` arrays), or — for a methylation array — a full-axis
            1-D beta ``np.ndarray`` (unit coverage). Must be aligned to the
            reference axis the model was fit on.
        contexts : str or iterable, optional
            Which cytosine contexts to use: ``'cg'`` (default; the only choice
            for arrays and the usual one for WGBS), ``'ch'``, or ``'cg+ch'``
            to use both channels jointly.
        weight_by_cov : bool, optional
            Weight each site by its bulk coverage (weighted least squares).
            Default True; ignored (no coverage) for array beta input.
        sum_to_one : bool, optional
            Constrain the fractions to sum to 1 (default True). False -> plain
            NNLS (fractions need not sum to 1).
        allow_unknown : bool, optional
            With ``sum_to_one=True``, relax to ``sum <= 1`` and report the
            remainder as an ``'unknown'`` fraction (an unmodelled compartment).
            Default False.
        min_cov : int, optional
            Only use bulk sites with coverage >= this (default 1).

        Returns
        -------
        pandas.Series
            Cell-type fractions indexed by cell type (plus ``'unknown'`` when
            ``allow_unknown``). ``.attrs`` carries ``'r2'`` (fit quality) and
            ``'n_sites'`` (sites used).
        """
        if self.cell_types is None:
            raise RuntimeError("call fit()/load() before deconvolve().")
        contexts = self._resolve_contexts(contexts)
        is_beta = isinstance(query, np.ndarray) or (
            isinstance(query, (tuple, list)) and len(query) == 1
            and isinstance(query[0], np.ndarray))
        query_chunks = self._deconv_query_chunks(query)
        theta, beta, weight = self._deconv_design(
            query_chunks, contexts,
            weight_by_cov=weight_by_cov and not is_beta, min_cov=min_cov)
        frac, r2 = _solve_fractions(
            theta, beta, weight if not is_beta else None,
            sum_to_one=sum_to_one, allow_unknown=allow_unknown)
        out = pd.Series(frac, index=self.cell_types, dtype=np.float64)
        if allow_unknown:
            out['unknown'] = max(0.0, 1.0 - float(frac.sum()))
        out.attrs['r2'] = r2
        out.attrs['n_sites'] = int(theta.shape[0])
        return out

    def deconvolve_batch(self, queries, contexts='cg', weight_by_cov=True,
                         sum_to_one=True, allow_unknown=False, min_cov=1,
                         n_jobs=None):
        """Deconvolve many bulk samples at once.

        Parameters
        ----------
        queries : dict {sample_id: query} or list
            One bulk sample per entry (each a ``.cz`` path, full-axis beta
            ``np.ndarray``, or ``(mc, cov)`` pair). See :meth:`deconvolve`.
        contexts, weight_by_cov, sum_to_one, allow_unknown, min_cov :
            see :meth:`deconvolve`.
        n_jobs : int or None, optional
            Threads used to deconvolve samples in parallel. ``None``/``1`` ->
            serial; ``-1`` -> all CPUs.

        Returns
        -------
        pandas.DataFrame
            One row per sample; columns are the cell-type fractions (plus
            ``'unknown'`` when ``allow_unknown``) followed by ``'r2'`` and
            ``'n_sites'``.
        """
        if self.cell_types is None:
            raise RuntimeError("call fit()/load() before deconvolve_batch().")
        qmap = _as_id_map(queries) if not isinstance(queries, dict) \
            else dict(queries)
        items = list(qmap.items())

        def _one(item):
            sid, q = item
            frac = self.deconvolve(
                q, contexts=contexts, weight_by_cov=weight_by_cov,
                sum_to_one=sum_to_one, allow_unknown=allow_unknown,
                min_cov=min_cov)
            row = frac.copy()
            row['r2'] = frac.attrs.get('r2', float('nan'))
            row['n_sites'] = frac.attrs.get('n_sites', 0)
            return sid, row.rename(sid)

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_one, items))
        else:
            results = [_one(it) for it in items]
        frac_df = pd.DataFrame([r for _, r in results])
        frac_df.index.name = 'sample_id'
        return frac_df

    def deconvolve_multicell(self, path, contexts='cg', weight_by_cov=True,
                             sum_to_one=True, allow_unknown=False, min_cov=1,
                             n_jobs=None):
        """Deconvolve every bulk sample packed in one concatenated ``.cz``.

        Like :meth:`predict_multicell`, but each packed sample is deconvolved
        into cell-type fractions instead of being classified. Returns the same
        DataFrame layout as :meth:`deconvolve_batch`.
        """
        if self.cell_types is None:
            raise RuntimeError(
                "call fit()/load() before deconvolve_multicell().")
        path = _abspath(path)
        needed = set()
        for chan in (self._cg, self._ch):
            if chan is not None:
                needed.update(chan['chunks'].keys())
        r = Reader(path)
        try:
            cols = list(r.header['columns'])
            if self.mc_col not in cols or self.cov_col not in cols:
                raise ValueError(
                    f"{path}: expected columns {self.mc_col!r} and "
                    f"{self.cov_col!r}, got {cols!r}")
            mc_i, cov_i = cols.index(self.mc_col), cols.index(self.cov_col)
            dims = list(r.header.get('chunk_dims', []) or [])
            keys = list(r.chunk_key2offset.keys())
        finally:
            r.close()
        if len(dims) <= 1:
            raise ValueError(
                f"{path}: not a multi-sample (cat) .cz (chunk_dims={dims}); "
                f"use deconvolve()/deconvolve_batch() for single files.")
        model_chroms = {k[0] for k in needed}
        chrom_idx, best_ov = 0, -1
        for i in range(len(dims)):
            ov = len({key[i] for key in keys} & model_chroms)
            if ov > best_ov:
                chrom_idx, best_ov = i, ov
        cell_idxs = [j for j in range(len(dims)) if j != chrom_idx]
        cells = {}
        for key in keys:
            chrom = (key[chrom_idx],)
            if chrom not in needed:
                continue
            cell = (key[cell_idxs[0]] if len(cell_idxs) == 1
                    else tuple(key[j] for j in cell_idxs))
            cells.setdefault(cell, {})[chrom] = key
        if not cells:
            raise ValueError(
                f"{path}: no chunks match the model's chromosomes; is it "
                f"aligned to the same reference axis as training?")
        items = list(cells.items())

        def _one(item):
            cell, keymap = item
            rr = Reader(path)
            try:
                query_chunks = {}
                for chrom, full_key in keymap.items():
                    arr = rr.chunk2numpy(full_key)
                    mc = np.asarray(arr[f'f{mc_i}'])
                    cov = np.asarray(arr[f'f{cov_i}'])
                    n_ref = self._chunk_lens.get(chrom)
                    if n_ref is not None and mc.size != int(n_ref):
                        raise ValueError(
                            f"{path}: chunk {chrom!r} of sample {cell!r} has "
                            f"{mc.size} rows but the reference axis has "
                            f"{int(n_ref)}; query must share the training axis.")
                    query_chunks[chrom] = (mc, cov)
            finally:
                rr.close()
            theta, beta, weight = self._deconv_design(
                query_chunks, self._resolve_contexts(contexts),
                weight_by_cov=weight_by_cov, min_cov=min_cov)
            frac, r2 = _solve_fractions(
                theta, beta, weight, sum_to_one=sum_to_one,
                allow_unknown=allow_unknown)
            sid = ('_'.join(str(v) for v in cell)
                   if isinstance(cell, tuple) else str(cell))
            row = pd.Series(frac, index=self.cell_types, dtype=np.float64)
            if allow_unknown:
                row['unknown'] = max(0.0, 1.0 - float(frac.sum()))
            row['r2'] = r2
            row['n_sites'] = int(theta.shape[0])
            return row.rename(sid)

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_one, items))
        else:
            results = [_one(it) for it in items]
        frac_df = pd.DataFrame(results)
        frac_df.index.name = 'sample_id'
        return frac_df

    # ----------------------------------------------------------------------
    def save(self, path):
        """Persist the fitted model to a directory of memory-mappable ``.npy``.

        Writes ``<path>/meta.json`` plus, per channel, the concatenated
        ``{cg,ch}_log_theta.npy`` / ``_log1m_theta.npy`` / ``_sites.npy``
        arrays (uncompressed, so :meth:`load` can ``mmap`` them). The arrays
        are copied straight from this model's on-disk store — nothing is
        pickled and the full table never has to be materialised in RAM.

        Parameters
        ----------
        path : str
            Output **directory** (created if needed). It becomes a
            self-contained, memory-mappable model store.

        Returns
        -------
        str
            The directory written.
        """
        if self.cell_types is None:
            raise RuntimeError("call fit() before save().")
        if not self._store_dir or not os.path.isdir(self._store_dir):
            raise RuntimeError("model store is gone (was close() called?).")
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        key2id = {k: i for i, k in enumerate(self._chunk_keys)}
        chan_meta = {}
        for name, chan in (('cg', self._cg), ('ch', self._ch)):
            if chan is None:
                chan_meta[name] = None
                continue
            for suffix in ('log_theta', 'log1m_theta', 'sites'):
                src = os.path.join(self._store_dir, f'{name}_{suffix}.npy')
                dst = os.path.join(path, f'{name}_{suffix}.npy')
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copyfile(src, dst)
            chan_meta[name] = {
                'alpha0': chan['alpha0'], 'beta0': chan['beta0'],
                'n_sites': chan['n_sites'],
                'chunks': {str(key2id[k]): [int(lo), int(hi)]
                           for k, (lo, hi) in chan['chunks'].items()},
            }
        meta = {
            'format': 'cytozip.CellTypeClassifier',
            'version': 3,
            'alpha0_cg': self.alpha0_cg, 'beta0_cg': self.beta0_cg,
            'alpha0_ch': self.alpha0_ch, 'beta0_ch': self.beta0_ch,
            'lambda_cg': self.lambda_cg, 'lambda_ch': self.lambda_ch,
            'mc_col': self.mc_col, 'cov_col': self.cov_col,
            'context_col': self.context_col,
            'cell_types': list(self.cell_types),
            'cell_counts': (dict(self.cell_counts)
                            if self.cell_counts is not None else None),
            'chunk_keys': [list(k) for k in self._chunk_keys],
            'chunk_lens': [int(self._chunk_lens[k]) for k in self._chunk_keys],
            'n_full': int(self._n_full),
            'cg': chan_meta['cg'],
            'ch': chan_meta['ch'],
        }
        with open(os.path.join(path, 'meta.json'), 'w') as fh:
            json.dump(meta, fh)
        logger.info(f"saved CellTypeClassifier to {path}")
        return path

    @classmethod
    def load(cls, path):
        """Load a model saved by :meth:`save` — arrays are memory-mapped.

        The returned classifier can immediately score a new single-cell
        ``.cz`` via :meth:`predict` / :meth:`predict_batch`, as long as the
        query is aligned to the same reference axis used at ``fit`` time. The
        per-channel ``log_theta`` / ``log1m_theta`` / ``sites`` arrays are
        ``mmap``-ed read-only, so scoring pages in only the chunk in use — the
        whole table never has to fit in RAM.

        Parameters
        ----------
        path : str
            Path to a model **directory** produced by :meth:`save`.

        Returns
        -------
        CellTypeClassifier
        """
        path = os.path.abspath(os.path.expanduser(path))
        with open(os.path.join(path, 'meta.json')) as fh:
            meta = json.load(fh)
        if meta.get('format') != 'cytozip.CellTypeClassifier':
            raise ValueError(f"{path}: not a CellTypeClassifier model dir.")
        obj = cls(
            lambda_cg=meta['lambda_cg'], lambda_ch=meta['lambda_ch'],
            mc_col=meta['mc_col'], cov_col=meta['cov_col'],
            context_col=meta['context_col'])
        obj.alpha0_cg = meta['alpha0_cg']
        obj.beta0_cg = meta['beta0_cg']
        obj.alpha0_ch = meta['alpha0_ch']
        obj.beta0_ch = meta['beta0_ch']
        obj.cell_types = list(meta['cell_types'])
        obj.cell_counts = meta['cell_counts']
        obj._chunk_keys = [tuple(k) for k in meta['chunk_keys']]
        obj._chunk_lens = {
            k: int(n) for k, n in
            zip(obj._chunk_keys, meta['chunk_lens'])}
        obj._n_full = int(meta['n_full'])
        # The store lives at ``path`` and is NOT owned (never auto-deleted).
        obj._store_dir = path
        obj._owns_store = False
        # Rebuild the per-chunk row spans on the full axis (for preloaded
        # (mc, cov) query arrays).
        span = {}
        off = 0
        for k in obj._chunk_keys:
            n_k = obj._chunk_lens[k]
            span[k] = (off, off + n_k)
            off += n_k
        obj._chunk_span = span

        def _load_channel(name):
            cm = meta.get(name)
            if cm is None:
                return None
            lt = np.load(os.path.join(path, f'{name}_log_theta.npy'),
                         mmap_mode='r')
            l1 = np.load(os.path.join(path, f'{name}_log1m_theta.npy'),
                         mmap_mode='r')
            si = np.load(os.path.join(path, f'{name}_sites.npy'),
                         mmap_mode='r')
            chunks = {obj._chunk_keys[int(idx)]: (int(v[0]), int(v[1]))
                      for idx, v in cm['chunks'].items()}
            return {'alpha0': cm['alpha0'], 'beta0': cm['beta0'],
                    'n_sites': cm['n_sites'], 'chunks': chunks,
                    'sites': si, 'log_theta': lt, 'log1m_theta': l1}

        obj._cg = _load_channel('cg')
        obj._ch = _load_channel('ch')
        return obj


def _as_id_map(queries):
    """Normalize a dict/list of query paths into a ``{id: path}`` dict."""
    if isinstance(queries, dict):
        return dict(queries)
    return {os.path.basename(str(p)).split('.')[0]: p for p in queries}


def _cz_stem(path):
    """Return a ``.cz`` file's basename without the ``.cz`` extension."""
    base = os.path.basename(str(path))
    return base[:-3] if base.endswith('.cz') else os.path.splitext(base)[0]


def _cz_is_multicell(path):
    """True if a ``.cz`` is a concatenated multi-cell file (>1 chunk dim)."""
    r = Reader(_abspath(path))
    try:
        return len(r.header.get('chunk_dims', []) or []) > 1
    finally:
        r.close()


def _model_store_complete(model_dir):
    """True if ``model_dir`` holds a complete, loadable model store.

    Checks for a valid ``meta.json`` plus, for every non-null channel it
    declares, the three memory-mappable arrays (``{cg,ch}_log_theta.npy`` /
    ``_log1m_theta.npy`` / ``_sites.npy``). Used by :func:`predict_cell_type`
    to skip ``fit`` and reuse a model saved by a previous run.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    meta_path = os.path.join(model_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    if meta.get('format') != 'cytozip.CellTypeClassifier':
        return False
    for name in ('cg', 'ch'):
        if meta.get(name) is None:
            continue
        for suffix in ('log_theta', 'log1m_theta', 'sites'):
            if not os.path.isfile(
                    os.path.join(model_dir, f'{name}_{suffix}.npy')):
                return False
    return True


def _resolve_pseudobulks(pseudobulks):
    """Backward-compatible module-level alias.

    Deprecated: use :meth:`CellTypeClassifier._resolve_pseudobulks`.
    """
    return CellTypeClassifier._resolve_pseudobulks(pseudobulks)


def _read_query_table(path):
    """Read a 2-column ``[cell_id, cz_path]`` table into ``{id: path}``.

    Delimiter is auto-detected; a header row is dropped when the 2nd cell of
    the first row is not a ``.cz`` path. Only the first two columns are used.
    """
    df = pd.read_csv(path, sep=None, engine='python', header=None,
                     dtype=str, comment='#')
    if df.shape[1] < 2:
        raise ValueError(
            f"query table {path!r} needs >= 2 columns [cell_id, cz_path].")
    if not str(df.iloc[0, 1]).endswith('.cz'):
        df = df.iloc[1:]  # drop header row
    return {str(cid): str(p) for cid, p in zip(df.iloc[:, 0], df.iloc[:, 1])}


def _resolve_queries(query):
    """Backward-compatible module-level alias.

    Deprecated: use :meth:`CellTypeClassifier._resolve_queries`.
    """
    return CellTypeClassifier._resolve_queries(query)


# ==========================================================
def predict_cell_type(query=None, pseudobulks=None, reference=None,
                      cell_counts=None, prior_alpha=0.0,
                      lambda_cg=1.0, lambda_ch=1.0,
                      max_query_cg=None, max_query_ch=None,
                      alpha0_cg=None, beta0_cg=None,
                      alpha0_ch=None, beta0_ch=None, prior_min_cov=2,
                      top_cg=None, top_ch=None,
                      min_range_cg=0.0, min_range_ch=0.0, top_per_class=None,
                      mc_col='mc', cov_col='cov', context_col='context',
                      abstain_threshold=None, contexts='cg+ch',
                      n_jobs=None, outdir=None, prefix=''):
    """One-shot convenience wrapper: fit cell-type files and predict cell(s).

    Fit per-type methylation frequencies from ``pseudobulks`` and predict the
    most likely cell type for one or many query cells. See
    :class:`CellTypeClassifier` for the parameter semantics.

    Parameters
    ----------
    query : str, (mc, cov) tuple, list, dict, or pandas.DataFrame
        The cell(s) to predict, in any of these forms:

        * a single all-cytosine ``.cz`` file (or preloaded ``(mc, cov)``
          arrays) -> returns a single prediction ``dict``;
        * a **concatenated (cat) ``.cz``** holding many cells (chunk keys
          carry a cell dimension) -> auto-detected and expanded, returns
          ``(labels, proba)`` like a batch;
        * a **directory** of ``.cz`` files (each file's stem is the cell id);
        * a **table** (``pandas.DataFrame`` or a delimited file) whose first
          two columns are ``[cell_id, cz_path]``;
        * a ``{cell_id: path}`` dict or a list of ``.cz`` paths.

        Every form except a single file/array is treated as a **batch** and
        returns ``(labels, proba)`` DataFrames.
    pseudobulks : dict or str
        Either a ``{cell_type: path}`` dict of per-type pseudobulk ``.cz``
        files (all cytosines, usually mc/cov-only), or a **directory** of
        such ``.cz`` files (each file's stem is the cell-type name).
    reference : str or None
        ``build_ref`` reference ``.cz`` supplying the per-row ``context``
        (required when the ``pseudobulks`` files lack a context column). See
        :meth:`CellTypeClassifier.fit`.
    cell_counts : dict {cell_type: int} or None, optional
        Number of reference cells pooled into each cell-type pseudobulk (the
        abundance of each type in the reference atlas). Only used together
        with ``prior_alpha`` to build the abundance prior at predict time; it
        does **not** affect the fitted frequencies. ``None`` (default) ->
        uniform prior. Keys must match the ``pseudobulks`` cell types.
    prior_alpha : float, optional
        Strength of the abundance prior (``pi_t ∝ cell_counts[t] **
        prior_alpha``). ``0.0`` (default) -> uniform prior over types (ignore
        ``cell_counts``); ``1.0`` -> full abundance prior (types weighted by
        their reference cell counts); values in between temper it. Has no
        effect when ``cell_counts`` is ``None``.
    max_query_cg, max_query_ch : int or None
        If given, randomly keep at most this many of each query cell's covered
        (``cov > 0``) CpG / CpH cytosines when scoring (drawn independently
        per cell). ``None`` (default) uses every covered site. This is a
        predict-time knob forwarded to :meth:`CellTypeClassifier.predict` /
        :meth:`CellTypeClassifier.predict_batch`.
    contexts : str, optional
        Which cytosine context(s) to classify on: ``'cg'``, ``'ch'``, or
        ``'both'`` / ``'cg+ch'`` (default, use both channels). The model is
        always fit on both contexts; this only selects which channel(s) score
        the query, so one trained model can be reused for cg-only, ch-only, or
        combined classification.
    outdir : str or None, optional
        If given, create the directory (if needed) and write the fitted model
        to ``<outdir>/model`` (a memory-mappable directory) plus the
        predictions to ``<outdir>/predictions.csv`` and
        ``<outdir>/predict_proba.csv``. The model's memory-mapped store is
        written **directly** under ``<outdir>/model`` (real disk), so a large
        model never spills into a small or RAM-backed (tmpfs) ``/tmp`` (which
        would crash with a **"Bus error" (SIGBUS)**). Without ``outdir`` the
        store falls back to ``$TMPDIR`` / ``/tmp`` — pass ``outdir`` (on real
        disk) if you hit a SIGBUS. If ``<outdir>/model`` already holds a
        complete saved model (``meta.json`` plus the per-channel ``.npy``
        arrays), fitting is **skipped** and that model is loaded and reused
        directly (so ``pseudobulks``/``reference`` need not be re-supplied).
    prefix : str, optional
        Prepended to the output filenames, i.e. ``<outdir>/<prefix>predictions.csv``
        and ``<outdir>/<prefix>predict_proba.csv``. Default ``''`` (no prefix).
        Pass a distinct prefix (e.g. ``'miseq_'``) when predicting a different
        query batch with the same model so its predictions do not overwrite an
        earlier batch's. Ignored when ``outdir`` is ``None``.
    n_jobs : int or None, optional
        Threads used to read pseudobulks and (for batch queries) score cells
        in parallel. ``None``/``1`` -> serial; ``-1`` -> all CPUs.
    (remaining) :
        Forwarded to :class:`CellTypeClassifier` / :meth:`fit` /
        :meth:`predict`.

    Returns
    -------
    dict or (pandas.DataFrame, pandas.DataFrame)
        For a single query: the :meth:`CellTypeClassifier.predict` ``dict``
        (``label``, ``confidence``, ``proba``, ``log_posterior``). For a
        batch query: ``(labels, proba)`` as in
        :meth:`CellTypeClassifier.predict_batch`.
    """
    model_dir = None
    if outdir is not None:
        outdir = os.path.abspath(os.path.expanduser(outdir))
        os.makedirs(outdir, exist_ok=True)
        model_dir = os.path.join(outdir, 'model')

    clf = CellTypeClassifier(
        lambda_cg=lambda_cg, lambda_ch=lambda_ch,
        mc_col=mc_col, cov_col=cov_col, context_col=context_col)
    # ``fit`` reuses a complete model already saved under <outdir>/model
    # (skips training and loads it), otherwise it trains. The memmap store is
    # routed straight to real disk (<outdir>/model) so a large model never
    # spills into a small/RAM-backed /tmp (-> SIGBUS). Without outdir it falls
    # back to a system temp dir. ``fit`` accepts a {cell_type: path} dict or a
    # directory of .cz files directly.
    clf.fit(pseudobulks=pseudobulks, reference=reference,
            cell_counts=cell_counts, top_cg=top_cg, top_ch=top_ch,
            min_range_cg=min_range_cg, min_range_ch=min_range_ch,
            top_per_class=top_per_class,
            alpha0_cg=alpha0_cg, beta0_cg=beta0_cg,
            alpha0_ch=alpha0_ch, beta0_ch=beta0_ch,
            prior_min_cov=prior_min_cov, contexts=contexts,
            n_jobs=n_jobs, outdir=model_dir)
    if outdir is not None and not _model_store_complete(model_dir):
        clf.save(model_dir)

    # ``predict`` dispatches on the query form (single / cat multi-cell /
    # directory / dict / table), returning a dict for a single cell or
    # (labels, proba) DataFrames otherwise.
    result = clf.predict(
        query, prior_alpha=prior_alpha, abstain_threshold=abstain_threshold,
        max_query_cg=max_query_cg, max_query_ch=max_query_ch, n_jobs=n_jobs,
        contexts=contexts)
    if outdir is not None:
        if isinstance(result, dict):
            cid = _cz_stem(query) if isinstance(query, str) else 'query'
            labels = pd.DataFrame(
                [[result['label'], result['confidence']]],
                index=[cid], columns=['label', 'confidence'])
            proba = result['proba'].rename(cid).to_frame().T
        else:
            labels, proba = result
        labels.index.name = 'cell_id'
        proba.index.name = 'cell_id'
        labels.to_csv(os.path.join(outdir, f'{prefix}predictions.csv'))
        proba.to_csv(os.path.join(outdir, f'{prefix}predict_proba.csv'))
        logger.info(f"wrote predictions to {outdir}")
    return result


# ==========================================================
def deconvolve_bulk(query=None, pseudobulks=None, reference=None,
                    contexts='cg', weight_by_cov=True, sum_to_one=True,
                    allow_unknown=False, min_cov=1,
                    alpha0_cg=None, beta0_cg=None,
                    alpha0_ch=None, beta0_ch=None, prior_min_cov=2,
                    top_cg=None, top_ch=None,
                    min_range_cg=0.0, min_range_ch=0.0,
                    mc_col='mc', cov_col='cov', context_col='context',
                    n_jobs=None, outdir=None):
    """One-shot convenience wrapper: fit cell-type references and deconvolve.

    Builds per-type methylation frequencies from ``pseudobulks`` (the same
    reference signature matrix :func:`predict_cell_type` uses) and deconvolves
    one or many **bulk** samples (WGBS ``.cz`` or methylation-array beta) into
    cell-type fractions. See :meth:`CellTypeClassifier.deconvolve` for the
    deconvolution knobs and :meth:`CellTypeClassifier.fit` for the reference /
    site-selection ones.

    Parameters
    ----------
    query : str, np.ndarray, (mc, cov) tuple, list, dict, or pandas.DataFrame
        The bulk sample(s) to deconvolve:

        * a single WGBS ``.cz`` (or preloaded ``(mc, cov)`` arrays, or a
          full-axis beta ``np.ndarray`` for an array) -> a single ``Series``;
        * a **concatenated (cat) ``.cz``** holding many samples -> auto-
          detected, returns a fractions ``DataFrame``;
        * a **directory** of ``.cz`` files, a **table** (DataFrame or delimited
          file with ``[sample_id, cz_path]``), a ``{sample_id: path}`` dict, or
          a list of paths -> a fractions ``DataFrame``.
    pseudobulks : dict or str
        ``{cell_type: path}`` per-type pseudobulk ``.cz`` files, or a directory
        of them (file stem = cell-type name). See
        :meth:`CellTypeClassifier.fit`.
    reference : str or None
        ``build_ref`` reference ``.cz`` supplying the per-row ``context``.
    contexts, weight_by_cov, sum_to_one, allow_unknown, min_cov :
        Deconvolution options, see :meth:`CellTypeClassifier.deconvolve`.
    outdir : str or None, optional
        If given, the fitted model is written to ``<outdir>/model`` (and reused
        if already complete) and the fractions to
        ``<outdir>/fractions.csv``.
    (remaining) :
        Forwarded to :class:`CellTypeClassifier` / :meth:`fit`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A single ``Series`` of fractions for one bulk sample, else a fractions
        ``DataFrame`` (one row per sample, cell-type columns + ``r2`` /
        ``n_sites``).
    """
    model_dir = None
    if outdir is not None:
        outdir = os.path.abspath(os.path.expanduser(outdir))
        os.makedirs(outdir, exist_ok=True)
        model_dir = os.path.join(outdir, 'model')

    if model_dir is not None and _model_store_complete(model_dir):
        logger.info(
            f"found existing model at {model_dir}; skipping fit and loading it")
        clf = CellTypeClassifier.load(model_dir)
    else:
        pseudobulks = _resolve_pseudobulks(pseudobulks)
        clf = CellTypeClassifier(
            mc_col=mc_col, cov_col=cov_col, context_col=context_col)
        clf.fit(pseudobulks=pseudobulks, reference=reference,
                top_cg=top_cg, top_ch=top_ch,
                min_range_cg=min_range_cg, min_range_ch=min_range_ch,
                alpha0_cg=alpha0_cg, beta0_cg=beta0_cg,
                alpha0_ch=alpha0_ch, beta0_ch=beta0_ch,
                prior_min_cov=prior_min_cov, contexts=contexts,
                n_jobs=n_jobs, outdir=model_dir)
        if outdir is not None:
            clf.save(model_dir)

    dec_kw = dict(contexts=contexts, weight_by_cov=weight_by_cov,
                  sum_to_one=sum_to_one, allow_unknown=allow_unknown,
                  min_cov=min_cov)
    # A bare full-axis beta array (methylation array) is a single sample that
    # _resolve_queries does not recognise, so short-circuit it here.
    if isinstance(query, np.ndarray):
        result = clf.deconvolve(query, **dec_kw)
        if outdir is not None:
            result.to_frame(name='fraction').to_csv(
                os.path.join(outdir, 'fractions.csv'))
            logger.info(f"wrote fractions to {outdir}")
        return result

    kind, q = _resolve_queries(query)
    if kind == 'single':
        result = clf.deconvolve(q, **dec_kw)
        if outdir is not None:
            sid = _cz_stem(q) if isinstance(q, str) else 'sample'
            result.rename(sid).to_frame(name='fraction').to_csv(
                os.path.join(outdir, 'fractions.csv'))
            logger.info(f"wrote fractions to {outdir}")
        return result
    if kind == 'multicell':
        result = clf.deconvolve_multicell(q, n_jobs=n_jobs, **dec_kw)
    else:
        result = clf.deconvolve_batch(q, n_jobs=n_jobs, **dec_kw)
    if outdir is not None:
        result.to_csv(os.path.join(outdir, 'fractions.csv'))
        logger.info(f"wrote fractions to {outdir}")
    return result


# ==========================================================
_DNA_COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def top_cytosine_fasta(importance, genome, out_fasta=None, split_dir=None,
                       flank=50, top_n=100, group_col='top_type',
                       contexts=('CG', 'CH'), pos_base=1, uppercase=True,
                       drop_incomplete=True):
    """Extract flanking DNA for each cell type's top cytosines into FASTA.

    Consumes a :meth:`CellTypeClassifier.site_importance` table (built with
    ``reference=`` so it carries ``pos``/``strand``) and, per cell type and per
    context (CG / CH), takes the ``top_n`` most important cytosines and pulls
    ``flank`` bp on each side of the cytosine from a genome FASTA. Minus-strand
    sites are reverse-complemented so the C stays centred and the sequence
    reads 5'->3' on the methylated strand.

    Parameters
    ----------
    importance : pandas.DataFrame
        Output of :meth:`CellTypeClassifier.site_importance` with
        ``reference=`` set (needs columns ``chrom``, ``pos``, ``strand``,
        ``context``, ``importance`` and ``group_col``).
    genome : str or pysam.FastaFile
        Indexed genome FASTA (a ``.fai`` beside it) or an open
        ``pysam.FastaFile``.
    out_fasta : str or None
        Write **all** records to this single FASTA (headers encode cell type
        and context).
    split_dir : str or None
        Write **one FASTA per (cell type, context)** as
        ``<split_dir>/<cell_type>.<context>.fa`` (handy for per-type motif
        discovery, e.g. MEME/HOMER). ``out_fasta`` and ``split_dir`` may be
        used together or separately.
    flank : int
        bp on each side of the cytosine (window length ``2*flank + 1``).
    top_n : int
        Per (cell type, context), keep this many top-importance sites.
    group_col : str
        Column defining the cell type (default ``'top_type'`` = the most
        methylated type; use ``'min_type'`` for the least methylated).
    contexts : iterable of str
        Which ``context`` values to include (default ``('CG', 'CH')``).
    pos_base : int
        1 if ``pos`` is 1-based (default), 0 if 0-based.
    uppercase : bool
        Upper-case the extracted sequence.
    drop_incomplete : bool
        Skip sites whose window runs off a chromosome end (default True).

    Returns
    -------
    list of (header, seq) tuples
        Also written to ``out_fasta`` / ``split_dir`` when given.
    """
    required = ('chrom', 'pos', 'strand', 'context', 'importance', group_col)
    missing = [c for c in required if c not in importance.columns]
    if missing:
        raise ValueError(
            f"importance table missing column(s) {missing}; call "
            f"site_importance(reference=...) so it carries pos/strand.")
    flank = int(flank)
    own = False
    if hasattr(genome, 'fetch'):
        fa = genome
    else:
        try:
            import pysam  # type: ignore
        except ImportError as e:
            raise ImportError(
                "top_cytosine_fasta needs pysam to read the genome FASTA; "
                "`pip install pysam` or `conda install -c bioconda pysam`."
            ) from e
        fa = pysam.FastaFile(os.path.abspath(os.path.expanduser(genome)))
        own = True
    chrom_len = {c: int(n) for c, n in zip(fa.references, fa.lengths)}
    missing_chroms = set()
    records = []                      # flat [(header, seq), ...]
    grouped = {}                      # (cell_type, context) -> [(header, seq)]
    try:
        for context in contexts:
            sub = importance[importance['context'] == context]
            if sub.empty:
                continue
            for cell_type, grp in sub.groupby(group_col, sort=False):
                top = grp.nlargest(top_n, 'importance')
                ct = '_'.join(str(cell_type).replace('|', '_').split())
                key = (ct, str(context))
                for _, row in top.iterrows():
                    chrom = str(row['chrom'])
                    clen = chrom_len.get(chrom)
                    if clen is None:
                        missing_chroms.add(chrom)
                        continue
                    c0 = int(row['pos']) - int(pos_base)   # 0-based centre
                    start, end = c0 - flank, c0 + flank + 1
                    if start < 0 or end > clen:
                        if drop_incomplete:
                            continue
                        start, end = max(0, start), min(clen, end)
                    seq = fa.fetch(chrom, start, end)
                    if uppercase:
                        seq = seq.upper()
                    strand = str(row['strand'])
                    if strand == '-':
                        seq = seq.translate(_DNA_COMPLEMENT)[::-1]
                    header = (f"{ct}|{context}|{chrom}:{int(row['pos'])}:"
                              f"{strand}|imp={float(row['importance']):.4f}")
                    records.append((header, seq))
                    grouped.setdefault(key, []).append((header, seq))
    finally:
        if own:
            fa.close()
    if missing_chroms:
        logger.warning(
            f"{len(missing_chroms)} chrom(s) not in the genome FASTA and "
            f"skipped (name mismatch?): {sorted(missing_chroms)[:5]}...")
    if out_fasta is not None:
        out_fasta = os.path.abspath(os.path.expanduser(out_fasta))
        os.makedirs(os.path.dirname(out_fasta) or '.', exist_ok=True)
        with open(out_fasta, 'w') as fh:
            for header, seq in records:
                fh.write(f">{header}\n{seq}\n")
        logger.info(f"wrote {len(records)} sequences to {out_fasta}")
    if split_dir is not None:
        split_dir = os.path.abspath(os.path.expanduser(split_dir))
        os.makedirs(split_dir, exist_ok=True)
        for (ct, context), recs in grouped.items():
            with open(os.path.join(split_dir, f"{ct}.{context}.fa"), 'w') as fh:
                for header, seq in recs:
                    fh.write(f">{header}\n{seq}\n")
        logger.info(
            f"wrote {len(grouped)} per-(cell type, context) FASTAs to "
            f"{split_dir}")
    return records
