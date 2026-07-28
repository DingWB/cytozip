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
import struct
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from .cz import Reader, np, pd


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


# ==========================================================
def _read_mc_cov(path, mc_col='mc', cov_col='cov', chunk_keys=None):
    """Read (mc, cov) as 1-D arrays concatenated over ``chunk_keys``.

    Parameters
    ----------
    path : str
        Path to a ``.cz`` file with ``mc_col`` / ``cov_col`` columns.
    mc_col, cov_col : str
        Column names holding the methylated count and coverage.
    chunk_keys : list of tuple or None
        Chunk keys (e.g. ``[('chr1',), ('chr2',), ...]``) in the desired
        concatenation order. ``None`` uses ``sorted(chunk_key2offset)``.

    Returns
    -------
    (chunk_keys, mc, cov) : (list, np.ndarray float64, np.ndarray float64)
        The keys actually used and the concatenated arrays.
    """
    path = os.path.abspath(os.path.expanduser(path))
    r = Reader(path)
    try:
        cols = list(r.header['columns'])
        if mc_col not in cols or cov_col not in cols:
            raise ValueError(
                f"{path}: expected columns {mc_col!r} and {cov_col!r}, "
                f"got {cols!r}")
        mi, ci = cols.index(mc_col), cols.index(cov_col)
        if chunk_keys is None:
            chunk_keys = sorted(r.chunk_key2offset.keys())
        mc_parts, cov_parts = [], []
        for k in chunk_keys:
            if k not in r.chunk_key2offset:
                raise ValueError(f"{path}: missing chunk {k!r}")
            arr = r.chunk2numpy(k)
            mc_parts.append(np.asarray(arr[f'f{mi}']))
            cov_parts.append(np.asarray(arr[f'f{ci}']))
    finally:
        r.close()
    if mc_parts:
        mc = np.concatenate(mc_parts).astype(np.float64)
        cov = np.concatenate(cov_parts).astype(np.float64)
    else:
        mc = np.zeros(0, dtype=np.float64)
        cov = np.zeros(0, dtype=np.float64)
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
        Name of the context column.
    chunk_keys : list of tuple or None
        Chunk order; ``None`` uses ``sorted(chunk_key2offset)``.

    Returns
    -------
    (chunk_keys, ctx) : (list, np.ndarray of ``S{n}`` bytes)
        Fixed-width byte strings such as ``b'CGN'`` / ``b'CAC'``.
    """
    path = os.path.abspath(os.path.expanduser(path))
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
        for k in chunk_keys:
            if k not in r.chunk_key2offset:
                raise ValueError(f"{path}: missing chunk {k!r}")
            arr = r.chunk2numpy(k)
            fld = np.asarray(arr[f'f{ci}'])
            parts.append(np.frombuffer(fld.tobytes(), dtype=f'S{width}'))
    finally:
        r.close()
    ctx = np.concatenate(parts) if parts else np.zeros(0, dtype='S1')
    return chunk_keys, ctx


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
        If ``None``, keep every site with score >= ``min_range``. If an
        ``int``, keep the top-``top`` sites by score. If a ``float`` in
        (0, 1], keep the top fraction of sites.
    min_range : float
        Minimum across-type range required to keep a site (default 0.0).

    Returns
    -------
    np.ndarray
        Sorted 1-D array of selected site indices (into the channel subset).
    """
    if theta.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    score = theta.max(axis=1) - theta.min(axis=1)
    keep = np.where(score >= min_range)[0]
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


# ==========================================================
class CellTypeClassifier:
    """Site-likelihood (naive-Bayes) cell-type classifier for methylation .cz.

    Estimate per-site methylation frequencies from cell-type pseudobulk
    ``.cz`` files, then score query cells by aggregated per-cytosine
    Bernoulli log-likelihood in the CpG and CpH channels. Each input ``.cz``
    holds *all* cytosines (CpG + CpH) as ``mc``/``cov``; the CpG vs CpH split
    comes from the ``context`` column, read from a ``build_ref`` ``reference``
    ``.cz`` (the cell-type files usually store only ``mc``/``cov``), so no
    pre-splitting into separate CG / CH files is needed.

    Parameters
    ----------
    alpha0_cg, beta0_cg : float or None
        Beta-prior pseudocounts for the CpG channel. ``None`` (default) ->
        estimate them empirically at ``fit`` time from the pooled reference
        counts of that context via :func:`estimate_beta_prior`
        (Beta-Binomial method of moments). Pass explicit floats to override
        the auto-estimate (both must be given to take effect).
    alpha0_ch, beta0_ch : float or None
        Beta-prior pseudocounts for the CpH channel. ``None`` (default) ->
        auto-estimated like the CpG channel (its own separate prior).
    lambda_cg, lambda_ch : float
        Log-space channel weights (default 1 each). CpH sites are far more
        numerous and often dominate implicitly; lower ``lambda_ch`` to
        rebalance.
    prior_min_cov : int
        Minimum coverage for a reference site to enter the empirical prior
        estimation (passed to :func:`estimate_beta_prior`; default 2).
    mc_col, cov_col, context_col : str
        Column names for methylated count, coverage, and sequence context
        in the ``.cz`` files.
    """

    def __init__(self, alpha0_cg=None, beta0_cg=None,
                 alpha0_ch=None, beta0_ch=None,
                 lambda_cg=1.0, lambda_ch=1.0, prior_min_cov=2,
                 mc_col='mc', cov_col='cov', context_col='context'):
        self.alpha0_cg = None if alpha0_cg is None else float(alpha0_cg)
        self.beta0_cg = None if beta0_cg is None else float(beta0_cg)
        self.alpha0_ch = None if alpha0_ch is None else float(alpha0_ch)
        self.beta0_ch = None if beta0_ch is None else float(beta0_ch)
        self.lambda_cg = float(lambda_cg)
        self.lambda_ch = float(lambda_ch)
        self.prior_min_cov = int(prior_min_cov)
        self.mc_col = mc_col
        self.cov_col = cov_col
        self.context_col = context_col

        self.cell_types = None
        self.cell_counts = None
        self._chunk_keys = None
        self._n_full = None
        # Per-channel fitted state (None if that context has no sites):
        #   mask        : bool over the full axis selecting this context
        #   sites       : indices into the masked subset (discriminative)
        #   log_theta / log1m_theta : (n_selected, n_types) float32
        self._cg = None
        self._ch = None

    # ----------------------------------------------------------------------
    def _build_channel(self, cell_types, mc_all, cov_all, mask,
                       alpha0, beta0, top, min_range, label,
                       prior_min_cov=2):
        """Estimate theta for one context and pick discriminative sites.

        When ``alpha0``/``beta0`` are ``None`` the Beta shrinkage prior is
        estimated empirically from this context's pooled (all-cell-type)
        reference counts via :func:`estimate_beta_prior`.
        """
        if mask.sum() == 0:
            logger.warning(f"{label}: no sites in this context; channel off.")
            return None
        # empirical-Bayes prior: estimate per context from the pooled reference
        # counts when not explicitly provided by the user.
        if alpha0 is None or beta0 is None:
            mc_pool = np.concatenate([mc_all[t][mask] for t in cell_types])
            cov_pool = np.concatenate([cov_all[t][mask] for t in cell_types])
            alpha0, beta0 = estimate_beta_prior(
                mc_pool, cov_pool, min_cov=prior_min_cov)
            logger.info(
                f"{label}: estimated Beta prior alpha0={alpha0:.4g}, "
                f"beta0={beta0:.4g} (mean={alpha0 / (alpha0 + beta0):.3f}, "
                f"kappa={alpha0 + beta0:.4g})")
        thetas = []
        for t in cell_types:
            thetas.append(estimate_theta(
                mc_all[t][mask], cov_all[t][mask], alpha0, beta0))
        theta = np.column_stack(thetas)  # (n_ctx, n_types)
        sites = _select_discriminative(theta, top=top, min_range=min_range)
        theta_sel = theta[sites]
        logger.info(
            f"{label}: {int(mask.sum()):,} context sites -> "
            f"{sites.size:,} discriminative sites across "
            f"{len(cell_types)} cell types")
        return {
            'mask': mask,
            'sites': sites,
            'alpha0': float(alpha0),
            'beta0': float(beta0),
            'log_theta': np.log(theta_sel).astype(np.float32),
            'log1m_theta': np.log1p(-theta_sel).astype(np.float32),
        }

    def fit(self, pseudobulks, reference=None, cell_counts=None,
            top_cg=None, top_ch=None, min_range_cg=0.0, min_range_ch=0.0,
            n_jobs=None):
        """Estimate per-type methylation frequencies from cell-type ``.cz`` files.

        Parameters
        ----------
        pseudobulks : dict {cell_type: path}
            Per-type pseudobulk ``.cz`` files, each holding *all* cytosines
            (CpG + CpH) as ``mc``/``cov`` aligned to a common reference axis.
            These typically store **only** ``mc``/``cov`` (no ``pos`` /
            ``context``), in which case ``reference`` is required.
        reference : str or None
            Path to the ``build_ref`` reference ``.cz`` (with ``pos, strand,
            context``) supplying the per-row ``context`` used to split CpG
            vs CpH. ``None`` (default) only works when the ``pseudobulks``
            files themselves carry a ``context_col`` column; otherwise pass
            the reference here.
        cell_counts : dict {cell_type: int}, optional
            Number of reference cells per type, used for the abundance /
            tempered prior at predict time. ``None`` (default) -> uniform
            prior only.
        top_cg, top_ch : None, int, or float, optional
            Discriminative-site selection for each channel. ``None`` keeps
            all sites with range >= ``min_range_*``; an int keeps the top-N
            sites; a float in (0, 1] keeps the top fraction.
        min_range_cg, min_range_ch : float, optional
            Minimum across-type frequency range to keep a site.
        n_jobs : int or None, optional
            Threads used to read the pseudobulk ``.cz`` files in parallel.
            ``None``/``1`` -> serial; ``-1`` -> all CPUs (see
            :func:`_resolve_n_jobs`).

        Returns
        -------
        self
        """
        if not pseudobulks:
            raise ValueError(
                "pseudobulks must be a non-empty {cell_type: path} dict.")
        cell_types = list(pseudobulks)

        # ---- Canonical axis + CpG/CpH masks from the context source ------
        # Cell-type files are usually mc/cov-only, so read context from the
        # build_ref `reference` when provided; otherwise fall back to the
        # pseudobulk files themselves (only works if they carry `context`).
        ctx_source = reference if reference is not None else pseudobulks[cell_types[0]]
        chunk_keys, ctx = _read_context(ctx_source, self.context_col, None)
        cg_mask, ch_mask = _context_masks(ctx)
        n_full = ctx.size

        # ---- Read every pseudobulk's mc/cov once (full axis) -------------
        # Reads/decompression release the GIL, so a thread pool reads the
        # per-type files concurrently when n_jobs > 1.
        def _read_one(t):
            _, mc, cov = _read_mc_cov(
                pseudobulks[t], self.mc_col, self.cov_col, chunk_keys)
            return t, mc, cov

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(cell_types) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                read_results = list(ex.map(_read_one, cell_types))
        else:
            read_results = [_read_one(t) for t in cell_types]

        mc_all, cov_all = {}, {}
        for t, mc, cov in read_results:
            if mc.size != n_full:
                raise ValueError(
                    f"pseudobulk {t!r} has {mc.size} sites but the context "
                    f"axis has {n_full}; all cell-type files (and the query) "
                    f"must be aligned to the same reference axis.")
            mc_all[t], cov_all[t] = mc, cov

        self._cg = self._build_channel(
            cell_types, mc_all, cov_all, cg_mask,
            self.alpha0_cg, self.beta0_cg, top_cg, min_range_cg, 'CG',
            self.prior_min_cov)
        self._ch = self._build_channel(
            cell_types, mc_all, cov_all, ch_mask,
            self.alpha0_ch, self.beta0_ch, top_ch, min_range_ch, 'CH',
            self.prior_min_cov)
        if self._cg is None and self._ch is None:
            raise ValueError("no CpG or CpH sites found; check context_col.")

        # record the effective (possibly auto-estimated) priors so save()/load()
        # and inspection see the actual numbers used.
        if self._cg is not None:
            self.alpha0_cg, self.beta0_cg = self._cg['alpha0'], self._cg['beta0']
        if self._ch is not None:
            self.alpha0_ch, self.beta0_ch = self._ch['alpha0'], self._ch['beta0']

        self.cell_types = cell_types
        self.cell_counts = cell_counts
        self._chunk_keys = chunk_keys
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
    def _score(chan, mc_full, cov_full):
        """Aggregated Bernoulli log-likelihood for one channel: (n_types,)."""
        mc = mc_full[chan['mask']][chan['sites']]
        cov = cov_full[chan['mask']][chan['sites']]
        umc = cov - mc  # unmethylated observations
        # (n_types,) = (n_sites,) . (n_sites, n_types)
        return mc @ chan['log_theta'] + umc @ chan['log1m_theta']

    def _load_query(self, query):
        """Return full-axis (mc, cov) for a query path or preloaded arrays."""
        if isinstance(query, (tuple, list)):
            mc = np.asarray(query[0], np.float64)
            cov = np.asarray(query[1], np.float64)
        else:
            _, mc, cov = _read_mc_cov(
                query, self.mc_col, self.cov_col, self._chunk_keys)
        if mc.size != self._n_full:
            raise ValueError(
                f"query has {mc.size} sites but the model was fit on "
                f"{self._n_full}; query must be aligned to the same reference "
                f"axis as the references.")
        return mc, cov

    def log_posterior(self, query, prior_alpha=0.0):
        """Unnormalized log-posterior per cell type: pandas Series.

        Parameters
        ----------
        query : str or (mc, cov) tuple
            The query cell's all-cytosine ``.cz`` (or preloaded full-axis
            ``(mc, cov)`` arrays).
        prior_alpha : float
            Temperature for the abundance prior (see :meth:`_log_prior`).
        """
        if self.cell_types is None:
            raise RuntimeError("call fit() before predicting.")
        mc, cov = self._load_query(query)
        logpost = self._log_prior(prior_alpha)
        if self._cg is not None:
            logpost = logpost + self.lambda_cg * self._score(self._cg, mc, cov)
        if self._ch is not None:
            logpost = logpost + self.lambda_ch * self._score(self._ch, mc, cov)
        return pd.Series(logpost, index=self.cell_types)

    @staticmethod
    def _softmax(logpost):
        """Numerically stable softmax of an array -> array."""
        vals = np.asarray(logpost, dtype=np.float64)
        vals = vals - vals.max()
        p = np.exp(vals)
        p /= p.sum()
        return p

    def predict_proba(self, query, prior_alpha=0.0):
        """Softmax posterior probabilities per cell type: pandas Series."""
        logpost = self.log_posterior(query, prior_alpha)
        return pd.Series(self._softmax(logpost.values), index=self.cell_types)

    def predict(self, query, prior_alpha=0.0, abstain_threshold=None):
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

        Returns
        -------
        dict
            ``{'label', 'confidence', 'proba' (Series), 'log_posterior'
            (Series)}``.
        """
        logpost = self.log_posterior(query, prior_alpha)
        proba = pd.Series(self._softmax(logpost.values), index=self.cell_types)
        top = proba.idxmax()
        conf = float(proba.max())
        label = top
        if abstain_threshold is not None and conf < abstain_threshold:
            label = 'unassigned'
        return {'label': label, 'confidence': conf,
                'proba': proba, 'log_posterior': logpost}

    def predict_batch(self, queries, prior_alpha=0.0, abstain_threshold=None,
                      n_jobs=None):
        """Predict many cells at once.

        Parameters
        ----------
        queries : dict {cell_id: path} or list of paths
            One all-cytosine ``.cz`` per query cell.
        prior_alpha, abstain_threshold : see :meth:`predict`.
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
            res = self.predict(path, prior_alpha=prior_alpha,
                               abstain_threshold=abstain_threshold)
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

    # ----------------------------------------------------------------------
    def save(self, path):
        """Persist the fitted model to a single ``.npz`` file.

        Stores all hyper-parameters, the reference axis (chunk order +
        length), the CpG/CpH masks and discriminative-site indices, and the
        pre-computed ``log_theta`` / ``log1m_theta`` matrices. Nothing is
        pickled — arrays are saved natively and the metadata is a JSON
        string — so :meth:`load` runs with ``allow_pickle=False``.

        Parameters
        ----------
        path : str
            Output ``.npz`` path. The exact name is used (no ``.npz`` is
            auto-appended).

        Returns
        -------
        str
            The path written.
        """
        if self.cell_types is None:
            raise RuntimeError("call fit() before save().")
        meta = {
            'format': 'cytozip.CellTypeClassifier',
            'version': 1,
            'alpha0_cg': self.alpha0_cg, 'beta0_cg': self.beta0_cg,
            'alpha0_ch': self.alpha0_ch, 'beta0_ch': self.beta0_ch,
            'lambda_cg': self.lambda_cg, 'lambda_ch': self.lambda_ch,
            'mc_col': self.mc_col, 'cov_col': self.cov_col,
            'context_col': self.context_col,
            'cell_types': list(self.cell_types),
            'cell_counts': (dict(self.cell_counts)
                            if self.cell_counts is not None else None),
            'chunk_keys': [list(k) for k in self._chunk_keys],
            'n_full': int(self._n_full),
            'has_cg': self._cg is not None,
            'has_ch': self._ch is not None,
        }
        arrays = {'meta': np.array(json.dumps(meta))}
        for name, chan in (('cg', self._cg), ('ch', self._ch)):
            if chan is None:
                continue
            arrays[f'{name}_mask'] = chan['mask']
            arrays[f'{name}_sites'] = chan['sites']
            arrays[f'{name}_log_theta'] = chan['log_theta']
            arrays[f'{name}_log1m_theta'] = chan['log1m_theta']
        path = os.path.abspath(os.path.expanduser(path))
        # Pass a file object so numpy does not auto-append ".npz".
        with open(path, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        logger.info(f"saved CellTypeClassifier to {path}")
        return path

    @classmethod
    def load(cls, path):
        """Load a model saved by :meth:`save` (no refitting needed).

        The returned classifier can immediately score a new single-cell
        ``.cz`` via :meth:`predict` / :meth:`predict_batch`, as long as the
        query is aligned to the same reference axis used at ``fit`` time.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` produced by :meth:`save`.

        Returns
        -------
        CellTypeClassifier
        """
        path = os.path.abspath(os.path.expanduser(path))
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['meta'].item()))
        if meta.get('format') != 'cytozip.CellTypeClassifier':
            raise ValueError(f"{path}: not a CellTypeClassifier model file.")
        obj = cls(
            alpha0_cg=meta['alpha0_cg'], beta0_cg=meta['beta0_cg'],
            alpha0_ch=meta['alpha0_ch'], beta0_ch=meta['beta0_ch'],
            lambda_cg=meta['lambda_cg'], lambda_ch=meta['lambda_ch'],
            mc_col=meta['mc_col'], cov_col=meta['cov_col'],
            context_col=meta['context_col'])
        obj.cell_types = list(meta['cell_types'])
        obj.cell_counts = meta['cell_counts']
        obj._chunk_keys = [tuple(k) for k in meta['chunk_keys']]
        obj._n_full = int(meta['n_full'])
        if meta['has_cg']:
            obj._cg = {
                'mask': data['cg_mask'], 'sites': data['cg_sites'],
                'log_theta': data['cg_log_theta'],
                'log1m_theta': data['cg_log1m_theta']}
        if meta['has_ch']:
            obj._ch = {
                'mask': data['ch_mask'], 'sites': data['ch_sites'],
                'log_theta': data['ch_log_theta'],
                'log1m_theta': data['ch_log1m_theta']}
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


def _resolve_pseudobulks(pseudobulks):
    """Resolve ``pseudobulks`` into a ``{cell_type: path}`` dict.

    Accepts either a ``{cell_type: path}`` dict (returned as a shallow copy)
    or a directory containing per-cell-type ``.cz`` files (each file's stem
    becomes the cell-type name).
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
                raise ValueError(f"no .cz files found in pseudobulk dir {d!r}")
            return {_cz_stem(f): f for f in files}
    raise ValueError(
        "pseudobulks must be a {cell_type: path} dict or a directory of "
        f".cz files; got {type(pseudobulks)!r}.")


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
    """Resolve ``query`` into ``('single', path_or_arrays)`` or
    ``('batch', {cell_id: path})``.

    Accepts:

    * a preloaded ``(mc, cov)`` array pair -> single;
    * a single ``.cz`` file path -> single;
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
        return 'batch', {str(cid): str(p)
                         for cid, p in zip(query.iloc[:, 0], query.iloc[:, 1])}
    if isinstance(query, dict):
        return 'batch', dict(query)
    if isinstance(query, (list, tuple)):
        return 'batch', _as_id_map(list(query))
    if isinstance(query, str):
        q = os.path.abspath(os.path.expanduser(query))
        if os.path.isdir(q):
            files = sorted(glob.glob(os.path.join(q, '*.cz')))
            if not files:
                raise ValueError(f"no .cz files found in query dir {q!r}")
            return 'batch', {_cz_stem(f): f for f in files}
        if q.endswith('.cz'):
            return 'single', q
        return 'batch', _read_query_table(q)  # any other file -> table
    raise ValueError(f"unsupported query type: {type(query)!r}.")


# ==========================================================
def predict_cell_type(query=None, pseudobulks=None, reference=None,
                      cell_counts=None, prior_alpha=0.0,
                      lambda_cg=1.0, lambda_ch=1.0,
                      alpha0_cg=None, beta0_cg=None,
                      alpha0_ch=None, beta0_ch=None, prior_min_cov=2,
                      top_cg=None, top_ch=None,
                      min_range_cg=0.0, min_range_ch=0.0,
                      mc_col='mc', cov_col='cov', context_col='context',
                      abstain_threshold=None, n_jobs=None, outdir=None):
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
    outdir : str or None, optional
        If given, create the directory (if needed) and write the fitted model
        to ``<outdir>/model.npz`` plus the predictions to
        ``<outdir>/predictions.csv`` and ``<outdir>/predict_proba.csv``.
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
    pseudobulks = _resolve_pseudobulks(pseudobulks)
    clf = CellTypeClassifier(
        alpha0_cg=alpha0_cg, beta0_cg=beta0_cg,
        alpha0_ch=alpha0_ch, beta0_ch=beta0_ch,
        lambda_cg=lambda_cg, lambda_ch=lambda_ch, prior_min_cov=prior_min_cov,
        mc_col=mc_col, cov_col=cov_col, context_col=context_col)
    clf.fit(pseudobulks=pseudobulks, reference=reference, cell_counts=cell_counts,
            top_cg=top_cg, top_ch=top_ch,
            min_range_cg=min_range_cg, min_range_ch=min_range_ch, n_jobs=n_jobs)

    if outdir is not None:
        outdir = os.path.abspath(os.path.expanduser(outdir))
        os.makedirs(outdir, exist_ok=True)
        clf.save(os.path.join(outdir, 'model.npz'))

    kind, q = _resolve_queries(query)
    if kind == 'single':
        res = clf.predict(q, prior_alpha=prior_alpha,
                          abstain_threshold=abstain_threshold)
        if outdir is not None:
            cid = _cz_stem(q) if isinstance(q, str) else 'query'
            labels = pd.DataFrame(
                [[res['label'], res['confidence']]],
                index=[cid], columns=['label', 'confidence'])
            labels.index.name = 'cell_id'
            proba = res['proba'].rename(cid).to_frame().T
            proba.index.name = 'cell_id'
            labels.to_csv(os.path.join(outdir, 'predictions.csv'))
            proba.to_csv(os.path.join(outdir, 'predict_proba.csv'))
            logger.info(f"wrote predictions to {outdir}")
        return res

    labels, proba = clf.predict_batch(
        q, prior_alpha=prior_alpha, abstain_threshold=abstain_threshold,
        n_jobs=n_jobs)
    if outdir is not None:
        labels.index.name = 'cell_id'
        proba.index.name = 'cell_id'
        labels.to_csv(os.path.join(outdir, 'predictions.csv'))
        proba.to_csv(os.path.join(outdir, 'predict_proba.csv'))
        logger.info(f"wrote predictions to {outdir}")
    return labels, proba
