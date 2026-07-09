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
import struct
from loguru import logger
from .cz import Reader, np, pd


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
    alpha0_cg, beta0_cg : float
        Beta-prior pseudocounts for the CpG channel. CpG is highly
        methylated, so a prior mean pulled high (``alpha0_cg > beta0_cg``)
        is reasonable; the defaults are neutral (1, 1).
    alpha0_ch, beta0_ch : float
        Beta-prior pseudocounts for the CpH channel. CpH is lowly
        methylated, so a prior mean pulled low (``beta0_ch > alpha0_ch``)
        is reasonable; the defaults are neutral (1, 1).
    lambda_cg, lambda_ch : float
        Log-space channel weights (default 1 each). CpH sites are far more
        numerous and often dominate implicitly; lower ``lambda_ch`` to
        rebalance.
    mc_col, cov_col, context_col : str
        Column names for methylated count, coverage, and sequence context
        in the ``.cz`` files.
    """

    def __init__(self, alpha0_cg=1.0, beta0_cg=1.0,
                 alpha0_ch=1.0, beta0_ch=1.0,
                 lambda_cg=1.0, lambda_ch=1.0,
                 mc_col='mc', cov_col='cov', context_col='context'):
        self.alpha0_cg = float(alpha0_cg)
        self.beta0_cg = float(beta0_cg)
        self.alpha0_ch = float(alpha0_ch)
        self.beta0_ch = float(beta0_ch)
        self.lambda_cg = float(lambda_cg)
        self.lambda_ch = float(lambda_ch)
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
                       alpha0, beta0, top, min_range, label):
        """Estimate theta for one context and pick discriminative sites."""
        if mask.sum() == 0:
            logger.warning(f"{label}: no sites in this context; channel off.")
            return None
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
            'log_theta': np.log(theta_sel).astype(np.float32),
            'log1m_theta': np.log1p(-theta_sel).astype(np.float32),
        }

    def fit(self, pseudobulks, reference=None, cell_counts=None,
            top_cg=None, top_ch=None, min_range_cg=0.0, min_range_ch=0.0):
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
        mc_all, cov_all = {}, {}
        for t in cell_types:
            _, mc, cov = _read_mc_cov(
                pseudobulks[t], self.mc_col, self.cov_col, chunk_keys)
            if mc.size != n_full:
                raise ValueError(
                    f"pseudobulk {t!r} has {mc.size} sites but the context "
                    f"axis has {n_full}; all cell-type files (and the query) "
                    f"must be aligned to the same reference axis.")
            mc_all[t], cov_all[t] = mc, cov

        self._cg = self._build_channel(
            cell_types, mc_all, cov_all, cg_mask,
            self.alpha0_cg, self.beta0_cg, top_cg, min_range_cg, 'CG')
        self._ch = self._build_channel(
            cell_types, mc_all, cov_all, ch_mask,
            self.alpha0_ch, self.beta0_ch, top_ch, min_range_ch, 'CH')
        if self._cg is None and self._ch is None:
            raise ValueError("no CpG or CpH sites found; check context_col.")

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

    def predict_batch(self, queries, prior_alpha=0.0, abstain_threshold=None):
        """Predict many cells at once.

        Parameters
        ----------
        queries : dict {cell_id: path} or list of paths
            One all-cytosine ``.cz`` per query cell.
        prior_alpha, abstain_threshold : see :meth:`predict`.

        Returns
        -------
        (labels, proba) : (pandas.DataFrame, pandas.DataFrame)
            ``labels`` has columns ``['label', 'confidence']`` indexed by
            cell id; ``proba`` is a cell x cell_type probability matrix.
        """
        qmap = _as_id_map(queries)
        rows, proba_rows = [], []
        for cid, path in qmap.items():
            res = self.predict(path, prior_alpha=prior_alpha,
                               abstain_threshold=abstain_threshold)
            rows.append((cid, res['label'], res['confidence']))
            proba_rows.append(res['proba'].rename(cid))
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


# ==========================================================
def predict_cell_type(query=None, pseudobulks=None, reference=None,
                      cell_counts=None, prior_alpha=0.0,
                      lambda_cg=1.0, lambda_ch=1.0,
                      alpha0_cg=1.0, beta0_cg=1.0,
                      alpha0_ch=1.0, beta0_ch=1.0,
                      top_cg=None, top_ch=None,
                      min_range_cg=0.0, min_range_ch=0.0,
                      mc_col='mc', cov_col='cov', context_col='context',
                      abstain_threshold=None):
    """One-shot convenience wrapper: fit cell-type files and predict one cell.

    Given a single-cell query ``.cz`` (all cytosines) and per-cell-type
    pseudobulk ``.cz`` files, return the most likely cell type. See
    :class:`CellTypeClassifier` for the parameter semantics.

    Parameters
    ----------
    query : str or (mc, cov) tuple
        Single-cell all-cytosine ``.cz`` (or preloaded full-axis arrays).
    pseudobulks : dict {cell_type: path}
        Per-type pseudobulk ``.cz`` files (all cytosines, usually mc/cov-only).
    reference : str or None
        ``build_ref`` reference ``.cz`` supplying the per-row ``context``
        (required when the ``pseudobulks`` files lack a context column). See
        :meth:`CellTypeClassifier.fit`.
    (remaining) :
        Forwarded to :class:`CellTypeClassifier` / :meth:`fit` /
        :meth:`predict`.

    Returns
    -------
    dict
        As returned by :meth:`CellTypeClassifier.predict`
        (``label``, ``confidence``, ``proba``, ``log_posterior``).
    """
    clf = CellTypeClassifier(
        alpha0_cg=alpha0_cg, beta0_cg=beta0_cg,
        alpha0_ch=alpha0_ch, beta0_ch=beta0_ch,
        lambda_cg=lambda_cg, lambda_ch=lambda_ch,
        mc_col=mc_col, cov_col=cov_col, context_col=context_col)
    clf.fit(pseudobulks=pseudobulks, reference=reference, cell_counts=cell_counts,
            top_cg=top_cg, top_ch=top_ch,
            min_range_cg=min_range_cg, min_range_ch=min_range_ch)
    return clf.predict(query=query, prior_alpha=prior_alpha,
                       abstain_threshold=abstain_threshold)
