#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features.py - Build cell x feature count matrices / AnnData from .cz files.

Public entry points:

* :func:`cz_to_anndata` - aggregate many single-cell ``.cz`` files (or one
  ``catcz``-merged ``.cz``) over a BED / DataFrame / genome-bins feature
  set and emit a single :class:`anndata.AnnData` with ``mc`` / ``cov``
  layers. ``.X`` holds a user-selected score (raw fraction, per-cell
  empirical-Bayes posterior fraction ``(mc + alpha)/(cov + alpha + beta)``
  from a Beta-Binomial prior, or a binomial hypo / hyper score).
* :func:`parse_features` - read a BED (plain or bgzipped) feature file
  into a DataFrame, normalizing columns.
* :func:`make_genome_bins` - tile every chromosome of a chrom-size /
  ``.fai`` file into fixed-size bins and return a feature DataFrame.

Design notes
------------
* Features are grouped by chrom for I/O locality; within each chrom we
  issue a ``Reader.fetch_chunk_bytes`` decode and vectorize region sums
  through numpy cumsum + ``searchsorted``.
* mc/cov aggregation is vectorized in numpy.
* Scoring (``prior_mean`` posterior-fraction, hypo/hyper) follows the
  ALLCools conventions (see ``ALLCools/mcds/utilities.py`` and
  ``ALLCools/count_matrix/dataset.py``).
"""
from __future__ import annotations

import os
import struct
from typing import List, Optional, Sequence, Union

from loguru import logger
import numpy as np
import pandas as pd

from .cz import Reader, _fmt_to_np_dtype


_VALID_SCORES = ("frac", "posterior_frac", "hypo-score", "hyper-score", "umc")
# mc / cov are intentionally excluded from the public score choices: they are
# always available in adata.layers['mc'] / ['cov'], so storing them in .X is
# redundant. _compute_score_matrix_sparse still handles them internally.
_VALID_NAN_POLICIES = ("auto", "zero", "nan", "prior_mean")


def _record_dtype_for(formats):
    """Build a numpy structured dtype matching a ``.cz`` record layout.

    Columns with non-numeric struct codes (e.g. ``'3s'``, ``'c'``) map to
    fixed-width byte strings so ``np.frombuffer`` keeps working.
    """
    fields = []
    for i, f in enumerate(formats):
        dt = _fmt_to_np_dtype(f[-1])
        if dt is None:
            dt = f"S{struct.calcsize(f)}"
        fields.append((f"c{i}", dt))
    return np.dtype(fields)


# ---------------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------------
def parse_features(
    features: Union[str, pd.DataFrame],
    name_col: Optional[int] = 3,
) -> pd.DataFrame:
    """Normalize a BED / bgz feature file into ``[chrom, start, end, name]``.

    Parameters
    ----------
    features : str or DataFrame
        - str: path to BED, BED.gz, or BED.bgz (tabular, tab-separated, no header).
        - DataFrame: used as-is; first three columns must be chrom/start/end.
    name_col : int or None
        0-based column index to use as feature name. If the column is
        missing or ``name_col=None``, falls back to ``chrom:start-end``.
    """
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        path = os.path.abspath(os.path.expanduser(features))
        # pandas auto-handles .gz; bgzip'd BED shares the .gz header format.
        df = pd.read_csv(path, sep="\t", header=None, comment="#",
                         dtype={0: str})
    if df.shape[1] < 3:
        raise ValueError("features must have at least 3 columns (chrom,start,end)")
    df = df.rename(columns={0: "chrom", 1: "start", 2: "end"})
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    if name_col is not None and df.shape[1] > name_col:
        df["name"] = df.iloc[:, name_col].astype(str)
    else:
        df["name"] = (df["chrom"].astype(str) + ":"
                      + df["start"].astype(str) + "-"
                      + df["end"].astype(str))
    # Guarantee uniqueness of feature names for AnnData.var_names
    if df["name"].duplicated().any():
        df["name"] = (df["name"].astype(str) + "_"
                      + df.groupby("name").cumcount().astype(str))
    return df[["chrom", "start", "end", "name"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# GTF parsing (gene-level features with optional flanking)
# ---------------------------------------------------------------------------
DEFAULT_GTF = "~/Ref/mm10/annotations/gencode.vM23.annotation.gtf"


def parse_gtf(
    gtf: str,
    flank_bp: int = 0,
    feature_type: str = "gene",
    id_col: str = "gene_name",
    exclude_chroms: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Parse a GENCODE / Ensembl GTF into a gene-level feature DataFrame.

    Extracts one row per ``feature_type`` (default ``'gene'``) record,
    optionally extends each interval by ``flank_bp`` on each side, and
    returns a DataFrame whose 4th column (``name``) is unique so it can
    be passed straight to :func:`cz_to_anndata` as ``features=``.

    Handling of duplicate ``gene_name``:
    GENCODE / Ensembl annotations frequently map a single ``gene_name``
    symbol to multiple distinct ``gene_id`` records (small RNAs like
    ``Y_RNA`` / ``Metazoa_SRP`` / ``5S_rRNA`` / ``Mir*`` reuse a symbol
    across dozens of unrelated loci, sometimes spanning megabases).
    Naive ``min(start), max(end)`` would collapse them into a single
    monstrous interval. Following Bioconductor / featureCounts /
    scanpy convention we instead **keep only the longest gene_id** per
    ``gene_name`` so every symbol maps to a single, biologically
    meaningful interval. ``gene_id`` is always unique by GTF spec, so
    when ``id_col='gene_id'`` no disambiguation is needed.

    Parameters
    ----------
    gtf : str
        Path to a GTF (plain or ``.gz`` / ``.bgz``).
    flank_bp : int, default 0
        Bases to extend on each side of the gene body. Use ``2000`` to
        reproduce the classical gene-body + 2 kb promoter window.
    feature_type : str, default ``'gene'``
        GTF ``feature`` (column 3) to keep.
    id_col : {'gene_name', 'gene_id'}, default ``'gene_name'``
        Which GTF attribute becomes the human-readable ``name`` (the
        4th column of the returned DataFrame, used as ``var_names`` on
        the AnnData built by :func:`cz_to_anndata`). With ``'gene_id'``
        every GTF record stays a row of its own; with ``'gene_name'``
        same-name records are de-duplicated by keeping the longest
        gene_id.
    exclude_chroms : sequence of str, optional
        Chromosomes to drop (e.g. ``['chrM']``).

    Returns
    -------
    DataFrame with columns
    ``['chrom', 'start', 'end', 'name', 'gene_id', 'gene_name',
    'gene_type', 'strand']``. ``name`` equals the chosen ``id_col`` and
    is unique.
    """
    if id_col not in ("gene_name", "gene_id"):
        raise ValueError(
            f"id_col must be 'gene_name' or 'gene_id', got {id_col!r}")
    path = os.path.abspath(os.path.expanduser(str(gtf)))
    df = pd.read_csv(
        path, sep="\t", header=None, comment="#",
        usecols=[0, 2, 3, 4, 6, 8],
        names=["chrom", "record_type", "start", "end", "strand", "info"],
        dtype={0: str},
    )
    df = df[df["record_type"] == feature_type].copy()
    if df.empty:
        raise ValueError(
            f"No '{feature_type}' rows found in {gtf!r}; try feature_type="
            f"'transcript' or check the GTF.")

    def _parse_info(s: str) -> dict:
        out = {}
        for item in s.replace('"', '').strip().rstrip(";").split(";"):
            item = item.strip()
            if not item:
                continue
            k, _, v = item.partition(" ")
            out[k.strip()] = v.strip()
        return out

    info = df["info"].map(_parse_info)
    df["gene_id"] = info.map(lambda d: d.get("gene_id", ""))
    df["gene_name"] = info.map(lambda d: d.get("gene_name", ""))
    df["gene_type"] = info.map(lambda d: d.get("gene_type",
                                               d.get("gene_biotype", "")))
    # GTF is 1-based inclusive on both ends; BED is 0-based half-open.
    df["start"] = (df["start"].astype(np.int64) - 1).clip(lower=0)
    df["end"] = df["end"].astype(np.int64)
    if exclude_chroms:
        df = df[~df["chrom"].isin(set(exclude_chroms))]

    agg = df[["chrom", "start", "end",
              "gene_id", "gene_name", "gene_type", "strand"]
             ].reset_index(drop=True)

    if id_col == "gene_name":
        # Resolve duplicate gene_names by keeping only the longest
        # gene_id per name. This collapses small-RNA symbol clutter
        # (Y_RNA × 100s of loci) to a single representative locus
        # without the megabase-span artefact a coordinate merge would
        # produce. ``idxmax`` on (end-start) gives us that row directly.
        spans = agg["end"] - agg["start"]
        winner_idx = (spans.groupby(agg["gene_name"]).idxmax()
                      .to_numpy())
        agg = agg.loc[winner_idx].reset_index(drop=True)

    # Apply flanking after dedup so flanking is symmetric per kept locus.
    if flank_bp and int(flank_bp) > 0:
        fb = int(flank_bp)
        agg["start"] = (agg["start"] - fb).clip(lower=0)
        agg["end"] = agg["end"] + fb

    agg["name"] = agg[id_col].astype(str)
    # Defensive: if id_col entries are somehow non-unique (e.g. empty
    # gene_name fields, malformed GTF), append a numeric suffix.
    if agg["name"].duplicated().any():
        agg["name"] = (agg["name"]
                       + "_"
                       + agg.groupby("name").cumcount().astype(str))
    return agg[["chrom", "start", "end", "name",
                "gene_id", "gene_name", "gene_type", "strand"]
               ].reset_index(drop=True)


def _looks_like_gtf(path: str) -> bool:
    """Detect a GTF path by extension (``.gtf`` / ``.gtf.gz`` / ``.gtf.bgz``)."""
    p = str(path).lower()
    for ext in (".gtf", ".gtf.gz", ".gtf.bgz"):
        if p.endswith(ext):
            return True
    return False


# ---------------------------------------------------------------------------
# Blacklist handling (ENCODE-style BED of regions to exclude)
# ---------------------------------------------------------------------------
def load_blacklist(
    blacklist: Union[str, pd.DataFrame],
) -> dict:
    """Read a blacklist BED / bed.gz into ``{chrom: (starts, ends)}``.

    Accepts a path or an already-loaded DataFrame (first three columns are
    chrom, start, end). Intervals are sorted per chrom and merged when
    overlapping so overlap tests are a single pair of ``searchsorted``
    lookups.
    """
    if isinstance(blacklist, pd.DataFrame):
        df = blacklist.iloc[:, :3].copy()
    else:
        path = os.path.abspath(os.path.expanduser(str(blacklist)))
        df = pd.read_csv(path, sep="\t", header=None, comment="#",
                         usecols=[0, 1, 2], dtype={0: str})
    df.columns = ["chrom", "start", "end"]
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)

    out: dict = {}
    for chrom, sub in df.groupby("chrom", sort=False):
        s = sub["start"].to_numpy()
        e = sub["end"].to_numpy()
        order = np.argsort(s)
        s = s[order]
        e = e[order]
        # Merge overlapping intervals in one pass.
        ms, me = [s[0]], [e[0]]
        for i in range(1, len(s)):
            if s[i] <= me[-1]:
                me[-1] = max(me[-1], e[i])
            else:
                ms.append(s[i])
                me.append(e[i])
        out[str(chrom)] = (np.asarray(ms, dtype=np.int64),
                           np.asarray(me, dtype=np.int64))
    return out


def _mask_features_by_blacklist(
    feat_df: pd.DataFrame,
    blacklist_map: dict,
) -> np.ndarray:
    """Return a boolean keep-mask of length ``len(feat_df)``.

    A feature is dropped if its ``[start, end)`` overlaps any merged
    blacklist interval on the same chrom. Overlap = ``feat.start <
    bl.end`` and ``feat.end > bl.start``.
    """
    keep = np.ones(len(feat_df), dtype=bool)
    chroms = feat_df["chrom"].to_numpy()
    starts = feat_df["start"].to_numpy(dtype=np.int64)
    ends = feat_df["end"].to_numpy(dtype=np.int64)
    # Group feature indices by chrom using a single argsort pass.
    order = np.argsort(chroms, kind="stable")
    sorted_chroms = chroms[order]
    # Slice out each contiguous chrom block.
    _, boundaries = np.unique(sorted_chroms, return_index=True)
    splits = np.append(boundaries, len(sorted_chroms))
    for k in range(len(boundaries)):
        chrom = str(sorted_chroms[boundaries[k]])
        bl = blacklist_map.get(chrom)
        if bl is None:
            continue
        bl_s, bl_e = bl
        idx = order[splits[k]:splits[k + 1]]
        fs = starts[idx]
        fe = ends[idx]
        # For each feature, find the rightmost bl whose start < fe;
        # check whether its end > fs -> overlap.
        pos = np.searchsorted(bl_s, fe, side="right") - 1
        valid = pos >= 0
        hit = np.zeros(len(idx), dtype=bool)
        if valid.any():
            hit[valid] = bl_e[pos[valid]] > fs[valid]
        keep[idx[hit]] = False
    return keep


# ---------------------------------------------------------------------------
# Genome-bin tiling
# ---------------------------------------------------------------------------
def make_genome_bins(
    chrom_size: Union[str, pd.DataFrame, dict],
    bin_size: int,
    exclude_chroms: Optional[Sequence[str]] = None,
    name_template: str = "{chrom}:{start}-{end}",
) -> pd.DataFrame:
    """Tile a genome into non-overlapping ``bin_size``-bp windows.

    Parameters
    ----------
    chrom_size
        One of:

        - Path to a chrom-size file (``chrom\\tlength``, e.g. UCSC
          ``.chrom.sizes``) or a samtools ``.fai`` index (first two
          columns are chrom and length).
        - DataFrame with columns ``['chrom', 'length']`` (extra columns
          are ignored; ``.fai`` shape is supported).
        - ``dict`` mapping ``chrom -> length``.
    bin_size
        Window size in bp (e.g. ``5000`` for 5 kb bins, ``100_000`` for
        100 kb bins).
    exclude_chroms
        Chromosomes to skip (e.g. ``['chrL']`` for the lambda spike-in).
    name_template
        Format string for the ``name`` column. Available fields:
        ``chrom``, ``start``, ``end``, ``i`` (per-chrom index).

    Returns
    -------
    DataFrame with columns ``['chrom', 'start', 'end', 'name']``.
    """
    if bin_size is None or int(bin_size) <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size!r}")
    bin_size = int(bin_size)

    if isinstance(chrom_size, dict):
        chroms = list(chrom_size.items())
    elif isinstance(chrom_size, pd.DataFrame):
        df = chrom_size
        chroms = list(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(int)))
    else:
        p = os.path.abspath(os.path.expanduser(str(chrom_size)))
        df = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1],
                         names=["chrom", "length"], dtype={0: str})
        chroms = list(zip(df["chrom"].astype(str), df["length"].astype(int)))

    exclude = set(exclude_chroms or ())
    rows = []
    for chrom, length in chroms:
        if chrom in exclude:
            continue
        length = int(length)
        for i, s in enumerate(range(0, length, bin_size)):
            e = min(s + bin_size, length)
            rows.append((chrom, s, e,
                         name_template.format(chrom=chrom, start=s, end=e, i=i)))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])


# ---------------------------------------------------------------------------
# Scoring (ALLCools conventions)
# ---------------------------------------------------------------------------
def _beta_binomial_alpha_beta(sum_mc, sum_cov, n_keep, X,
                              df_correction=True, eps=1e-6):
    """Finish the Beta-Binomial MoM from per-row aggregate statistics.

    Given per-row sums over the *kept* sites (``sum_mc``, ``sum_cov``, the
    count ``n_keep``) and the Pearson over-dispersion statistic
    ``X = sum (mc_i - cov_i*mu)^2 / (cov_i*mu*(1-mu))``, solve for the Beta
    concentration and return ``(alpha, beta, prior_mean)`` as float32,
    NaN on degenerate rows. See ``docs/beta_binomial_prior.md`` (section 3).
    """
    n = np.asarray(sum_cov).shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(sum_cov > 0, sum_mc / sum_cov, np.nan)  # = alpha/(alpha+beta)
    sum_cov_m1 = sum_cov - n_keep                 # = sum(cov_i - 1) over kept
    n_eff = (n_keep - 1.0) if df_correction else n_keep
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (X - n_eff) / sum_cov_m1            # intra-class correlation
    ok = (np.isfinite(mu) & (mu > 0.0) & (mu < 1.0)
          & (n_keep >= 2) & (sum_cov_m1 > 0.0) & np.isfinite(rho))
    alpha = np.full(n, np.nan, dtype=np.float64)
    beta = np.full(n, np.nan, dtype=np.float64)
    if ok.any():
        r = np.clip(rho[ok], eps, 1.0 - eps)
        kappa = (1.0 - r) / r                     # = alpha + beta
        m = mu[ok]
        alpha[ok] = np.maximum(m * kappa, 1e-6)
        beta[ok] = np.maximum((1.0 - m) * kappa, 1e-6)
    prior_mean = alpha / (alpha + beta)
    return (alpha.astype(np.float32),
            beta.astype(np.float32),
            prior_mean.astype(np.float32))


# ---------------------------------------------------------------------------
# Sparse-streaming helpers (memory-efficient path for hundreds of thousands
# of cells). All scoring / Beta-prior estimation runs directly on the
# streamed CSR mc/cov layers; there is no dense code path.
# ---------------------------------------------------------------------------
class _StreamingSparseBuilder:
    """Append-one-cell-at-a-time CSR builder.

    Avoids materialising the dense ``(n_cells, n_feat)`` matrices that
    otherwise dominate memory for large cohorts. Each cell is appended
    by passing its ``(n_feat, 2)`` int64 array; only the (mc, cov)
    entries on positions where ``cov > 0`` are kept (single-cell
    methylation is typically <5% covered, so this is the bulk of the
    saving).

    Build the final sparse layers with :meth:`finalize`.
    """

    __slots__ = ("n_feat", "_mc_data", "_cov_data", "_indices",
                 "_indptr", "_n_rows")

    def __init__(self, n_feat: int):
        self.n_feat = int(n_feat)
        self._mc_data: list = []
        self._cov_data: list = []
        self._indices: list = []
        # ``indptr`` always starts with 0 and grows by one entry per row.
        self._indptr: list = [0]
        self._n_rows = 0

    def append(self, arr) -> None:
        """Append one cell's ``(n_feat, 2)`` int64 array.

        Only entries with ``cov > 0`` are stored. Since ``mc <= cov``,
        the same sparsity pattern works for both layers.
        """
        cov = arr[:, 1]
        # nonzero is a *fast* C-level select; no Python-loop overhead.
        nz = np.flatnonzero(cov).astype(np.int32, copy=False)
        if nz.size:
            # Cast counts to uint32 — methylation calls fit easily.
            self._mc_data.append(arr[nz, 0].astype(np.uint32, copy=False))
            self._cov_data.append(cov[nz].astype(np.uint32, copy=False))
            self._indices.append(nz)
        self._n_rows += 1
        self._indptr.append(self._indptr[-1] + int(nz.size))

    def finalize(self):
        """Materialise ``(mc_csr, cov_csr)`` shape ``(n_rows, n_feat)``.

        After this call the internal buffers are dropped so the builder
        cannot be reused.
        """
        import scipy.sparse as ss
        n_rows = self._n_rows
        shape = (n_rows, self.n_feat)
        # ``indptr`` may exceed int32 for >2.1B nonzeros total — pick
        # the narrowest dtype that still covers the value range.
        last = self._indptr[-1]
        indptr_dtype = np.int64 if last > np.iinfo(np.int32).max else np.int32
        indptr = np.asarray(self._indptr, dtype=indptr_dtype)
        if self._indices:
            indices = np.concatenate(self._indices)
            mc_data = np.concatenate(self._mc_data)
            cov_data = np.concatenate(self._cov_data)
        else:
            indices = np.empty(0, dtype=np.int32)
            mc_data = np.empty(0, dtype=np.uint32)
            cov_data = np.empty(0, dtype=np.uint32)
        # Drop the chunk lists so concatenation buffers are reclaimed
        # before two more arrays are allocated for the second layer.
        self._mc_data.clear()
        self._cov_data.clear()
        self._indices.clear()
        # Note: ``indices`` is shared between the two CSRs because both
        # layers have the same sparsity pattern.
        mc_csr = ss.csr_matrix((mc_data, indices, indptr), shape=shape)
        cov_csr = ss.csr_matrix((cov_data, indices, indptr), shape=shape)
        # Indicate the layers are already canonical (sorted, no dups).
        mc_csr.has_sorted_indices = True
        cov_csr.has_sorted_indices = True
        mc_csr.has_canonical_format = True
        cov_csr.has_canonical_format = True
        return mc_csr, cov_csr


def _compute_beta_params_sparse(mc_csr, cov_csr, min_cov=2, df_correction=True):
    """Per-cell Beta(alpha, beta) via the coverage-aware Beta-Binomial MoM.

    Operates only on the stored (non-zero-coverage) entries of each row via
    two ``reduceat`` passes (first the per-row mean, then the Pearson
    over-dispersion statistic), never densifying. Assumes ``mc_csr`` and
    ``cov_csr`` share the same sparsity structure (same ``indptr`` /
    ``indices``), as produced upstream.
    """
    n_rows = cov_csr.shape[0]
    indptr = np.asarray(cov_csr.indptr)
    counts = np.diff(indptr).astype(np.int64)     # stored entries per row
    nz_rows = counts > 0

    covd = cov_csr.data.astype(np.float64)
    mcd = mc_csr.data.astype(np.float64)
    w = (covd >= min_cov).astype(np.float64)      # keep-mask as 0/1 weights

    # ---- first pass: per-row sums over kept entries -> mu ----
    sum_cov = np.zeros(n_rows, dtype=np.float64)
    sum_mc = np.zeros(n_rows, dtype=np.float64)
    n_keep = np.zeros(n_rows, dtype=np.float64)
    if nz_rows.any():
        starts = indptr[:-1][nz_rows]
        sum_cov[nz_rows] = np.add.reduceat(covd * w, starts)
        sum_mc[nz_rows] = np.add.reduceat(mcd * w, starts)
        n_keep[nz_rows] = np.add.reduceat(w, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(sum_cov > 0, sum_mc / sum_cov, np.nan)
    mu_c = np.clip(mu, 1e-6, 1.0 - 1e-6)
    denom = mu_c * (1.0 - mu_c)

    # ---- second pass: Pearson over-dispersion statistic X ----
    mu_entry = np.repeat(mu_c, counts)            # broadcast row mu to entries
    den_entry = np.repeat(denom, counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(w > 0,
                        (mcd - covd * mu_entry) ** 2 / (covd * den_entry),
                        0.0)
    X = np.zeros(n_rows, dtype=np.float64)
    if nz_rows.any():
        starts = indptr[:-1][nz_rows]
        X[nz_rows] = np.add.reduceat(term, starts)

    return _beta_binomial_alpha_beta(sum_mc, sum_cov, n_keep, X, df_correction)


def _compute_score_matrix_sparse(mc_csr, cov_csr, score, score_cutoff,
                                 alpha=None, beta=None):
    """Compute the requested score matrix from the streamed CSR mc/cov layers.

    Supported scores: ``'frac'``, ``'posterior_frac'``, ``'hypo-score'``,
    ``'hyper-score'``, ``'mc'``, ``'cov'``, ``'umc'`` (see :func:`cz_to_anndata`).
    Returns a ``csr_matrix`` of shape ``(n_cells, n_features)`` with
    ``float32`` data. Zero-coverage entries are *implicit* zeros — they
    are not materialised, which is the whole point of the streaming
    path. For ``'posterior_frac'`` the per-cell posterior mean is computed
    only on the stored (``cov > 0``) entries, so the sparsity pattern is
    identical to ``'frac'`` (uncovered entries stay implicit ``0``).
    """
    import scipy.sparse as ss
    indptr = np.asarray(cov_csr.indptr)
    indices = np.asarray(cov_csr.indices)
    shape = cov_csr.shape

    if score == "mc":
        return mc_csr.astype(np.float32)
    if score == "cov":
        return cov_csr.astype(np.float32)
    if score == "umc":
        umc = (cov_csr.data.astype(np.int64)
               - mc_csr.data.astype(np.int64))
        np.maximum(umc, 0, out=umc)
        return ss.csr_matrix((umc.astype(np.float32), indices, indptr),
                             shape=shape, dtype=np.float32)
    if score == "frac":
        data = (mc_csr.data.astype(np.float32)
                / np.maximum(cov_csr.data, 1).astype(np.float32))
        return ss.csr_matrix((data, indices, indptr),
                             shape=shape, dtype=np.float32)

    if score == "posterior_frac":
        # per-cell empirical-Bayes posterior mean (mc + a)/(cov + a + b) on
        # the stored (cov>0) entries; broadcast the per-row alpha/beta out to
        # per-nonzero via repeat. Zero-cov entries remain implicit 0.
        if alpha is None or beta is None:
            alpha, beta, _ = _compute_beta_params_sparse(mc_csr, cov_csr)
        counts = np.diff(indptr).astype(np.int64)
        a_nz = np.repeat(alpha.astype(np.float64), counts)
        b_nz = np.repeat(beta.astype(np.float64), counts)
        mcd = mc_csr.data.astype(np.float64)
        covd = cov_csr.data.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            post = (mcd + a_nz) / (covd + a_nz + b_nz)
            raw = mcd / np.maximum(covd, 1.0)
        # degenerate cells (NaN alpha/beta) fall back to the raw fraction
        post = np.where(np.isfinite(post), post, raw).astype(np.float32)
        return ss.csr_matrix((post, indices, indptr),
                             shape=shape, dtype=np.float32)

    if score in ("hypo-score", "hyper-score"):
        from scipy.stats import binom
        n_rows = shape[0]
        counts = np.diff(indptr).astype(np.int64)
        nz_rows = counts > 0
        tot_mc = np.zeros(n_rows, dtype=np.float64)
        tot_cov = np.zeros(n_rows, dtype=np.float64)
        if nz_rows.any():
            starts = indptr[:-1][nz_rows]
            tot_mc[nz_rows] = np.add.reduceat(
                mc_csr.data.astype(np.float64), starts)
            tot_cov[nz_rows] = np.add.reduceat(
                cov_csr.data.astype(np.float64), starts)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cell = tot_mc / (tot_cov + 1e-6)
        valid = (np.isfinite(p_cell) & (p_cell > 0) & (p_cell < 1)
                 & (tot_cov > 0))
        # Broadcast per-row stats out to per-nz arrays via repeat.
        p_per_nz = np.repeat(p_cell, counts)
        valid_per_nz = np.repeat(valid, counts)
        sf = binom.sf(mc_csr.data, cov_csr.data, p_per_nz)
        pv = (1.0 - sf) if score == "hyper-score" else sf
        pv = np.where(valid_per_nz & (pv >= float(score_cutoff)),
                      pv, 0.0).astype(np.float32)
        return ss.csr_matrix((pv, indices, indptr),
                             shape=shape, dtype=np.float32)

    raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")


def _compute_hvf_var_stats_sparse(mc_csr, cov_csr, alpha=None, beta=None,
                                  method="posterior"):
    """Per-feature additive HVF accumulators via column reductions.

    For each feature (column ``j``), over the cells covering it (``cov > 0``),
    computes three additive statistics of the methylation fraction ``f``:

    * ``hvf_n_cov[j]``   = number of covering cells,
    * ``hvf_sum[j]``     = ``sum_i f_ij``,
    * ``hvf_sum_sq[j]``  = ``sum_i f_ij^2``.

    With ``method='posterior'`` (default) ``f`` is the per-cell empirical-Bayes
    posterior mean ``(mc + a)/(cov + a + b)`` (falling back to raw ``mc/cov``
    for degenerate cells whose ``alpha``/``beta`` are NaN); ``method='raw'``
    uses ``mc/cov`` directly. All stored entries have ``cov > 0`` (the builder
    drops zero-coverage), so the column non-zero count equals the covered-cell
    count.

    These three vectors are additive across cells (and across merged
    datasets), so the per-feature mean, variance, dispersion and mean-binned
    normalized dispersion can be reconstructed later without re-reading the
    matrix (e.g. by ``pym3c`` ``MultiAdata.select_hvf``):

    .. code-block:: text

        mean = hvf_sum / hvf_n_cov
        var  = hvf_sum_sq / hvf_n_cov - mean**2
        dispersion = var / mean
        normalized_dispersion = z-score of dispersion within mean bins

    Returns three ``(n_feat,)`` float64 arrays
    ``(hvf_n_cov, hvf_sum, hvf_sum_sq)``.
    """
    n_feat = cov_csr.shape[1]
    indptr = np.asarray(cov_csr.indptr)
    indices = np.asarray(cov_csr.indices)
    covd = cov_csr.data.astype(np.float64)
    mcd = mc_csr.data.astype(np.float64)
    # all stored entries are covered (cov > 0), so raw is always finite
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = mcd / np.maximum(covd, 1.0)
    if method == "posterior" and alpha is not None and beta is not None:
        counts = np.diff(indptr).astype(np.int64)
        a_nz = np.repeat(alpha.astype(np.float64), counts)
        b_nz = np.repeat(beta.astype(np.float64), counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (mcd + a_nz) / (covd + a_nz + b_nz)
        frac = np.where(np.isfinite(frac), frac, raw)
    else:
        frac = raw
    hvf_n_cov = np.bincount(indices, minlength=n_feat).astype(np.float64)
    hvf_sum = np.bincount(indices, weights=frac, minlength=n_feat)
    hvf_sum_sq = np.bincount(indices, weights=frac * frac, minlength=n_feat)
    return hvf_n_cov, hvf_sum, hvf_sum_sq


# ---------------------------------------------------------------------------
# Per-reader aggregation (numpy fast path)
# ---------------------------------------------------------------------------
def _detect_chrom_axis(reader: Reader, known_chroms: set) -> int:
    """Find which position of the dim tuple holds the chrom.

    Different producers stamp the chrom in different slots:

    * Single-cell (``chunk_dims=['chrom']``): position 0.
    * ``catcz(key_added=...)``: appends the new key at the *end* of the
      source dim tuple, so chrom keeps its original position (typically 0)
      and the added cell id lives at the last slot.
    * User-built merged files may declare any order.

    We probe by scanning the first handful of chunk dims and picking the
    index whose values match the most features-by-chrom keys.
    """
    samples = list(reader.chunk_key2offset.keys())[:32]
    if not samples:
        return -1
    n = len(samples[0])
    best_idx = -1
    best_hits = -1
    for i in range(n):
        hits = sum(1 for dim in samples if dim[i] in known_chroms)
        if hits > best_hits:
            best_hits = hits
            best_idx = i
    return best_idx


def _aggregate_one_reader(
    reader: Reader,
    features_by_chrom: dict,
    pos_col: Optional[int],
    mc_col: int,
    cov_col: int,
    cell_prefix: Optional[tuple] = None,
    chrom_axis: Optional[int] = None,
    ref_pos_map: Optional[dict] = None,
    index_ids_by_chrom: Optional[dict] = None,
) -> np.ndarray:
    """Return shape ``(n_features, 2)`` int64 array ``[sum_mc, sum_cov]``.

    Fast path: per-chrom ``fetch_chunk_bytes`` + ``np.frombuffer`` decode,
    then ``np.searchsorted`` on the sorted position array to sum each
    region via cumulative-sum subtraction.

    Parameters
    ----------
    pos_col : int or None
        0-based index of the ``pos`` column in the record layout. ``None``
        for ``[mc, cov]``-only cells that rely on ``ref_pos_map``.
    cell_prefix : tuple, optional
        Non-chrom dim values (in their positional order) identifying the
        cell in a merged reader. For a single-cell reader, pass ``None``.
    chrom_axis : int, optional
        Index of the chrom slot in the reader's dim tuple (see
        :func:`_detect_chrom_axis`). Defaults to the *last* position.
    ref_pos_map : dict, optional
        ``{chrom: int64 positions}`` for reference-aligned cells.
    index_ids_by_chrom : dict, optional
        ``{chrom: 1-based int64 IDs}`` for an optional context filter
        (CG / CH / ...). When given, the cell array (and reference
        positions, if any) are gathered to only the index-selected rows
        before per-feature summation. The IDs apply equally to row-aligned
        cell + reference data, so positions and counts stay aligned.
    """
    n_feat = sum(
        v["n_bins"] if v.get("tiled") else len(v["indices"])
        for v in features_by_chrom.values()
    )
    out = np.zeros((n_feat, 2), dtype=np.int64)
    chunk_dims = reader.header["chunk_dims"]
    n_keys = len(chunk_dims)

    if chrom_axis is None:
        chrom_axis = n_keys - 1

    record_dtype = _record_dtype_for(reader.header["formats"])

    for chrom, info in features_by_chrom.items():
        # Build the full dim tuple.
        if n_keys == 1:
            dim = (chrom,)
        else:
            # Splice chrom into the chrom_axis slot; fill the remaining
            # slots from cell_prefix (ordered by their original position).
            if cell_prefix is None:
                continue
            dim_list = list(cell_prefix)
            # cell_prefix lists the non-chrom values in their positional
            # order; insert chrom at its slot.
            dim_list.insert(chrom_axis, chrom)
            dim = tuple(dim_list)
        if dim not in reader.chunk_key2offset:
            continue

        raw = reader.fetch_chunk_bytes(dim)
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=record_dtype)
        if arr.size == 0:
            continue

        mc_vals = arr[f"c{mc_col}"].astype(np.int64, copy=False)
        cov_vals = arr[f"c{cov_col}"].astype(np.int64, copy=False)

        if pos_col is not None:
            positions = arr[f"c{pos_col}"].astype(np.int64, copy=False)
        elif ref_pos_map is not None and chrom in ref_pos_map:
            # ref_pos_map may be a plain dict (legacy) or a
            # _LazyRefPositions (current bam.py); both expose ``.get`` and
            # ``__contains__``, but only the dict supports ``[]``.
            getter = getattr(ref_pos_map, "get", None)
            positions = (getter(chrom) if callable(getter)
                         else ref_pos_map[chrom])
            if positions is None or positions.size != arr.size:
                continue
            positions = np.asarray(positions, dtype=np.int64)
        else:
            continue

        # Optional 1-D context-index filter (e.g. CG / CH only).
        # Gather positions, mc and cov together so they stay aligned.
        if index_ids_by_chrom is not None:
            ids = index_ids_by_chrom.get(chrom)
            if ids is None or ids.size == 0:
                continue
            n_rows = positions.size
            if ids.min() < 1 or ids.max() > n_rows:
                # Be lenient: clip OOB IDs (shorter cell than ref) instead
                # of erroring, since per-cell .cz may have fewer rows.
                ids = ids[(ids >= 1) & (ids <= n_rows)]
                if ids.size == 0:
                    continue
            gather = ids - 1
            positions = positions[gather]
            mc_vals = mc_vals[gather]
            cov_vals = cov_vals[gather]

        # Cumulative sums -> O(1) range sums via subtraction.
        # Fast path: if features for this chrom are equal-width
        # non-overlapping tiles (from make_genome_bins), use bincount on
        # the bin index; ~2-3x faster than cumsum+searchsorted and more
        # cache-friendly for large feature counts.
        if info.get("tiled", False):
            bs = int(info["bin_size"])
            n_bins_chrom = int(info["n_bins"])
            first = int(info["first_index"])
            # .allc/.cz positions are 1-based; BED tiles are 0-based
            # half-open. pos=1 belongs to tile 0 under bin [0, bs).
            idx = ((positions.astype(np.int64) - 1) // bs)
            # Clip so out-of-bounds positions (e.g. reference pos beyond
            # the last short tile) fold into the last bin harmlessly.
            np.clip(idx, 0, n_bins_chrom - 1, out=idx)
            mc_sum = np.bincount(idx, weights=mc_vals,
                                 minlength=n_bins_chrom)[:n_bins_chrom]
            cov_sum = np.bincount(idx, weights=cov_vals,
                                  minlength=n_bins_chrom)[:n_bins_chrom]
            out[first:first + n_bins_chrom, 0] = mc_sum.astype(np.int64)
            out[first:first + n_bins_chrom, 1] = cov_sum.astype(np.int64)
            continue

        mc_cum = np.concatenate(([0], np.cumsum(mc_vals)))
        cov_cum = np.concatenate(([0], np.cumsum(cov_vals)))

        starts = np.asarray(info["starts"], dtype=np.int64)
        ends = np.asarray(info["ends"], dtype=np.int64)
        # BED is half-open [start, end); pos is 1-based. Include pos where
        # start < pos <= end.
        lo = np.searchsorted(positions, starts, side="right")
        hi = np.searchsorted(positions, ends, side="right")
        region_mc = mc_cum[hi] - mc_cum[lo]
        region_cov = cov_cum[hi] - cov_cum[lo]

        feat_idx = np.asarray(info["indices"], dtype=np.int64)
        out[feat_idx, 0] = region_mc
        out[feat_idx, 1] = region_cov
    return out


# ---------------------------------------------------------------------------
# Parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------
_WORKER_STATE: dict = {}


def _pool_init(features_by_chrom, pos_col, mc_col, cov_col, reference,
               index_ids_by_chrom=None):
    """Per-process initializer: stash shared read-only state.

    Each worker loads the reference position map lazily (only if needed)
    and then reuses it across every cell it receives.
    """
    _WORKER_STATE.clear()
    _WORKER_STATE["features_by_chrom"] = features_by_chrom
    _WORKER_STATE["pos_col"] = pos_col
    _WORKER_STATE["mc_col"] = mc_col
    _WORKER_STATE["cov_col"] = cov_col
    _WORKER_STATE["reference"] = reference
    _WORKER_STATE["ref_pos_map"] = None
    _WORKER_STATE["chroms_set"] = set(features_by_chrom.keys())
    _WORKER_STATE["index_ids_by_chrom"] = index_ids_by_chrom


def _pool_get_ref_pos_map(hint_path=None):
    if _WORKER_STATE["ref_pos_map"] is not None:
        return _WORKER_STATE["ref_pos_map"]
    ref_path = _WORKER_STATE["reference"] or hint_path
    if ref_path is None:
        raise ValueError(
            "Cell .cz has no 'pos' column; pass reference= to "
            "provide the coordinate reference.")
    from .bam import _LazyRefPositions
    _WORKER_STATE["ref_pos_map"] = _LazyRefPositions(ref_path)
    return _WORKER_STATE["ref_pos_map"]


def _pool_process_file(cz_path):
    """Worker entry: aggregate a whole per-cell ``.cz`` file."""
    fbc = _WORKER_STATE["features_by_chrom"]
    # Whole-file per-cell aggregation (all chroms once) -> sequential.
    r = Reader(cz_path, mmap_advise="sequential")
    try:
        cols = r.header["columns"]
        pi = cols.index(_WORKER_STATE["pos_col"]) \
            if _WORKER_STATE["pos_col"] in cols else None
        mc_i = cols.index(_WORKER_STATE["mc_col"])
        cov_i = cols.index(_WORKER_STATE["cov_col"])
        rpm = _pool_get_ref_pos_map(r.header.get("message")) \
            if pi is None else None
        chrom_axis = _detect_chrom_axis(r, _WORKER_STATE["chroms_set"])
        arr = _aggregate_one_reader(
            r, fbc, pi, mc_i, cov_i,
            chrom_axis=chrom_axis, ref_pos_map=rpm,
            index_ids_by_chrom=_WORKER_STATE.get("index_ids_by_chrom"))
    finally:
        r.close()
    return arr


def _pool_process_prefix(args):
    """Worker entry: aggregate one cell prefix inside a merged ``.cz``.

    ``args = (cz_path, prefix_tuple)``. The Reader is opened once per
    worker process and cached; cell prefixes streamed in reuse it.
    """
    cz_path, prefix = args
    fbc = _WORKER_STATE["features_by_chrom"]
    # Cache the merged Reader per worker.
    r = _WORKER_STATE.get("_merged_reader")
    if r is None or _WORKER_STATE.get("_merged_path") != cz_path:
        if r is not None:
            r.close()
        r = Reader(cz_path, mmap_advise="sequential")
        _WORKER_STATE["_merged_reader"] = r
        _WORKER_STATE["_merged_path"] = cz_path
        cols = r.header["columns"]
        _WORKER_STATE["_pi"] = (cols.index(_WORKER_STATE["pos_col"])
                                if _WORKER_STATE["pos_col"] in cols else None)
        _WORKER_STATE["_mc_i"] = cols.index(_WORKER_STATE["mc_col"])
        _WORKER_STATE["_cov_i"] = cols.index(_WORKER_STATE["cov_col"])
        _WORKER_STATE["_chrom_axis"] = _detect_chrom_axis(
            r, _WORKER_STATE["chroms_set"])
    pi = _WORKER_STATE["_pi"]
    rpm = _pool_get_ref_pos_map(r.header.get("message")) if pi is None else None
    arr = _aggregate_one_reader(
        r, fbc, pi, _WORKER_STATE["_mc_i"], _WORKER_STATE["_cov_i"],
        cell_prefix=prefix, chrom_axis=_WORKER_STATE["_chrom_axis"],
        ref_pos_map=rpm,
        index_ids_by_chrom=_WORKER_STATE.get("index_ids_by_chrom"))
    return arr


# ---------------------------------------------------------------------------
# Shared feature-prep / anndata-assembly helpers (used by cz_to_anndata and
# cz_to_anndata_multiref)
# ---------------------------------------------------------------------------
def _prepare_feature_groups(features, chrom_size, exclude_chroms, blacklist,
                            flank_bp, gtf_id_col):
    """Parse ``features`` into ``(feat_df, features_by_chrom, n_feat,
    gtf_meta_df)``.

    ``features_by_chrom`` carries either per-region ``starts/ends/indices``
    or, for genome tiling, ``tiled/bin_size/first_index/n_bins`` metadata.
    Shared by :func:`cz_to_anndata` and :func:`cz_to_anndata_multiref`.
    """
    # Resolve features -> DataFrame.
    tiled_bin_size: Optional[int] = None
    # Metadata columns (gene_id, gene_name, gene_type, strand) extracted
    # from the GTF for var_df construction. None for non-GTF inputs.
    gtf_meta_df: Optional[pd.DataFrame] = None
    if isinstance(features, str) and _looks_like_gtf(features):
        gene_df = parse_gtf(features, flank_bp=flank_bp,
                            id_col=gtf_id_col,
                            exclude_chroms=exclude_chroms)
        feat_df = gene_df[["chrom", "start", "end", "name"]].copy()
        gtf_meta_df = gene_df.set_index("name")[
            ["gene_id", "gene_name", "gene_type", "strand"]]
    elif isinstance(features, (int, np.integer)):
        if chrom_size is None:
            raise ValueError(
                "features=<int bin_size> requires chrom_size= (path to a "
                "chrom-size or .fai file, DataFrame, or dict).")
        tiled_bin_size = int(features)
        feat_df = make_genome_bins(chrom_size, tiled_bin_size,
                                   exclude_chroms=exclude_chroms)
    else:
        feat_df = parse_features(features)
    n_feat = len(feat_df)

    # Optionally exclude blacklisted regions (ENCODE-style BED). Applied
    # *before* chrom-grouping so both the aggregation and downstream
    # scoring see the pruned feature set.
    if blacklist is not None:
        bl_map = load_blacklist(blacklist)
        keep_mask = _mask_features_by_blacklist(feat_df, bl_map)
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            logger.info(
                f"[cytozip] blacklist: dropped {n_dropped}/{len(feat_df)} "
                f"features overlapping blacklist"
            )
            feat_df = feat_df.loc[keep_mask].reset_index(drop=True)
            n_feat = len(feat_df)
            # When tiled, blacklist breaks contiguity; force the slow
            # (but correct) cumsum+searchsorted path by clearing the tag.
            if isinstance(features, (int, np.integer)):
                tiled_bin_size = None

    # Pre-group by chrom, keeping original feature order via `indices`.
    # When tiled_bin_size is set (features produced by make_genome_bins),
    # we also flag each chrom with tiling metadata so the aggregation
    # fast path (np.bincount) can be used.
    features_by_chrom: dict = {}
    if tiled_bin_size is not None:
        # Features from make_genome_bins come out sorted per chrom and
        # contiguous; recover (first_index, n_bins) in one pass.
        chroms = feat_df["chrom"].to_numpy()
        # First-occurrence index per chrom.
        _, first_idx, counts = np.unique(chroms, return_index=True,
                                         return_counts=True)
        order = np.argsort(first_idx)
        for k in order:
            c = str(chroms[first_idx[k]])
            features_by_chrom[c] = {
                "tiled": True,
                "bin_size": tiled_bin_size,
                "first_index": int(first_idx[k]),
                "n_bins": int(counts[k]),
            }
    else:
        # Vectorized equivalent of the per-row iterrows loop: group by chrom
        # in first-occurrence order, preserving each feature's original index
        # label (what iterrows yielded as ``i``).
        _chroms = feat_df["chrom"].to_numpy()
        _starts = feat_df["start"].to_numpy()
        _ends = feat_df["end"].to_numpy()
        _index_labels = feat_df.index.to_numpy()
        for c in pd.unique(_chroms):
            sel = np.nonzero(_chroms == c)[0]
            features_by_chrom[c] = {
                "starts": _starts[sel].astype(int).tolist(),
                "ends": _ends[sel].astype(int).tolist(),
                "indices": _index_labels[sel].tolist(),
            }
    return feat_df, features_by_chrom, n_feat, gtf_meta_df


def _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                         score, score_cutoff, hvf_frac, obs, output,
                         reference=None, nan_policy="auto"):
    """Assemble an :class:`anndata.AnnData` from a filled streaming builder.

    Shared tail of :func:`cz_to_anndata` and :func:`cz_to_anndata_multiref`:
    materialises the sparse mc/cov layers, computes the requested ``.X``
    score (plus the per-cell Beta prior / per-feature HVF accumulators for
    ``posterior_frac``), and writes the ``.h5ad`` when ``output`` is given.
    ``reference`` (a path or list of paths) is recorded in
    ``adata.uns['cytozip_reference']``.

    ``nan_policy`` controls how uncovered (``cov == 0``) entries appear in
    ``.X`` (see :func:`_apply_nan_policy`).
    """
    import anndata

    # Materialise sparse layers (no dense intermediate). mc_sp / cov_sp are the
    # (n_cells, n_features) CSR uint32 matrices of methylated counts and total
    # coverage; only covered (cov > 0) entries are stored, so their shared
    # non-zero structure is the covered mask.
    mc_sp, cov_sp = builder.finalize()

    # Build var_df. For GTF inputs we attach gene_id / gene_type /
    # strand alongside the coordinates; for BED / bins var carries just
    # the coordinates. `name` is always used verbatim as var_names.
    var_df = feat_df.set_index("name")
    if gtf_meta_df is not None:
        var_df = var_df.join(gtf_meta_df)

    # The per-cell Beta prior (alpha, beta) and the per-feature HVF
    # accumulators are only needed for the posterior_frac score, so they are
    # computed *only* in that branch to avoid wasted work on the other scores.
    # alpha/beta are estimated once here and shared by the score transform, the
    # HVF accumulators and adata.obs (no double estimation).
    obs_df = pd.DataFrame(index=obs_names)
    prior_mean = None
    if score == "posterior_frac":
        alpha, beta, prior_mean = _compute_beta_params_sparse(mc_sp, cov_sp)
        X = _compute_score_matrix_sparse(mc_sp, cov_sp, score,
                                         score_cutoff=score_cutoff,
                                         alpha=alpha, beta=beta)
        # Per-feature additive HVF accumulators -> adata.var. Additive across
        # cells (and merged datasets), so mean / var / dispersion / normalized
        # dispersion can be reconstructed downstream (e.g. pym3c
        # MultiAdata.select_hvf) without re-reading the matrix.
        hvf_n_cov, hvf_sum, hvf_sum_sq = _compute_hvf_var_stats_sparse(
            mc_sp, cov_sp, alpha=alpha, beta=beta, method=hvf_frac)
        var_df["hvf_n_cov"] = hvf_n_cov
        var_df["hvf_sum"] = hvf_sum
        var_df["hvf_sum_sq"] = hvf_sum_sq
        # per-cell Beta prior + prior_mean + rho for downstream posterior use.
        obs_df["alpha"] = alpha
        obs_df["beta"] = beta
        obs_df["prior_mean"] = prior_mean
        # rho = 1 / (alpha + beta + 1)  == 1/(kappa+1): the intra-class
        # correlation (over-dispersion) of the per-cell Beta-Binomial fit. It
        # is coverage-independent (unlike a raw variance), so it doubles as a
        # per-cell QC handle (flag degenerate/low-complexity cells) and lets
        # downstream code recover the shrinkage strength kappa = (1 - rho)/rho.
        # NaN wherever alpha/beta are NaN (degenerate rows).
        obs_df["rho"] = (1.0 / (alpha.astype(np.float64)
                                + beta.astype(np.float64) + 1.0)).astype(np.float32)
    else:
        X = _compute_score_matrix_sparse(mc_sp, cov_sp, score,
                                         score_cutoff=score_cutoff)
    # Resolve nan_policy (incl. 'auto') to an effective per-score policy, then
    # apply it to uncovered (cov==0) entries of .X. 'zero' keeps the sparse
    # implicit 0; 'nan' / 'prior_mean' densify .X to float32.
    eff_policy = nan_policy
    if nan_policy == "auto":
        if score == "posterior_frac":
            eff_policy = "prior_mean"
        elif score in ("frac", "umc"):
            eff_policy = "nan"
        else:  # hypo-score / hyper-score have no missing entries; stay sparse
            eff_policy = "zero"
    if eff_policy != "zero":
        if eff_policy in ("nan", "prior_mean"):
            logger.warning(
                f"nan_policy={eff_policy!r} densifies .X to float32 "
                f"(n_cells * n_features * 4 bytes, peak ~1.5-2x during build "
                f"and h5ad write) and can OOM for large cohorts / genome-wide "
                f"bins. If you hit an out-of-memory error, switch to "
                f"nan_policy='zero', which keeps .X sparse (implicit 0). For "
                f"sparse single-cell data 'zero' is recommended: it saves "
                f"both RAM and on-disk space, and you can restore NaN on "
                f"demand / block-wise from layers['cov'] via "
                f"cytozip.features.to_dense_nan(adata).")
        if eff_policy == "prior_mean" and prior_mean is None:
            _, _, prior_mean = _compute_beta_params_sparse(mc_sp, cov_sp)
        X = _apply_nan_policy(X, cov_sp, eff_policy, prior_mean=prior_mean)
    if obs is not None:
        obs_df = obs_df.join(obs, how="left")

    adata = anndata.AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
        layers={"mc": mc_sp, "cov": cov_sp},
    )
    adata.uns["cytozip_score"] = {
        "score": score,
        "score_cutoff": float(score_cutoff),
        "hvf_frac": hvf_frac,
    }
    # Uncovered (cov==0) features carry no measurement. How they appear in .X
    # depends on nan_policy (resolved to eff_policy); the raw layers['mc'] /
    # ['cov'] always keep the sparse implicit 0, so the covered mask is exactly
    # the non-zero structure of layers['cov'] (covered iff cov>0). Restore NaN
    # on demand with :func:`covered_mask` / :func:`to_dense_nan`.
    adata.uns["cytozip_coverage"] = {
        "nan_policy": eff_policy,
        "nan_policy_requested": nan_policy,
        "covered_mask": "layers['cov'] != 0",
        "note": ("Uncovered (cov==0) entries in layers are implicit 0 (not "
                 "real zeros). In .X they follow nan_policy. For posterior_frac "
                 "the uncovered posterior equals obs['prior_mean']. Covered "
                 "mask == nonzero structure of layers['cov']. Restore NaN with "
                 "cytozip.features.to_dense_nan(adata)."),
    }
    if reference is not None:
        adata.uns["cytozip_reference"] = reference
    if output:
        adata.write_h5ad(os.path.abspath(os.path.expanduser(output)),
                         compression='lzf')
    return adata


# ---------------------------------------------------------------------------
# Coverage-mask helpers (restore the covered / uncovered NaN distinction the
# sparse .X collapses to 0). See uns['cytozip_coverage'].
# ---------------------------------------------------------------------------
def _uncovered_mask(cov_sp, shape):
    """Dense boolean ``cov == 0`` mask built from the cov sparsity structure.

    Avoids densifying ``cov`` to uint32 (4 B/entry) just to test ``== 0``: the
    mask is a bool array (1 B/entry) seeded ``True`` and cleared at the stored
    (covered) coordinates, which are only ~5% of entries for single-cell data.
    """
    import scipy.sparse as ss
    if ss.issparse(cov_sp):
        uncovered = np.ones(shape, dtype=bool)
        cc = cov_sp.tocoo()
        uncovered[cc.row, cc.col] = False
        return uncovered
    return np.asarray(cov_sp) == 0


def _apply_nan_policy(X, cov_sp, policy, prior_mean=None):
    """Densify ``X`` to ``float32`` filling uncovered (``cov == 0``) per policy.

    ``'nan'`` fills ``np.nan``; ``'prior_mean'`` fills each cell's Beta prior
    mean (per-row broadcast, NaN fallback for degenerate cells). ``'zero'`` is
    handled by the caller (keeps ``X`` sparse) and never reaches here.
    """
    import scipy.sparse as ss
    # X is the freshly-built sparse score matrix here, so toarray() already
    # yields an owned float32 array — avoid a second redundant copy.
    if ss.issparse(X):
        dense = X.toarray()
        if dense.dtype != np.float32:
            dense = dense.astype(np.float32, copy=False)
    else:
        dense = np.array(X, dtype=np.float32)
    uncovered = _uncovered_mask(cov_sp, dense.shape)
    if policy == "nan":
        dense[uncovered] = np.nan
    elif policy == "prior_mean":
        if prior_mean is None:
            raise ValueError(
                "nan_policy='prior_mean' requires per-cell prior_mean")
        fill = np.broadcast_to(
            np.asarray(prior_mean, dtype=np.float32).reshape(-1, 1),
            dense.shape)
        dense[uncovered] = fill[uncovered]
    else:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {policy!r}")
    return dense


def covered_mask(adata):
    """Sparse boolean mask that is ``True`` exactly where a feature is covered.

    An entry is *covered* iff ``cov > 0``. Because :func:`cz_to_anndata`
    stores only ``cov > 0`` entries, the covered mask equals the stored
    (non-zero) structure of ``adata.layers['cov']``. Uncovered entries are
    the implicit zeros of ``.X`` / every layer and represent **missing data**,
    not measured zeros.

    Returns
    -------
    scipy.sparse.csr_matrix (bool)
        Same shape as ``adata.X``; explicit ``True`` at covered entries only.
    """
    import scipy.sparse as ss
    cov = adata.layers["cov"]
    if ss.issparse(cov):
        m = cov.tocsr(copy=True)
        m.eliminate_zeros()
        m.data = np.ones_like(m.data, dtype=bool)
        return m
    return ss.csr_matrix(np.asarray(cov) != 0)


def to_dense_nan(adata, layer=None):
    """Dense ``float32`` view of ``.X`` (or a layer) with uncovered = NaN.

    Thin wrapper over :func:`to_dense` with ``fill='nan'``. Warning: allocates
    ``n_cells * n_features * 4`` bytes.

    Parameters
    ----------
    layer : str or None
        Which matrix to densify. ``None`` (default) uses ``adata.X``;
        otherwise ``adata.layers[layer]`` (e.g. ``'mc'`` / ``'cov'``).
    """
    return to_dense(adata, fill="nan", layer=layer)


def to_dense(adata, fill="prior_mean", layer=None):
    """Dense ``float32`` view of ``.X`` (or a layer) with uncovered entries filled.

    The memory-cheap counterpart to storing a dense ``.X``: keep ``.X`` sparse
    (uncovered = implicit 0) and only materialise a filled dense matrix when a
    downstream step actually needs it. Warning: allocates
    ``n_cells * n_features * 4`` bytes.

    Parameters
    ----------
    fill : {'prior_mean', 'nan', 'zero'}, default ``'prior_mean'``
        How uncovered (``cov == 0``) entries are filled. ``'prior_mean'`` uses
        each cell's Beta prior mean from ``adata.obs['prior_mean']`` — the exact
        ``posterior_frac`` value with no data (requires that column, i.e. a
        ``score='posterior_frac'`` build). ``'nan'`` marks them missing;
        ``'zero'`` leaves the implicit 0.
    layer : str or None
        Which matrix to densify. ``None`` (default) uses ``adata.X``;
        otherwise ``adata.layers[layer]`` (e.g. ``'mc'`` / ``'cov'``).
    """
    import scipy.sparse as ss
    src = adata.X if layer is None else adata.layers[layer]
    if ss.issparse(src):
        dense = src.toarray()
        dense = dense.astype(np.float32, copy=(dense.dtype != np.float32))
    else:
        dense = np.array(src, dtype=np.float32)
    if fill == "zero":
        return dense
    uncovered = _uncovered_mask(adata.layers["cov"], dense.shape)
    if fill == "nan":
        dense[uncovered] = np.nan
    elif fill == "prior_mean":
        if "prior_mean" not in adata.obs:
            raise ValueError(
                "fill='prior_mean' requires adata.obs['prior_mean'] "
                "(only present for score='posterior_frac' builds)")
        pm = np.asarray(adata.obs["prior_mean"].to_numpy(), dtype=np.float32)
        pm_col = np.broadcast_to(pm.reshape(-1, 1), dense.shape)
        dense[uncovered] = pm_col[uncovered]
    else:
        raise ValueError(
            f"fill must be 'prior_mean', 'nan', or 'zero'; got {fill!r}")
    return dense


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def cz_to_anndata(
    cz_inputs: Union[str, Sequence[str]],
    features: Union[str, pd.DataFrame, int],
    output: Optional[str] = None,
    use_samples: Optional[Sequence[str]] = None,
    ext: str = ".cz",
    pos_col: str = "pos",
    mc_col: str = "mc",
    cov_col: str = "cov",
    obs: Optional[pd.DataFrame] = None,
    reference: Optional[str] = None,
    index: Optional[Union[str, "Reader", dict]] = None,
    chrom_size: Optional[Union[str, pd.DataFrame, dict]] = None,
    exclude_chroms: Optional[Sequence[str]] = ("chrL",),
    blacklist: Optional[Union[str, pd.DataFrame]] = None,
    flank_bp: int = 2000,
    gtf_id_col: str = "gene_name",
    score: str = "posterior_frac",
    score_cutoff: float = 0.9,
    hvf_frac: str = "posterior",
    nan_policy: str = "auto",
    jobs: int = 1,
):
    """Build an :class:`anndata.AnnData` of shape ``(n_cells, n_features)``.

    Parameters
    ----------
    cz_inputs : str or list[str]
        Either:

        - A list of single-cell ``.cz`` paths.
        - A directory path (all ``*.cz`` inside are used).
        - A single merged ``.cz`` (from ``catcz``); the first chunk-key is
          treated as the cell id.
    features : str, DataFrame, or int
        - str: BED / BED.gz / BED.bgz path, **or** a GTF / GTF.gz path.
          When a GTF is passed, gene-level features are extracted
          automatically (one row per ``gene`` record), the interval is
          expanded by ``flank_bp`` on each side, and ``gene_name``
          becomes the unique ``var_names`` of the returned AnnData
          (switch to Ensembl IDs via ``gtf_id_col='gene_id'``).
          Detection is by extension (``.gtf`` / ``.gtf.gz`` /
          ``.gtf.bgz``).
        - BED path: the 4th column is used verbatim as each feature's
          ``var_name`` (whether it holds a gene symbol, Ensembl ID, or
          coordinate string). Pre-build the BED with the names you want.
        - DataFrame: first three columns must be chrom/start/end.
        - int: bin size in bp (e.g. ``5000`` for 5kb, ``100_000`` for
          100kb genome-wide bins). Requires ``chrom_size=``.

        Parsed via :func:`parse_features` / :func:`make_genome_bins` /
        :func:`parse_gtf`.
    output : str, optional
        If given, write the AnnData to this ``.h5ad`` path.
    use_samples : list of str, optional
        Whitelist of sample names to include. ``None`` (default) merges
        ALL samples found in ``cz_inputs``. Sample names are matched
        against:

        * For per-file inputs: each file's basename with ``ext`` stripped.
        * For a single pre-catcz'd ``.cz``: the non-chrom chunk-key value
          (or ``/``-joined values when there are >1 non-chrom dims).

        Use this to subset a large directory or merged file without
        having to pre-filter the inputs yourself.
    ext : str, default ``'.cz'``
        Suffix stripped from per-file basenames to derive each sample
        name. The sample name is what gets compared to ``use_samples``
        and what becomes the row label in ``adata.obs``. For merged
        inputs the chunk-key is used as-is and ``ext`` has no effect.
    pos_col, mc_col, cov_col : str
        Column names (in the ``.cz`` ``header['columns']``) to use for
        position / methylated count / coverage.
    obs : DataFrame, optional
        Per-cell metadata (cluster, sample, donor, ...). Joined on cell id.
    reference : str, optional
        Reference ``.cz`` supplying positions for cells written with
        ``bam_to_cz(mode='mc_cov')`` / ``allc2cz(reference=...)``.
    index : str, ``Reader``, dict or None, optional
        1-D context index built by :func:`cytozip.index_context`
        (e.g. ``mm10.allc.cz.CGN.index`` for CG-only or ``.CHN.index``
        for CH-only). When supplied, every per-cell aggregation is
        restricted to the index-selected positions. The index file is
        read **once** for the whole job (even with ``jobs>1`` it is
        materialised to ``{chrom: int64 IDs}`` in the parent and shipped
        to workers via the pool initializer), so the per-chrom cost is
        a single zero-copy gather on top of the existing decode pipeline.
    chrom_size : str or DataFrame or dict, optional
        Required when ``features`` is an int: chrom-size / ``.fai`` for
        genome tiling.
    exclude_chroms : list, optional
        Chromosomes to drop from genome-bin tiling (default ``['chrL']``).
        Ignored when ``features`` is a BED / DataFrame.
    blacklist : str or DataFrame, optional
        BED / bed.gz path (or DataFrame with ``[chrom, start, end]``) of
        genomic regions to exclude. Any feature whose ``[start, end)``
        overlaps a blacklist interval is dropped *before* aggregation.
        Useful for ENCODE-style mappability / hypervariable blacklists
        when building 100 kb / 5 kb bin matrices.
    flank_bp : int, default 2000
        Only used when ``features`` is a GTF path. Extends each gene
        interval by this many bp on both sides before aggregation
        (gene body + promoter window; set to ``0`` for the bare
        gene body).
    gtf_id_col : {'gene_name', 'gene_id'}, default ``'gene_name'``
        Only used when ``features`` is a GTF path. Which GTF attribute
        becomes ``var_names``. When ``'gene_name'`` (the default), GENCODE
        records that share a symbol on one chrom are merged into a single
        interval so the output has exactly one row per gene symbol.
    score : {'frac', 'posterior_frac', 'hypo-score', 'hyper-score', 'umc'}
        What to store in ``.X``:

        - ``'frac'``: raw ``mc/cov`` fraction. Uncovered features follow
          ``nan_policy`` (``np.nan`` under the default ``'auto'``; use
          ``nan_policy='zero'`` to keep ``.X`` sparse).
        - ``'posterior_frac'`` (default): per-cell empirical-Bayes posterior mean
          ``(mc + alpha) / (cov + alpha + beta)`` using the per-cell Beta
          prior estimated by the Beta-Binomial method of moments (see
          ``docs/Methods.md``). Shrinks noisy low-coverage estimates toward
          the cell's prior mean. Uncovered features' posterior equals that
          prior mean (a per-cell constant kept in ``obs['prior_mean']``); under
          the default ``nan_policy='auto'`` uncovered ``.X`` entries are filled
          with that per-cell prior mean (i.e. resolves to ``'prior_mean'``; use
          ``nan_policy='zero'`` to keep ``.X`` **sparse** with implicit ``0``
          and reconstruct the prior-filled dense matrix on demand via
          :func:`to_dense` (``fill='prior_mean'``)).
        - ``'hypo-score'``: per-cell binomial survival function
          ``P(X > mc | Binomial(cov, p_cell))``, with ``p_cell = total_mc /
          total_cov`` for that cell. Values below ``score_cutoff`` are set
          to zero. High values mark hypo-methylated sites.
        - ``'hyper-score'``: ``1 - sf`` with the same sparsification;
          high values mark hyper-methylated sites.
        - ``'umc'``: raw unmethylated count = ``cov - mc`` per
          (cell, feature). Useful when you want to assemble ``mc`` and
          ``umc`` matrices side-by-side for downstream Beta-binomial /
          ALLCools-style modeling without recomputing.

        Raw ``mc`` / ``cov`` counts always live in ``.layers['mc']`` /
        ``.layers['cov']`` regardless of ``score`` (which only changes ``.X``),
        so they are not offered as ``score`` choices.

        Only when ``score='posterior_frac'`` are the per-cell Beta prior
        columns ``['alpha', 'beta', 'prior_mean', 'rho']`` written to
        ``adata.obs`` and the per-feature accumulators ``['hvf_n_cov',
        'hvf_sum', 'hvf_sum_sq']`` written to ``adata.var`` (they reuse the
        same per-cell prior, estimated once). For the other scores these are
        skipped to avoid the extra Beta-Binomial estimation.
    score_cutoff : float
        Sparsification threshold for hypo/hyper scores. Default 0.9.
    hvf_frac : {'posterior', 'raw'}, default ``'posterior'``
        Only used when ``score='posterior_frac'``. Which per-(cell, feature)
        fraction the per-feature HVF accumulators
        (``adata.var['hvf_n_cov' / 'hvf_sum' / 'hvf_sum_sq']``) are computed
        on. ``'posterior'`` uses the empirical-Bayes posterior mean
        ``(mc + alpha) / (cov + alpha + beta)`` (raw ``mc/cov`` fallback for
        degenerate cells); ``'raw'`` uses ``mc/cov``. These three additive
        accumulators let mean / var / dispersion / normalized dispersion be
        reconstructed downstream (see ``docs/Methods.md``).
    nan_policy : {'auto', 'zero', 'nan', 'prior_mean'}, default ``'auto'``
        How uncovered (``cov == 0``) entries are represented in ``.X``.

        - ``'auto'`` (default): pick per ``score`` — ``'posterior_frac'`` ->
          ``'prior_mean'`` (uncovered posterior equals the cell prior mean),
          ``'frac'`` / ``'umc'`` -> ``'nan'``, and ``'hypo-score'`` /
          ``'hyper-score'`` -> these have no missing entries, so
          ``.X`` stays sparse.
        - ``'zero'`` (**memory-cheap — switch to this if you OOM on large
          cohorts / genome-wide bins**): keep ``.X`` a sparse CSR with implicit
          ``0`` everywhere. Reconstruct a filled dense matrix on demand with
          :func:`to_dense` (``fill='prior_mean'`` / ``'nan'``).
        - ``'nan'``: densify ``.X`` and set uncovered entries to ``np.nan``.
        - ``'prior_mean'``: densify ``.X`` and fill uncovered entries with each
          cell's Beta prior mean (computed on demand for non-``posterior_frac``
          scores). Degenerate cells fall back to ``np.nan``.

        .. warning::
           ``'nan'`` / ``'prior_mean'`` (and the ``'auto'`` cases that resolve
           to them) force a **dense** ``float32`` ``.X`` of
           ``n_cells * n_features * 4`` bytes (peak ~1.5-2x that during
           construction and h5ad write), so they can easily **OOM** for large
           cell counts or genome-wide bins (e.g. 1M cells x 540k 5kb bins ≈
           2 TB dense). A warning is emitted when a dense policy is used. If
           you hit OOM, rebuild with ``nan_policy='zero'`` (sparse ``.X``) and
           materialise NaN lazily / block-wise from ``layers['cov']`` via
           :func:`to_dense` / :func:`to_dense_nan`.

        ``.layers['mc']`` / ``.layers['cov']`` always stay sparse with implicit
        ``0``; the covered mask equals their non-zero structure
        (see :func:`covered_mask`).
    jobs : int
        Number of worker processes (CPUs) for parallel per-cell aggregation.
        ``1`` (default) runs serially in-process. ``>1`` uses a
        :class:`concurrent.futures.ProcessPoolExecutor`: for a list of
        per-cell ``.cz`` files each worker opens its own Reader and
        processes one cell; for a merged ``.cz`` each worker handles
        a subset of cell prefixes. ``0`` or negative falls back to
        serial.

    Returns
    -------
    anndata.AnnData
        ``.X`` holds the requested score as ``float32``. Under the default
        ``nan_policy='auto'`` uncovered entries are filled per score
        (``prior_mean`` for ``posterior_frac``, ``np.nan`` for ``frac`` /
        ``umc``, implicit ``0`` kept sparse for hypo/hyper scores); ``'zero'``
        keeps ``.X`` a CSR sparse matrix with implicit-``0`` uncovered
        features; ``'nan'`` / ``'prior_mean'`` force a dense array.
        ``.layers['mc']`` and ``.layers['cov']`` hold the raw integer counts
        (``uint32``) as CSR sparse matrices.
    """
    if score not in _VALID_SCORES:
        raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")
    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {nan_policy!r}")

    feat_df, features_by_chrom, n_feat, gtf_meta_df = _prepare_feature_groups(
        features, chrom_size, exclude_chroms, blacklist, flank_bp, gtf_id_col)

    # Lazily load reference positions when needed.
    ref_pos_map_cache = {"loaded": False, "map": None}

    def _get_ref_pos_map(hint_path=None):
        if ref_pos_map_cache["loaded"]:
            return ref_pos_map_cache["map"]
        ref_path = reference or hint_path
        if ref_path is None:
            raise ValueError(
                "Cell .cz has no 'pos' column; pass reference= to "
                "provide the coordinate reference."
            )
        # Local import to avoid circular deps at module load time.
        from .bam import _LazyRefPositions
        ref_pos_map_cache["map"] = _LazyRefPositions(ref_path)
        ref_pos_map_cache["loaded"] = True
        return ref_pos_map_cache["map"]

    def _resolve_cols(r):
        cols = r.header["columns"]
        if pos_col in cols:
            pi = cols.index(pos_col)
        else:
            pi = None  # mc_cov-only layout
        return pi, cols.index(mc_col), cols.index(cov_col)

    # Resolve input mode.
    paths = _resolve_inputs(cz_inputs)
    n_workers = int(jobs) if jobs and int(jobs) > 1 else 1

    # Pre-materialise the optional context index once for the whole job.
    # Reading the .cz index here means: (a) workers don't each re-open it,
    # (b) the dict ships cleanly through ProcessPoolExecutor, (c) per-cell
    # aggregation only pays a zero-copy gather (no extra I/O / decompress).
    index_ids_by_chrom: Optional[dict] = None
    if index is not None:
        index_ids_by_chrom = _resolve_index_to_dict(
            index, chroms=set(features_by_chrom.keys()))

    # Streaming sparse builder: each cell's (n_feat, 2) array is appended
    # immediately and dropped, so peak memory is dominated by the sparse
    # CSRs (~nnz * 8 B) instead of n_cells * n_feat * 8 B.
    builder = _StreamingSparseBuilder(n_feat)
    obs_names: List[str] = []

    # Cap pool chunksize so the executor's in-flight buffer cannot scale
    # with n_cells. Default ``len // (n_workers*4)`` blew up to thousands
    # of arrays buffered for cohorts >100k cells. ``32`` keeps in-flight
    # memory bounded at ~ ``n_workers * 32 * (n_feat * 16 B)``.
    _POOL_CHUNK_CAP = 32

    def _run_parallel_files(paths_):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_pool_init,
                initargs=(features_by_chrom, pos_col, mc_col, cov_col,
                          reference, index_ids_by_chrom)) as ex:
            chunk = max(1, min(_POOL_CHUNK_CAP,
                               len(paths_) // (n_workers * 4) or 1))
            for arr in ex.map(_pool_process_file, paths_, chunksize=chunk):
                yield arr

    def _run_parallel_prefixes(cz_path, prefix_list):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_pool_init,
                initargs=(features_by_chrom, pos_col, mc_col, cov_col,
                          reference, index_ids_by_chrom)) as ex:
            args = [(cz_path, pref) for pref in prefix_list]
            chunk = max(1, min(_POOL_CHUNK_CAP,
                               len(args) // (n_workers * 4) or 1))
            for arr in ex.map(_pool_process_prefix, args, chunksize=chunk):
                yield arr

    # Whitelist set: O(1) membership tests; None = include all.
    sample_set = set(use_samples) if use_samples is not None else None

    def _basename_sample(p: str) -> str:
        """Derive sample name from a per-cell .cz path: strip ``ext``."""
        b = os.path.basename(p)
        return b[:-len(ext)] if ext and b.endswith(ext) else b

    if len(paths) == 1:
        # ---- Single-file mode: per-cell or pre-catcz'd merged file.
        r = Reader(paths[0], mmap_advise="sequential")
        try:
            n_keys = len(r.header["chunk_dims"])
            pi, mc_i, cov_i = _resolve_cols(r)
            rpm = _get_ref_pos_map(r.header.get("message")) if pi is None else None
            chrom_axis = _detect_chrom_axis(r, set(features_by_chrom.keys()))
            if n_keys >= 2:
                # Merged file: enumerate cells, filter by whitelist.
                cell_prefixes = [
                    (pref, lab) for pref, lab in
                    _enumerate_cell_prefixes(r, chrom_axis)
                    if sample_set is None or lab in sample_set
                ]
                if n_workers > 1 and len(cell_prefixes) > 1:
                    r.close()  # workers reopen independently
                    r = None
                    prefixes = [p for p, _ in cell_prefixes]
                    labels = [lab for _, lab in cell_prefixes]
                    for arr, label in zip(
                            _run_parallel_prefixes(paths[0], prefixes),
                            labels):
                        builder.append(arr)
                        obs_names.append(label)
                        del arr
                else:
                    for prefix, label in cell_prefixes:
                        arr = _aggregate_one_reader(
                            r, features_by_chrom, pi, mc_i, cov_i,
                            cell_prefix=prefix, chrom_axis=chrom_axis,
                            ref_pos_map=rpm,
                            index_ids_by_chrom=index_ids_by_chrom)
                        builder.append(arr)
                        obs_names.append(label)
                        del arr
            else:
                # Single per-cell file: sample name from filename.
                label = _basename_sample(paths[0])
                if sample_set is None or label in sample_set:
                    arr = _aggregate_one_reader(
                        r, features_by_chrom, pi, mc_i, cov_i,
                        chrom_axis=chrom_axis, ref_pos_map=rpm,
                        index_ids_by_chrom=index_ids_by_chrom)
                    builder.append(arr)
                    obs_names.append(label)
                    del arr
        finally:
            if r is not None:
                r.close()
    else:
        # ---- Multi-file mode: derive sample names from basenames; filter.
        keep_paths = []
        labels = []
        for p in paths:
            lab = _basename_sample(p)
            if sample_set is not None and lab not in sample_set:
                continue
            keep_paths.append(p)
            labels.append(lab)
        if n_workers > 1:
            for arr, label in zip(_run_parallel_files(keep_paths), labels):
                builder.append(arr)
                obs_names.append(label)
                del arr
        else:
            for p, label in zip(keep_paths, labels):
                r = Reader(p, mmap_advise="sequential")
                try:
                    pi, mc_i, cov_i = _resolve_cols(r)
                    rpm = _get_ref_pos_map(r.header.get("message")) \
                        if pi is None else None
                    chrom_axis = _detect_chrom_axis(
                        r, set(features_by_chrom.keys()))
                    arr = _aggregate_one_reader(
                        r, features_by_chrom, pi, mc_i, cov_i,
                        chrom_axis=chrom_axis, ref_pos_map=rpm,
                        index_ids_by_chrom=index_ids_by_chrom)
                finally:
                    r.close()
                builder.append(arr)
                obs_names.append(label)
                del arr

    return _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                                score, score_cutoff, hvf_frac, obs, output,
                                reference=(os.path.abspath(
                                    os.path.expanduser(reference))
                                    if reference else None),
                                nan_policy=nan_policy)


def cz_to_anndata_multiref(
    cz_table: Union[str, pd.DataFrame],
    features: Union[str, pd.DataFrame, int],
    output: Optional[str] = None,
    use_samples: Optional[Sequence[str]] = None,
    ext: str = ".cz",
    pos_col: str = "pos",
    mc_col: str = "mc",
    cov_col: str = "cov",
    obs: Optional[pd.DataFrame] = None,
    chrom_size: Optional[Union[str, pd.DataFrame, dict]] = None,
    exclude_chroms: Optional[Sequence[str]] = ("chrL",),
    blacklist: Optional[Union[str, pd.DataFrame]] = None,
    flank_bp: int = 2000,
    gtf_id_col: str = "gene_name",
    score: str = "posterior_frac",
    score_cutoff: float = 0.9,
    hvf_frac: str = "posterior",
    nan_policy: str = "auto",
    jobs: int = 1,
):
    """Build an ``AnnData`` from per-cell .cz with *different* references.

    The multi-reference analogue of :func:`cz_to_anndata`. When cells come
    from different donors, each was mapped to a donor-specific genome and is
    therefore row-aligned to its *own* reference .cz (different C positions,
    possibly different context at the same coordinate). A single shared
    ``reference=`` (as in :func:`cz_to_anndata`) would attach the wrong
    coordinates. Here every cell declares its own ``reference`` (and optional
    context ``index``) per row of ``cz_table`` and is aggregated over
    ``features`` using that reference for positions.

    Because the output is per-cell (cells × features), no cross-cell
    coordinate pooling is needed: every cell independently maps its mc/cov
    onto the feature intervals via its own reference. Context mismatch is
    handled naturally when a per-cell ``index`` (CG / CH) is supplied (each
    cell's index reflects that donor's genome), so e.g. a position that is
    CG in one donor but CH in another is counted correctly for each.

    Parameters
    ----------
    cz_table : DataFrame or path
        Per-cell table (DataFrame or headered TSV) with at least ``path``
        (per-cell reference-less ``.cz``) and ``reference`` (that cell's
        reference ``.cz``) columns; an optional ``index`` column gives each
        cell's 1-D context index ``.cz`` (CG / CH), and an optional
        ``cell_id`` column sets the row label (default: basename of ``path``
        with ``ext`` stripped). See
        :func:`cytozip.merge._resolve_cz_table`. A merged (catcz'd)
        multi-donor ``.cz`` is NOT supported — list per-cell files.
    features : str, DataFrame, or int
        Same as :func:`cz_to_anndata` (BED / GTF path, DataFrame, or bin
        size). See there for details.
    output, use_samples, ext, pos_col, mc_col, cov_col, obs, chrom_size,
    exclude_chroms, blacklist, flank_bp, gtf_id_col, score, score_cutoff,
    hvf_frac, nan_policy, jobs
        Same meaning as in :func:`cz_to_anndata`. ``use_samples`` filters on
        the table's ``cell_id`` column.

    Returns
    -------
    anndata.AnnData
        Same layout as :func:`cz_to_anndata`.
    """
    if score not in _VALID_SCORES:
        raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")
    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {nan_policy!r}")

    feat_df, features_by_chrom, n_feat, gtf_meta_df = _prepare_feature_groups(
        features, chrom_size, exclude_chroms, blacklist, flank_bp, gtf_id_col)
    chroms_set = set(features_by_chrom.keys())

    # Resolve the per-cell table (DataFrame or headered TSV).
    from .merge import _resolve_cz_table
    df = _resolve_cz_table(cz_table, ext=ext, require=("path", "reference"))
    if use_samples is not None:
        df = df.loc[df["cell_id"].isin(set(use_samples))]
    if len(df) == 0:
        raise ValueError("cz_to_anndata_multiref: no cells to process")
    for p in df["path"]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"cz_to_anndata_multiref: cell not found: {p}")

    # Group cells by (reference, index) so each reference position map and
    # context index is resolved once and reused across its cells.
    idx_col = df["index"].tolist() if "index" in df.columns else [None] * len(df)
    groups: dict = {}
    for p, ref, idx, lab in zip(df["path"], df["reference"], idx_col,
                                df["cell_id"]):
        groups.setdefault((ref, idx), []).append((p, lab))

    from .bam import _LazyRefPositions

    builder = _StreamingSparseBuilder(n_feat)
    obs_names: List[str] = []
    n_workers = int(jobs) if jobs and int(jobs) > 1 else 1
    _POOL_CHUNK_CAP = 32

    # Process each (reference, index) group so every cell uses its own
    # reference / context index.
    for (ref_path, idx_path), cells in groups.items():
        idx_dict = None
        if idx_path is not None:
            idx_dict = _resolve_index_to_dict(idx_path, chroms=chroms_set)
        if n_workers > 1 and len(cells) > 1:
            # Fresh pool per group: each worker's _pool_init pins THIS
            # group's reference + index, so no cross-reference contamination.
            from concurrent.futures import ProcessPoolExecutor
            paths_ = [p for p, _ in cells]
            labels_ = [lab for _, lab in cells]
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_pool_init,
                    initargs=(features_by_chrom, pos_col, mc_col, cov_col,
                              ref_path, idx_dict)) as ex:
                chunk = max(1, min(_POOL_CHUNK_CAP,
                                   len(paths_) // (n_workers * 4) or 1))
                for arr, lab in zip(
                        ex.map(_pool_process_file, paths_, chunksize=chunk),
                        labels_):
                    builder.append(arr)
                    obs_names.append(lab)
                    del arr
        else:
            rpm = _LazyRefPositions(ref_path)
            try:
                for p, label in cells:
                    r = Reader(p, mmap_advise="sequential")
                    try:
                        cols = r.header["columns"]
                        pi = cols.index(pos_col) if pos_col in cols else None
                        mc_i = cols.index(mc_col)
                        cov_i = cols.index(cov_col)
                        chrom_axis = _detect_chrom_axis(r, chroms_set)
                        arr = _aggregate_one_reader(
                            r, features_by_chrom, pi, mc_i, cov_i,
                            chrom_axis=chrom_axis,
                            ref_pos_map=(rpm if pi is None else None),
                            index_ids_by_chrom=idx_dict)
                    finally:
                        r.close()
                    builder.append(arr)
                    obs_names.append(label)
                    del arr
            finally:
                rpm.close()

    return _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                                score, score_cutoff, hvf_frac, obs, output,
                                reference=sorted(set(df["reference"])),
                                nan_policy=nan_policy)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_index_to_dict(index, chroms=None) -> dict:
    """Materialise a 1-D context index to ``{chrom_str: int64 IDs}``.

    Used by :func:`cz_to_anndata` to read the index .cz **once** in the
    parent process and ship a plain dict to workers via ``initargs``.
    Avoids one open / decompress per worker.

    Accepted forms mirror :meth:`Reader._resolve_index_ids` (str path,
    open ``Reader``, or pre-built dict).
    """
    if isinstance(index, dict):
        return {k: np.asarray(v, dtype=np.int64) for k, v in index.items()}
    close_after = False
    if isinstance(index, str):
        ir = Reader(index, mmap_advise="random")
        close_after = True
    elif isinstance(index, Reader):
        ir = index
    else:
        raise TypeError(
            f"index must be None, str path, Reader, or dict; got {type(index)}")
    try:
        if not (ir.header["columns"] == ["ID"]
                and ir.header["formats"] == ["I"]):
            raise ValueError(
                "index= requires a 1-D context index "
                "(columns=['ID'], formats=['I']); got "
                f"columns={ir.header['columns']}, "
                f"formats={ir.header['formats']}")
        out: dict = {}
        for dim in ir.chunk_key2offset.keys():
            chrom = dim[0] if isinstance(dim, tuple) else dim
            if chroms is not None and chrom not in chroms:
                continue
            ids = ir.get_ids_from_index(dim)
            if ids.ndim != 1:
                raise ValueError(
                    "index= requires a 1-D context index; got 2-D")
            out[chrom] = ids.astype(np.int64, copy=False)
        return out
    finally:
        if close_after:
            ir.close()


def _resolve_inputs(cz_inputs) -> List[str]:
    if isinstance(cz_inputs, str):
        path = os.path.abspath(os.path.expanduser(cz_inputs))
        if os.path.isdir(path):
            return sorted(os.path.join(path, f) for f in os.listdir(path)
                          if f.endswith(".cz"))
        return [path]
    return [os.path.abspath(os.path.expanduser(p)) for p in cz_inputs]


def _enumerate_cell_prefixes(reader: Reader, chrom_axis: int):
    """Yield ``(cell_prefix_tuple, label)`` pairs for each cell.

    The cell prefix is the dim tuple with the ``chrom_axis`` slot removed,
    preserving the original ordering of the other slots. ``label`` is
    the prefix joined with ``/`` when there are >1 non-chrom dims, or
    the single value otherwise. Sample-name filtering is the caller's
    responsibility.
    """
    n_keys = len(reader.header["chunk_dims"])
    if n_keys < 2:
        return
    seen = set()
    for dim in reader.chunk_key2offset.keys():
        prefix = tuple(v for j, v in enumerate(dim) if j != chrom_axis)
        seen.add(prefix)
    for prefix in sorted(seen):
        label = "/".join(prefix) if len(prefix) > 1 else prefix[0]
        yield prefix, label
