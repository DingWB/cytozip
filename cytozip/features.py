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
* Features are grouped by chrom for I/O locality. Reference-aligned cells
  (no ``pos`` column) are aggregated through a site -> feature map that the
  parent resolves once from the reference (:func:`_build_site_feature_map`);
  workers then only sum mc/cov over precomputed row runs and never load
  reference positions. Cells carrying their own ``pos`` column use a
  ``Reader.fetch_chunk_bytes`` decode + numpy cumsum / ``searchsorted``.
* Worker count is capped to the memory budget, tasks are submitted through
  a bounded window, and the finalize stage is row-blocked, so peak memory
  is ~ (sparse layers) + (one decompressed chunk per worker).
* Scoring (``prior_mean`` posterior-fraction, hypo/hyper) follows the
  ALLCools conventions (see ``ALLCools/mcds/utilities.py`` and
  ``ALLCools/count_matrix/dataset.py``).
"""
from __future__ import annotations

import os
import re
import struct
from typing import List, Optional, Sequence, Union

from loguru import logger
import numpy as np
import pandas as pd

from .cz import Reader, _fmt_to_np_dtype, _make_np_dtype, _find_pos_col


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
        nz, mc_nz, cov_nz = _to_sparse_triplet(np.asarray(arr))
        if nz.size:
            self._mc_data.append(mc_nz)
            self._cov_data.append(cov_nz)
            self._indices.append(nz)
        self._n_rows += 1
        self._indptr.append(self._indptr[-1] + int(nz.size))

    def append_sparse(self, indices, mc, cov) -> None:
        """Append one cell given only its covered entries: sorted column
        ``indices`` (int) with their ``mc`` / ``cov`` counts (``cov > 0``)."""
        n = int(len(indices))
        if n:
            self._indices.append(np.asarray(indices, dtype=np.int32))
            self._mc_data.append(np.asarray(mc, dtype=np.uint32))
            self._cov_data.append(np.asarray(cov, dtype=np.uint32))
        self._n_rows += 1
        self._indptr.append(self._indptr[-1] + n)

    def finalize(self):
        """Materialise ``(mc_csr, cov_csr)`` shape ``(n_rows, n_feat)``.

        The three output arrays are allocated once at their final size and
        the per-cell chunks are copied in and released one by one, so the
        peak is ~1x the result (not the 2x of ``np.concatenate``). After
        this call the internal buffers are dropped so the builder cannot be
        reused.
        """
        import scipy.sparse as ss
        n_rows = self._n_rows
        shape = (n_rows, self.n_feat)
        # ``indptr`` may exceed int32 for >2.1B nonzeros total — pick
        # the narrowest dtype that still covers the value range.
        last = self._indptr[-1]
        indptr_dtype = np.int64 if last > np.iinfo(np.int32).max else np.int32
        indptr = np.asarray(self._indptr, dtype=indptr_dtype)
        indices = np.empty(last, dtype=np.int32)
        mc_data = np.empty(last, dtype=np.uint32)
        cov_data = np.empty(last, dtype=np.uint32)
        off = 0
        for k in range(len(self._indices)):
            idx = self._indices[k]
            n = idx.size
            indices[off:off + n] = idx
            mc_data[off:off + n] = self._mc_data[k]
            cov_data[off:off + n] = self._cov_data[k]
            off += n
            # Release each chunk as soon as it has been copied.
            self._indices[k] = self._mc_data[k] = self._cov_data[k] = None
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


# Target number of stored entries handled per row block in the finalize
# stage; bounds every float64 temporary to ~8 * _BLOCK_NNZ bytes.
_BLOCK_NNZ = 4_000_000


def _row_blocks(indptr, target_nnz=_BLOCK_NNZ):
    """Yield ``(r0, r1)`` row ranges whose stored entries total at most
    ``target_nnz`` (always at least one row per block)."""
    indptr = np.asarray(indptr)
    n_rows = indptr.size - 1
    r0 = 0
    while r0 < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r0] + target_nnz,
                                 side="right")) - 1
        r1 = min(n_rows, max(r1, r0 + 1))
        yield r0, r1
        r0 = r1


def _compute_beta_params_sparse(mc_csr, cov_csr, min_cov=2, df_correction=True):
    """Per-cell Beta(alpha, beta) via the coverage-aware Beta-Binomial MoM.

    Operates only on the stored (non-zero-coverage) entries of each row via
    two ``reduceat`` passes (first the per-row mean, then the Pearson
    over-dispersion statistic), never densifying. Rows are processed in
    blocks of ~``_BLOCK_NNZ`` stored entries so the float64 temporaries do
    not scale with the whole matrix. Assumes ``mc_csr`` and ``cov_csr``
    share the same sparsity structure (same ``indptr`` / ``indices``), as
    produced upstream.
    """
    n_rows = cov_csr.shape[0]
    indptr = np.asarray(cov_csr.indptr)
    counts = np.diff(indptr).astype(np.int64)     # stored entries per row
    sum_cov = np.zeros(n_rows, dtype=np.float64)
    sum_mc = np.zeros(n_rows, dtype=np.float64)
    n_keep = np.zeros(n_rows, dtype=np.float64)
    X = np.zeros(n_rows, dtype=np.float64)
    mc_all, cov_all = mc_csr.data, cov_csr.data
    for r0, r1 in _row_blocks(indptr):
        s, e = int(indptr[r0]), int(indptr[r1])
        if e == s:
            continue
        covd = cov_all[s:e].astype(np.float64)
        mcd = mc_all[s:e].astype(np.float64)
        w = (covd >= min_cov).astype(np.float64)  # keep-mask as 0/1 weights
        cnt = counts[r0:r1]
        nz = cnt > 0
        starts = (indptr[r0:r1][nz] - s)
        # ---- first pass: per-row sums over kept entries -> mu ----
        sc = np.zeros(r1 - r0, dtype=np.float64)
        sm = np.zeros(r1 - r0, dtype=np.float64)
        nk = np.zeros(r1 - r0, dtype=np.float64)
        sc[nz] = np.add.reduceat(covd * w, starts)
        sm[nz] = np.add.reduceat(mcd * w, starts)
        nk[nz] = np.add.reduceat(w, starts)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(sc > 0, sm / sc, np.nan)
        mu_c = np.clip(mu, 1e-6, 1.0 - 1e-6)
        denom = mu_c * (1.0 - mu_c)
        # ---- second pass: Pearson over-dispersion statistic X ----
        mu_entry = np.repeat(mu_c, cnt)            # broadcast row mu to entries
        den_entry = np.repeat(denom, cnt)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(w > 0,
                            (mcd - covd * mu_entry) ** 2 / (covd * den_entry),
                            0.0)
        Xb = np.zeros(r1 - r0, dtype=np.float64)
        Xb[nz] = np.add.reduceat(term, starts)
        sum_cov[r0:r1] = sc
        sum_mc[r0:r1] = sm
        n_keep[r0:r1] = nk
        X[r0:r1] = Xb
    return _beta_binomial_alpha_beta(sum_mc, sum_cov, n_keep, X, df_correction)


def _compute_score_matrix_sparse(mc_csr, cov_csr, score, score_cutoff,
                                 alpha=None, beta=None):
    """Compute the requested score matrix from the streamed sparse layers.

    Supported scores: ``'frac'``, ``'posterior_frac'``, ``'hypo-score'``,
    ``'hyper-score'``, ``'mc'``, ``'cov'``, ``'umc'`` (see :func:`cz_to_anndata`).
    Returns a ``csr_matrix`` of shape ``(n_cells, n_features)`` with
    ``float32`` data. Zero-coverage entries are *implicit* zeros — they
    are not materialised, which is the whole point of the streaming
    path. For ``'posterior_frac'`` the per-cell posterior mean is computed
    only on the stored (``cov > 0``) entries, so the sparsity pattern is
    identical to ``'frac'`` (uncovered entries stay implicit ``0``).

    The float32 output is allocated once; all float64 temporaries live only
    for one row block (~``_BLOCK_NNZ`` entries) at a time.
    """
    import scipy.sparse as ss
    if score not in _VALID_SCORES and score not in ("mc", "cov"):
        raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")
    indptr = np.asarray(cov_csr.indptr)
    indices = np.asarray(cov_csr.indices)
    shape = cov_csr.shape
    mc_all, cov_all = mc_csr.data, cov_csr.data
    nnz = int(indptr[-1])
    out = np.empty(nnz, dtype=np.float32)

    if score == "posterior_frac" and (alpha is None or beta is None):
        alpha, beta, _ = _compute_beta_params_sparse(mc_csr, cov_csr)
    counts = None
    p_cell = valid = None
    if score in ("hypo-score", "hyper-score"):
        counts = np.diff(indptr).astype(np.int64)
        n_rows = shape[0]
        tot_mc = np.zeros(n_rows, dtype=np.float64)
        tot_cov = np.zeros(n_rows, dtype=np.float64)
        for r0, r1 in _row_blocks(indptr):
            s, e = int(indptr[r0]), int(indptr[r1])
            cnt = counts[r0:r1]
            nz = cnt > 0
            if e == s:
                continue
            starts = indptr[r0:r1][nz] - s
            tm = np.zeros(r1 - r0); tc = np.zeros(r1 - r0)
            tm[nz] = np.add.reduceat(mc_all[s:e].astype(np.float64), starts)
            tc[nz] = np.add.reduceat(cov_all[s:e].astype(np.float64), starts)
            tot_mc[r0:r1] = tm
            tot_cov[r0:r1] = tc
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cell = tot_mc / (tot_cov + 1e-6)
        valid = (np.isfinite(p_cell) & (p_cell > 0) & (p_cell < 1)
                 & (tot_cov > 0))
    elif score == "posterior_frac":
        counts = np.diff(indptr).astype(np.int64)

    for r0, r1 in _row_blocks(indptr):
        s, e = int(indptr[r0]), int(indptr[r1])
        if e == s:
            continue
        mcb, covb = mc_all[s:e], cov_all[s:e]
        if score == "mc":
            out[s:e] = mcb
        elif score == "cov":
            out[s:e] = covb
        elif score == "umc":
            umc = covb.astype(np.int64) - mcb.astype(np.int64)
            np.maximum(umc, 0, out=umc)
            out[s:e] = umc
        elif score == "frac":
            out[s:e] = (mcb.astype(np.float32)
                        / np.maximum(covb, 1).astype(np.float32))
        elif score == "posterior_frac":
            # per-cell empirical-Bayes posterior mean (mc + a)/(cov + a + b)
            # on the stored (cov>0) entries; broadcast the per-row alpha/beta
            # to per-nonzero via repeat. Zero-cov entries remain implicit 0.
            cnt = counts[r0:r1]
            a_nz = np.repeat(alpha[r0:r1].astype(np.float64), cnt)
            b_nz = np.repeat(beta[r0:r1].astype(np.float64), cnt)
            mcd = mcb.astype(np.float64)
            covd = covb.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                post = (mcd + a_nz) / (covd + a_nz + b_nz)
                raw = mcd / np.maximum(covd, 1.0)
            # degenerate cells (NaN alpha/beta) fall back to the raw fraction
            out[s:e] = np.where(np.isfinite(post), post, raw)
        else:  # hypo / hyper
            from scipy.stats import binom
            cnt = counts[r0:r1]
            p_per_nz = np.repeat(p_cell[r0:r1], cnt)
            valid_per_nz = np.repeat(valid[r0:r1], cnt)
            sf = binom.sf(mcb, covb, p_per_nz)
            pv = (1.0 - sf) if score == "hyper-score" else sf
            out[s:e] = np.where(valid_per_nz & (pv >= float(score_cutoff)),
                                pv, 0.0)
    return ss.csr_matrix((out, indices, indptr), shape=shape, dtype=np.float32)


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
    counts = np.diff(indptr).astype(np.int64)
    use_post = (method == "posterior" and alpha is not None
                and beta is not None)
    hvf_n_cov = np.bincount(indices, minlength=n_feat).astype(np.float64)
    hvf_sum = np.zeros(n_feat, dtype=np.float64)
    hvf_sum_sq = np.zeros(n_feat, dtype=np.float64)
    mc_all, cov_all = mc_csr.data, cov_csr.data
    for r0, r1 in _row_blocks(indptr):
        s, e = int(indptr[r0]), int(indptr[r1])
        if e == s:
            continue
        covd = cov_all[s:e].astype(np.float64)
        mcd = mc_all[s:e].astype(np.float64)
        # all stored entries are covered (cov > 0), so raw is always finite
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = mcd / np.maximum(covd, 1.0)
        if use_post:
            cnt = counts[r0:r1]
            a_nz = np.repeat(alpha[r0:r1].astype(np.float64), cnt)
            b_nz = np.repeat(beta[r0:r1].astype(np.float64), cnt)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = (mcd + a_nz) / (covd + a_nz + b_nz)
            frac = np.where(np.isfinite(frac), frac, raw)
        else:
            frac = raw
        idx = indices[s:e]
        hvf_sum += np.bincount(idx, weights=frac, minlength=n_feat)
        hvf_sum_sq += np.bincount(idx, weights=frac * frac, minlength=n_feat)
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
        # The decoded bytes are owned by ``raw``; the mmap pages behind the
        # chunk can go back to the kernel right away.
        reader.release_chunk(dim)
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=record_dtype)
        if arr.size == 0:
            continue
        n_rows = arr.size

        if pos_col is not None:
            positions = None  # taken from the (possibly gathered) records
        elif ref_pos_map is not None and chrom in ref_pos_map:
            # ref_pos_map may be a plain dict (legacy) or a
            # _LazyRefPositions (current bam.py); both expose ``.get`` and
            # ``__contains__``, but only the dict supports ``[]``.
            getter = getattr(ref_pos_map, "get", None)
            positions = (getter(chrom) if callable(getter)
                         else ref_pos_map[chrom])
            if positions is None or positions.size != n_rows:
                continue
        else:
            continue

        # Optional 1-D context-index filter (e.g. CG / CH only). The gather
        # runs on the raw (narrow) records *before* the int64 up-cast so the
        # transient footprint scales with the selected rows, not the chrom.
        if index_ids_by_chrom is not None:
            ids = index_ids_by_chrom.get(chrom)
            if ids is None or ids.size == 0:
                continue
            if ids.min() < 1 or ids.max() > n_rows:
                # Be lenient: clip OOB IDs (shorter cell than ref) instead
                # of erroring, since per-cell .cz may have fewer rows.
                ids = ids[(ids >= 1) & (ids <= n_rows)]
                if ids.size == 0:
                    continue
            gather = ids - 1
            arr = arr[gather]
            if positions is not None:
                positions = np.asarray(positions)[gather]
        del raw

        mc_vals = arr[f"c{mc_col}"].astype(np.int64, copy=False)
        cov_vals = arr[f"c{cov_col}"].astype(np.int64, copy=False)
        if positions is None:
            positions = arr[f"c{pos_col}"].astype(np.int64, copy=False)
        else:
            positions = np.asarray(positions, dtype=np.int64)
        del arr

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
# Memory / process-pool helpers
# ---------------------------------------------------------------------------
_MEM_UNITS = {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "m": 1 << 20,
              "mb": 1 << 20, "g": 1 << 30, "gb": 1 << 30, "t": 1 << 40,
              "tb": 1 << 40}


def _parse_mem(value) -> Optional[int]:
    """``'200G'`` / ``'512MB'`` / ``int`` bytes -> bytes (``None`` passthrough)."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer, float)):
        return int(value)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(value))
    if not m or m.group(2).lower() not in _MEM_UNITS:
        raise ValueError(f"cannot parse memory size {value!r} (e.g. '200G')")
    return int(float(m.group(1)) * _MEM_UNITS[m.group(2).lower()])


def _read_int_file(path) -> Optional[int]:
    try:
        with open(path) as fh:
            txt = fh.read().strip()
        return None if txt in ("max", "") else int(txt)
    except (OSError, ValueError):
        return None


def _cgroup_memory_headroom() -> Optional[int]:
    """``limit - usage`` of the tightest memory cgroup this process sits in
    (v2 or v1, walking up the hierarchy -- e.g. SLURM sets the job limit on
    a parent of the task cgroup). ``None`` when unlimited / unknown."""
    try:
        with open("/proc/self/cgroup") as fh:
            lines = [ln.strip().split(":", 2) for ln in fh if ln.strip()]
    except OSError:
        return None
    paths = []   # (mount_dir, rel_path, limit_file, usage_file)
    for parts in lines:
        if len(parts) != 3:
            continue
        hid, ctrls, rel = parts
        if hid == "0" and ctrls == "":           # cgroup v2 unified
            paths.append(("/sys/fs/cgroup", rel, "memory.max",
                          "memory.current"))
        elif "memory" in ctrls.split(","):       # cgroup v1
            paths.append(("/sys/fs/cgroup/memory", rel,
                          "memory.limit_in_bytes", "memory.usage_in_bytes"))
    best = None
    for mount, rel, lim_f, use_f in paths:
        rel = rel.strip("/")
        comps = rel.split("/") if rel else []
        # Walk from the leaf up to the mount root.
        for depth in range(len(comps), -1, -1):
            d = os.path.join(mount, *comps[:depth]) if depth else mount
            lim = _read_int_file(os.path.join(d, lim_f))
            if lim is None or lim >= (1 << 60):
                continue
            use = _read_int_file(os.path.join(d, use_f)) or 0
            head = max(0, lim - use)
            best = head if best is None else min(best, head)
    return best


def _available_memory_bytes() -> Optional[int]:
    """Best-effort free-memory estimate: min(free physical RAM, cgroup
    headroom). Returns ``None`` when nothing can be determined."""
    cands = []
    try:
        cands.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    head = _cgroup_memory_headroom()
    if head is not None:
        cands.append(head)
    return min(cands) if cands else None


def _fmt_bytes(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# Rough fixed per-process cost of a worker (interpreter + numpy + cytozip).
_WORKER_BASE_BYTES = 150 << 20


def _cap_workers(jobs, est_worker_bytes, max_mem=None, parent_bytes=0,
                 what="worker") -> int:
    """Clamp ``jobs`` so ``jobs * est_worker_bytes + parent_bytes`` fits in
    the memory budget (``max_mem`` or 80% of detected free memory)."""
    n = int(jobs) if jobs and int(jobs) > 1 else 1
    if n <= 1 or not est_worker_bytes:
        return n
    budget = _parse_mem(max_mem)
    if budget is None:
        avail = _available_memory_bytes()
        if avail is None:
            return n
        budget = int(avail * 0.8)
    fit = max(1, int((budget - parent_bytes) // est_worker_bytes))
    if fit < n:
        logger.warning(
            f"[cytozip] jobs={n} would need ~{_fmt_bytes(n * est_worker_bytes)} "
            f"(~{_fmt_bytes(est_worker_bytes)} per {what}) but the memory "
            f"budget is {_fmt_bytes(budget)}; using jobs={fit}. Pass "
            f"max_mem= to override the budget.")
        return fit
    return n


def _make_pool(n_workers, initializer, initargs):
    """``ProcessPoolExecutor`` preferring ``fork`` so the (possibly large)
    read-only ``initargs`` are inherited copy-on-write instead of pickled
    once per worker."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = None
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, initializer=initializer,
                               initargs=initargs, mp_context=ctx)


def _bounded_map(ex, fn, items, window):
    """Ordered ``ex.map`` with at most ``window`` tasks in flight, so
    finished-but-unconsumed results cannot pile up with ``len(items)``."""
    from collections import deque
    it = iter(items)
    pending = deque()
    for _ in range(max(1, int(window))):
        try:
            pending.append(ex.submit(fn, next(it)))
        except StopIteration:
            break
    while pending:
        res = pending.popleft().result()
        try:
            pending.append(ex.submit(fn, next(it)))
        except StopIteration:
            pass
        yield res


def _cgroup_usage_file() -> Optional[str]:
    """Usage file of the tightest limited memory cgroup (see
    :func:`_cgroup_memory_headroom`), or ``None``."""
    try:
        with open("/proc/self/cgroup") as fh:
            lines = [ln.strip().split(":", 2) for ln in fh if ln.strip()]
    except OSError:
        return None
    best = None
    for parts in lines:
        if len(parts) != 3:
            continue
        hid, ctrls, rel = parts
        if hid == "0" and ctrls == "":
            mount, lim_f, use_f = "/sys/fs/cgroup", "memory.max", "memory.current"
        elif "memory" in ctrls.split(","):
            mount, lim_f, use_f = ("/sys/fs/cgroup/memory",
                                   "memory.limit_in_bytes",
                                   "memory.usage_in_bytes")
        else:
            continue
        comps = rel.strip("/").split("/") if rel.strip("/") else []
        for depth in range(len(comps), -1, -1):
            d = os.path.join(mount, *comps[:depth]) if depth else mount
            lim = _read_int_file(os.path.join(d, lim_f))
            if lim is not None and lim < (1 << 60):
                f = os.path.join(d, use_f)
                if best is None or best[0] > lim:
                    best = (lim, f)
    return best[1] if best else None


def _child_pids(root_pid):
    """Direct children of ``root_pid`` via ``/proc/<pid>/task/*/children``
    (O(children)); falls back to scanning ``/proc`` on kernels without it."""
    pids = []
    try:
        task_dir = f"/proc/{root_pid}/task"
        for tid in os.listdir(task_dir):
            with open(f"{task_dir}/{tid}/children") as fh:
                pids.extend(int(x) for x in fh.read().split())
        return pids
    except (OSError, ValueError):
        pass
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    st = fh.read()
                # field 4 (after the parenthesised comm) is the ppid
                ppid = int(st[st.rfind(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == root_pid:
                pids.append(int(d))
    except OSError:
        pass
    return pids


def _proc_tree_pss_bytes(root_pid) -> int:
    """Physical memory of ``root_pid`` + its direct children: the sum of
    PSS (shared pages split between sharers, so forked workers are not
    double counted); falls back to RSS when ``smaps_rollup`` is missing."""
    pids = [root_pid] + _child_pids(root_pid)
    total = 0
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    for pid in pids:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as fh:
                for ln in fh:
                    if ln.startswith("Pss:"):
                        total += int(ln.split()[1]) * 1024
                        break
        except OSError:
            try:
                with open(f"/proc/{pid}/statm") as fh:
                    total += int(fh.read().split()[1]) * page
            except (OSError, ValueError, IndexError):
                pass
    return total


def _progress(iterable=None, total=None, desc="", enabled=False, unit="it"):
    """Progress bar over ``iterable`` (or a manual bar when ``iterable`` is
    ``None``) shown only when ``enabled``. Uses ``tqdm`` when installed,
    otherwise logs at ~10% steps; returns a plain iterator / no-op bar when
    disabled, so call sites need no branching."""
    if not enabled:
        return _NullBar(iterable)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _LogBar(iterable, total=total, desc=desc)
    return tqdm(iterable, total=total, desc=f"[cytozip] {desc}", unit=unit,
                dynamic_ncols=True, leave=True, mininterval=0.5)


class _NullBar:
    """No-op stand-in for a tqdm bar (iteration passes straight through)."""

    def __init__(self, iterable=None):
        self._it = iterable

    def __iter__(self):
        return iter(self._it if self._it is not None else ())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        pass

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix_str(self, s="", refresh=True):
        pass

    def close(self):
        pass


class _LogBar(_NullBar):
    """Fallback progress reporter (no ``tqdm``): one ``logger.info`` line per
    ~10% of ``total`` (or every 1000 items when ``total`` is unknown)."""

    def __init__(self, iterable=None, total=None, desc=""):
        super().__init__(iterable)
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total = total
        self.desc = desc
        self.n = 0
        self._next = 0
        self._last = None          # (desc, n) of the last emitted line
        self._step = max(1, total // 10) if total else 1000

    def __iter__(self):
        for x in (self._it if self._it is not None else ()):
            yield x
            self.update(1)
        self.close()

    def update(self, n=1):
        self.n += n
        if self.n >= self._next:
            self._emit()
            self._next = self.n + self._step

    def set_description(self, desc=None, refresh=True):
        # tqdm-compatible: callers may pass the full "[cytozip] ..." label.
        if desc:
            self.desc = desc[len("[cytozip] "):] if desc.startswith("[cytozip] ") else desc
        self._emit()

    def set_postfix_str(self, s="", refresh=True):
        pass

    def _emit(self):
        key = (self.desc, self.n)
        if key == self._last:
            return
        self._last = key
        if self.total:
            logger.info(f"[cytozip] {self.desc}: {self.n}/{self.total} "
                        f"({100.0 * self.n / self.total:.0f}%)")
        else:
            logger.info(f"[cytozip] {self.desc}: {self.n}")

    def close(self):
        if self.total and self.n < self.total:
            self.n = self.total
        self._emit()


class _ResourceReport:
    """Context manager that logs wall time and peak memory when ``enabled``.

    Peak memory is sampled every ``interval`` seconds from a daemon thread:
    the physical footprint of this process plus its worker children (PSS,
    so pages shared through ``fork`` are counted once) and, when the job
    runs under a limited memory cgroup (containers, SLURM), that cgroup's
    usage -- the number the OOM killer actually compares against the
    limit. ``ru_maxrss`` of the parent and of the largest worker are
    reported as well.
    """

    def __init__(self, enabled, label, jobs=1, interval=0.5):
        self.enabled = bool(enabled)
        self.label = label
        self.jobs = int(jobs) if jobs else 1
        self.interval = float(interval)
        self.elapsed = None
        self.peak_parent = None
        self.peak_child = None
        self.peak_tree = 0
        self.peak_cgroup = 0
        self._stop = None
        self._thread = None

    @staticmethod
    def _maxrss(who):
        import resource
        import sys
        v = resource.getrusage(who).ru_maxrss
        return int(v) if sys.platform == "darwin" else int(v) * 1024

    def _sample(self, pid, cg_file):
        self.peak_tree = max(self.peak_tree, _proc_tree_pss_bytes(pid))
        if cg_file is not None:
            self.peak_cgroup = max(self.peak_cgroup,
                                   _read_int_file(cg_file) or 0)

    def _run(self, pid, cg_file):
        import time
        interval = self.interval
        while not self._stop.wait(interval):
            try:
                t0 = time.perf_counter()
                self._sample(pid, cg_file)
                # PSS of many large workers is a page-table walk; keep the
                # sampler below ~5% of one core by backing off.
                interval = max(self.interval, 20.0 * (time.perf_counter() - t0))
            except Exception:  # sampling must never break the run
                pass

    def __enter__(self):
        import threading
        import time
        self._t0 = time.perf_counter()
        if self.enabled and os.path.isdir("/proc"):
            self._cg_file = _cgroup_usage_file()
            self._pid = os.getpid()
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(self._pid, self._cg_file),
                name="cytozip-resource-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        import resource
        import time
        self.elapsed = time.perf_counter() - self._t0
        self.peak_parent = self._maxrss(resource.RUSAGE_SELF)
        self.peak_child = self._maxrss(resource.RUSAGE_CHILDREN)
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            try:
                self._sample(self._pid, self._cg_file)
            except Exception:
                pass
        if not self.enabled:
            return False
        mins, secs = divmod(self.elapsed, 60.0)
        hrs, mins = divmod(int(mins), 60)
        clock = (f"{hrs}h{mins:02d}m{secs:04.1f}s" if hrs
                 else f"{mins}m{secs:04.1f}s")
        parts = [f"elapsed {clock}"]
        if self.peak_tree:
            parts.append(
                f"peak memory {_fmt_bytes(self.peak_tree)} "
                f"(parent + {self.jobs} worker(s), PSS)")
        if self.peak_cgroup:
            parts.append(
                f"cgroup peak {_fmt_bytes(self.peak_cgroup)} (incl. page cache)")
        parts.append(f"parent max RSS {_fmt_bytes(self.peak_parent)}")
        if self.jobs > 1 and self.peak_child:
            parts.append(f"largest worker max RSS {_fmt_bytes(self.peak_child)}"
                         " (includes pages shared with the parent)")
        logger.info(f"[cytozip] {self.label}: " + "; ".join(parts))
        return False


# ---------------------------------------------------------------------------
# Site -> feature map (built once in the parent, shared read-only by workers)
# ---------------------------------------------------------------------------
# Rows are processed in slabs of this many records so the accumulator
# up-cast of a whole chromosome (e.g. 8e7 rows -> 640 MB) is never
# materialised; 2M keeps the per-slab buffers L2/L3-friendly.
_RUN_SLAB = 2_000_000


def _chrom_feature_intervals(info):
    """``(starts, ends, feat_idx)`` int64 arrays for one chrom entry of
    ``features_by_chrom`` (tiled or explicit intervals)."""
    if info.get("tiled", False):
        bs = int(info["bin_size"])
        n = int(info["n_bins"])
        first = int(info["first_index"])
        starts = np.arange(n, dtype=np.int64) * bs
        return starts, starts + bs, first + np.arange(n, dtype=np.int64)
    return (np.asarray(info["starts"], dtype=np.int64),
            np.asarray(info["ends"], dtype=np.int64),
            np.asarray(info["indices"], dtype=np.int64))


class _LazyIndexIds:
    """Per-chrom 1-based IDs of a 1-D context index, read on demand (so a
    genome-wide CH index -- ~1e9 IDs, 8 GB as int64 -- is never held whole).
    Accepts a path, an open ``Reader`` or a pre-built ``{chrom: ids}``."""

    def __init__(self, index):
        self._dict = index if isinstance(index, dict) else None
        self._own = False
        self._reader = None
        if isinstance(index, str):
            self._reader = Reader(index, mmap_advise="sequential")
            self._own = True
        elif isinstance(index, Reader):
            self._reader = index
        elif self._dict is None:
            raise TypeError(
                f"index must be None, str path, Reader, or dict; got {type(index)}")
        if self._reader is not None:
            ir = self._reader
            if not (ir.header["columns"] == ["ID"]
                    and ir.header["formats"] == ["I"]):
                raise ValueError(
                    "index= requires a 1-D context index "
                    "(columns=['ID'], formats=['I']); got "
                    f"columns={ir.header['columns']}, "
                    f"formats={ir.header['formats']}")
            self._dims = {(d[0] if isinstance(d, tuple) else d): d
                          for d in ir.chunk_key2offset.keys()}

    def get(self, chrom):
        if self._dict is not None:
            v = self._dict.get(chrom)
            return None if v is None else np.asarray(v)
        dim = self._dims.get(chrom)
        if dim is None:
            return None
        ids = self._reader.get_ids_from_index(dim)
        self._reader.release_chunk(dim)
        if ids.ndim != 1:
            raise ValueError("index= requires a 1-D context index; got 2-D")
        return ids

    def close(self):
        if self._own and self._reader is not None:
            self._reader.close()
            self._reader = None


def _build_site_feature_map(reference, features_by_chrom,
                            index_ids_by_chrom=None, verbose=False,
                            index=None) -> dict:
    """Pre-resolve, per chrom, which (index-selected) reference rows fall in
    which feature, so workers never touch the reference positions.

    For every chrom the result holds

    * ``n_ref``: expected row count of the cell chunk (= reference rows),
    * ``gather``: 0-based rows kept by the context index (``None`` = all);
      for *dense* selections (> 50% of the rows, e.g. a CH index) the map
      instead stores the excluded complement ``excl`` with each excluded
      row's run id ``excl_run``: workers sum all rows and subtract the
      complement, avoiding a near-full gather per cell and a per-row map,
    * ``bounds``: sorted unique run boundaries in the kept-row space
      (``bounds[0] == 0``, ``bounds[-1] == n_sel``). Summing mc / cov over
      each run and taking a cumulative sum lets every feature (even
      overlapping ones) be read off as ``cum[hi_run] - cum[lo_run]``,
    * ``feat_idx`` / ``lo_run`` / ``hi_run``: per-feature output column and
      run range.

    Memory is ``O(n_features)`` per chrom plus the ``gather`` rows (int32
    when they fit), typically tens of MB genome-wide instead of the ~4 GB
    of cached positions a per-worker ``_LazyRefPositions`` ends up holding.
    The reference is read one chrom at a time and released immediately.
    """
    from .bam import _LazyRefPositions
    rpm = _LazyRefPositions(reference)
    out = {"chroms": {}, "max_rows": 0, "reference": reference}
    # ``index`` (path / Reader / dict) is read one chrom at a time; a
    # pre-materialised ``index_ids_by_chrom`` dict is accepted for callers
    # that already hold one.
    lazy_ids = None
    if index is not None:
        lazy_ids = _LazyIndexIds(index)
    elif index_ids_by_chrom is not None:
        lazy_ids = _LazyIndexIds(index_ids_by_chrom)
    try:
        for chrom, info in _progress(list(features_by_chrom.items()),
                                     desc="site -> feature map",
                                     enabled=verbose, unit="chrom"):
            if chrom not in rpm:
                continue
            pos = rpm.get(chrom)
            if pos is None:
                continue
            n_ref = int(pos.size)
            row_dtype = np.int32 if n_ref <= np.iinfo(np.int32).max else np.int64
            gather = excl = None
            if lazy_ids is not None:
                ids = lazy_ids.get(chrom)
                if ids is None or ids.size == 0:
                    rpm.drop(chrom)
                    continue
                ids = np.asarray(ids, dtype=np.int64)
                if ids.min() < 1 or ids.max() > n_ref:
                    ids = ids[(ids >= 1) & (ids <= n_ref)]
                    if ids.size == 0:
                        rpm.drop(chrom)
                        continue
                rows = ids - 1
                if rows.size * 2 > n_ref:
                    # Dense selection (e.g. a CH index keeps ~96% of the
                    # sites): sum ALL rows and subtract the small excluded
                    # complement instead of gathering ~every row per cell.
                    keep = np.zeros(n_ref, dtype=bool)
                    keep[rows] = True
                    excl = np.flatnonzero(~keep).astype(row_dtype)
                    del keep, rows
                    pos_sel = pos
                else:
                    gather = rows
                    pos_sel = pos[gather]
                    gather = gather.astype(row_dtype)
            else:
                pos_sel = pos
            n_sel = int(pos_sel.size)
            starts, ends, feat_idx = _chrom_feature_intervals(info)
            # BED half-open [start, end) vs 1-based pos: start < pos <= end.
            # Query in the positions' own dtype so searchsorted does not
            # up-cast (copy) the whole position array to int64.
            qdt = pos_sel.dtype
            if (starts.size and starts.min() >= 0
                    and ends.max() <= np.iinfo(qdt).max):
                q_starts, q_ends = starts.astype(qdt), ends.astype(qdt)
            else:
                q_starts, q_ends = starts, ends
            lo = np.searchsorted(pos_sel, q_starts, side="right")
            hi = np.searchsorted(pos_sel, q_ends, side="right")
            del pos_sel
            rpm.drop(chrom)
            bounds = np.unique(np.concatenate(
                ([0, n_sel], lo, hi))).astype(np.int64)
            run_dtype = (np.int32 if bounds.size <= np.iinfo(np.int32).max
                         else np.int64)
            entry = {
                "n_ref": n_ref,
                "gather": gather,
                "bounds": bounds,
                "feat_idx": feat_idx,
                "lo_run": np.searchsorted(bounds, lo).astype(run_dtype),
                "hi_run": np.searchsorted(bounds, hi).astype(run_dtype),
                "excl": None,
                "excl_run": None,
            }
            if excl is not None and excl.size:
                # run id of every excluded row (full-row space == bounds space)
                entry["excl"] = excl
                entry["excl_run"] = (np.searchsorted(bounds, excl, side="right")
                                     - 1).astype(run_dtype)
            out["chroms"][chrom] = entry
            out["max_rows"] = max(out["max_rows"], n_ref)
    finally:
        rpm.close()
        if lazy_ids is not None:
            lazy_ids.close()
    return out


try:  # optional Cython kernels (see cz_accel.c_run_sums / c_run_sums_rows)
    from .cz_accel import c_run_sums as _c_run_sums
    from .cz_accel import c_run_sums_rows as _c_run_sums_rows
except Exception:  # pragma: no cover - pure-python fallback
    _c_run_sums = _c_run_sums_rows = None


def _kernel_layout(arr, mc_field, cov_field):
    """``(buf, unit, mc_off, mc_size, cov_off, cov_size)`` for the Cython
    kernels when ``arr`` is a C-contiguous record array whose two fields are
    unsigned little-endian ints; ``None`` otherwise (-> numpy fallback)."""
    dt = arr.dtype
    if dt.fields is None or not arr.flags.c_contiguous:
        return None
    f_mc, f_cov = dt.fields[mc_field], dt.fields[cov_field]
    for f in (f_mc, f_cov):
        if not (f[0].kind == "u" and f[0].itemsize in (1, 2, 4, 8)
                and f[0].byteorder in ("|", "=", "<")):
            return None
    return (arr.view(np.uint8).reshape(-1), dt.itemsize,
            f_mc[1], f_mc[0].itemsize, f_cov[1], f_cov[0].itemsize)


def _run_sums(arr, mc_field, cov_field, bounds, slab=_RUN_SLAB):
    """Per-run int64 sums of ``arr[mc_field]`` / ``arr[cov_field]`` over the
    runs ``[bounds[i], bounds[i+1])``.

    Fast path: the Cython kernel ``c_run_sums`` walks the packed record
    bytes once with both accumulators (no temporaries). Fallback:
    ``reduceat`` straight on the (strided) record field with an explicit
    accumulator dtype -- ``uint32`` for 1-byte counts (a 2M-row slab of
    ``uint8`` sums to <= 5.1e8) and ``int64`` otherwise -- processed in
    slabs so no up-cast copy of a whole chromosome is ever made.
    """
    n_runs = bounds.size - 1
    mc_runs = np.zeros(n_runs, dtype=np.int64)
    cov_runs = np.zeros(n_runs, dtype=np.int64)
    n = int(arr.shape[0])
    if n == 0 or n_runs <= 0:
        return mc_runs, cov_runs
    lay = _kernel_layout(arr, mc_field, cov_field) if _c_run_sums else None
    if lay is not None:
        bnd = np.ascontiguousarray(bounds, dtype=np.int64)
        _c_run_sums(*lay, bnd, mc_runs, cov_runs)
        return mc_runs, cov_runs
    mc_v, cov_v = arr[mc_field], arr[cov_field]

    def _acc(v):
        return (np.uint32 if (v.dtype.kind == "u" and v.dtype.itemsize == 1
                              and slab <= 1 << 24) else np.int64)
    acc_mc, acc_cov = _acc(mc_v), _acc(cov_v)
    a = 0
    while a < n:
        b = min(a + slab, n)
        # Runs overlapping [a, b): the run containing a .. the last run
        # starting before b.
        lo = int(np.searchsorted(bounds, a, side="right")) - 1
        hi = int(np.searchsorted(bounds, b, side="left"))
        if hi <= lo:
            a = b
            continue
        starts = bounds[lo:hi] - a
        starts[0] = 0
        mc_runs[lo:hi] += np.add.reduceat(mc_v[a:b], starts, dtype=acc_mc)
        cov_runs[lo:hi] += np.add.reduceat(cov_v[a:b], starts, dtype=acc_cov)
        a = b
    return mc_runs, cov_runs


def _subtract_rows(arr, mc_field, cov_field, rows, run_ids, mc_runs, cov_runs):
    """``mc_runs[run_ids] -= arr[mc_field][rows]`` (and cov), in place."""
    lay = _kernel_layout(arr, mc_field, cov_field) if _c_run_sums_rows else None
    if lay is not None:
        neg_mc = np.zeros_like(mc_runs)
        neg_cov = np.zeros_like(cov_runs)
        _c_run_sums_rows(*lay, np.ascontiguousarray(rows, dtype=np.int64),
                         np.ascontiguousarray(run_ids, dtype=np.int64),
                         neg_mc, neg_cov)
        mc_runs -= neg_mc
        cov_runs -= neg_cov
        return
    sub = arr[rows]
    n_runs = mc_runs.size
    mc_runs -= np.bincount(run_ids, weights=sub[mc_field],
                           minlength=n_runs).astype(np.int64)
    cov_runs -= np.bincount(run_ids, weights=sub[cov_field],
                            minlength=n_runs).astype(np.int64)


def _aggregate_mapped(reader, site_map, mc_col, cov_col, n_feat,
                      cell_prefix=None, chrom_axis=None, record_dtype=None):
    """``(n_features, 2)`` int64 ``[sum_mc, sum_cov]`` for one reference-
    aligned cell using a pre-built :func:`_build_site_feature_map`.

    Per chrom this decodes the cell chunk once, applies the context-index
    gather on the raw (narrow) records, sums in slabs and releases the
    chunk, so the transient footprint is ~ the decompressed chunk size.
    """
    out = np.zeros((n_feat, 2), dtype=np.int64)
    chunk_dims = reader.header["chunk_dims"]
    n_keys = len(chunk_dims)
    if chrom_axis is None:
        chrom_axis = n_keys - 1
    if record_dtype is None:
        record_dtype = _record_dtype_for(reader.header["formats"])
    mc_field, cov_field = f"c{mc_col}", f"c{cov_col}"
    for chrom, m in site_map["chroms"].items():
        if n_keys == 1:
            dim = (chrom,)
        else:
            if cell_prefix is None:
                continue
            dim_list = list(cell_prefix)
            dim_list.insert(chrom_axis, chrom)
            dim = tuple(dim_list)
        if dim not in reader.chunk_key2offset:
            continue
        raw = reader.fetch_chunk_bytes(dim)
        reader.release_chunk(dim)
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=record_dtype)
        if arr.shape[0] != m["n_ref"]:
            key = (str(reader.input), dim)
            if key not in _WARNED_MISMATCH:
                _WARNED_MISMATCH.add(key)
                logger.warning(
                    f"{reader.input} {dim}: {arr.shape[0]} rows but "
                    f"reference has {m['n_ref']}; chrom skipped")
            continue
        gather = m["gather"]
        if gather is not None:
            arr = arr[gather]
        del raw
        mc_runs, cov_runs = _run_sums(arr, mc_field, cov_field, m["bounds"])
        excl = m.get("excl")
        if excl is not None:
            # dense index selection: subtract the excluded rows' contribution
            _subtract_rows(arr, mc_field, cov_field, excl, m["excl_run"],
                           mc_runs, cov_runs)
        del arr
        cum_mc = np.concatenate(([0], np.cumsum(mc_runs)))
        cum_cov = np.concatenate(([0], np.cumsum(cov_runs)))
        lo, hi = m["lo_run"], m["hi_run"]
        fi = m["feat_idx"]
        out[fi, 0] = cum_mc[hi] - cum_mc[lo]
        out[fi, 1] = cum_cov[hi] - cum_cov[lo]
    return out


_U32_MAX = np.iinfo(np.uint32).max


def _check_u32(cov_sums):
    """Sparse layers store ``uint32``; warn (once) and clip if a feature's
    coverage sum exceeds it (only plausible for bulk-like cells over huge
    features)."""
    if cov_sums.size and int(cov_sums.max()) > _U32_MAX:
        if "u32" not in _WARNED_MISMATCH:
            _WARNED_MISMATCH.add("u32")
            logger.warning("[cytozip] per-feature counts exceed uint32; "
                           "values are clipped to 4294967295 in the sparse "
                           "mc / cov layers")
        return True
    return False


def _to_sparse_triplet(out):
    """``(n_feat, 2)`` int64 -> ``(cols int32, mc uint32, cov uint32)`` of
    the covered (``cov > 0``) entries; what workers ship back."""
    cov = out[:, 1]
    nz = np.flatnonzero(cov).astype(np.int32, copy=False)
    mc_nz, cov_nz = out[nz, 0], cov[nz]
    if _check_u32(cov_nz):
        mc_nz = np.minimum(mc_nz, _U32_MAX)
        cov_nz = np.minimum(cov_nz, _U32_MAX)
    return (nz, mc_nz.astype(np.uint32, copy=False),
            cov_nz.astype(np.uint32, copy=False))


def _worker_single_thread_inflate():
    """Pin the Cython chunk inflate to one thread inside pool workers: the
    default (up to 4 OpenMP threads per process) oversubscribes the node
    when ``jobs`` workers inflate concurrently (measured ~1.8x slower)."""
    os.environ.setdefault("CYTOZIP_INFLATE_THREADS", "1")
    try:
        from . import cz_accel
        cz_accel.set_inflate_threads(1)
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------
_WORKER_STATE: dict = {}
# (reader.input, dim) pairs already reported as length-mismatched, so a
# cohort with one odd chromosome does not emit one warning per cell.
_WARNED_MISMATCH: set = set()


def _pool_init(features_by_chrom, pos_col, mc_col, cov_col, reference,
               index_ids_by_chrom=None, site_map=None, index_path=None):
    """Per-process initializer: stash shared read-only state.

    ``site_map`` (see :func:`_build_site_feature_map`) is the normal path
    for reference-aligned cells; the per-worker lazy reference loader is
    only a fallback for cells without a ``pos`` column when no map was
    built, and it drops each chrom right after use.
    """
    _worker_single_thread_inflate()
    _WORKER_STATE.clear()
    _WORKER_STATE["features_by_chrom"] = features_by_chrom
    _WORKER_STATE["pos_col"] = pos_col
    _WORKER_STATE["mc_col"] = mc_col
    _WORKER_STATE["cov_col"] = cov_col
    _WORKER_STATE["reference"] = reference
    _WORKER_STATE["ref_pos_map"] = None
    _WORKER_STATE["chroms_set"] = set(features_by_chrom.keys())
    _WORKER_STATE["index_ids_by_chrom"] = index_ids_by_chrom
    _WORKER_STATE["index_path"] = index_path
    _WORKER_STATE["site_map"] = site_map
    _WORKER_STATE["n_feat"] = sum(
        v["n_bins"] if v.get("tiled") else len(v["indices"])
        for v in features_by_chrom.values())


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


def _pool_index_ids(pi):
    """Raw index IDs for a position-column cell (``pi`` not None) in a
    worker: the shared dict when the parent built one, else resolved once
    per worker from the index path (rare mixed-input case)."""
    st = _WORKER_STATE
    if pi is None:
        return st.get("index_ids_by_chrom")
    if st.get("index_ids_by_chrom") is None and st.get("index_path"):
        st["index_ids_by_chrom"] = _resolve_index_to_dict(
            st["index_path"], chroms=st["chroms_set"])
    return st.get("index_ids_by_chrom")


def _aggregate_cell(reader, pi, mc_i, cov_i, cell_prefix=None,
                    chrom_axis=None, site_map=None, ref_pos_map=None,
                    features_by_chrom=None, index_ids_by_chrom=None,
                    n_feat=None):
    """Dispatch one cell to the mapped (reference-aligned) or the
    position-column aggregation path; returns ``(n_feat, 2)`` int64."""
    if pi is None and site_map is not None:
        return _aggregate_mapped(reader, site_map, mc_i, cov_i, n_feat,
                                 cell_prefix=cell_prefix,
                                 chrom_axis=chrom_axis)
    arr = _aggregate_one_reader(
        reader, features_by_chrom, pi, mc_i, cov_i,
        cell_prefix=cell_prefix, chrom_axis=chrom_axis,
        ref_pos_map=ref_pos_map, index_ids_by_chrom=index_ids_by_chrom)
    if pi is None and ref_pos_map is not None and hasattr(ref_pos_map, "drop"):
        # Fallback path: do not let the per-worker position cache grow to
        # the whole genome across cells.
        for chrom in list(getattr(ref_pos_map, "_cache", {}).keys()):
            ref_pos_map.drop(chrom)
    return arr


def _pool_process_file(cz_path):
    """Worker entry: aggregate a whole per-cell ``.cz`` file; returns the
    covered entries as ``(cols, mc, cov)`` (see :func:`_to_sparse_triplet`)."""
    st = _WORKER_STATE
    r = Reader(cz_path, mmap_advise="sequential")
    try:
        cols = r.header["columns"]
        pi = cols.index(st["pos_col"]) if st["pos_col"] in cols else None
        mc_i = cols.index(st["mc_col"])
        cov_i = cols.index(st["cov_col"])
        site_map = st.get("site_map")
        rpm = None
        if pi is None and site_map is None:
            rpm = _pool_get_ref_pos_map(r.header.get("message"))
        chrom_axis = _detect_chrom_axis(r, st["chroms_set"])
        arr = _aggregate_cell(
            r, pi, mc_i, cov_i, chrom_axis=chrom_axis, site_map=site_map,
            ref_pos_map=rpm, features_by_chrom=st["features_by_chrom"],
            index_ids_by_chrom=_pool_index_ids(pi),
            n_feat=st["n_feat"])
    finally:
        r.close()
    return _to_sparse_triplet(arr)


def _pool_process_prefix(args):
    """Worker entry: aggregate one cell prefix inside a merged ``.cz``.

    ``args = (cz_path, prefix_tuple)``. The Reader is opened once per
    worker process and cached; cell prefixes streamed in reuse it.
    """
    cz_path, prefix = args
    st = _WORKER_STATE
    r = st.get("_merged_reader")
    if r is None or st.get("_merged_path") != cz_path:
        if r is not None:
            r.close()
        r = Reader(cz_path, mmap_advise="sequential")
        st["_merged_reader"] = r
        st["_merged_path"] = cz_path
        cols = r.header["columns"]
        st["_pi"] = (cols.index(st["pos_col"])
                     if st["pos_col"] in cols else None)
        st["_mc_i"] = cols.index(st["mc_col"])
        st["_cov_i"] = cols.index(st["cov_col"])
        st["_chrom_axis"] = _detect_chrom_axis(r, st["chroms_set"])
    pi = st["_pi"]
    site_map = st.get("site_map")
    rpm = None
    if pi is None and site_map is None:
        rpm = _pool_get_ref_pos_map(r.header.get("message"))
    arr = _aggregate_cell(
        r, pi, st["_mc_i"], st["_cov_i"], cell_prefix=prefix,
        chrom_axis=st["_chrom_axis"], site_map=site_map, ref_pos_map=rpm,
        features_by_chrom=st["features_by_chrom"],
        index_ids_by_chrom=_pool_index_ids(pi), n_feat=st["n_feat"])
    return _to_sparse_triplet(arr)


def _estimate_worker_bytes(site_map, unit_size, paths=None) -> int:
    """Rough per-worker peak for :func:`_cap_workers`: fixed base + the
    largest decompressed cell chunk (+ slab / gather scratch)."""
    est = _WORKER_BASE_BYTES
    if site_map is not None and site_map.get("max_rows"):
        chunk = int(site_map["max_rows"]) * max(1, int(unit_size))
        gather = max((m["gather"].nbytes for m in site_map["chroms"].values()
                      if m["gather"] is not None), default=0)
        excl = max((m["excl"].nbytes for m in site_map["chroms"].values()
                    if m.get("excl") is not None), default=0)
        est += int(chunk * 1.25) + gather + excl + 2 * _RUN_SLAB * 8
    elif paths:
        # Position-column cells: compressed size x a generous inflate ratio.
        try:
            largest = max(os.path.getsize(p) for p in paths[:2000])
        except OSError:
            largest = 0
        est += 6 * largest
    return est


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
        # numpy arrays (not Python lists): ~8 B per value instead of ~36 B,
        # and every consumer goes through np.asarray anyway.
        for c in pd.unique(_chroms):
            sel = np.nonzero(_chroms == c)[0]
            features_by_chrom[c] = {
                "starts": _starts[sel].astype(np.int64),
                "ends": _ends[sel].astype(np.int64),
                "indices": _index_labels[sel].astype(np.int64),
            }
    return feat_df, features_by_chrom, n_feat, gtf_meta_df


def _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                         score, score_cutoff, hvf_frac, obs, output,
                         reference=None, nan_policy="auto", max_mem=None,
                         verbose=False):
    """Assemble an :class:`anndata.AnnData` from a filled streaming builder.

    Shared tail of :func:`cz_to_anndata` and :func:`cz_to_anndata_multiref`:
    materialises the sparse mc/cov layers, computes the requested ``.X``
    score (plus the per-cell Beta prior / per-feature HVF accumulators for
    ``posterior_frac``), and writes the ``.h5ad`` when ``output`` is given.
    ``reference`` (a path or list of paths) is recorded in
    ``adata.uns['cytozip_reference']``.

    ``nan_policy`` controls how uncovered (``cov == 0``) entries appear in
    ``.X`` (see :func:`_apply_nan_policy`); under ``'auto'`` the dense fill
    is skipped (falls back to ``'zero'``) when the dense matrix would exceed
    the memory budget (``max_mem`` or half of the detected free memory).
    """
    import anndata

    # Materialise sparse layers (no dense intermediate). mc_sp / cov_sp are the
    # (n_cells, n_features) CSR uint32 matrices of methylated counts and total
    # coverage; only covered (cov > 0) entries are stored, so their shared
    # non-zero structure is the covered mask.
    n_phase = 2 + (3 if score == "posterior_frac" else 1) + (1 if output else 0)
    bar = _progress(total=n_phase, desc="finalize: sparse layers",
                    enabled=verbose, unit="step")
    mc_sp, cov_sp = builder.finalize()
    bar.update(1)

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
        bar.set_description("[cytozip] finalize: Beta prior")
        alpha, beta, prior_mean = _compute_beta_params_sparse(mc_sp, cov_sp)
        bar.update(1)
        bar.set_description(f"[cytozip] finalize: score {score}")
        X = _compute_score_matrix_sparse(mc_sp, cov_sp, score,
                                         score_cutoff=score_cutoff,
                                         alpha=alpha, beta=beta)
        bar.update(1)
        bar.set_description("[cytozip] finalize: HVF stats")
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
        bar.set_description(f"[cytozip] finalize: score {score}")
        X = _compute_score_matrix_sparse(mc_sp, cov_sp, score,
                                         score_cutoff=score_cutoff)
    bar.update(1)
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
    if eff_policy != "zero" and nan_policy == "auto":
        # Dense .X is n_cells * n_features * 4 B; when that does not fit the
        # memory budget comfortably, 'auto' keeps .X sparse instead of OOM.
        dense_bytes = int(mc_sp.shape[0]) * int(mc_sp.shape[1]) * 4
        budget = _parse_mem(max_mem)
        if budget is None:
            avail = _available_memory_bytes()
            budget = None if avail is None else int(avail * 0.5)
        if budget is not None and dense_bytes > budget:
            logger.warning(
                f"[cytozip] nan_policy='auto' would densify .X to "
                f"{_fmt_bytes(dense_bytes)} (> budget {_fmt_bytes(budget)}); "
                f"keeping .X sparse (nan_policy='zero'). Uncovered entries "
                f"are implicit 0; restore NaN / prior_mean on demand with "
                f"cytozip.features.to_dense(adata). Pass nan_policy="
                f"{eff_policy!r} explicitly to force the dense matrix.")
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
        bar.set_description(f"[cytozip] finalize: dense .X ({eff_policy})")
        if eff_policy == "prior_mean" and prior_mean is None:
            _, _, prior_mean = _compute_beta_params_sparse(mc_sp, cov_sp)
        X = _apply_nan_policy(X, cov_sp, eff_policy, prior_mean=prior_mean)
    bar.set_description("[cytozip] finalize: AnnData")
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
    bar.update(1)
    if output:
        bar.set_description(f"[cytozip] finalize: write {os.path.basename(output)}")
        adata.write_h5ad(os.path.abspath(os.path.expanduser(output)),
                         compression='lzf')
        bar.update(1)
    bar.set_description("[cytozip] finalize: done")
    bar.close()
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


def _scatter_csr_rows(dense, csr, r0, r1):
    """``dense[rows, cols] = data`` for the rows ``r0:r1`` of ``csr``."""
    indptr = csr.indptr
    s, e = int(indptr[r0]), int(indptr[r1])
    if e == s:
        return
    rows = np.repeat(np.arange(r0, r1), np.diff(indptr[r0:r1 + 1]))
    dense[rows, csr.indices[s:e]] = csr.data[s:e]


def _apply_nan_policy(X, cov_sp, policy, prior_mean=None):
    """Densify ``X`` to ``float32`` filling uncovered (``cov == 0``) per policy.

    ``'nan'`` fills ``np.nan``; ``'prior_mean'`` fills each cell's Beta prior
    mean (per-row broadcast, NaN fallback for degenerate cells). ``'zero'`` is
    handled by the caller (keeps ``X`` sparse) and never reaches here.

    The dense result is allocated once and pre-filled with the policy value
    (a per-row broadcast), then the covered entries are scattered in row
    blocks -- no dense boolean mask, no COO copy, so the peak is ~1x the
    dense ``float32`` matrix instead of ~2.5x.
    """
    import scipy.sparse as ss
    if policy not in ("nan", "prior_mean"):
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {policy!r}")
    if policy == "prior_mean" and prior_mean is None:
        raise ValueError(
            "nan_policy='prior_mean' requires per-cell prior_mean")
    if not ss.issparse(X):
        dense = np.array(X, dtype=np.float32, copy=True)
        uncovered = _uncovered_mask(cov_sp, dense.shape)
        if policy == "nan":
            dense[uncovered] = np.nan
        else:
            fill = np.broadcast_to(
                np.asarray(prior_mean, dtype=np.float32).reshape(-1, 1),
                dense.shape)
            dense[uncovered] = fill[uncovered]
        return dense

    X = X.tocsr()
    cov = cov_sp.tocsr() if ss.issparse(cov_sp) else None
    dense = np.empty(X.shape, dtype=np.float32)
    if policy == "nan":
        dense.fill(np.nan)
    else:
        dense[:] = np.asarray(prior_mean, dtype=np.float32).reshape(-1, 1)
    same_structure = (cov is not None and cov.nnz == X.nnz
                      and np.array_equal(cov.indptr, X.indptr)
                      and np.array_equal(cov.indices, X.indices))
    for r0, r1 in _row_blocks(X.indptr):
        if cov is not None and not same_structure:
            # covered entries missing from X are measured zeros, not NaN
            zeros = cov[r0:r1].copy()
            zeros.data = np.zeros_like(zeros.data, dtype=np.float32)
            _scatter_csr_rows(dense[r0:r1], zeros, 0, r1 - r0)
        _scatter_csr_rows(dense, X, r0, r1)
    if cov is None:
        uncovered = np.asarray(cov_sp) == 0
        if policy == "nan":
            dense[uncovered] = np.nan
        else:
            fill = np.broadcast_to(
                np.asarray(prior_mean, dtype=np.float32).reshape(-1, 1),
                dense.shape)
            dense[uncovered] = fill[uncovered]
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
    max_mem: Optional[Union[str, int]] = None,
    verbose: bool = False,
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
        restricted to the index-selected positions. For reference-aligned
        cells the index is streamed chrom by chrom into the parent's
        site -> feature map (sparse selections such as CG become a row
        gather, dense ones such as CH a small excluded complement), so even
        a genome-wide CH index (~1e9 IDs) is never materialised whole; for
        cells with their own ``pos`` column it is read once into a shared
        ``{chrom: IDs}`` dict (copy-on-write across workers).
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
          When score is posterior_frac, the raw frac could be easily calculated 
          on the fly:
          ```
            frac=adata.to_df(layer='mc').astype(np.float32) / np.maximum(adata.to_df(layer='cov').astype(np.float32), 1)
            uncovered = adata.to_df(layer='cov')==0
            frac[uncovered] = np.nan
        ```
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
        serial. The effective worker count is capped so that
        ``jobs x (estimated per-worker peak)`` fits the memory budget (see
        ``max_mem``); a warning reports the cap.
    max_mem : str or int, optional
        Memory budget (e.g. ``'200G'``, ``'512MB'`` or bytes) used to cap
        ``jobs`` and to decide whether ``nan_policy='auto'`` may densify
        ``.X``. Default: derived from the free RAM / cgroup limit at run
        time (80% for the worker cap, 50% for the dense ``.X`` decision).
    verbose : bool, default False
        Show progress bars (site-map build, per-cell aggregation, finalize
        phases, h5ad write; ``tqdm`` when installed, else periodic log
        lines) and log the wall-clock time and peak memory (process tree
        PSS, cgroup usage, parent / largest-worker RSS) when the run
        finishes.

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

    with _ResourceReport(verbose, "cz_to_anndata", jobs) as rep:
        adata = _cz_to_anndata_impl(
            cz_inputs, feat_df, features_by_chrom, n_feat, gtf_meta_df,
            output, use_samples, ext, pos_col, mc_col, cov_col, obs,
            reference, index, score, score_cutoff, hvf_frac, nan_policy,
            jobs, max_mem, rep, verbose)
    return adata


def _cz_to_anndata_impl(cz_inputs, feat_df, features_by_chrom, n_feat,
                        gtf_meta_df, output, use_samples, ext, pos_col,
                        mc_col, cov_col, obs, reference, index, score,
                        score_cutoff, hvf_frac, nan_policy, jobs, max_mem,
                        rep, verbose=False):
    # Lazily load reference positions when needed (serial fallback only).
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
    if not paths:
        raise ValueError(f"cz_to_anndata: no .cz inputs found in {cz_inputs!r}")

    # Pre-materialise the optional context index once for the whole job.
    # Reading the .cz index here means: (a) workers don't each re-open it,
    # (b) the dict ships cleanly through the pool initializer, (c) per-cell
    # aggregation only pays a zero-copy gather (no extra I/O / decompress).
    # Peek at the first input's header: cells without a ``pos`` column are
    # row-aligned to a reference, for which the parent pre-resolves the
    # site -> feature map ONCE (workers then never load reference
    # positions; see _build_site_feature_map).
    r0 = Reader(paths[0], mmap_advise="random")
    try:
        pi0, _, _ = _resolve_cols(r0)
        unit_size = int(r0._unit_size)
        hint_ref = r0.header.get("message")
    finally:
        r0.close()
    site_map = None
    index_for_workers: Optional[dict] = None
    index_path_for_workers = index if isinstance(index, str) else None
    if pi0 is None:
        ref_path = reference or hint_ref
        if ref_path is None or not os.path.isfile(str(ref_path)):
            raise ValueError(
                "Cell .cz has no 'pos' column; pass reference= to "
                "provide the coordinate reference.")
        logger.info("[cytozip] resolving site -> feature map from "
                    f"{ref_path}")
        # The index is streamed chrom by chrom into the map (a genome-wide
        # CH index is ~1e9 IDs; it is never materialised as a dict here).
        site_map = _build_site_feature_map(ref_path, features_by_chrom,
                                           verbose=verbose, index=index)
        if index is not None and not isinstance(index, str):
            # Non-path index (Reader / dict) cannot be re-opened lazily by a
            # worker, so keep a dict for any position-column cell.
            index_for_workers = _resolve_index_to_dict(
                index, chroms=set(features_by_chrom.keys()))
    elif index is not None:
        # Position-column cells filter by the index themselves; read it
        # once in the parent and share it (fork -> copy-on-write).
        index_for_workers = _resolve_index_to_dict(
            index, chroms=set(features_by_chrom.keys()))

    est_worker = _estimate_worker_bytes(site_map, unit_size, paths)
    n_workers = _cap_workers(jobs, est_worker, max_mem=max_mem)
    rep.jobs = n_workers

    # Streaming sparse builder: each cell's covered entries are appended
    # immediately and dropped, so peak memory is dominated by the sparse
    # CSRs (~nnz * 12 B) instead of n_cells * n_feat * 8 B.
    builder = _StreamingSparseBuilder(n_feat)
    obs_names: List[str] = []

    # Keep at most this many tasks in flight per worker so finished results
    # waiting for in-order consumption stay bounded regardless of n_cells.
    _INFLIGHT_PER_WORKER = 4

    def _run_parallel_files(paths_):
        with _make_pool(n_workers, _pool_init,
                        (features_by_chrom, pos_col, mc_col, cov_col,
                         reference, index_for_workers, site_map,
                         index_path_for_workers)) as ex:
            yield from _bounded_map(ex, _pool_process_file, paths_,
                                    n_workers * _INFLIGHT_PER_WORKER)

    def _run_parallel_prefixes(cz_path, prefix_list):
        with _make_pool(n_workers, _pool_init,
                        (features_by_chrom, pos_col, mc_col, cov_col,
                         reference, index_for_workers, site_map,
                         index_path_for_workers)) as ex:
            args = [(cz_path, pref) for pref in prefix_list]
            yield from _bounded_map(ex, _pool_process_prefix, args,
                                    n_workers * _INFLIGHT_PER_WORKER)

    def _serial_cell(r, pi, mc_i, cov_i, chrom_axis, cell_prefix=None):
        nonlocal index_for_workers
        rpm = None
        if pi is None and site_map is None:
            rpm = _get_ref_pos_map(r.header.get("message"))
        if (pi is not None and index_for_workers is None
                and index_path_for_workers is not None):
            # mixed inputs: a position-column cell needs the raw IDs
            index_for_workers = _resolve_index_to_dict(
                index_path_for_workers, chroms=set(features_by_chrom.keys()))
        return _aggregate_cell(
            r, pi, mc_i, cov_i, cell_prefix=cell_prefix,
            chrom_axis=chrom_axis, site_map=site_map, ref_pos_map=rpm,
            features_by_chrom=features_by_chrom,
            index_ids_by_chrom=index_for_workers, n_feat=n_feat)

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
                    for (cols_, mc_, cov_), label in _progress(
                            zip(_run_parallel_prefixes(paths[0], prefixes),
                                labels),
                            total=len(labels), desc="aggregating cells",
                            enabled=verbose, unit="cell"):
                        builder.append_sparse(cols_, mc_, cov_)
                        obs_names.append(label)
                else:
                    for prefix, label in _progress(
                            cell_prefixes, desc="aggregating cells",
                            enabled=verbose, unit="cell"):
                        arr = _serial_cell(r, pi, mc_i, cov_i, chrom_axis,
                                           cell_prefix=prefix)
                        builder.append(arr)
                        obs_names.append(label)
                        del arr
            else:
                # Single per-cell file: sample name from filename.
                label = _basename_sample(paths[0])
                if sample_set is None or label in sample_set:
                    arr = _serial_cell(r, pi, mc_i, cov_i, chrom_axis)
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
        if n_workers > 1 and len(keep_paths) > 1:
            for (cols_, mc_, cov_), label in _progress(
                    zip(_run_parallel_files(keep_paths), labels),
                    total=len(labels), desc="aggregating cells",
                    enabled=verbose, unit="cell"):
                builder.append_sparse(cols_, mc_, cov_)
                obs_names.append(label)
        else:
            for p, label in _progress(list(zip(keep_paths, labels)),
                                      desc="aggregating cells",
                                      enabled=verbose, unit="cell"):
                r = Reader(p, mmap_advise="sequential")
                try:
                    pi, mc_i, cov_i = _resolve_cols(r)
                    chrom_axis = _detect_chrom_axis(
                        r, set(features_by_chrom.keys()))
                    arr = _serial_cell(r, pi, mc_i, cov_i, chrom_axis)
                finally:
                    r.close()
                builder.append(arr)
                obs_names.append(label)
                del arr
    if ref_pos_map_cache["map"] is not None:
        ref_pos_map_cache["map"].close()

    return _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                                score, score_cutoff, hvf_frac, obs, output,
                                reference=(os.path.abspath(
                                    os.path.expanduser(reference))
                                    if reference else None),
                                nan_policy=nan_policy, max_mem=max_mem,
                                verbose=verbose)


# ---------------------------------------------------------------------------
# Single-cytosine resolution matrix (cells x sites)
# ---------------------------------------------------------------------------
_SITES_STATE: dict = {}


def _sites_pool_init(state):
    _worker_single_thread_inflate()
    _SITES_STATE.clear()
    _SITES_STATE.update(state)


def _sites_read_cell(reader, prefix, st):
    """Covered entries ``(global_site_idx int32, mc uint32, cov uint32)`` of
    one cell over the site axis described by ``st``."""
    mc_col, cov_col, min_cov = st['mc_col'], st['cov_col'], st['min_cov']
    chrom_axis = st['chrom_axis']
    idx_parts, mc_parts, cov_parts = [], [], []
    dt = None
    for chrom, offset, rows, n_ref in st['axis']:
        if prefix is None:
            dim = (chrom,)
        else:
            d = list(prefix)
            d.insert(chrom_axis, chrom)
            dim = tuple(d)
        if dim not in reader.chunk_key2offset:
            continue
        if dt is None:
            cols = reader.header['columns']
            if mc_col not in cols or cov_col not in cols:
                raise ValueError(
                    f"{reader.input}: columns {cols} lack {mc_col!r}/{cov_col!r}"
                    " -- not a per-cell mc/cov .cz (is a reference / index "
                    "file mixed into the input directory?)")
            dt = _make_np_dtype(reader.header['formats'], cols)
        if rows is not None and rows.size * 64 < n_ref:
            # very sparse selection (<~1.5% of rows): block-wise gather
            # beats inflating the whole chunk; denser selections (e.g. a
            # CGN index, ~8-10%) are faster via the parallel chunk inflate.
            arr = reader.gather_rows(dim, rows, dtype=dt)
        else:
            raw = reader.fetch_chunk_bytes(dim)
            arr = np.frombuffer(raw, dtype=dt) if raw else np.empty(0, dtype=dt)
            if arr.shape[0] != n_ref:
                logger.warning(f"{reader.input} {dim}: {arr.shape[0]} rows but "
                               f"reference has {n_ref}; chrom skipped")
                reader.release_chunk(dim)
                continue
            if rows is not None:
                arr = arr[rows]
            reader.release_chunk(dim)
        cov = arr[cov_col]
        nz = np.flatnonzero(cov >= min_cov) if min_cov > 1 else np.flatnonzero(cov)
        if nz.size:
            idx_parts.append((nz + offset).astype(np.int32, copy=False))
            mc_parts.append(arr[mc_col][nz].astype(np.uint32, copy=False))
            cov_parts.append(cov[nz].astype(np.uint32, copy=False))
    if not idx_parts:
        e = np.empty(0, dtype=np.int32)
        return e, e.astype(np.uint32), e.astype(np.uint32)
    return (np.concatenate(idx_parts), np.concatenate(mc_parts),
            np.concatenate(cov_parts))


def _sites_pool_file(path):
    r = Reader(path, mmap_advise="sequential")
    try:
        return _sites_read_cell(r, None, _SITES_STATE)
    finally:
        r.close()


def _sites_pool_prefix(args):
    path, prefix = args
    r = _SITES_STATE.get('_reader')
    if r is None or r.input != path:
        if r is not None:
            r.close()
        r = Reader(path, mmap_advise="random")
        _SITES_STATE['_reader'] = r
    return _sites_read_cell(r, prefix, _SITES_STATE)


def cz_to_anndata_sites(
    cz_inputs: Union[str, Sequence[str]],
    reference: str,
    output: Optional[str] = None,
    index: Optional[Union[str, "Reader", dict]] = None,
    chroms: Optional[Sequence[str]] = None,
    use_samples: Optional[Sequence[str]] = None,
    ext: str = ".cz",
    mc_col: str = "mc",
    cov_col: str = "cov",
    min_cov: int = 1,
    score: str = "posterior_frac",
    nan_policy: str = "zero",
    obs: Optional[pd.DataFrame] = None,
    var_names: str = "coord",
    jobs: int = 1,
    max_mem: Optional[Union[str, int]] = None,
    verbose: bool = False,
):
    """Single-cytosine resolution ``(n_cells, n_sites)`` AnnData.

    The fastest way to get per-site methylation levels for many cells: each
    cell's chunk is bulk-decompressed once (or block-gathered when the
    context index is sparse), only **covered** sites are kept, and the
    result is assembled as sparse CSR layers — no dense ``n_cells x n_sites``
    intermediate. Single-cell data is typically <5-10% covered, so memory is
    ``~ nnz * 12 B`` (int32 index + uint32 mc + uint32 cov).

    Parameters
    ----------
    cz_inputs : str or list
        Directory of per-cell ``.cz``, list of paths, or one ``catcz``'d
        multi-cell ``.cz`` (chunk_dims ``['chrom', 'cell_id']``). All cells
        must be row-aligned to ``reference``.
    reference : str
        Reference ``.cz`` (``pos`` + optional ``strand`` / ``context``).
    index : str / Reader / dict, optional
        1-D context index (e.g. ``*.CGN.cz``) restricting the site axis to
        those cytosines. Strongly recommended (CG only is ~10x fewer sites).
    chroms : list of str, optional
        Restrict to these chromosomes (in this order). Default: all
        reference chunks. Use this to build the matrix per chromosome when
        the whole genome does not fit in memory.
    min_cov : int, default 1
        Sites with ``cov < min_cov`` are treated as uncovered (not stored).
    score : {'frac', 'posterior_frac', 'mc', 'cov', 'umc'}
        ``.X`` content on covered sites: raw ``mc/cov`` or the per-cell
        empirical-Bayes posterior mean ``(mc + a_c) / (cov + a_c + b_c)``
        (``a_c, b_c`` from a Beta-Binomial fit over that cell's covered
        sites, stored in ``obs['alpha'/'beta'/'prior_mean'/'rho']``).
    nan_policy : {'zero', 'nan', 'prior_mean'}, default 'zero'
        ``'zero'`` keeps ``.X`` sparse (uncovered = implicit 0; the covered
        mask is ``layers['cov'] != 0``, see :func:`to_dense`). The other two
        densify ``.X`` -- only sensible for small matrices.
    var_names : {'coord', 'pos'}
        ``'coord'`` -> ``"chr1:10469"``; ``'pos'`` -> position only (unique
        only within one chromosome; use with ``chroms=[one]``).
    jobs : int
        Worker processes (one cell per task); capped to the memory budget
        like :func:`cz_to_anndata`.
    max_mem, verbose
        See :func:`cz_to_anndata`.

    Returns
    -------
    anndata.AnnData
        ``obs`` = cells, ``var`` = sites (``chrom``, ``start``, ``end``,
        ``pos`` + ``strand``/``context`` when present in the reference;
        string columns are stored as ``category`` to keep ``var`` small),
        ``layers['mc']`` / ``layers['cov']`` sparse counts, ``.X`` = score.
    """
    if score not in ("frac", "posterior_frac", "mc", "cov", "umc"):
        raise ValueError("score must be one of frac, posterior_frac, mc, cov, umc")
    with _ResourceReport(verbose, "cz_to_anndata_sites", jobs) as rep:
        adata = _cz_to_anndata_sites_impl(
            cz_inputs, reference, output, index, chroms, use_samples, ext,
            mc_col, cov_col, min_cov, score, nan_policy, obs, var_names,
            jobs, max_mem, rep, verbose)
    return adata


def _bytes_to_categorical(v):
    """``S``-dtype array -> ``pd.Categorical`` decoding only the uniques
    (an object array of per-site Python strings costs ~60 B/site)."""
    uniq, inv = np.unique(v, return_inverse=True)
    cats = [u.decode("utf-8", "replace") for u in uniq.tolist()]
    return pd.Categorical.from_codes(inv.astype(np.int64, copy=False), cats)


def _cz_to_anndata_sites_impl(cz_inputs, reference, output, index, chroms,
                              use_samples, ext, mc_col, cov_col, min_cov,
                              score, nan_policy, obs, var_names, jobs,
                              max_mem, rep, verbose=False):
    reference = os.path.abspath(os.path.expanduser(reference))
    paths = _resolve_inputs(cz_inputs)
    if not paths:
        raise ValueError(f"cz_to_anndata_sites: no .cz inputs in {cz_inputs!r}")

    # ---- site axis from the reference -----------------------------------
    rr = Reader(reference, mmap_advise="sequential")
    try:
        ref_cols = rr.header['columns']
        ref_dt = _make_np_dtype(rr.header['formats'], ref_cols)
        pos_name = ref_cols[_find_pos_col(rr)]
        ref_chroms = [d[0] for d in rr.chunk_key2offset]
        if chroms is not None:
            want = [str(c) for c in chroms]
            missing = [c for c in want if c not in set(ref_chroms)]
            if missing:
                raise ValueError(f"chroms not in reference: {missing}")
            ref_chroms = want
        rows_by_chrom = (_resolve_index_to_dict(index, chroms=set(ref_chroms))
                         if index is not None else None)
        axis = []          # (chrom, offset, rows_or_None, n_ref_rows)
        var_parts = []
        offset = 0
        max_chunk_rows = 0
        for chrom in _progress(ref_chroms, desc="site axis (reference)",
                               enabled=verbose, unit="chrom"):
            raw = rr.fetch_chunk_bytes((chrom,))
            ref_arr = np.frombuffer(raw, dtype=ref_dt) if raw else np.empty(0, ref_dt)
            n_ref = ref_arr.shape[0]
            rows = None
            if rows_by_chrom is not None:
                ids = rows_by_chrom.get(chrom)
                if ids is None:
                    rr.release_chunk((chrom,))
                    continue
                rows = ids.astype(np.int64, copy=False) - 1
                rows = rows[(rows >= 0) & (rows < n_ref)]
                sel = ref_arr[rows]
            else:
                sel = ref_arr
            part = {'chrom': chrom,
                    'pos': sel[pos_name].astype(np.int64)}
            for c in ref_cols:
                if c == pos_name:
                    continue
                v = sel[c]
                part[c] = (_bytes_to_categorical(v)
                           if v.dtype.kind == 'S' else np.asarray(v))
            var_parts.append(pd.DataFrame(part))
            axis.append((chrom, offset, rows, n_ref))
            offset += int(sel.shape[0])
            max_chunk_rows = max(max_chunk_rows, n_ref)
            rr.release_chunk((chrom,))
    finally:
        rr.close()
    n_sites = offset
    if n_sites == 0:
        raise ValueError("no sites selected (empty reference / index)")
    feat_df = pd.concat(var_parts, ignore_index=True) if var_parts else pd.DataFrame()
    feat_df.insert(1, 'start', feat_df['pos'] - 1)
    feat_df.insert(2, 'end', feat_df['pos'])
    if var_names == 'pos':
        feat_df['name'] = feat_df['pos'].astype(str)
    else:
        feat_df['name'] = (feat_df['chrom'].astype(str) + ':'
                           + feat_df['pos'].astype(str))
    logger.info(f"cz_to_anndata_sites: {n_sites:,} sites over "
                f"{len(axis)} chrom(s)")

    # ---- cells ------------------------------------------------------------
    sample_set = set(use_samples) if use_samples is not None else None

    def _basename_sample(p):
        b = os.path.basename(p)
        return b[:-len(ext)] if ext and b.endswith(ext) else b

    builder = _StreamingSparseBuilder(n_sites)
    obs_names: List[str] = []
    _INFLIGHT_PER_WORKER = 4

    r0 = Reader(paths[0], mmap_advise="random")
    try:
        cols0 = r0.header['columns']
        if mc_col not in cols0 or cov_col not in cols0:
            raise ValueError(f"columns {cols0} lack {mc_col!r}/{cov_col!r}")
        unit_size = int(r0._unit_size)
        n_keys = len(r0.header['chunk_dims'])
        chrom_axis = _detect_chrom_axis(r0, set(ref_chroms)) if n_keys >= 2 else 0
        cell_prefixes = None
        if len(paths) == 1 and n_keys >= 2:
            cell_prefixes = [(pref, lab) for pref, lab in
                             _enumerate_cell_prefixes(r0, chrom_axis)
                             if sample_set is None or lab in sample_set]
    finally:
        r0.close()
    state = dict(axis=axis, mc_col=mc_col, cov_col=cov_col,
                 min_cov=int(min_cov), chrom_axis=chrom_axis)
    # Per-worker peak ~ decompressed chunk + the selected-rows copy.
    sel_bytes = max((r.nbytes for _, _, r, _ in axis if r is not None),
                    default=0)
    est_worker = (_WORKER_BASE_BYTES + int(max_chunk_rows * unit_size * 1.25)
                  + 2 * sel_bytes)
    n_workers = _cap_workers(jobs, est_worker, max_mem=max_mem)
    rep.jobs = n_workers

    def _consume(results, labels):
        for (idx, mc, cov), label in _progress(
                zip(results, labels), total=len(labels),
                desc="reading cells", enabled=verbose, unit="cell"):
            builder.append_sparse(idx, mc, cov)
            obs_names.append(label)

    if cell_prefixes is not None:
        prefixes = [p for p, _ in cell_prefixes]
        labels = [lab for _, lab in cell_prefixes]
        if n_workers > 1 and len(prefixes) > 1:
            with _make_pool(n_workers, _sites_pool_init, (state,)) as ex:
                args = [(paths[0], pref) for pref in prefixes]
                _consume(_bounded_map(ex, _sites_pool_prefix, args,
                                      n_workers * _INFLIGHT_PER_WORKER), labels)
        else:
            r = Reader(paths[0], mmap_advise="random")
            try:
                _consume((_sites_read_cell(r, pref, state) for pref in prefixes),
                         labels)
            finally:
                r.close()
    else:
        keep, labels = [], []
        for p in paths:
            lab = _basename_sample(p)
            if sample_set is not None and lab not in sample_set:
                continue
            keep.append(p)
            labels.append(lab)
        if n_workers > 1 and len(keep) > 1:
            with _make_pool(n_workers, _sites_pool_init, (state,)) as ex:
                _consume(_bounded_map(ex, _sites_pool_file, keep,
                                      n_workers * _INFLIGHT_PER_WORKER), labels)
        else:
            def _gen():
                for p in keep:
                    r = Reader(p, mmap_advise="sequential")
                    try:
                        yield _sites_read_cell(r, None, state)
                    finally:
                        r.close()
            _consume(_gen(), labels)

    return _finalize_cz_anndata(builder, obs_names, feat_df, None,
                                score, 0.9, "posterior", obs, output,
                                reference=reference, nan_policy=nan_policy,
                                max_mem=max_mem, verbose=verbose)


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
    max_mem: Optional[Union[str, int]] = None,
    verbose: bool = False,
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
    hvf_frac, nan_policy, jobs, max_mem, verbose
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

    with _ResourceReport(verbose, "cz_to_anndata_multiref", jobs) as rep:
        adata = _cz_to_anndata_multiref_impl(
            df, feat_df, features_by_chrom, n_feat, gtf_meta_df, chroms_set,
            output, pos_col, mc_col, cov_col, obs, score, score_cutoff,
            hvf_frac, nan_policy, jobs, max_mem, rep, verbose)
    return adata


_MULTIREF_STATE: dict = {}


def _multiref_pool_init(features_by_chrom, pos_col, mc_col, cov_col,
                        group_maps, group_refs, group_idx, group_idx_paths=None):
    """One pool for every (reference, index) group: ``group_maps[g]`` is
    that group's site map (or ``None`` for position-column cells)."""
    _worker_single_thread_inflate()
    _MULTIREF_STATE.clear()
    _MULTIREF_STATE.update(features_by_chrom=features_by_chrom,
                           pos_col=pos_col, mc_col=mc_col, cov_col=cov_col,
                           group_maps=group_maps, group_refs=group_refs,
                           group_idx=group_idx,
                           group_idx_paths=group_idx_paths
                           or [None] * len(group_maps))


def _multiref_pool_file(args):
    cz_path, g = args
    st = _MULTIREF_STATE
    # Re-point the shared worker state at this cell's group (only when the
    # group changes), then reuse the standard per-file worker.
    if st.get("_group") != g:
        _pool_init(st["features_by_chrom"], st["pos_col"], st["mc_col"],
                   st["cov_col"], st["group_refs"][g], st["group_idx"][g],
                   st["group_maps"][g], st["group_idx_paths"][g])
        st["_group"] = g
    return _pool_process_file(cz_path)


def _cz_to_anndata_multiref_impl(df, feat_df, features_by_chrom, n_feat,
                                 gtf_meta_df, chroms_set, output, pos_col,
                                 mc_col, cov_col, obs, score, score_cutoff,
                                 hvf_frac, nan_policy, jobs, max_mem, rep,
                                 verbose=False):
    # Group cells by (reference, index) so each reference position map and
    # context index is resolved once and reused across its cells.
    idx_col = df["index"].tolist() if "index" in df.columns else [None] * len(df)
    groups: dict = {}
    for p, ref, idx, lab in zip(df["path"], df["reference"], idx_col,
                                df["cell_id"]):
        groups.setdefault((ref, idx), []).append((p, lab))

    builder = _StreamingSparseBuilder(n_feat)
    obs_names: List[str] = []

    # Resolve every group's context index + site map ONCE in the parent
    # (one reference pass per group, released chrom by chrom); workers
    # receive them read-only and never load reference positions.
    group_keys = list(groups.keys())
    group_refs, group_idx, group_maps = [], [], []
    unit_size = 1
    group_idx_paths = []
    for ref_path, idx_path in _progress(group_keys, desc="reference groups",
                                        enabled=verbose, unit="ref"):
        first = groups[(ref_path, idx_path)][0][0]
        r0 = Reader(first, mmap_advise="random")
        try:
            has_pos = pos_col in r0.header["columns"]
            unit_size = max(unit_size, int(r0._unit_size))
        finally:
            r0.close()
        site_map = idx_dict = None
        if not has_pos:
            logger.info("[cytozip] resolving site -> feature map from "
                        f"{ref_path}")
            # index streamed into the map; position-column cells of this
            # group (mixed inputs) resolve the dict lazily in the worker
            site_map = _build_site_feature_map(ref_path, features_by_chrom,
                                               verbose=verbose, index=idx_path)
        elif idx_path is not None:
            idx_dict = _resolve_index_to_dict(idx_path, chroms=chroms_set)
        group_refs.append(ref_path)
        group_idx.append(idx_dict)
        group_idx_paths.append(idx_path)
        group_maps.append(site_map)

    est_worker = max(_estimate_worker_bytes(m, unit_size,
                                            [p for p, _ in groups[k]])
                     for k, m in zip(group_keys, group_maps))
    n_workers = _cap_workers(jobs, est_worker, max_mem=max_mem)
    rep.jobs = n_workers
    tasks = [(p, g) for g, k in enumerate(group_keys) for p, _ in groups[k]]
    labels = [lab for k in group_keys for _, lab in groups[k]]

    if n_workers > 1 and len(tasks) > 1:
        with _make_pool(n_workers, _multiref_pool_init,
                        (features_by_chrom, pos_col, mc_col, cov_col,
                         group_maps, group_refs, group_idx,
                         group_idx_paths)) as ex:
            for (cols_, mc_, cov_), lab in _progress(
                    zip(_bounded_map(ex, _multiref_pool_file, tasks,
                                     n_workers * 4), labels),
                    total=len(labels), desc="aggregating cells",
                    enabled=verbose, unit="cell"):
                builder.append_sparse(cols_, mc_, cov_)
                obs_names.append(lab)
    else:
        from .bam import _LazyRefPositions
        bar = _progress(total=len(labels), desc="aggregating cells",
                        enabled=verbose, unit="cell")
        for g, k in enumerate(group_keys):
            ref_path, _ = k
            site_map, idx_dict = group_maps[g], group_idx[g]
            rpm = None
            try:
                for p, label in groups[k]:
                    r = Reader(p, mmap_advise="sequential")
                    try:
                        cols = r.header["columns"]
                        pi = cols.index(pos_col) if pos_col in cols else None
                        mc_i = cols.index(mc_col)
                        cov_i = cols.index(cov_col)
                        chrom_axis = _detect_chrom_axis(r, chroms_set)
                        if pi is None and site_map is None and rpm is None:
                            rpm = _LazyRefPositions(ref_path)
                        if (pi is not None and idx_dict is None
                                and group_idx_paths[g] is not None):
                            idx_dict = _resolve_index_to_dict(
                                group_idx_paths[g], chroms=chroms_set)
                        arr = _aggregate_cell(
                            r, pi, mc_i, cov_i, chrom_axis=chrom_axis,
                            site_map=site_map, ref_pos_map=rpm,
                            features_by_chrom=features_by_chrom,
                            index_ids_by_chrom=idx_dict, n_feat=n_feat)
                    finally:
                        r.close()
                    builder.append(arr)
                    obs_names.append(label)
                    bar.update(1)
                    del arr
            finally:
                if rpm is not None:
                    rpm.close()
        bar.close()

    return _finalize_cz_anndata(builder, obs_names, feat_df, gtf_meta_df,
                                score, score_cutoff, hvf_frac, obs, output,
                                reference=sorted(set(df["reference"])),
                                nan_policy=nan_policy, max_mem=max_mem,
                                verbose=verbose)


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
        return {k: np.asarray(v) for k, v in index.items()}
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
            # int32 halves the footprint of genome-wide (CH) indexes; every
            # consumer re-casts / subtracts 1 in int64 anyway.
            out[chrom] = ids.astype(np.int32 if ids.size == 0
                                    or int(ids.max()) <= np.iinfo(np.int32).max
                                    else np.int64, copy=False)
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
