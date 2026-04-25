#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features.py - Build cell x feature count matrices / AnnData from .cz files.

Public entry points:

* :func:`cz_to_anndata` - aggregate many single-cell ``.cz`` files (or one
  ``catcz``-merged ``.cz``) over a BED / DataFrame / genome-bins feature
  set and emit a single :class:`anndata.AnnData` with ``mc`` / ``cov``
  layers. ``.X`` holds a user-selected score (raw fraction, posterior
  fraction normalized by the cell's prior mean a la ALLCools, or a
  binomial hypo / hyper score).
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
from typing import Iterable, List, Optional, Sequence, Union

from loguru import logger
import numpy as np
import pandas as pd

from .cz import Reader, _fmt_to_np_dtype


_VALID_SCORES = ("frac", "hypo-score", "hyper-score")


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
    returns a DataFrame whose 4th column (``name``) is guaranteed unique
    so it can be passed straight to :func:`cz_to_anndata` as
    ``features=``.

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
        Which GTF attribute becomes the unique feature ``name`` and
        ``var_names`` on the final AnnData. When ``'gene_name'`` (the
        default) and multiple gene_ids share the same name on the same
        chrom, their intervals are merged (min start / max end) so the
        output has one row per gene symbol; any residual duplicate on a
        different chrom is kept as-is by falling back to a
        ``<name>|<chrom>`` suffix to preserve uniqueness.
    exclude_chroms : sequence of str, optional
        Chromosomes to drop (e.g. ``['chrM']``).

    Returns
    -------
    DataFrame with columns
    ``['chrom', 'start', 'end', 'name', 'gene_id', 'gene_name',
    'gene_type', 'strand']``. ``name`` equals the chosen ``id_col``
    (``gene_name`` by default) and is unique.
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

    # Collapse records sharing the chosen id (gene_name by default) on
    # the same chrom into a single interval spanning all of them. This
    # is the fix for GENCODE entries where one gene_name is split over
    # multiple gene_ids (e.g. readthrough / PAR-aliased records).
    agg = (df.groupby(["chrom", id_col], sort=False, as_index=False)
             .agg(start=("start", "min"),
                  end=("end", "max"),
                  gene_id=("gene_id", "first"),
                  gene_name=("gene_name", "first"),
                  gene_type=("gene_type", "first"),
                  strand=("strand", "first")))
    # Apply flanking after the merge so flanking is symmetric per gene.
    if flank_bp and int(flank_bp) > 0:
        fb = int(flank_bp)
        agg["start"] = (agg["start"] - fb).clip(lower=0)
        agg["end"] = agg["end"] + fb

    # Guarantee uniqueness of `name`: if the same id survives on more
    # than one chrom (e.g. PAR), disambiguate by appending '|<chrom>'.
    name_counts = agg[id_col].value_counts()
    dup = set(name_counts.index[name_counts > 1])
    agg["name"] = np.where(
        agg[id_col].isin(dup),
        agg[id_col].astype(str) + "|" + agg["chrom"].astype(str),
        agg[id_col].astype(str),
    )
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
def _compute_beta_params(mc_mat, cov_mat):
    """Per-cell method-of-moments Beta(alpha, beta) from raw mc/cov.

    Mirrors ALLCools ``calculate_posterior_mc_frac`` without applying the
    posterior transform. Returns ``(alpha, beta, prior_mean)`` as
    ``(n_cells,)`` float32 vectors. Rows whose raw fraction is
    degenerate (all-NaN, zero variance, mean outside ``(0, 1)``) are
    flagged with NaN so downstream code can identify them.

    Writing these to ``adata.obs`` lets users reconstruct the posterior
    fraction later:

    .. code-block:: python

        a = adata.obs["alpha"].values[:, None]
        b = adata.obs["beta"].values[:, None]
        prior = adata.obs["prior_mean"].values[:, None]
        mc = adata.layers["mc"].toarray()
        cov = adata.layers["cov"].toarray()
        post = (mc + a) / (cov + a + b) / prior  # ALLCools post_frac
    """
    mc = mc_mat.astype(np.float64, copy=False)
    cov = cov_mat.astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(cov > 0, mc / np.maximum(cov, 1), np.nan)
    with np.errstate(invalid="ignore"), \
            np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        mean = np.nanmean(raw, axis=1)
        var = np.nanvar(raw, axis=1)
    ok = (np.isfinite(mean) & np.isfinite(var)
          & (mean > 0) & (mean < 1) & (var > 0))
    alpha = np.full(mc.shape[0], np.nan, dtype=np.float64)
    beta = np.full(mc.shape[0], np.nan, dtype=np.float64)
    if ok.any():
        m = mean[ok]
        v = var[ok]
        a = (1.0 - m) * m * m / v - m
        b = a * (1.0 / m - 1.0)
        alpha[ok] = np.maximum(a, 1e-6)
        beta[ok] = np.maximum(b, 1e-6)
    prior_mean = alpha / (alpha + beta)
    return (alpha.astype(np.float32),
            beta.astype(np.float32),
            prior_mean.astype(np.float32))


def _compute_score_matrix(mc_mat, cov_mat, score, score_cutoff):
    """Fully-vectorized dispatch to the requested scoring function.

    Supported scores: ``'frac'``, ``'hypo-score'``, ``'hyper-score'``.
    The ALLCools posterior-fraction transform is intentionally *not*
    computed here; per-cell Beta(alpha, beta) parameters are instead
    written to ``adata.obs`` by :func:`cz_to_anndata` so users can
    reconstruct the posterior fraction on demand.
    """
    if score == "frac":
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(cov_mat > 0,
                         mc_mat.astype(np.float32) / np.maximum(cov_mat, 1),
                         0.0).astype(np.float32)
        return x

    mc = mc_mat.astype(np.float64, copy=False)
    cov = cov_mat.astype(np.float64, copy=False)

    if score in ("hypo-score", "hyper-score"):
        from scipy.stats import binom
        tot_mc = mc.sum(axis=1)
        tot_cov = cov.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = tot_mc / (tot_cov + 1e-6)
        valid = np.isfinite(p) & (p > 0) & (p < 1) & (tot_cov > 0)
        out = np.zeros_like(mc, dtype=np.float32)
        if valid.any():
            # binom.sf broadcasts (k, n_feat) vs (k, 1) -> (k, n_feat).
            sf = binom.sf(mc[valid], cov[valid], p[valid, None])
            pv = (1.0 - sf) if score == "hyper-score" else sf
            pv = np.where(cov[valid] > 0, pv, 0.0)
            pv = np.where(pv >= float(score_cutoff), pv, 0.0)
            out[valid] = pv.astype(np.float32)
        return out

    raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")


# ---------------------------------------------------------------------------
# Per-reader aggregation (numpy fast path)
# ---------------------------------------------------------------------------
def _detect_chrom_axis(reader: Reader, known_chroms: set) -> int:
    """Find which position of the dim tuple holds the chrom.

    Different producers stamp the chrom in different slots:

    * Single-cell (``chunk_dims=['chrom']``): position 0.
    * ``catcz(add_key=True)``: appends the new key at the *end* of the
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
) -> np.ndarray:
    """Return shape ``(n_features, 2)`` int64 array ``[sum_mc, sum_cov]``.

    Fast path: per-chrom ``fetch_chunk_bytes`` + ``np.frombuffer`` decode,
    then ``np.searchsorted`` on the sorted position array to sum each
    region via cumulative-sum subtraction.

    Parameters
    ----------
    pos_col : int or None
        Index of the ``pos`` column in the record layout. ``None`` for
        ``[mc, cov]``-only cells that rely on ``ref_pos_map``.
    cell_prefix : tuple, optional
        Non-chrom dim values (in their positional order) identifying the
        cell in a merged reader. For a single-cell reader, pass ``None``.
    chrom_axis : int, optional
        Index of the chrom slot in the reader's dim tuple (see
        :func:`_detect_chrom_axis`). Defaults to the *last* position.
    ref_pos_map : dict, optional
        ``{chrom: int64 positions}`` for reference-aligned cells.
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
            positions = ref_pos_map[chrom]
            if positions.size != arr.size:
                continue
        else:
            continue

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


def _pool_init(features_by_chrom, pos_col, mc_col, cov_col, reference):
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


def _pool_get_ref_pos_map(hint_path=None):
    if _WORKER_STATE["ref_pos_map"] is not None:
        return _WORKER_STATE["ref_pos_map"]
    ref_path = _WORKER_STATE["reference"] or hint_path
    if ref_path is None:
        raise ValueError(
            "Cell .cz has no 'pos' column; pass reference= to "
            "provide the coordinate reference.")
    from .bam import _load_reference_positions
    _WORKER_STATE["ref_pos_map"] = _load_reference_positions(ref_path)
    return _WORKER_STATE["ref_pos_map"]


def _pool_process_file(cz_path):
    """Worker entry: aggregate a whole per-cell ``.cz`` file."""
    fbc = _WORKER_STATE["features_by_chrom"]
    r = Reader(cz_path)
    try:
        cols = r.header["columns"]
        pi = cols.index(_WORKER_STATE["pos_col"]) \
            if _WORKER_STATE["pos_col"] in cols else None
        mc_i = cols.index(_WORKER_STATE["mc_col"])
        cov_i = cols.index(_WORKER_STATE["cov_col"])
        rpm = _pool_get_ref_pos_map(r.header.get("message")) \
            if pi is None else None
        chrom_axis = _detect_chrom_axis(r, _WORKER_STATE["chroms_set"])
        arr = _aggregate_one_reader(r, fbc, pi, mc_i, cov_i,
                                    chrom_axis=chrom_axis, ref_pos_map=rpm)
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
        r = Reader(cz_path)
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
        ref_pos_map=rpm)
    return arr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def cz_to_anndata(
    cz_inputs: Union[str, Sequence[str]],
    features: Union[str, pd.DataFrame, int],
    output: Optional[str] = None,
    cell_ids: Optional[Sequence[str]] = None,
    pos_col: str = "pos",
    mc_col: str = "mc",
    cov_col: str = "cov",
    obs: Optional[pd.DataFrame] = None,
    reference: Optional[str] = None,
    chrom_size: Optional[Union[str, pd.DataFrame, dict]] = None,
    exclude_chroms: Optional[Sequence[str]] = ("chrL",),
    blacklist: Optional[Union[str, pd.DataFrame]] = None,
    flank_bp: int = 2000,
    gtf_id_col: str = "gene_name",
    score: str = "frac",
    score_cutoff: float = 0.9,
    threads: int = 1,
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
    cell_ids : list, optional
        Override cell ids. For per-file input the default is the file's
        basename with ``.cz`` stripped; for merged input the default is
        the first chunk-key value.
    pos_col, mc_col, cov_col : str
        Column names in the ``.cz`` header to use for position / methylated
        count / coverage.
    obs : DataFrame, optional
        Per-cell metadata (cluster, sample, donor, ...). Joined on cell id.
    reference : str, optional
        Reference ``.cz`` supplying positions for cells written with
        ``bam_to_cz(mode='mc_cov')`` / ``allc2cz(reference=...)``.
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
    score : {'frac', 'hypo-score', 'hyper-score'}
        What to store in ``.X``:

        - ``'frac'`` (default): raw ``mc/cov`` fraction. Zero-cov
          features are ``0``.
        - ``'hypo-score'``: per-cell binomial survival function
          ``P(X > mc | Binomial(cov, p_cell))``, with ``p_cell = total_mc /
          total_cov`` for that cell. Values below ``score_cutoff`` are set
          to zero. High values mark hypo-methylated sites.
        - ``'hyper-score'``: ``1 - sf`` with the same sparsification;
          high values mark hyper-methylated sites.

        The ALLCools posterior-fraction transform is intentionally *not*
        offered as a score; instead, per-cell Beta(alpha, beta) are
        written to ``adata.obs`` as ``['alpha', 'beta', 'prior_mean']``
        so users can recover the posterior fraction downstream:
        ``(mc + alpha) / (cov + alpha + beta) / prior_mean``.
    score_cutoff : float
        Sparsification threshold for hypo/hyper scores. Default 0.9.
    threads : int
        Number of worker processes for parallel per-cell aggregation.
        ``1`` (default) runs serially in-process. ``>1`` uses a
        :class:`concurrent.futures.ProcessPoolExecutor`: for a list of
        per-cell ``.cz`` files each worker opens its own Reader and
        processes one cell; for a merged ``.cz`` each worker handles
        a subset of cell prefixes. ``0`` or negative falls back to
        serial.

    Returns
    -------
    anndata.AnnData
        ``.X`` holds the requested score (dense ``float32``).
        ``.layers['mc']`` and ``.layers['cov']`` hold the raw integer
        counts (``uint32``) as CSR sparse matrices.
    """
    import anndata
    import scipy.sparse as ss

    if score not in _VALID_SCORES:
        raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")

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
        for i, row in feat_df.iterrows():
            c = row["chrom"]
            if c not in features_by_chrom:
                features_by_chrom[c] = {"starts": [], "ends": [],
                                        "indices": []}
            features_by_chrom[c]["starts"].append(int(row["start"]))
            features_by_chrom[c]["ends"].append(int(row["end"]))
            features_by_chrom[c]["indices"].append(i)

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
        from .bam import _load_reference_positions
        ref_pos_map_cache["map"] = _load_reference_positions(ref_path)
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
    n_workers = int(threads) if threads and int(threads) > 1 else 1

    cell_arrays = []
    obs_names: List[str] = []

    def _run_parallel_files(paths_):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_pool_init,
                initargs=(features_by_chrom, pos_col, mc_col, cov_col,
                          reference)) as ex:
            chunk = max(1, len(paths_) // (n_workers * 4) or 1)
            for arr in ex.map(_pool_process_file, paths_, chunksize=chunk):
                yield arr

    def _run_parallel_prefixes(cz_path, prefix_list):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_pool_init,
                initargs=(features_by_chrom, pos_col, mc_col, cov_col,
                          reference)) as ex:
            args = [(cz_path, pref) for pref in prefix_list]
            chunk = max(1, len(args) // (n_workers * 4) or 1)
            for arr in ex.map(_pool_process_prefix, args, chunksize=chunk):
                yield arr

    if len(paths) == 1:
        r = Reader(paths[0])
        try:
            n_keys = len(r.header["chunk_dims"])
            pi, mc_i, cov_i = _resolve_cols(r)
            rpm = _get_ref_pos_map(r.header.get("message")) if pi is None else None
            chrom_axis = _detect_chrom_axis(r, set(features_by_chrom.keys()))
            if n_keys >= 2:
                cell_prefixes = list(
                    _enumerate_cell_prefixes(r, chrom_axis,
                                             cell_ids=cell_ids))
                prefixes = [p for p, _ in cell_prefixes]
                labels = [lab for _, lab in cell_prefixes]
                if n_workers > 1 and len(prefixes) > 1:
                    r.close()  # workers reopen independently
                    for arr, label in zip(
                            _run_parallel_prefixes(paths[0], prefixes),
                            labels):
                        cell_arrays.append(arr)
                        obs_names.append(label)
                    r = None
                else:
                    for prefix, label in cell_prefixes:
                        arr = _aggregate_one_reader(
                            r, features_by_chrom, pi, mc_i, cov_i,
                            cell_prefix=prefix, chrom_axis=chrom_axis,
                            ref_pos_map=rpm)
                        cell_arrays.append(arr)
                        obs_names.append(label)
            else:
                arr = _aggregate_one_reader(r, features_by_chrom,
                                            pi, mc_i, cov_i,
                                            chrom_axis=chrom_axis,
                                            ref_pos_map=rpm)
                cell_arrays.append(arr)
                label = (cell_ids[0] if cell_ids else
                         os.path.basename(paths[0]).rsplit(".cz", 1)[0])
                obs_names.append(label)
        finally:
            if r is not None:
                r.close()
    else:
        labels = [
            (cell_ids[i] if cell_ids and i < len(cell_ids) else
             os.path.basename(p).rsplit(".cz", 1)[0])
            for i, p in enumerate(paths)
        ]
        if n_workers > 1:
            for arr, label in zip(_run_parallel_files(paths), labels):
                cell_arrays.append(arr)
                obs_names.append(label)
        else:
            for p, label in zip(paths, labels):
                r = Reader(p)
                try:
                    pi, mc_i, cov_i = _resolve_cols(r)
                    rpm = _get_ref_pos_map(r.header.get("message")) \
                        if pi is None else None
                    chrom_axis = _detect_chrom_axis(
                        r, set(features_by_chrom.keys()))
                    arr = _aggregate_one_reader(
                        r, features_by_chrom, pi, mc_i, cov_i,
                        chrom_axis=chrom_axis, ref_pos_map=rpm)
                finally:
                    r.close()
                cell_arrays.append(arr)
                obs_names.append(label)

    # Stack into dense -> sparse matrices.
    n_cells = len(cell_arrays)
    mc_mat = np.zeros((n_cells, n_feat), dtype=np.uint32)
    cov_mat = np.zeros((n_cells, n_feat), dtype=np.uint32)
    for i, arr in enumerate(cell_arrays):
        mc_mat[i] = arr[:, 0].astype(np.uint32)
        cov_mat[i] = arr[:, 1].astype(np.uint32)

    mc_sp = ss.csr_matrix(mc_mat)
    cov_sp = ss.csr_matrix(cov_mat)

    # Build var_df. For GTF inputs we attach gene_id / gene_type /
    # strand alongside the coordinates; for BED / bins var carries just
    # the coordinates. `name` is always used verbatim as var_names.
    var_df = feat_df.set_index("name")
    if gtf_meta_df is not None:
        var_df = var_df.join(gtf_meta_df)

    X = _compute_score_matrix(mc_mat, cov_mat, score,
                              score_cutoff=score_cutoff)

    # Per-cell Beta(alpha, beta) + prior_mean for ALLCools-style
    # posterior fraction reconstruction downstream.
    alpha, beta, prior_mean = _compute_beta_params(mc_mat, cov_mat)

    obs_df = pd.DataFrame(index=obs_names)
    obs_df["alpha"] = alpha
    obs_df["beta"] = beta
    obs_df["prior_mean"] = prior_mean
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
    }
    if output:
        adata.write_h5ad(os.path.abspath(os.path.expanduser(output)))
    return adata


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_inputs(cz_inputs) -> List[str]:
    if isinstance(cz_inputs, str):
        path = os.path.abspath(os.path.expanduser(cz_inputs))
        if os.path.isdir(path):
            return sorted(os.path.join(path, f) for f in os.listdir(path)
                          if f.endswith(".cz"))
        return [path]
    return [os.path.abspath(os.path.expanduser(p)) for p in cz_inputs]


def _enumerate_cell_prefixes(reader: Reader, chrom_axis: int, cell_ids=None):
    """Yield ``(cell_prefix_tuple, label)`` pairs for each cell.

    The cell prefix is the dim tuple with the ``chrom_axis`` slot removed,
    preserving the original ordering of the other slots.
    """
    n_keys = len(reader.header["chunk_dims"])
    if n_keys < 2:
        return
    seen = set()
    for dim in reader.chunk_key2offset.keys():
        prefix = tuple(v for j, v in enumerate(dim) if j != chrom_axis)
        seen.add(prefix)
    selected = sorted(seen)
    id_set = set(cell_ids) if cell_ids is not None else None
    for prefix in selected:
        label = "/".join(prefix) if len(prefix) > 1 else prefix[0]
        if id_set is not None and label not in id_set:
            continue
        yield prefix, label
