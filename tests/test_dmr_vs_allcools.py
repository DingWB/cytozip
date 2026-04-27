"""Sanity test: cytozip.call_dmr vs ALLCools RMS test.

Runs cytozip's DMR caller on the example data and compares per-site
p-values against ALLCools' ``permute_root_mean_square_test`` kernel
applied to the IDENTICAL contingency tables.

Both sides use random permutations, so exact p-values won't match;
we check:
  * cytozip output exists with valid schema
  * sign of frac_delta agrees site-by-site (deterministic)
  * Pearson correlation of -log10(p) is high (> 0.7)
  * fraction of sites where both call DMS @ p<=0.01 agrees (> 0.7)

Run:  python tests/test_dmr_vs_allcools.py
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cytozip as czip
from cytozip import Reader

# ALLCools kernel
from ALLCools.dmr.rms_test import (
    downsample_table,
    init_rms_functions,
    permute_root_mean_square_test,
)


HERE = Path(__file__).resolve().parent
EX = HERE.parent / "cytozip_example_data" / "output"
CELLS = sorted(glob.glob(str(EX / "cz" / "*.cz")))
REF = str(EX / "mm10_with_chrL.allc.cz")
CHROMS = ["chrL", "chrM"]
N_PERMUTE = 3000
MIN_PVALUE = 0.01
MAX_ROW_COUNT = 50
MAX_TOTAL_COUNT = 3000
MIN_COV = 1
MIN_SAMPLES = 2


def part1_run_czip(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cytozip.call_dmr and assert output schema."""
    print("=" * 70)
    print(f"[part 1] running cytozip.call_dmr on {len(CELLS)} cells")
    a, b = CELLS[:5], CELLS[5:]
    dmr_path = out_dir / "cz_dmr.tsv"
    dms_path = out_dir / "cz_dms.tsv"
    t0 = time.time()
    czip.call_dmr(
        group_a=a,
        group_b=b,
        reference=REF,
        output=str(dmr_path),
        dms_output=str(dms_path),
        min_cov=MIN_COV,
        min_samples_per_group=MIN_SAMPLES,
        p_value_cutoff=0.05,
        frac_delta_cutoff=0.3,
        n_permute=N_PERMUTE,
        min_pvalue=MIN_PVALUE,
        max_row_count=MAX_ROW_COUNT,
        max_total_count=MAX_TOTAL_COUNT,
        chroms=CHROMS,
        jobs=4,
    )
    print(f"[part 1] elapsed: {time.time()-t0:.1f}s")

    dmr = pd.read_csv(dmr_path, sep="\t")
    dms = pd.read_csv(dms_path, sep="\t")

    expected_dmr = {"chrom", "start", "end", "n_dms",
                    "p_min", "frac_delta_mean", "state"}
    expected_dms = {"chrom", "pos", "p", "delta"}
    assert expected_dmr.issubset(dmr.columns), \
        f"DMR columns missing: {expected_dmr - set(dmr.columns)}"
    assert expected_dms.issubset(dms.columns), \
        f"DMS columns missing: {expected_dms - set(dms.columns)}"
    assert len(dms) > 0, "no DMS produced"
    assert len(dmr) > 0, "no DMR produced"
    assert (dmr["start"] < dmr["end"]).all(), "invalid BED coords"
    assert (dmr["n_dms"] >= 1).all(), "n_dms < 1"
    assert dmr["state"].isin([-1, 0, 1]).all(), "bad state values"
    print(f"[part 1] OK: {len(dms)} DMS -> {len(dmr)} DMRs, schema valid")
    return dms, dmr


def _load_site_matrix(chrom: str):
    """Return (pos, mc[N,L], cov[N,L]) for all 9 cells on `chrom`."""
    ref = Reader(REF)
    ref_arr = ref.chunk2numpy((chrom,))
    # ref dtype uses ('f0','f1','f2') after structured cast — use first field
    pos = np.asarray(ref_arr[ref_arr.dtype.names[0]], dtype=np.int64)
    n_sites = pos.size
    ref.close()

    n_cells = len(CELLS)
    mc = np.zeros((n_cells, n_sites), dtype=np.int64)
    cov = np.zeros((n_cells, n_sites), dtype=np.int64)
    # per-cell cz files are aligned 1:1 to the reference chunk (same length)
    for i, cz in enumerate(CELLS):
        rd = Reader(cz)
        if (chrom,) in rd.chunk_key2offset:
            arr = rd.chunk2numpy((chrom,))
            if arr.shape[0] == n_sites:
                mc[i] = np.asarray(arr[arr.dtype.names[0]], dtype=np.int64)
                cov[i] = np.asarray(arr[arr.dtype.names[1]], dtype=np.int64)
        rd.close()
    return pos, mc, cov


def part2_compare_rms(dms_df: pd.DataFrame, max_sites: int = 500) -> dict:
    """Recompute p-values with ALLCools kernel on identical tables.

    We sample czip's DMS sites (already passing min_cov filter) and run
    ALLCools' permute_root_mean_square_test on the same contingency table.
    """
    print("=" * 70)
    print(f"[part 2] re-running ALLCools RMS on up to {max_sites} sites")
    init_rms_functions()  # warm-up numba

    # build per-chrom site matrices once
    chrom2data = {c: _load_site_matrix(c) for c in CHROMS}

    # subset to czip-reported DMS sites
    rng = np.random.default_rng(0)
    dms = dms_df.copy()
    if len(dms) > max_sites:
        dms = dms.sample(n=max_sites, random_state=0).reset_index(drop=True)

    cz_p = dms["p"].to_numpy()
    cz_d = dms["delta"].to_numpy()
    ac_p = np.full(len(dms), np.nan)
    ac_d = np.full(len(dms), np.nan)

    for i, row in dms.iterrows():
        chrom = row["chrom"]
        pos_target = int(row["pos"])
        pos, mc, cov = chrom2data[chrom]
        # binary search
        j = np.searchsorted(pos, pos_target)
        if j >= pos.size or pos[j] != pos_target:
            continue
        mc_col = mc[:, j]
        cov_col = cov[:, j]
        # build [mc, c=cov-mc] table; group A first 5, group B remaining
        table = np.column_stack([mc_col, cov_col - mc_col]).astype(np.float64)
        table = downsample_table(
            table,
            max_row_count=MAX_ROW_COUNT,
            max_total_count=MAX_TOTAL_COUNT,
        )
        p = permute_root_mean_square_test(
            table, n_permute=N_PERMUTE, min_pvalue=MIN_PVALUE
        )
        ac_p[i] = p
        # frac_delta: same definition as cytozip
        # mean(mc_a/cov_a) - mean(mc_b/cov_b), only where cov>0
        with np.errstate(divide="ignore", invalid="ignore"):
            f = mc_col / np.where(cov_col > 0, cov_col, 1)
            f[cov_col == 0] = np.nan
        a_mean = np.nanmean(f[:5])
        b_mean = np.nanmean(f[5:])
        ac_d[i] = a_mean - b_mean

    valid = ~np.isnan(ac_p)
    n_valid = int(valid.sum())
    assert n_valid >= 10, f"too few valid sites recomputed ({n_valid})"

    # sign agreement
    sign_match = float(np.mean(
        np.sign(cz_d[valid]) == np.sign(ac_d[valid])
    ))
    # delta numeric match (should be exact — same data, same formula)
    delta_max_abs = float(np.max(np.abs(cz_d[valid] - ac_d[valid])))
    # p-value correlation on -log10
    eps = 1e-6
    cz_lp = -np.log10(np.clip(cz_p[valid], eps, 1.0))
    ac_lp = -np.log10(np.clip(ac_p[valid], eps, 1.0))
    if cz_lp.std() < 1e-9 or ac_lp.std() < 1e-9:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(cz_lp, ac_lp)[0, 1])
    # call agreement at p<=MIN_PVALUE (czip reports only p<=cutoff in DMS)
    cz_call = cz_p[valid] <= MIN_PVALUE
    ac_call = ac_p[valid] <= MIN_PVALUE
    call_agree = float(np.mean(cz_call == ac_call))

    print(f"[part 2] valid sites compared: {n_valid}")
    print(f"[part 2] frac_delta sign agreement : {sign_match:.3f}")
    print(f"[part 2] frac_delta max |diff|     : {delta_max_abs:.3e}")
    print(f"[part 2] -log10(p) Pearson r       : {pearson:.3f}")
    print(f"[part 2] DMS-call agreement (p<={MIN_PVALUE}): {call_agree:.3f}")
    print(f"[part 2] cz   p median: {np.median(cz_p[valid]):.4f}")
    print(f"[part 2] ALLC p median: {np.median(ac_p[valid]):.4f}")

    # assertions
    assert sign_match >= 0.99, f"frac_delta sign mismatch ({sign_match:.3f})"
    assert delta_max_abs < 1e-6, \
        f"frac_delta numeric mismatch ({delta_max_abs:.3e})"
    assert np.isnan(pearson) or pearson >= 0.7, \
        f"p-value correlation too low ({pearson:.3f})"
    assert call_agree >= 0.7, \
        f"DMS-call agreement too low ({call_agree:.3f})"
    return dict(
        n=n_valid,
        sign=sign_match,
        delta_max=delta_max_abs,
        pearson=pearson,
        call_agree=call_agree,
    )


def main() -> int:
    assert len(CELLS) == 9, f"expected 9 example cells, got {len(CELLS)}"
    assert os.path.exists(REF), f"reference missing: {REF}"
    with tempfile.TemporaryDirectory(prefix="cz_dmr_test_") as tmp:
        tmp = Path(tmp)
        dms, dmr = part1_run_czip(tmp)
        stats = part2_compare_rms(dms, max_sites=500)
    print("=" * 70)
    print("PASS:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
