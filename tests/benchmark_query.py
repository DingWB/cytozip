#!/usr/bin/env python
"""Benchmark region queries across all .cz files under ``output/cz/``.

For every cell we measure four backends on the same genomic region:

* ``tabix`` CLI on the original ``.allc.tsv.gz`` (cold — one subprocess),
* ``czip query`` CLI on the ref-aligned ``.cz`` (cold — one subprocess),
* In-process ``pytabix.open().query()`` on an already-open handle
  (warm — NQ repetitions, per-call average),
* In-process ``cytozip.Reader.query()`` on an already-open reader
  (warm — NQ repetitions, per-call average).

Outputs:

  cytozip_example_data/output/query_benchmark/query_benchmark.tsv
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# Prefer the in-repo cytozip over any pip-installed copy in site-packages.
sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "cytozip_example_data"
OUT = DATA / "output"
ALLC_DIR = OUT / "allc"
CZ_DIR = OUT / "cz"
BENCH = OUT / "query_benchmark"
REF_CZ = OUT / "mm10_with_chrL.allc.cz"

PY = sys.executable
CZIP = f"{Path(PY).parent}/czip"
TIME_BIN = "/usr/bin/time"
YAP_BIN = "/home/x-wding2/Software/conda/yap/bin"
TABIX = f"{YAP_BIN}/tabix"

# 5-kb region near the start of chr9 (matches the dnam notebook example).
REGION = dict(chrom="chr9", start=3000294, end=3005294)
NQ = 100  # repetitions for warm / in-process timings


def _time_cli(cmd):
    """Run ``cmd`` once under /usr/bin/time; return (ok, wall_s, rss_mb, n)."""
    wrapped = [TIME_BIN, "-f", "__TIME__ %e %M", "--"] + list(cmd)
    res = subprocess.run(wrapped, capture_output=True, text=True)
    wall_s, rss_mb = float("nan"), float("nan")
    for line in res.stderr.splitlines()[::-1]:
        if line.startswith("__TIME__"):
            parts = line.split()
            wall_s = float(parts[1])
            rss_mb = float(parts[2]) / 1024.0
            break
    n = res.stdout.count("\n") if res.returncode == 0 else -1
    return res.returncode == 0, wall_s, rss_mb, n


def _warm_time(fn, n=NQ):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    BENCH.mkdir(parents=True, exist_ok=True)

    cz_files = sorted(CZ_DIR.glob("*.cz"))
    if not cz_files:
        raise SystemExit(f"no .cz files under {CZ_DIR}")
    region_str = f"{REGION['chrom']}:{REGION['start']}-{REGION['end']}"
    print(f"[bench] region={region_str}  cells={len(cz_files)}  NQ={NQ}")

    import tabix as pytabix
    import cytozip as czip

    # Open the reference Reader ONCE — passing it as an object avoids
    # re-opening + re-mmapping the (~1.3 GB mm10) reference per call,
    # which previously dominated the warm query_numpy timing.
    ref_reader = czip.Reader(str(REF_CZ))

    rows = []
    correctness_rows = []
    for cz in cz_files:
        cid = cz.stem
        allc = ALLC_DIR / f"{cid}.allc.tsv.gz"
        if not allc.exists():
            print(f"[bench] skip {cid}: missing {allc.name}")
            continue

        tbi = Path(str(allc) + ".tbi")
        if not tbi.exists() or tbi.stat().st_mtime < allc.stat().st_mtime:
            subprocess.run([TABIX, "-f", "-s", "1", "-b", "2", "-e", "2",
                            str(allc)], check=True)

        # 1) tabix CLI (cold)
        ok, t, m, n = _time_cli([TABIX, str(allc), region_str])
        rows.append(dict(cell=cid, tool="tabix CLI (cold)",
                         time_s=t, peak_rss_mb=m, n=n))

        # 2) czip query CLI (cold)
        ok, t, m, n = _time_cli([CZIP, "query", "-I", str(cz),
                                 "-r", str(REF_CZ),
                                 "-D", REGION["chrom"],
                                 "-s", str(REGION["start"]),
                                 "-e", str(REGION["end"])])
        rows.append(dict(cell=cid, tool="czip query CLI (cold)",
                         time_s=t, peak_rss_mb=m, n=n))

        # 3) pytabix .query() (warm — handle already open)
        tb = pytabix.open(str(allc))
        tb_n = sum(1 for _ in tb.query(
            REGION["chrom"], REGION["start"], REGION["end"]))
        t_tb = _warm_time(lambda: list(tb.query(
            REGION["chrom"], REGION["start"], REGION["end"])))
        rows.append(dict(cell=cid, tool=f"pytabix .query() (warm avg/{NQ})",
                         time_s=t_tb, peak_rss_mb=float("nan"), n=tb_n))

        # 4) cytozip Reader.query() (warm — reader already open)
        reader = czip.Reader(str(cz))
        cz_n = sum(1 for _ in reader.query(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=str(REF_CZ), printout=False))
        t_cz = _warm_time(lambda: list(reader.query(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=str(REF_CZ), printout=False)))
        rows.append(dict(cell=cid, tool=f"cytozip Reader.query() (warm avg/{NQ})",
                         time_s=t_cz, peak_rss_mb=float("nan"), n=cz_n))

        # 5) cytozip Reader.query_numpy() (warm — vectorized, returns ndarray)
        # Prime the chunk caches once, then time pure searchsorted hot path.
        # When reference= is given, query_numpy returns (positions, records);
        # use len(records) for the hit count.
        qn_out = reader.query_numpy(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=ref_reader)
        qn_n = (0 if qn_out is None
                else (len(qn_out[1]) if isinstance(qn_out, tuple)
                      else len(qn_out)))
        t_qn = _warm_time(lambda: reader.query_numpy(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=ref_reader))

        # ---- Correctness check (besides count_fmt='B' truncation) ----
        # The cz files are written with --count_fmt B (uint8), so any
        # ALLC row with mc or cov >= 256 would be saturated at 255 in
        # the .cz. Anything else differing between pytabix and cytozip
        # is a genuine mismatch worth flagging.
        # NOTE: pytabix.query() uses 0-based half-open [start, end);
        # cytozip query_numpy uses 1-based inclusive [start, end].
        # Pass start-1 to pytabix so both span the same positions.
        tb_rows = list(tb.query(
            REGION["chrom"], REGION["start"] - 1, REGION["end"]))
        tb_map = {int(r[1]): (int(r[4]), int(r[5])) for r in tb_rows}
        cz_pos, cz_recs = qn_out  # (positions, structured ndarray)
        # Records have unnamed fields (f0=mc, f1=cov per --count_fmt B,B
        # written by bam_to_cz mode=mc_cov).
        mc_field, cov_field = cz_recs.dtype.names[:2]
        cz_map = {int(p): (int(rec[mc_field]), int(rec[cov_field]))
                  for p, rec in zip(cz_pos, cz_recs)}
        truncated = mismatched = 0
        examples = []
        for p, (mc_a, cov_a) in tb_map.items():
            cz_v = cz_map.get(p)
            if cz_v is None:
                mismatched += 1
                if len(examples) < 3:
                    examples.append((p, (mc_a, cov_a), None))
                continue
            mc_z, cov_z = cz_v
            if (mc_z, cov_z) == (min(mc_a, 255), min(cov_a, 255)):
                if mc_a >= 256 or cov_a >= 256:
                    truncated += 1
            else:
                mismatched += 1
                if len(examples) < 3:
                    examples.append((p, (mc_a, cov_a), (mc_z, cov_z)))
        only_in_cz_covered = sum(1 for p, v in cz_map.items()
                                 if p not in tb_map and v != (0, 0))
        only_in_cz_zero = sum(1 for p, v in cz_map.items()
                              if p not in tb_map and v == (0, 0))
        msg = (f"[check] {cid}  pytabix={len(tb_map)}  cz={len(cz_map)}  "
               f"truncated(>=256)={truncated}  mismatched={mismatched}  "
               f"only_in_cz_zero={only_in_cz_zero}  "
               f"only_in_cz_covered={only_in_cz_covered}")
        if examples:
            msg += f"  examples={examples}"
        print(msg)

        correctness_rows.append(dict(
            cell=cid,
            region=region_str,
            pytabix_n=len(tb_map),
            cz_n=len(cz_map),
            truncated_ge256=truncated,
            mismatched=mismatched,
            only_in_cz_zero=only_in_cz_zero,
            only_in_cz_covered=only_in_cz_covered,
        ))

        reader.close()
        rows.append(dict(cell=cid, tool=f"cytozip Reader.query_numpy() (warm avg/{NQ})",
                         time_s=t_qn, peak_rss_mb=float("nan"), n=qn_n))

        print(f"[bench] {cid}  tabix={rows[-5]['time_s']:.3f}s  "
              f"czip_cli={rows[-4]['time_s']:.3f}s  "
              f"pytabix_warm={t_tb*1e3:.2f}ms  "
              f"czip_warm={t_cz*1e3:.2f}ms  "
              f"qnumpy_warm={t_qn*1e6:.1f}us")

    tsv = BENCH / "query_benchmark.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "tool", "time_s",
                                          "peak_rss_mb", "n"],
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({"cell": r["cell"], "tool": r["tool"],
                        "time_s": f"{r['time_s']:.6f}",
                        "peak_rss_mb": ("" if r["peak_rss_mb"] != r["peak_rss_mb"]
                                        else f"{r['peak_rss_mb']:.1f}"),
                        "n": r["n"]})
    print(f"\n[bench] wrote {tsv}\n")

    if correctness_rows:
        ctsv = BENCH / "query_correctness.tsv"
        with ctsv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(correctness_rows[0].keys()),
                               delimiter="\t")
            w.writeheader()
            for r in correctness_rows:
                w.writerow(r)
        n_bad = sum(r['mismatched'] for r in correctness_rows)
        n_only = sum(r['only_in_cz_covered'] for r in correctness_rows)
        n_trunc = sum(r['truncated_ge256'] for r in correctness_rows)
        print(f"[check] cells={len(correctness_rows)}  total_truncated(>=256)={n_trunc}  "
              f"total_mismatched={n_bad}  total_only_in_cz_covered={n_only}")
        print(f"[check] wrote {ctsv}\n")

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["tool"]].append(r["time_s"])
    print(f"{'tool':<42} {'n_cells':>8} {'mean_time_s':>12}")
    for tool, ts in agg.items():
        mean = sum(ts) / len(ts)
        print(f"{tool:<42} {len(ts):>8} {mean:>12.6f}")

    # --------------------------------------------------------------
    # Region-size scaling benchmark (single representative cell)
    # --------------------------------------------------------------
    # For warm queries the per-record Python overhead in pytabix grows
    # ~linearly with N hits, while query_numpy is essentially a chunk
    # decode (cached after first call) + np.searchsorted, so it should
    # become *relatively* faster as the queried region grows.
    cell_path = cz_files[len(cz_files) // 2]
    cid = cell_path.stem
    allc = ALLC_DIR / f"{cid}.allc.tsv.gz"
    if allc.exists():
        # Use chr9 instead of chr1: it sits much further down the
        # reference .cz on disk, so any cold-cache effect would show up
        # more clearly. (Warm timings here should still be steady.)
        chrom = "chr9"
        center = 50_000_000
        # Region half-widths (bp); region size = 2 * half + 1.
        half_widths = [2_500, 25_000, 250_000, 2_500_000, 25_000_000]
        scale_rows = []

        tb = pytabix.open(str(allc))
        reader = czip.Reader(str(cell_path))

        # Prime caches once on the largest region so the searchsorted
        # path is purely warm-CPU for every measurement below.
        max_w = max(half_widths)
        reader.query_numpy(chunk_key=chrom,
                           start=center - max_w, end=center + max_w,
                           reference=ref_reader)

        print(f"\n[bench-scale] cell={cid}  chrom={chrom}  center={center}  reps=10 (per-row)")
        N_REPLICATES = 10  # rows per size: each row is its own _warm_time avg
        for hw in half_widths:
            s, e = center - hw, center + hw
            size_bp = e - s

            tb_n = sum(1 for _ in tb.query(chrom, s, e))
            # Inner repetitions per replicate: larger regions => fewer.
            n_rep = 100 if hw <= 250_000 else (20 if hw <= 2_500_000 else 5)

            for rep in range(N_REPLICATES):
                t_tb = _warm_time(
                    lambda s=s, e=e: list(tb.query(chrom, s, e)), n=n_rep)
                t_qn = _warm_time(
                    lambda s=s, e=e: reader.query_numpy(
                        chunk_key=chrom, start=s, end=e,
                        reference=ref_reader),
                    n=n_rep)
                speedup = (t_tb / t_qn) if t_qn > 0 else float("nan")
                scale_rows.append(dict(
                    cell=cid, chrom=chrom, start=s, end=e, size_bp=size_bp,
                    n_records=tb_n, n_rep=n_rep, replicate=rep,
                    pytabix_time_s=t_tb, qnumpy_time_s=t_qn,
                    speedup=speedup,
                ))

            # Print summary across replicates for this size.
            tb_arr = np.array([r['pytabix_time_s'] for r in scale_rows
                               if r['size_bp'] == size_bp])
            qn_arr = np.array([r['qnumpy_time_s'] for r in scale_rows
                               if r['size_bp'] == size_bp])
            print(f"[bench-scale] size={size_bp:>10} bp  N={tb_n:>7}  "
                  f"pytabix={tb_arr.mean()*1e3:>9.3f}\u00b1{tb_arr.std()*1e3:.3f} ms  "
                  f"qnumpy={qn_arr.mean()*1e3:>9.3f}\u00b1{qn_arr.std()*1e3:.3f} ms")
        reader.close()

        tsv2 = BENCH / "query_scale_benchmark.tsv"
        with tsv2.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys()),
                               delimiter="\t")
            w.writeheader()
            for r in scale_rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"[bench-scale] wrote {tsv2}")

    ref_reader.close()


if __name__ == "__main__":
    main()
