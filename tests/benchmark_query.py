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

    rows = []
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
        qn_arr = reader.query_numpy(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=str(REF_CZ))
        qn_n = 0 if qn_arr is None else len(qn_arr)
        t_qn = _warm_time(lambda: reader.query_numpy(
            chunk_key=REGION["chrom"],
            start=REGION["start"], end=REGION["end"],
            reference=str(REF_CZ)))
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

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["tool"]].append(r["time_s"])
    print(f"{'tool':<42} {'n_cells':>8} {'mean_time_s':>12}")
    for tool, ts in agg.items():
        mean = sum(ts) / len(ts)
        print(f"{tool:<42} {len(ts):>8} {mean:>12.6f}")


if __name__ == "__main__":
    main()
