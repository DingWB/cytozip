#!/usr/bin/env python
"""Benchmark ALLCools ``bam_to_allc`` vs cytozip ``bam_to_cz``.

Runs both tools on every ``*.hisat3n_dna.all_reads.deduped.bam`` under
``cytozip_example_data/bam/`` as isolated subprocesses, with wall-clock
and peak-RSS measured by ``/usr/bin/time -f '%e %M'``. Writes:

  cytozip_example_data/output/allc/<cell>.allc.tsv.gz       (+ .tbi)
  cytozip_example_data/output/cz/<cell>.cz
  cytozip_example_data/output/bam_benchmark/bam_benchmark.tsv
  cytozip_example_data/output/bam_benchmark/bam_benchmark.txt

Reference FASTA  : ``~/Ref/mm10/mm10_ucsc_with_chrL.fa``
Reference .cz    : ``cytozip_example_data/output/mm10_with_chrL.allc.cz``
                   (auto-built when missing or older than the FASTA)

Usage:
    python tests/benchmark_bam_to_cz.py -j 9
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "cytozip_example_data"
BAM_DIR = DATA / "bam"
OUT = DATA / "output"
ALLC_DIR = OUT / "allc"
CZ_DIR = OUT / "cz"
BENCH = OUT / "bam_benchmark"
REF_FA = Path.home() / "Ref/mm10/mm10_ucsc_with_chrL.fa"
REF_CZ = OUT / "mm10_with_chrL.allc.cz"

PY = sys.executable
CZIP = f"{Path(PY).parent}/czip"
TIME_BIN = "/usr/bin/time"

YAP_BIN = "/home/x-wding2/Software/conda/yap/bin"
SAMTOOLS = f"{YAP_BIN}/samtools"
# Make `samtools` discoverable inside child subprocesses (ALLCools shells out
# to the bare `samtools` name).
os.environ["PATH"] = YAP_BIN + os.pathsep + os.environ.get("PATH", "")


_ALLC_CALL = (
    "from ALLCools._bam_to_allc import bam_to_allc; "
    "bam_to_allc(bam_path={bam!r}, genome={fa!r}, "
    "output_path={out!r}, cpu=1, num_upstr_bases=0, num_downstr_bases=2, "
    "min_mapq=10, min_base_quality=20, compress_level=5)"
)
_CZ_CALL = (
    "from cytozip.bam import bam_to_cz; "
    "bam_to_cz(bam_path={bam!r}, genome={fa!r}, "
    "output={out!r}, mode='mc_cov', count_fmt='B', reference={ref!r}, "
    "min_mapq=10, min_base_quality=20)"
)


def _time_child(cmd):
    wrapped = [TIME_BIN, "-f", "__TIME__ %e %M", "--"] + list(cmd)
    res = subprocess.run(wrapped, capture_output=True, text=True)
    wall_s, rss_mb = float("nan"), float("nan")
    for line in res.stderr.splitlines()[::-1]:
        if line.startswith("__TIME__"):
            parts = line.split()
            wall_s = float(parts[1])
            rss_mb = float(parts[2]) / 1024.0
            break
    return {
        "ok": res.returncode == 0,
        "wall_s": wall_s,
        "rss_mb": rss_mb,
        "stderr_tail": "\n".join(res.stderr.splitlines()[-8:]),
    }


def _ensure_index(bam: Path):
    bai = Path(str(bam) + ".bai")
    if not bai.exists() or bai.stat().st_mtime < bam.stat().st_mtime:
        subprocess.run([SAMTOOLS, "index", str(bam)], check=True)


def _count_reads(bam: Path) -> int:
    r = subprocess.run([SAMTOOLS, "view", "-c", "-F", "4", str(bam)],
                       check=True, capture_output=True, text=True)
    return int(r.stdout.strip())


def _ensure_ref_cz(threads: int) -> None:
    if REF_CZ.exists() and REF_CZ.stat().st_mtime >= REF_FA.stat().st_mtime:
        return
    REF_CZ.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([CZIP, "build_ref", "-g", str(REF_FA),
                    "-O", str(REF_CZ), "-t", str(threads)], check=True)


def bench_one(bam_path: str) -> dict:
    bam = Path(bam_path)
    _ensure_index(bam)
    cid = bam.name.split(".hisat3n_dna")[0]
    allc_out = ALLC_DIR / f"{cid}.allc.tsv.gz"
    cz_out = CZ_DIR / f"{cid}.cz"
    for p in (allc_out, Path(str(allc_out) + ".tbi"), cz_out):
        if p.exists():
            p.unlink()

    r_a = _time_child([PY, "-c", _ALLC_CALL.format(
        bam=str(bam), fa=str(REF_FA), out=str(allc_out))])
    r_c = _time_child([PY, "-c", _CZ_CALL.format(
        bam=str(bam), fa=str(REF_FA), out=str(cz_out), ref=str(REF_CZ))])

    return dict(
        cell=cid,
        bam=bam.name,
        bam_size_mb=bam.stat().st_size / 1e6,
        n_reads=_count_reads(bam),
        allc_ok=r_a["ok"], allc_wall_s=r_a["wall_s"], allc_rss_mb=r_a["rss_mb"],
        allc_size_mb=(allc_out.stat().st_size / 1e6) if allc_out.exists() else 0,
        cz_ok=r_c["ok"], cz_wall_s=r_c["wall_s"], cz_rss_mb=r_c["rss_mb"],
        cz_size_mb=(cz_out.stat().st_size / 1e6) if cz_out.exists() else 0,
        allc_err=r_a["stderr_tail"], cz_err=r_c["stderr_tail"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-j", "--jobs", type=int, default=9,
                    help="parallel BAMs (each worker uses ~2 CPUs serially)")
    ap.add_argument("--ref_threads", type=int, default=20,
                    help="threads for build_ref if reference .cz is missing")
    args = ap.parse_args()

    for d in (ALLC_DIR, CZ_DIR, BENCH):
        d.mkdir(parents=True, exist_ok=True)
    _ensure_ref_cz(args.ref_threads)

    bams = sorted(BAM_DIR.glob("*.hisat3n_dna.all_reads.deduped.bam"))
    if not bams:
        raise SystemExit(f"no BAMs under {BAM_DIR}")
    print(f"[bench] {len(bams)} BAMs, {args.jobs} parallel workers")
    for b in bams:
        _ensure_index(b)

    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(bench_one, str(b)): b for b in bams}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            ok = "OK" if (r["allc_ok"] and r["cz_ok"]) else "FAIL"
            print(f"[{i:>2}/{len(bams)}] {ok:4s} {r['cell']:<32}  "
                  f"allc={r['allc_wall_s']:6.1f}s  cz={r['cz_wall_s']:6.1f}s",
                  flush=True)
    print(f"[bench] total wall-clock: {time.perf_counter()-t0:.1f}s")

    rows.sort(key=lambda r: r["cell"])
    cols = ["cell", "bam_size_mb", "n_reads",
            "allc_wall_s", "allc_rss_mb", "allc_size_mb",
            "cz_wall_s", "cz_rss_mb", "cz_size_mb"]
    tsv = BENCH / "bam_benchmark.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["speedup", "size_ratio"],
                           delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["speedup"] = r["allc_wall_s"] / r["cz_wall_s"] if r["cz_wall_s"] else float("nan")
            r["size_ratio"] = r["cz_size_mb"] / r["allc_size_mb"] if r["allc_size_mb"] else float("nan")
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    tot_reads = sum(r["n_reads"] for r in rows)
    tot_at = sum(r["allc_wall_s"] for r in rows)
    tot_ct = sum(r["cz_wall_s"] for r in rows)
    tot_as = sum(r["allc_size_mb"] for r in rows)
    tot_cs = sum(r["cz_size_mb"] for r in rows)
    txt = BENCH / "bam_benchmark.txt"
    txt.write_text(
        "cytozip bam_to_cz  vs  ALLCools bam_to_allc\n"
        "============================================\n"
        f"reference FASTA : {REF_FA}\n"
        f"reference .cz   : {REF_CZ}\n"
        f"BAMs            : {len(rows)}   total reads = {tot_reads:,}\n"
        "\n"
        f"ALLCools : time={tot_at:8.1f} s   size={tot_as:8.1f} MB\n"
        f"cytozip  : time={tot_ct:8.1f} s   size={tot_cs:8.1f} MB\n"
        "\n"
        f"speedup (allc / cz time)  = {tot_at / max(tot_ct, 1e-9):5.2f}x\n"
        f"compression (cz / allc)   = {tot_cs / max(tot_as, 1e-9) * 100:5.1f}%\n"
    )
    print(f"[bench] wrote {tsv}\n[bench] wrote {txt}")
    print(txt.read_text())


if __name__ == "__main__":
    main()
