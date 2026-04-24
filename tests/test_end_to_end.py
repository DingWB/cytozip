#!/usr/bin/env python
"""End-to-end test + benchmark for cytozip.

Covers the full pipeline documented in ``notebooks/2.dnam.ipynb`` and
``dev.md``:

    1. build the mm10 allc reference  (``czip build_ref``)
    2. pack a single-cell allc.tsv.gz **with** coordinates     (``allc2cz``)
    3. pack the same allc.tsv.gz **without** coordinates        (``allc2cz -r ref``)
    4. run a regional query with ``tabix`` and with ``czip query``
       (both the coordinate-carrying .cz and the reference-less one)
       and report timing.

All generated artifacts and timing logs are written to
``czip_example_data/test/`` so re-running the script is idempotent
against an existing example data directory.

The script is intentionally self-contained: no pytest required.
Run with:

    python tests/test_end_to_end.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "czip_example_data"
TEST_DIR = EXAMPLE_DIR / "test"

REFERENCE_FA = Path.home() / "Ref/mm10/mm10_ucsc_with_chrL.fa"
ALLC_GZ = EXAMPLE_DIR / "FC_E17a_3C_1-1-I3-F13.allc.tsv.gz"

# Outputs under TEST_DIR — do NOT touch the pre-existing files in EXAMPLE_DIR.
REF_CZ = TEST_DIR / "mm10_with_chrL.allc.cz"
CZ_WITH_COORD = TEST_DIR / "FC_E17a_3C_1-1-I3-F13.with_coordinate.cz"
CZ_NO_COORD = TEST_DIR / "FC_E17a_3C_1-1-I3-F13.cz"
REPORT = TEST_DIR / "benchmark.txt"

# Query region documented in dev.md / 2.dnam.ipynb.
QUERY_CHROM = "chr9"
QUERY_START = 3_000_294
QUERY_END = 3_005_294

# Number of repetitions for timing so short runs don't get dominated by
# process startup noise.
REPEATS = 5


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, raising on non-zero exit."""
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def time_cmd(cmd: list[str], repeats: int = REPEATS) -> tuple[float, int, bytes]:
    """Run *cmd* repeatedly and return (median_seconds, nrecords, last_stdout).

    ``nrecords`` is the line count of stdout, matching what ``wc -l`` would
    report. We use median to sidestep occasional OS jitter.
    """
    times: list[float] = []
    last_out = b""
    nrec = 0
    for i in range(repeats):
        t0 = time.perf_counter()
        res = subprocess.run(cmd, check=True, capture_output=True)
        dt = time.perf_counter() - t0
        times.append(dt)
        last_out = res.stdout
        nrec = last_out.count(b"\n")
    times.sort()
    median = times[len(times) // 2]
    return median, nrec, last_out


def ensure_inputs() -> None:
    if not REFERENCE_FA.exists():
        raise SystemExit(f"Reference FASTA missing: {REFERENCE_FA}")
    if not ALLC_GZ.exists():
        raise SystemExit(
            f"Example allc missing: {ALLC_GZ} — run "
            "`figshare download 31953882 -o czip_example_data` first."
        )
    tabix = shutil.which("tabix")
    czip = shutil.which("czip")
    if tabix is None:
        raise SystemExit("tabix not on PATH")
    if czip is None:
        raise SystemExit("czip not on PATH (pip install -e . in the repo?)")
    log(f"tabix:  {tabix}")
    log(f"czip:   {czip}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_reference() -> None:
    """Step 1 — generate mm10_with_chrL.allc.cz from a FASTA."""
    if REF_CZ.exists():
        log(f"reference exists, skip build_ref: {REF_CZ}")
        return
    log("building reference (this takes ~45 s with n=20)...")
    run([
        "czip", "build_ref",
        "-g", str(REFERENCE_FA),
        "-O", str(REF_CZ),
        "-t", "20",
    ])
    assert REF_CZ.exists(), f"build_ref did not produce {REF_CZ}"


def pack_with_coordinates() -> None:
    """Step 2 — allc.tsv.gz -> .cz including pos, mc, cov."""
    if CZ_WITH_COORD.exists():
        log(f"with-coord .cz exists, skip: {CZ_WITH_COORD}")
        return
    log("packing allc.tsv.gz with coordinates...")
    run([
        "czip", "allc2cz",
        "-I", str(ALLC_GZ),
        "-O", str(CZ_WITH_COORD),
        "-F", "Q,H,H",
        "-C", "pos,mc,cov",
        "-u", "1,4,5",
    ])
    assert CZ_WITH_COORD.exists()


def pack_without_coordinates() -> None:
    """Step 3 — allc.tsv.gz -> .cz without coordinates (uses reference)."""
    if CZ_NO_COORD.exists():
        log(f"no-coord .cz exists, skip: {CZ_NO_COORD}")
        return
    log("packing allc.tsv.gz without coordinates (uses reference)...")
    run([
        "czip", "allc2cz",
        "-I", str(ALLC_GZ),
        "-O", str(CZ_NO_COORD),
        "-r", str(REF_CZ),
    ])
    assert CZ_NO_COORD.exists()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(out_lines: list[str]) -> None:
    region = f"{QUERY_CHROM}:{QUERY_START}-{QUERY_END}"

    # ---- tabix ------------------------------------------------------------
    tabix_cmd = ["tabix", str(ALLC_GZ), region]
    t_tabix, n_tabix, _ = time_cmd(tabix_cmd)
    out_lines.append(f"tabix           {region}  median={t_tabix*1000:7.2f} ms   records={n_tabix}")

    # ---- cytozip (with coordinates, sort_col indexed) ---------------------
    cz_cmd = [
        "czip", "query",
        "-I", str(CZ_WITH_COORD),
        "-D", QUERY_CHROM,
        "-s", str(QUERY_START),
        "-e", str(QUERY_END),
    ]
    t_cz, n_cz, _ = time_cmd(cz_cmd)
    out_lines.append(f"czip (coord)    {region}  median={t_cz*1000:7.2f} ms   records={n_cz-1}  (minus header)")

    # ---- cytozip (no coordinates, requires reference) ---------------------
    cz_ref_cmd = [
        "czip", "query",
        "-I", str(CZ_NO_COORD),
        "-r", str(REF_CZ),
        "-D", QUERY_CHROM,
        "-s", str(QUERY_START),
        "-e", str(QUERY_END),
    ]
    t_cz_ref, n_cz_ref, _ = time_cmd(cz_ref_cmd)
    out_lines.append(f"czip (ref)      {region}  median={t_cz_ref*1000:7.2f} ms   records={n_cz_ref-1}  (minus header)")

    out_lines.append("")
    out_lines.append(f"speed vs tabix (wall-clock, CLI startup included):")
    out_lines.append(f"  czip (coord): {t_tabix / t_cz:5.2f}x   (>1 means czip faster)")
    out_lines.append(f"  czip (ref):   {t_tabix / t_cz_ref:5.2f}x")

    # ---- In-process timing (strips CLI startup overhead from Python) ------
    # The CLI numbers above include ~100 ms of Python interpreter startup on
    # every call, which dwarfs small regional queries. The numbers below
    # measure only the actual query work inside a warm Python process, so
    # they isolate the library's true per-query cost.
    import pysam
    import tabix as pytabix  # pip install pytabix
    from cytozip.cz import Reader

    # pysam tabix: equivalent to running `tabix` inside the same process.
    tbx = pysam.TabixFile(str(ALLC_GZ))
    # pytabix: the pure-C wrapper from the `pytabix` PyPI package.
    pytbx = pytabix.open(str(ALLC_GZ))

    def bench(fn, repeats=50):
        # Warm-up to populate caches.
        for _ in range(3):
            fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2]

    def q_tabix():
        return list(tbx.fetch(QUERY_CHROM, QUERY_START, QUERY_END))

    def q_pytabix():
        # pytabix.querys returns an iterator over tab-split lists.
        return list(pytbx.query(QUERY_CHROM, QUERY_START, QUERY_END))

    r_coord = Reader(str(CZ_WITH_COORD))
    def q_cz_coord():
        r = Reader(str(CZ_WITH_COORD))
        out = list(r.query(dimension=QUERY_CHROM, start=QUERY_START,
                           end=QUERY_END, printout=False))
        r.close()
        return out

    def q_cz_coord_warm():
        # Reuse a single Reader — no re-open / re-scan overhead. This
        # matches how a library user would run many queries in a loop.
        return list(r_coord.query(dimension=QUERY_CHROM, start=QUERY_START,
                                  end=QUERY_END, printout=False))

    t_tabix_ip = bench(q_tabix)
    t_pytabix_ip = bench(q_pytabix)
    t_cz_ip = bench(q_cz_coord)
    t_cz_warm = bench(q_cz_coord_warm)

    # Record counts (sanity check — all three should agree).
    n_tabix_ip = len(q_tabix())
    n_pytabix_ip = len(q_pytabix())
    n_cz_warm = len(q_cz_coord_warm())

    r_coord.close()
    tbx.close()

    out_lines.append("")
    out_lines.append("in-process timing (strips CLI startup):")
    out_lines.append("---------------------------------------")
    out_lines.append(f"pysam.TabixFile.fetch            median={t_tabix_ip*1000:7.3f} ms   records={n_tabix_ip}")
    out_lines.append(f"pytabix Tabix.query              median={t_pytabix_ip*1000:7.3f} ms   records={n_pytabix_ip}")
    out_lines.append(f"cytozip Reader open+query        median={t_cz_ip*1000:7.3f} ms")
    out_lines.append(f"cytozip Reader.query (warm)      median={t_cz_warm*1000:7.3f} ms   records={n_cz_warm}")
    out_lines.append("")
    out_lines.append(f"speed vs pysam tabix (in-process):")
    out_lines.append(f"  pytabix:         {t_tabix_ip / t_pytabix_ip:5.2f}x")
    out_lines.append(f"  czip open+query: {t_tabix_ip / t_cz_ip:5.2f}x")
    out_lines.append(f"  czip warm:       {t_tabix_ip / t_cz_warm:5.2f}x")


def save_example_outputs() -> None:
    """Capture tabix + czip stdout to files for easy inspection."""
    region = f"{QUERY_CHROM}:{QUERY_START}-{QUERY_END}"
    (TEST_DIR / "query.tabix.txt").write_bytes(
        subprocess.run(["tabix", str(ALLC_GZ), region],
                       check=True, capture_output=True).stdout)
    (TEST_DIR / "query.cz_coord.txt").write_bytes(
        subprocess.run(["czip", "query", "-I", str(CZ_WITH_COORD),
                        "-D", QUERY_CHROM,
                        "-s", str(QUERY_START), "-e", str(QUERY_END)],
                       check=True, capture_output=True).stdout)
    (TEST_DIR / "query.cz_ref.txt").write_bytes(
        subprocess.run(["czip", "query", "-I", str(CZ_NO_COORD),
                        "-r", str(REF_CZ),
                        "-D", QUERY_CHROM,
                        "-s", str(QUERY_START), "-e", str(QUERY_END)],
                       check=True, capture_output=True).stdout)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    ensure_inputs()

    build_reference()
    pack_with_coordinates()
    pack_without_coordinates()

    # Sizes report
    size_lines = [
        "",
        f"file sizes:",
        f"  ref .cz:             {REF_CZ.stat().st_size/1e6:8.2f} MB",
        f"  with-coordinate .cz: {CZ_WITH_COORD.stat().st_size/1e6:8.2f} MB",
        f"  no-coordinate .cz:   {CZ_NO_COORD.stat().st_size/1e6:8.2f} MB",
        f"  allc.tsv.gz:         {ALLC_GZ.stat().st_size/1e6:8.2f} MB",
        "",
    ]
    for line in size_lines:
        log(line)

    out_lines: list[str] = [
        "cytozip end-to-end benchmark",
        "============================",
        f"query region: {QUERY_CHROM}:{QUERY_START}-{QUERY_END}",
        f"repeats per cmd: {REPEATS} (median reported)",
        "",
    ] + size_lines + [
        "timing:",
        "-------",
    ]

    benchmark(out_lines)
    save_example_outputs()

    REPORT.write_text("\n".join(out_lines) + "\n")
    log(f"report saved to {REPORT}")
    print()
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
