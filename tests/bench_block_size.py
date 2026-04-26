"""Empirical sweep of `_BLOCK_MAX_LEN` to find the sweet spot.

For each candidate block size (uncompressed cap), this script:
  1. Patches `_BLOCK_MAX_LEN` in both `cytozip/cz.py` (line ~78) and
     `cytozip/cz_accel.pyx` (line ~36).
  2. Rebuilds the Cython extension.
  3. Runs a subprocess that: writes a synthetic delta-encoded .cz with
     a known dataset, then measures
       - file size
       - end-to-end write time
       - whole-chunk decode time
       - random point-query time (`pos2id` x N)
  4. Restores the originals at the end.

Run from the repo root:
    /home/x-wding2/Software/conda/m3c/bin/python tests/bench_block_size.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/x-wding2/Software/conda/m3c/bin/python"
CZ_PY = REPO_ROOT / "cytozip" / "cz.py"
CZ_PYX = REPO_ROOT / "cytozip" / "cz_accel.pyx"

# Block sizes to sweep (uncompressed bytes).  All <= (1<<20)-1 because
# the virtual-offset format reserves 20 bits for within-block offsets.
SIZES = [
    ("64K",   (1 << 16) - 1),
    ("128K",  (1 << 17) - 1),
    ("256K",  (1 << 18) - 1),
    ("512K",  (1 << 19) - 1),
    ("1M",    (1 << 20) - 1),  # current default
]


def patch_const(path: Path, value: int):
    txt = path.read_text()
    if path.suffix == ".py":
        new = re.sub(
            r"^_BLOCK_MAX_LEN\s*=.*$",
            f"_BLOCK_MAX_LEN = {value}    # patched by bench_block_size",
            txt, count=1, flags=re.M)
    else:
        new = re.sub(
            r"^cdef unsigned long _BLOCK_MAX_LEN\s*=.*$",
            f"cdef unsigned long _BLOCK_MAX_LEN = {value}  # patched",
            txt, count=1, flags=re.M)
    if new == txt:
        raise RuntimeError(f"failed to patch {path}")
    path.write_text(new)


def rebuild():
    subprocess.run(
        [PY, "setup.py", "build_ext", "--inplace"],
        cwd=REPO_ROOT, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


BENCH_CODE = r"""
import sys, os, tempfile, time, hashlib, pickle, random
sys.path.insert(0, %REPO%)
import numpy as np, pandas as pd, cytozip as cz

# Build a 4M-row dataset resembling mC allc (sorted pos + 2 uint8).
N = 4_000_000
rng = np.random.RandomState(0)
gaps = rng.randint(1, 30, size=N).astype(np.uint32)
pos = np.cumsum(gaps).astype(np.uint32)
mc = rng.randint(0, 50, N).astype(np.uint8)
cv = (mc + rng.randint(0, 50, N)).astype(np.uint8)
df = pd.DataFrame({'chrom':'chr1', 'pos':pos, 'mc':mc, 'cov':cv})

with tempfile.NamedTemporaryFile(suffix='.cz', delete=False) as tmp:
    dst = tmp.name
try:
    # ---- write
    t0 = time.perf_counter()
    w = cz.Writer(dst, formats=['I','B','B'], columns=['pos','mc','cov'],
                  chunk_dims=['chrom'], sort_col=0, delta_cols=['pos'])
    w.tocz(df, usecols=['pos','mc','cov'], key_cols=['chrom'])
    write_t = time.perf_counter() - t0
    file_size = os.path.getsize(dst)

    # ---- whole-chunk decode (avg of 5)
    decode_times = []
    for _ in range(5):
        r = cz.Reader(dst)
        t0 = time.perf_counter()
        arr = r.chunk2numpy(('chr1',))
        decode_times.append(time.perf_counter() - t0)
        r.close()

    # ---- point query (pos2id of 200 random positions)
    sample_ids = sorted(rng.choice(N, 200, replace=False))
    sample_pos = pos[sample_ids].tolist()
    r = cz.Reader(dst)
    # warm
    list(r.pos2id(('chr1',), [(int(p), int(p)) for p in sample_pos[:5]]))
    qtimes = []
    for _ in range(5):
        t0 = time.perf_counter()
        list(r.pos2id(('chr1',), [(int(p), int(p)) for p in sample_pos]))
        qtimes.append(time.perf_counter() - t0)
    r.close()

    # ---- block count introspection
    r = cz.Reader(dst)
    r._load_chunk(r.chunk_key2offset[('chr1',)], jump=False)
    n_blocks = len(r._chunk_block_1st_record_virtual_offsets)
    r.close()

    print('RESULT', N, write_t, file_size, min(decode_times), min(qtimes), n_blocks)
finally:
    os.unlink(dst)
"""


def run_bench():
    code = BENCH_CODE.replace("%REPO%", repr(str(REPO_ROOT)))
    out = subprocess.run([PY, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][-1]
    parts = line.split()
    return {
        "n":          int(parts[1]),
        "write_s":    float(parts[2]),
        "size_B":     int(parts[3]),
        "decode_s":   float(parts[4]),
        "qry_s":      float(parts[5]),
        "n_blocks":   int(parts[6]),
    }


def main():
    cz_py_orig = CZ_PY.read_text()
    cz_pyx_orig = CZ_PYX.read_text()
    results = []
    try:
        for label, val in SIZES:
            print(f"\n=== block size = {label} ({val} B) ===", flush=True)
            patch_const(CZ_PY, val)
            patch_const(CZ_PYX, val)
            rebuild()
            r = run_bench()
            r["label"] = label
            r["block_B"] = val
            results.append(r)
            print(f"  size={r['size_B']/1e6:.2f} MB  "
                  f"write={r['write_s']:.3f}s  "
                  f"decode={r['decode_s']*1000:.1f} ms  "
                  f"200-qry={r['qry_s']*1000:.1f} ms  "
                  f"blocks={r['n_blocks']}",
                  flush=True)
    finally:
        CZ_PY.write_text(cz_py_orig)
        CZ_PYX.write_text(cz_pyx_orig)
        print("\nrestored originals; rebuilding...", flush=True)
        rebuild()

    # Summary
    base = next(r for r in results if r["label"] == "1M")
    print("\n" + "=" * 88)
    print(f"{'block':>6}  {'size_MB':>8}  {'vs1M':>6}  "
          f"{'wr_s':>6}  {'wr×':>5}  "
          f"{'dec_ms':>7}  {'dec×':>5}  "
          f"{'qry_ms':>7}  {'qry×':>5}  {'#blk':>6}")
    print("-" * 88)
    for r in results:
        print(f"{r['label']:>6}  {r['size_B']/1e6:>8.2f}  "
              f"{r['size_B']/base['size_B']:>6.3f}  "
              f"{r['write_s']:>6.2f}  "
              f"{base['write_s']/r['write_s']:>5.2f}  "
              f"{r['decode_s']*1000:>7.1f}  "
              f"{base['decode_s']/r['decode_s']:>5.2f}  "
              f"{r['qry_s']*1000:>7.1f}  "
              f"{base['qry_s']/r['qry_s']:>5.2f}  "
              f"{r['n_blocks']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
