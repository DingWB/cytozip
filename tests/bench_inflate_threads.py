"""Focused benchmark: measure multi-threaded inflate speedup on a bulk
chunk decompression path.

Sets ``CYTOZIP_INFLATE_THREADS`` BEFORE importing cytozip so the choice
sticks (the value is cached on first use). Forks subprocesses per setting
and reports wall-time for repeated whole-chunk reads.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CZ = REPO / "cytozip_example_data" / "output" / "all_cells.cz"


def _run_one(threads: int, ntrials: int = 3) -> float:
    code = f"""
import os, sys, time
os.environ['CYTOZIP_INFLATE_THREADS'] = '{threads}'
sys.path.insert(0, r'{REPO}')
import cytozip.cz as cz
r = cz.Reader(r'{CZ}')
ck = list(r.chunk_key2offset.keys())[0]
# warm-up
r._load_chunk(r.chunk_key2offset[ck], jump=False)
data = r.fetch_chunk_bytes(ck)
n = len(data)
t0 = time.perf_counter()
for _ in range({ntrials}):
    r._load_chunk(r.chunk_key2offset[ck], jump=False)
    d = r.fetch_chunk_bytes(ck)
    assert len(d) == n
elapsed = (time.perf_counter() - t0) / {ntrials}
print(f'OK threads={threads} bytes={{n}} mean_s={{elapsed:.4f}}')
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(1)
    line = p.stdout.strip().splitlines()[-1]
    print(line)
    return float(line.split("mean_s=")[1])


if __name__ == "__main__":
    t1 = _run_one(1)
    t2 = _run_one(2)
    t4 = _run_one(4)
    t8 = _run_one(8)
    print(f"\nspeedup t1/t2={t1 / t2:.2f}x, t1/t4={t1 / t4:.2f}x, t1/t8={t1 / t8:.2f}x")
