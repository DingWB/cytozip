"""Benchmark Reader.iter_chunks_bytes async prefetch (P1 #4)."""
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cytozip as cz

SRC = REPO_ROOT / "cytozip_example_data" / "output" / "all_cells.cz"


def time_sync():
    r = cz.Reader(str(SRC))
    h = hashlib.md5()
    n = 0
    t0 = time.perf_counter()
    for ck in list(r.chunk_key2offset.keys()):
        data = r.fetch_chunk_bytes(ck)
        h.update(data)
        n += 1
    elapsed = time.perf_counter() - t0
    r.close()
    return elapsed, h.hexdigest(), n


def time_prefetch(prefetch):
    r = cz.Reader(str(SRC))
    h = hashlib.md5()
    n = 0
    t0 = time.perf_counter()
    for ck, data in r.iter_chunks_bytes(prefetch=prefetch):
        h.update(data)
        n += 1
    elapsed = time.perf_counter() - t0
    r.close()
    return elapsed, h.hexdigest(), n


def main():
    print(f"source: {SRC} ({SRC.stat().st_size/1e6:.1f} MB)")
    s_t, s_md5, s_n = time_sync()
    print(f"sync (fetch_chunk_bytes loop): {s_t:.3f}s  n={s_n}  md5={s_md5}")
    for p in (1, 2, 4):
        t, md, n = time_prefetch(p)
        print(f"prefetch={p}:  {t:.3f}s  speedup={s_t/t:.2f}x  md5={md}  ok={md == s_md5}")


if __name__ == "__main__":
    main()
