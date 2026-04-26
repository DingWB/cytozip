"""Bench query_numpy_multi parallel — cold-cache scenario (P1 #5).

Each region hits a different chunk so chunk decompression dominates.
Reader is reopened per measurement to avoid warm cache.
"""
import sys, time, random
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import cytozip as cz

SRC = REPO_ROOT / "cytozip_example_data" / "output" / "mm10_with_chrL.allc.cz"

def measure(workers, regions):
    r = cz.Reader(str(SRC))  # fresh = cold cache
    t0 = time.perf_counter()
    out = r.query_numpy_multi(regions, max_workers=workers)
    t = time.perf_counter() - t0
    tot = sum(a.size for a in out)
    r.close()
    return t, tot

def main():
    r = cz.Reader(str(SRC))
    sc = r.header['sort_col']
    cks = list(r.chunk_key2offset.keys())
    print(f"sort_col={sc}, chunks={len(cks)}")
    rng = random.Random(0)
    regions = []
    # one query per chunk, sample 30 distinct chunks
    for ck in cks[:30]:
        pos = r._cached_sort_column(ck, sc)
        if not len(pos):
            continue
        pmin, pmax = int(pos[0]), int(pos[-1])
        s = rng.randint(pmin, max(pmin, pmax - 1_000_000))
        regions.append((ck, s, s + 1_000_000))
    r.close()
    print(f"n_regions={len(regions)} (one chunk each)")

    seq_t, seq_tot = measure(1, regions)
    print(f"seq (1t):  {seq_t:.3f}s  records={seq_tot}")
    for w in (2, 4, 8):
        t, tot = measure(w, regions)
        print(f"par ({w}t): {t:.3f}s  speedup={seq_t/t:.2f}x  records={tot}  ok={tot==seq_tot}")

if __name__ == "__main__":
    main()
