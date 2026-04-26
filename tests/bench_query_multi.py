"""Bench query_numpy_multi parallel (P1 #5)."""
import sys, time, random
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import cytozip as cz

SRC = REPO_ROOT / "cytozip_example_data" / "output" / "mm10_with_chrL.allc.cz"

def main():
    r = cz.Reader(str(SRC))
    sc = r.header.get('sort_col')
    print(f"sort_col={sc}, chunks={len(r.chunk_key2offset)}")
    if sc is None or sc < 0:
        print("file not self-positioned; skipping")
        return
    # Pick first 10 chunks; build random regions of length 1Mb across each.
    cks = list(r.chunk_key2offset.keys())[:10]
    rng = random.Random(0)
    regions = []
    for ck in cks:
        arr = r._chunk2numpy_cached(ck)
        if not len(arr):
            continue
        pos = r._cached_sort_column(ck, sc)
        if not len(pos):
            continue
        pmin, pmax = int(pos[0]), int(pos[-1])
        for _ in range(200):
            s = rng.randint(pmin, max(pmin, pmax - 1_000_000))
            regions.append((ck, s, s + 1_000_000))
    print(f"n_regions={len(regions)}")

    # Warm cache
    r.query_numpy_multi(regions[:5], max_workers=1)

    # Sequential
    t0 = time.perf_counter()
    seq = r.query_numpy_multi(regions, max_workers=1)
    seq_t = time.perf_counter() - t0
    seq_total = sum(a.size for a in seq)
    print(f"seq (1t):  {seq_t:.3f}s  total_records={seq_total}")

    for w in (2, 4, 8):
        t0 = time.perf_counter()
        out = r.query_numpy_multi(regions, max_workers=w)
        t = time.perf_counter() - t0
        tot = sum(a.size for a in out)
        ok = (tot == seq_total)
        print(f"par ({w}t): {t:.3f}s  speedup={seq_t/t:.2f}x  total={tot}  ok={ok}")
    r.close()

if __name__ == "__main__":
    main()
