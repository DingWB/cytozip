"""Benchmark Writer-side parallel deflate (P0 #2).

Round-trip test:
  1. Read a sample CZ file's chunks into memory (raw decompressed records).
  2. Re-write with cytozip Writer using N deflate threads.
  3. Time the write and verify md5 by re-reading.

Usage: python tests/bench_deflate_threads.py [src.cz]
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SRC = REPO_ROOT / "cytozip_example_data" / "output" / "all_cells.cz"


def md5_of_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_records(p):
    """Hash the decompressed concatenated records of every chunk."""
    import cytozip as cz
    r = cz.Reader(str(p))
    h = hashlib.md5()
    for ck in list(r.chunk_key2offset.keys()):
        data = r.fetch_chunk_bytes(ck)
        h.update(data)
    r.close()
    return h.hexdigest()


def write_with_threads(src, dst, threads):
    """Subprocess to honor CYTOZIP_DEFLATE_THREADS at module init."""
    code = f"""
import sys, time
sys.path.insert(0, {str(REPO_ROOT)!r})
import cytozip as cz
r = cz.Reader({str(src)!r})
header = r.header
w = cz.Writer({str(dst)!r},
              formats=header['formats'],
              columns=header['columns'],
              chunk_dims=header['chunk_dims'],
              sort_col=header['sort_col'],
              delta_cols=header['delta_cols'] or None)
t0 = time.perf_counter()
for ck in list(r.chunk_key2offset.keys()):
    data = r.fetch_chunk_bytes(ck)
    dims = list(ck) if isinstance(ck, tuple) else [ck]
    w.write_chunk(data, dims)
w.close()
elapsed = time.perf_counter() - t0
r.close()
print(f"WRITE_TIME {{elapsed:.4f}}")
"""
    env = os.environ.copy()
    env["CYTOZIP_DEFLATE_THREADS"] = str(threads)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("WRITE_TIME")][0]
    return float(line.split()[1])


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        sys.exit(1)
    print(f"source: {src}  size={src.stat().st_size/1e6:.1f} MB")

    src_md5 = md5_of_records(src)
    print(f"src records md5: {src_md5}")

    results = []
    for n in (1, 2, 4, 8):
        with tempfile.NamedTemporaryFile(suffix=".cz", delete=False) as tmp:
            dst = Path(tmp.name)
        try:
            t = write_with_threads(src, dst, n)
            dst_md5 = md5_of_records(dst)
            ok = dst_md5 == src_md5
            print(f"threads={n:>2}  write={t:.4f}s  md5={dst_md5}  ok={ok}")
            results.append((n, t, ok))
        finally:
            try:
                dst.unlink()
            except FileNotFoundError:
                pass

    if results:
        t1 = results[0][1]
        print()
        for n, t, ok in results:
            print(f"  threads={n}: {t:.4f}s  speedup={t1/t:.2f}x  ok={ok}")


if __name__ == "__main__":
    main()
