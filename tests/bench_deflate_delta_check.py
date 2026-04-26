"""Delta-encoded round-trip sanity check for parallel deflate (P0 #2)."""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_cz(dst, threads):
    code = f"""
import sys, os
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np, pandas as pd, cytozip as cz
parts = []
for chrom in ['chr1','chr2','chr3']:
    n = 200_000
    pos = np.sort(np.random.RandomState(hash(chrom) & 0xFFFF).randint(0, 1<<30, n)).astype(np.uint32)
    df = pd.DataFrame({{'chrom': chrom, 'pos': pos,
                        'mc': np.full(n, 5, np.uint8),
                        'cov': np.full(n, 10, np.uint8)}})
    parts.append(df)
big = pd.concat(parts, ignore_index=True)
w = cz.Writer({str(dst)!r}, formats=['I','B','B'],
              columns=['pos','mc','cov'], chunk_dims=['chrom'],
              sort_col=0, delta_cols=['pos'])
w.tocz(big, usecols=['pos','mc','cov'], key_cols=['chrom'])
# tocz closes the writer.
"""
    env = os.environ.copy()
    env["CYTOZIP_DEFLATE_THREADS"] = str(threads)
    env["PYTHONHASHSEED"] = "0"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def _records_md5(p):
    sys.path.insert(0, str(REPO_ROOT))
    import cytozip as cz
    r = cz.Reader(str(p))
    h = hashlib.md5()
    for ck in list(r.chunk_key2offset.keys()):
        h.update(r.fetch_chunk_bytes(ck))
    r.close()
    return h.hexdigest()


def main():
    digests = {}
    for n in (1, 2, 4, 8):
        with tempfile.NamedTemporaryFile(suffix=".cz", delete=False) as tmp:
            dst = Path(tmp.name)
        try:
            _make_cz(dst, n)
            d = _records_md5(dst)
            digests[n] = d
            print(f"threads={n}  md5={d}")
        finally:
            dst.unlink(missing_ok=True)
    unique = set(digests.values())
    print(f"all_equal: {len(unique) == 1}")
    return 0 if len(unique) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
