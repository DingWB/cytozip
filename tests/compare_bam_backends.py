"""Compare bam_to_cz pysam vs samtools-mpileup backends.

Runs the same BAM through both backends with identical parameters and
verifies the resulting count_df agrees on per-context mC / cov totals.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ensure samtools is on PATH for the mpileup backend.
SAMTOOLS_DIR = "/home/x-wding2/Software/conda/m3c/bin"
os.environ["PATH"] = SAMTOOLS_DIR + os.pathsep + os.environ.get("PATH", "")

from cytozip.bam import bam_to_cz
import cytozip as cz
import hashlib


BAM = REPO_ROOT / "cytozip_example_data" / "bam" / \
    "FC_E17b_3C_5-5-I24-A21.hisat3n_dna.all_reads.deduped.bam"
GENOME = "/anvil/projects/x-mcb130189/manoj/prtbsq/mm10_prtbsq_refs/Orgnls/mm10_with_chrl.fa"


def _records_md5(p):
    r = cz.Reader(str(p))
    h = hashlib.md5()
    for ck in list(r.chunk_key2offset.keys()):
        h.update(r.fetch_chunk_bytes(ck))
    r.close()
    return h.hexdigest(), len(r.chunk_key2offset) if False else 0


def _records_md5_and_n(p):
    r = cz.Reader(str(p))
    h = hashlib.md5()
    n_chunks = len(r.chunk_key2offset)
    for ck in list(r.chunk_key2offset.keys()):
        h.update(r.fetch_chunk_bytes(ck))
    r.close()
    return h.hexdigest(), n_chunks


def run(backend):
    if backend == "mpileup":
        os.environ["CYTOZIP_BAM_BACKEND_MPILEUP"] = "1"
    else:
        os.environ["CYTOZIP_BAM_BACKEND_MPILEUP"] = "0"
    with tempfile.NamedTemporaryFile(suffix=".cz", delete=False) as tmp:
        out = tmp.name
    t0 = time.perf_counter()
    df = bam_to_cz(
        bam_path=str(BAM),
        genome=GENOME,
        output=out,
        mode="pos_mc_cov",
        count_fmt="H",
        min_mapq=10,
        min_base_quality=20,
        batch_size=5000,
    )
    elapsed = time.perf_counter() - t0
    md5, n = _records_md5_and_n(out)
    return {
        "backend": backend,
        "elapsed": elapsed,
        "md5": md5,
        "n_chunks": n,
        "out": out,
        "df": df,
    }


def main():
    if not BAM.exists():
        print(f"BAM not found: {BAM}", file=sys.stderr)
        sys.exit(1)
    if not Path(GENOME).exists():
        print(f"genome fasta not found: {GENOME}", file=sys.stderr)
        sys.exit(1)

    results = []
    for backend in ("mpileup", "pysam"):
        try:
            r = run(backend)
        except Exception as e:
            print(f"{backend}: FAILED — {e}")
            raise
        print(f"{backend:8s} elapsed={r['elapsed']:.2f}s  "
              f"chunks={r['n_chunks']}  md5={r['md5']}")
        results.append(r)
        # Print top-context summary
        if r["df"] is not None and not r["df"].empty:
            print(r["df"].sort_values("cov", ascending=False).head(8))
        print()

    # Compare: identical contexts, identical mc/cov totals expected.
    a, b = results
    print(f"speedup mpileup/pysam = {a['elapsed']/b['elapsed']:.2f}x")
    print(f"records md5 identical: {a['md5'] == b['md5']}")
    if a["df"] is not None and b["df"] is not None:
        common = sorted(set(a["df"].index) & set(b["df"].index))
        if common:
            ca = a["df"].loc[common, ["mc","cov"]]
            cb = b["df"].loc[common, ["mc","cov"]]
            diff = (ca - cb).abs()
            print(f"per-context mc max abs diff:  {int(diff['mc'].max())}")
            print(f"per-context cov max abs diff: {int(diff['cov'].max())}")
            tot_a_mc, tot_a_cov = int(ca['mc'].sum()), int(ca['cov'].sum())
            tot_b_mc, tot_b_cov = int(cb['mc'].sum()), int(cb['cov'].sum())
            print(f"total mc: mpileup={tot_a_mc}  pysam={tot_b_mc}  "
                  f"rel_diff={(tot_b_mc-tot_a_mc)/max(tot_a_mc,1):+.4%}")
            print(f"total cov: mpileup={tot_a_cov}  pysam={tot_b_cov} "
                  f"rel_diff={(tot_b_cov-tot_a_cov)/max(tot_a_cov,1):+.4%}")

    for r in results:
        try:
            os.unlink(r["out"])
        except OSError:
            pass


if __name__ == "__main__":
    main()
