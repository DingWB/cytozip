"""Parallel region-query benchmark worker: cytozip ``query_numpy`` vs ``pytabix``.

Imported by ``benchmark_query.ipynb``. Kept as an importable module (rather than
notebook-defined functions) so the worker is picklable by
``ProcessPoolExecutor``.

Memory trick: the genome-wide reference ``.cz`` chunk (~2.6 GB when a whole
chromosome is decoded) is primed **once** in the parent process
(:func:`prime_reference`). If the process pool is created with the ``fork``
start method, every worker inherits that already-decoded array copy-on-write
instead of decoding its own copy, so hundreds of workers stay within memory.
"""
from __future__ import annotations

import glob
import os
import sys
import time
import traceback

# Prefer the in-repo cytozip over any pip-installed copy in site-packages.
_REPO = os.path.expanduser("~/Projects/Github/cytozip")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cytozip as czip          # noqa: E402
import tabix as pytabix          # noqa: E402

# ---- config (populated by configure()) ----------------------------------
REF_CZ = None
CZ_DIR = None
ALLC_DIR = None
CHROM = None
START = None
END = None
NQ = 100

# Primed reference reader — shared with fork()ed workers copy-on-write.
_REF_READER = None


def configure(ref_cz, cz_dir, allc_dir, chrom, start, end, nq=100):
    """Set the module-level benchmark configuration (call in the parent)."""
    global REF_CZ, CZ_DIR, ALLC_DIR, CHROM, START, END, NQ
    REF_CZ = str(ref_cz)
    CZ_DIR = str(cz_dir)
    ALLC_DIR = str(allc_dir)
    CHROM = str(chrom)
    START = int(start)
    END = int(end)
    NQ = int(nq)


def prime_reference():
    """Open the reference and decode the queried chunk once, in this process.

    Call this in the parent BEFORE creating the (fork) process pool so every
    worker inherits the decoded chunk copy-on-write.
    """
    global _REF_READER
    _REF_READER = czip.Reader(REF_CZ)
    any_cz = sorted(glob.glob(f"{CZ_DIR}/*.cz"))[0]
    r = czip.Reader(any_cz)
    r.query_numpy(chunk_key=CHROM, start=START, end=END, reference=_REF_READER)
    r.close()
    return _REF_READER


def _warm_time(fn, n):
    """Mean per-call wall time of ``fn`` over ``n`` repetitions (seconds)."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def bench_cell(cid):
    """Time warm region queries for one cell: cz ``query_numpy`` vs ``pytabix``.

    Returns a dict with per-call wall times (ms) and the number of records
    each backend returns. ``cz_n`` counts every reference cytosine in the
    window (covered + uncovered); ``tabix_n`` counts only covered rows.
    """
    cz_path = f"{CZ_DIR}/{cid}.cz"
    allc_path = f"{ALLC_DIR}/{cid}.allc.tsv.gz"
    try:
        r = czip.Reader(cz_path)
        tb = pytabix.open(allc_path)

        # cytozip Reader.query_numpy() (warm — vectorized, ndarray out)
        pos, recs = r.query_numpy(chunk_key=CHROM, start=START, end=END,
                                  reference=_REF_READER)
        cz_n = len(recs)
        t_qn = _warm_time(lambda: r.query_numpy(
            chunk_key=CHROM, start=START, end=END, reference=_REF_READER), NQ)

        # pytabix .query() (warm — handle already open)
        tb_n = sum(1 for _ in tb.query(CHROM, START, END))
        t_tb = _warm_time(lambda: list(tb.query(CHROM, START, END)), NQ)
        r.close()

        return dict(cell=cid, ok=True, cz_n=cz_n, tabix_n=tb_n,
                    cz_ms=t_qn * 1e3, tabix_ms=t_tb * 1e3,
                    speedup=(t_tb / t_qn if t_qn else float("nan")), err="")
    except Exception:
        return dict(cell=cid, ok=False, cz_n=-1, tabix_n=-1,
                    cz_ms=float("nan"), tabix_ms=float("nan"),
                    speedup=float("nan"), err=traceback.format_exc()[-400:])
