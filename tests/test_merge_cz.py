"""Correctness regression test for ``merge_cz``.

Re-runs ``merge_cz`` on the 9 single-cell .cz files in
``cytozip_example_data/output/cz/`` and verifies that, for every chromosome
chunk, the merged mc/cov pair equals the (clipped) sum of the per-cell
mc/cov pairs computed independently with NumPy.

Usage:
    pytest tests/test_merge_cz.py -s
or:
    python tests/test_merge_cz.py

The test is skipped automatically if the example data directory is missing.
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

import numpy as np
import pytest

from cytozip.cz import Reader
from cytozip.merge import merge_cz


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CELLS_DIR = os.path.join(REPO_ROOT, 'cytozip_example_data', 'output', 'cz')


def _struct_dtype(fmt_codes):
    # Map struct format chars -> numpy dtype chars.
    char_map = {'B': 'u1', 'H': 'u2', 'I': 'u4', 'Q': 'u8',
                'b': 'i1', 'h': 'i2', 'i': 'i4', 'q': 'i8',
                'f': 'f4', 'd': 'f8'}
    return np.dtype([(f'f{i}', char_map[c]) for i, c in enumerate(fmt_codes)])


@pytest.mark.skipif(
    not os.path.isdir(CELLS_DIR),
    reason=f"example data not found at {CELLS_DIR}",
)
def test_merge_cz_sum_matches_per_cell_sum(tmp_path):
    cells = sorted(f for f in os.listdir(CELLS_DIR) if f.endswith('.cz'))
    assert len(cells) >= 2, "need at least 2 single-cell .cz files"

    out = tmp_path / "merged.cz"
    merge_cz(
        input=[os.path.join(CELLS_DIR, c) for c in cells],
        output=str(out),
        jobs=4,
        formats=['H', 'H'],
        bgzip=False,
        keep_cat=False,
        temp=False,
    )
    assert out.exists(), f"merge_cz did not produce {out}"

    # Read merged file
    rm = Reader(str(out))
    out_fmts = rm.header['formats']
    out_dt = _struct_dtype(out_fmts)
    out_max = np.iinfo(out_dt['f0']).max  # both columns same family for now
    out_chroms = sorted(rm.chunk_key2offset.keys())

    # Pick a per-cell dtype from the first cell (assume all cells share fmts).
    r0 = Reader(os.path.join(CELLS_DIR, cells[0]))
    cell_fmts = r0.header['formats']
    r0.close()
    cell_dt = _struct_dtype(cell_fmts)

    total_records = 0
    total_bad = 0
    bad_chroms: list[tuple] = []

    for dim in out_chroms:
        per_cell_arr = []
        N = None
        for f in cells:
            r = Reader(os.path.join(CELLS_DIR, f))
            if dim not in r.chunk_key2offset:
                r.close()
                continue
            arr = np.frombuffer(r.fetch_chunk_bytes(dim), dtype=cell_dt)
            r.close()
            if N is None:
                N = arr.size
            assert arr.size == N, (
                f"{dim}: cell {f} has {arr.size} records, expected {N}"
            )
            per_cell_arr.append(arr)
        assert per_cell_arr, f"no cells contain {dim}"

        mc_sum = np.zeros(N, dtype=np.int64)
        cov_sum = np.zeros(N, dtype=np.int64)
        for a in per_cell_arr:
            mc_sum += a['f0'].astype(np.int64)
            cov_sum += a['f1'].astype(np.int64)
        mc_truth = np.minimum(mc_sum, out_max).astype(np.int64)
        cov_truth = np.minimum(cov_sum, out_max).astype(np.int64)

        arr_pb = np.frombuffer(rm.fetch_chunk_bytes(dim), dtype=out_dt)
        assert arr_pb.size == N, (
            f"{dim}: merged size {arr_pb.size} != cell size {N}"
        )
        bad = int(((arr_pb['f0'].astype(np.int64) != mc_truth) |
                   (arr_pb['f1'].astype(np.int64) != cov_truth)).sum())
        total_records += N
        total_bad += bad
        if bad:
            bad_chroms.append((dim, N, bad))

    rm.close()
    print(f"[merge_cz test] {len(out_chroms)} chroms, "
          f"{total_records} records, mismatches={total_bad}")
    assert total_bad == 0, (
        f"merge_cz produced {total_bad} mismatched records "
        f"in {len(bad_chroms)} chroms; first: {bad_chroms[:3]}"
    )


if __name__ == "__main__":
    if not os.path.isdir(CELLS_DIR):
        print(f"SKIP: example data not at {CELLS_DIR}")
        sys.exit(0)
    tmp = tempfile.mkdtemp(prefix='merge_cz_test_')
    try:
        class _P:
            def __init__(self, p):
                self._p = p

            def __truediv__(self, other):
                import pathlib
                return pathlib.Path(self._p) / other

        test_merge_cz_sum_matches_per_cell_sum(_P(tmp))
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
