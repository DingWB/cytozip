"""Correctness regression test for ``Writer.catcz``.

Concatenates the 9 single-cell .cz files in
``cytozip_example_data/output/cz/`` into one catcz output (with
``key_added="cell_id"``, producing chunk_dims=['chrom','cell_id']), and
verifies that for every ``(chrom, cell_basename)`` chunk the raw
decoded bytes are identical to the corresponding ``(chrom,)`` chunk
in the original per-cell .cz file.

Usage:
    pytest tests/test_catcz.py -s
or:
    python tests/test_catcz.py
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

import pytest

from cytozip.cz import Reader, Writer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CELLS_DIR = os.path.join(REPO_ROOT, 'cytozip_example_data', 'output', 'cz')


@pytest.mark.skipif(
    not os.path.isdir(CELLS_DIR),
    reason=f"example data not found at {CELLS_DIR}",
)
def test_catcz_bytes_identical_to_originals(tmp_path):
    cells = sorted(f for f in os.listdir(CELLS_DIR) if f.endswith('.cz'))
    assert len(cells) >= 2

    out_path = str(tmp_path / 'cat.cz')

    # Pull header from the first cell and create a Writer with matching
    # formats / columns / chunk_dims. ``key_added="cell_id"`` appends the
    # basename (sans .cz) as a new chunk dim called 'cell_id'.
    r0 = Reader(os.path.join(CELLS_DIR, cells[0]))
    h = r0.header
    r0.close()

    w = Writer(output=out_path, formats=h['formats'], columns=h['columns'],
               chunk_dims=h['chunk_dims'], message='catcz_test')
    w.catcz(input=[os.path.join(CELLS_DIR, c) for c in cells], key_added="cell_id")
    del w  # close+flush

    rcat = Reader(out_path)
    assert rcat.header['chunk_dims'] == h['chunk_dims'] + ['cell_id'], (
        f"chunk_dims after catcz: {rcat.header['chunk_dims']}"
    )

    total = 0
    mismatches: list[tuple] = []
    for cell in cells:
        rc = Reader(os.path.join(CELLS_DIR, cell))
        base_no_ext = cell[:-3]
        for cdim in list(rc.chunk_key2offset.keys()):
            chrom = cdim[0] if isinstance(cdim, tuple) else cdim
            orig = rc.fetch_chunk_bytes(cdim)
            cat_dim = (chrom, base_no_ext)
            assert cat_dim in rcat.chunk_key2offset, (
                f"{cat_dim} missing in catcz output (cell={cell})"
            )
            new = rcat.fetch_chunk_bytes(cat_dim)
            if orig != new:
                mismatches.append((cell, cat_dim, len(orig), len(new)))
            total += 1
        rc.close()
    rcat.close()

    print(f"[catcz test] {len(cells)} cells, {total} chunks, "
          f"mismatches={len(mismatches)}")
    assert not mismatches, (
        f"catcz produced {len(mismatches)} byte-mismatched chunks; "
        f"first: {mismatches[:3]}"
    )


if __name__ == "__main__":
    if not os.path.isdir(CELLS_DIR):
        print(f"SKIP: example data not at {CELLS_DIR}")
        sys.exit(0)
    tmp = tempfile.mkdtemp(prefix='catcz_test_')
    try:
        import pathlib
        test_catcz_bytes_identical_to_originals(pathlib.Path(tmp))
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
