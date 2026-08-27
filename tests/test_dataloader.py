"""Smoke + correctness test for :mod:`cytozip.dataloader`.

Uses the benchmark single-cell .cz files and the hg38 reference / CGN index:

    cells     : ~/Projects/test_cytozip/benchmark/cz/
    reference : ~/Ref/hg38/hg38_with_chrL.allc.cz
    CGN index : ~/Ref/hg38/hg38_with_chrL.CGN.cz

Run:
    pytest tests/test_dataloader.py -s
or:
    python tests/test_dataloader.py

Skipped automatically when the data paths are missing.
"""
from __future__ import annotations

import os
import glob
import time

import numpy as np
import pytest

from cytozip.cz import Reader
from cytozip import dataloader as dl


CELLS_DIR = os.path.expanduser("~/Projects/test_cytozip/benchmark/cz")
REFERENCE = os.path.expanduser("~/Ref/hg38/hg38_with_chrL.allc.cz")
CGN_INDEX = os.path.expanduser("~/Ref/hg38/hg38_with_chrL.CGN.cz")
CHROM = "chr1"
BINSIZE = 100_000

_have_data = (os.path.isdir(CELLS_DIR)
              and os.path.isfile(REFERENCE)
              and os.path.isfile(CGN_INDEX)
              and len(glob.glob(os.path.join(CELLS_DIR, "*.cz"))) >= 2)

pytestmark = pytest.mark.skipif(
    not _have_data, reason="benchmark data / hg38 reference not found")


def _first_cells(n=6):
    return sorted(glob.glob(os.path.join(CELLS_DIR, "*.cz")))[:n]


def test_load_chrom_matrix_shapes():
    cells = _first_cells()
    cell_ids, pos, mc, cov = dl.load_chrom_matrix(
        REFERENCE, cells, chrom=CHROM, index=CGN_INDEX, threads=4)
    assert len(cell_ids) == len(cells)
    assert mc.shape == cov.shape == (len(cells), pos.shape[0])
    assert pos.ndim == 1 and pos.shape[0] > 0
    # positions must be strictly ascending (reference order / CG subset)
    assert np.all(np.diff(pos) > 0)


def test_matrix_matches_bruteforce():
    """The loaded per-cell CG columns must equal a direct chunk2numpy gather."""
    cells = _first_cells(4)
    cell_ids, pos, mc, cov = dl.load_chrom_matrix(
        REFERENCE, cells, chrom=CHROM, index=CGN_INDEX, threads=1)

    ix = Reader(CGN_INDEX)
    ids = ix.get_ids_from_index((CHROM,)).astype(np.int64) - 1  # 0-based
    ix.close()

    for i, path in enumerate(cells):
        r = Reader(path)
        arr = r.chunk2numpy((CHROM,))
        cols = r.header["columns"]
        mc_name, cov_name = f"f{cols.index(cols[0])}", f"f{cols.index(cols[-1])}"
        # chunk2numpy fields are f0..fn positional
        exp_mc = arr["f0"][ids]
        exp_cov = arr[f"f{len(cols) - 1}"][ids]
        r.close()
        assert np.array_equal(mc[i], exp_mc.astype(mc.dtype))
        assert np.array_equal(cov[i], exp_cov.astype(cov.dtype))


def test_iter_windows_partition():
    """Windows must partition the chromosome's sites with no overlap/gap."""
    cells = _first_cells(4)
    _, pos, mc, cov = dl.load_chrom_matrix(
        REFERENCE, cells, chrom=CHROM, index=CGN_INDEX, threads=2)

    total = 0
    prev_end = -1
    loader = dl.CzWindowLoader(REFERENCE, cells, index=CGN_INDEX, threads=2)
    try:
        for w in loader.iter_windows(CHROM, binsize=BINSIZE):
            assert w.mc.shape == w.cov.shape == (len(cells), w.pos.shape[0])
            assert np.all(w.pos >= w.start) and np.all(w.pos < w.end)
            assert w.start >= prev_end  # non-overlapping, ascending
            prev_end = w.end
            total += w.pos.shape[0]
    finally:
        loader.close()
    assert total == pos.shape[0]  # every site assigned to exactly one window


def test_bin_timing():
    """Report the number of cells and the time to fetch the first 10 bins."""
    cells = sorted(glob.glob(os.path.join(CELLS_DIR, "*.cz")))
    print(f"\n[dataloader] #cells = {len(cells)}")

    t0 = time.perf_counter()
    loader = dl.CzWindowLoader(REFERENCE, cells, index=CGN_INDEX, threads=8)
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"[dataloader] loader build: {build_ms:.1f} ms, "
          f"{len(loader.chroms)} chroms")
    try:
        # Time to obtain the first 10 bins (bin 0 also pays the whole-chromosome
        # decompression across all cells; the rest are in-RAM slices).
        t0 = time.perf_counter()
        bins = []
        for w in loader.iter_windows(CHROM, binsize=BINSIZE):
            bins.append(w)
            if len(bins) >= 10:
                break
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        print(f"[dataloader] first {len(bins)} bins on {CHROM}: "
              f"{elapsed_ms:.1f} ms "
              f"(bin0: {bins[0].mc.shape[0]} cells x {bins[0].mc.shape[1]} sites)")
    finally:
        loader.close()


if __name__ == "__main__":
    if not _have_data:
        print("data not found; skipping")
    else:
        test_load_chrom_matrix_shapes()
        test_matrix_matches_bruteforce()
        test_iter_windows_partition()
        test_bin_timing()
        print("dataloader tests passed")
