"""Regression harness for the performance-optimisation work.

Builds a small, fully synthetic dataset (reference fasta, reference .cz,
per-cell allc / .cz files) and exercises every code path touched by the
P1/P2 optimisations. Two uses:

1. ``pytest tests/test_perf_regression.py`` — self-contained invariant
   tests (fast paths vs. reference implementations, round-trips).
2. ``python tests/test_perf_regression.py capture OUTDIR`` — dumps the
   outputs of every exercised API into ``OUTDIR`` (plus ``timings.json``)
   so two code versions can be diffed with
   ``python tests/test_perf_regression.py compare OLD NEW``.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from cytozip.cz import Reader, Writer, _VO_OFFSET_BITS as _VO_BITS  # noqa: E402

CHROM_SIZES = {"chr1": 60000, "chr2": 30000, "chrM": 2000}
N_CELLS = 4
SEED = 7


# ----------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------
def _random_seq(rng, n):
    # Enrich CpG so CG / CH contexts are both well represented.
    alphabet = np.frombuffer(b"ACGTCG", dtype=np.uint8)
    return rng.choice(alphabet, size=n).tobytes().decode()


def build_dataset(root, n_cells=N_CELLS, chrom_sizes=CHROM_SIZES, seed=SEED):
    """Create every synthetic input under *root*; return a dict of paths."""
    from cytozip.allc import AllC, allc2cz

    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    paths = {"root": root}

    fasta = os.path.join(root, "genome.fa")
    with open(fasta, "w") as fh:
        for chrom, n in chrom_sizes.items():
            fh.write(f">{chrom}\n")
            seq = _random_seq(rng, n)
            for i in range(0, n, 60):
                fh.write(seq[i:i + 60] + "\n")
    pysam.faidx(fasta)
    paths["fasta"] = fasta

    ref = os.path.join(root, "ref.cz")
    AllC(genome=fasta, output=ref, jobs=2).run()
    paths["ref"] = ref

    rref = Reader(ref)
    ref_rows = {}
    for dim in rref.chunk_key2offset:
        arr = np.frombuffer(rref.fetch_chunk_bytes(dim),
                            dtype=[("pos", "<u8"), ("strand", "S1"),
                                   ("context", "S3")])
        ref_rows[dim[0]] = arr.copy()
    rref.close()

    cells, allcs, tsvs, nr_cells = [], [], [], []
    for ci in range(n_cells):
        name = f"cell{ci}"
        lines = []
        for chrom, arr in ref_rows.items():
            keep = rng.random(arr.size) < 0.35
            sub = arr[keep]
            cov = rng.integers(1, 6, size=sub.size)
            mc = rng.binomial(cov, 0.7 if ci % 2 == 0 else 0.3)
            for p, s, c, m, v in zip(sub["pos"], sub["strand"],
                                     sub["context"], mc, cov):
                lines.append(f"{chrom}\t{p}\t{s.decode()}\t{c.decode()}"
                             f"\t{m}\t{v}\t1\n")
        tsv = os.path.join(root, f"{name}.allc.tsv")
        with open(tsv, "w") as fh:
            fh.writelines(lines)
        gz = pysam.tabix_index(tsv, seq_col=0, start_col=1, end_col=1,
                               force=True, keep_original=True)
        allcs.append(gz)
        tsvs.append(tsv)
        out = os.path.join(root, f"{name}.cz")
        allc2cz(gz, out, reference=ref, formats=["B", "B"])
        cells.append(out)
        # Reference-free cell .cz (exercises tocz/_int_slab_parser + delta).
        nr = os.path.join(root, f"{name}.noref.cz")
        w = Writer(nr, formats=["Q", "H", "H"], columns=["pos", "mc", "cov"],
                   chunk_dims=["chrom"], sort_col="pos", delta_cols=["pos"])
        w.tocz(input=tsv, usecols=[1, 4, 5], key_cols=[0])
        w.close()
        nr_cells.append(nr)
    paths.update(cells=cells, allcs=allcs, tsvs=tsvs, noref_cells=nr_cells)

    regions = os.path.join(root, "regions.bed")
    with open(regions, "w") as fh:
        for chrom, n in chrom_sizes.items():
            for s in range(1, n, max(1, n // 7)):
                fh.write(f"{chrom}\t{s}\t{min(n, s + 900)}\n")
    paths["regions"] = regions
    return paths


# ----------------------------------------------------------------------
# Golden capture
# ----------------------------------------------------------------------
@contextlib.contextmanager
def _stdout_to(path):
    old = sys.stdout
    with open(path, "w") as fh:
        sys.stdout = fh
        try:
            yield
        finally:
            sys.stdout = old


def _cz_digest(path):
    """Header + decoded per-chunk bytes digest (robust to block layout)."""
    r = Reader(path)
    h = hashlib.md5()
    h.update(repr((r.header["formats"], r.header["columns"],
                   r.header["chunk_dims"])).encode())
    for dim in r.chunk_key2offset:
        h.update(repr(dim).encode())
        h.update(r.fetch_chunk_bytes(dim))
    r.close()
    return h.hexdigest()


def _gz_text(path):
    with gzip.open(path, "rt") as fh:
        return fh.read()


def capture(outdir, data=None):
    """Run every exercised API and store outputs + timings in *outdir*."""
    os.makedirs(outdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="czperf_")
    timings, digests = {}, {}

    def _t(name, fn):
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as e:  # record failures so the baseline still completes
            timings[name] = f"error: {type(e).__name__}: {e}"
            return None
        timings[name] = round(time.perf_counter() - t0, 4)
        return res

    try:
        d = data or _t("build_dataset", lambda: build_dataset(
            os.path.join(work, "data")))
        digests["ref.cz"] = _cz_digest(d["ref"])
        for p in d["cells"] + d["noref_cells"]:
            digests[os.path.basename(p)] = _cz_digest(p)

        # Reader.query / summaries (stdout)
        r = Reader(d["noref_cells"][0])
        _t("query_regions_print", lambda: _with_stdout(
            os.path.join(outdir, "query_regions.tsv"),
            lambda: r.query(regions=d["regions"], printout=True)))
        _t("summary_blocks", lambda: _with_stdout(
            os.path.join(outdir, "summary_blocks.tsv"),
            lambda: r.summary_blocks(printout=True)))
        _t("summary_chunks", lambda: _with_stdout(
            os.path.join(outdir, "summary_chunks.tsv"),
            lambda: r.summary_chunks(printout=True)))
        r.close()
        rc = Reader(d["cells"][0])
        _t("query_ref_print", lambda: _with_stdout(
            os.path.join(outdir, "query_ref.tsv"),
            lambda: rc.query(regions=d["regions"], reference=d["ref"],
                             printout=True)))
        rc.close()

        # Random block access on the delta-encoded reference
        rr = Reader(d["ref"])
        rng = np.random.default_rng(1)
        h = hashlib.md5()

        def _rand():
            for _ in range(3):
                for dim in rr.chunk_key2offset:
                    rr._load_chunk(rr.chunk_key2offset[dim], jump=False)
                    starts = [vo >> _VO_BITS for vo in
                              rr._chunk_block_1st_record_virtual_offsets]
                    for i in rng.permutation(len(starts)):
                        rr._block_start_offset = -1
                        rr._load_block(starts[i])
                        h.update(bytes(rr._buffer))
        _t("random_block_loads", _rand)
        digests["random_block_loads"] = h.hexdigest()
        rr.close()

        # to_bgzip
        tb = os.path.join(work, "cell0.tsv.gz")
        rc = Reader(d["cells"][0])
        _t("to_bgzip", lambda: rc.to_bgzip(tb, reference=d["ref"]))
        rc.close()
        with open(os.path.join(outdir, "to_bgzip.tsv"), "w") as fh:
            fh.write(_gz_text(tb))

        # merge_cz (text output with bgzip/tabix)
        from cytozip.merge import merge_cz
        mo = os.path.join(work, "merged.cz")
        _t("merge_cz", lambda: merge_cz(input=d["cells"], output=mo, jobs=2,
                                        formats=["H", "H"], bgzip=False))
        digests["merged.cz"] = _cz_digest(mo)

        # pivot
        from cytozip.pivot import pivot_fraction, pivot_fisher
        for name, fn in (("pivot_fraction", pivot_fraction),
                         ("pivot_fisher", pivot_fisher)):
            po = os.path.join(work, f"{name}.txt")
            try:
                _t(name, lambda: fn(indir=d["root"], cz_paths=[
                    os.path.basename(c) for c in d["cells"]], output=po,
                    jobs=2, reference=d["ref"], bgzip=True))
                src = po + ".gz" if os.path.exists(po + ".gz") else po
                text = _gz_text(src) if src.endswith(".gz") else open(src).read()
                with open(os.path.join(outdir, f"{name}.tsv"), "w") as fh:
                    fh.write(text)
            except ImportError as e:  # fast_fisher missing
                timings[name] = f"skipped: {e}"

        # DMR 1-vs-1 Fisher kernel
        from cytozip.dmr import _fisher_1v1_sites
        rng = np.random.default_rng(3)
        cov = rng.integers(0, 30, size=(2, 200000))
        mc = rng.binomial(cov, 0.5)
        keep = np.arange(cov.shape[1])
        p, da = _t("fisher_1v1", lambda: _fisher_1v1_sites(mc, cov, keep))
        np.savez(os.path.join(outdir, "fisher_1v1.npz"), p=p, da=da)

        # peaks pseudo-reads
        from cytozip.peaks import _write_pseudo_reads
        pos = np.sort(rng.choice(10_000_000, 200000, replace=False)) + 1
        sig = rng.integers(0, 6, size=pos.size)
        buf = io.StringIO()
        _t("pseudo_reads", lambda: _write_pseudo_reads(buf, "chr1", pos, sig,
                                                       150))
        with open(os.path.join(outdir, "pseudo_reads.bed"), "w") as fh:
            fh.write(buf.getvalue())

        # dataloader
        from cytozip.dataloader import CzWindowLoader
        ld = CzWindowLoader(d["ref"], d["cells"], threads=2)
        sums = []

        def _dl():
            for chrom in CHROM_SIZES:
                for w in ld.iter_windows(chrom, binsize=500):
                    sums.append(int(np.asarray(w.mc).sum())
                                + int(np.asarray(w.cov).sum()))
                pos_, mc_, cov_ = ld.load_chrom(chrom)
                sums.append(int(mc_.sum()) + int(cov_.sum()))
        _t("dataloader", _dl)
        ld.close()
        with open(os.path.join(outdir, "dataloader.json"), "w") as fh:
            json.dump(sums, fh)

        # bam reference sequence loader
        from cytozip.bam import _get_chromosome_sequence_upper, _read_faidx
        fai = _read_faidx(d["fasta"] + ".fai")
        seqs = _t("bam_fasta", lambda: [
            _get_chromosome_sequence_upper(d["fasta"], fai, c)
            for c in CHROM_SIZES])
        digests["bam_fasta"] = hashlib.md5("".join(seqs).encode()).hexdigest()

        # model deconvolution solver
        from cytozip.model import _solve_fractions
        fr = []

        def _solve():
            for k in range(40):
                rg = np.random.default_rng(100 + k)
                th = rg.random((20000, 6))
                f = rg.dirichlet(np.ones(6))
                b = th @ f + rg.normal(0, 0.05, 20000)
                w = rg.integers(1, 20, 20000)
                fr.append(_solve_fractions(th, b, weight=w)[0].tolist())
                fr.append(_solve_fractions(th, b, weight=w,
                                           allow_unknown=True)[0].tolist())
        _t("solve_fractions", _solve)
        with open(os.path.join(outdir, "solve_fractions.json"), "w") as fh:
            json.dump(fr, fh)

        # array2cz reference
        from cytozip.array import array2cz
        man = pd.DataFrame({
            "probe_id": [f"cg{i:07d}" for i in range(3000)],
            "chrom": np.repeat(["chr1", "chr2", "chr3"], 1000),
            "pos": np.tile(np.arange(1, 1001) * 7, 3),
            "gene": [f"G{i % 37}" for i in range(3000)],
        })
        df = pd.DataFrame({"sample_id": "s1", "probe_id": man.probe_id,
                           "beta": np.linspace(0, 1, 3000)})
        aref = os.path.join(work, "array_ref.cz")
        _t("array2cz", lambda: array2cz(df, man, os.path.join(work, "arr"),
                                        output_reference=aref,
                                        overwrite=True))
        digests["array_ref.cz"] = _cz_digest(aref)

        with open(os.path.join(outdir, "digests.json"), "w") as fh:
            json.dump(digests, fh, indent=1, sort_keys=True)
        with open(os.path.join(outdir, "timings.json"), "w") as fh:
            json.dump(timings, fh, indent=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return timings


def _with_stdout(path, fn):
    with _stdout_to(path):
        return fn()


def compare(old, new):
    """Print a per-artifact equality report between two capture dirs."""
    ok = True
    for name in sorted(set(os.listdir(old)) | set(os.listdir(new))):
        if name == "timings.json":
            continue
        a, b = os.path.join(old, name), os.path.join(new, name)
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"MISSING {name}")
            ok = False
            continue
        same = open(a, "rb").read() == open(b, "rb").read()
        print(f"{'SAME' if same else 'DIFF'} {name}")
        ok &= same
    ta = json.load(open(os.path.join(old, "timings.json")))
    tb = json.load(open(os.path.join(new, "timings.json")))
    print(f"\n{'step':<22}{'old(s)':>10}{'new(s)':>10}{'speedup':>9}")
    for k in ta:
        x, y = ta[k], tb.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            print(f"{k:<22}{x:>10.4f}{y:>10.4f}{x / max(y, 1e-9):>8.2f}x")
    return ok


# ----------------------------------------------------------------------
# pytest invariants
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_dataset(str(tmp_path_factory.mktemp("czperf")))


def test_noref_tocz_roundtrip(dataset):
    for tsv, cz in zip(dataset["tsvs"], dataset["noref_cells"]):
        df = pd.read_csv(tsv, sep="\t", header=None)
        r = Reader(cz)
        got = r.to_df()
        r.close()
        assert len(got) == len(df)
        assert np.array_equal(got["pos"].to_numpy(np.int64),
                              df[1].to_numpy(np.int64))
        assert np.array_equal(got["mc"].to_numpy(np.int64),
                              df[4].to_numpy(np.int64))
        assert np.array_equal(got["cov"].to_numpy(np.int64),
                              df[5].to_numpy(np.int64))


def test_delta_random_block_access(dataset):
    """Random-access block loads on a delta file match a full chunk decode."""
    r = Reader(dataset["ref"])
    assert r._delta_cols
    rng = np.random.default_rng(0)
    for dim in r.chunk_key2offset:
        full = r.fetch_chunk_bytes(dim)
        r._load_chunk(r.chunk_key2offset[dim], jump=False)
        vos = list(r._chunk_block_1st_record_virtual_offsets)
        starts = [vo >> _VO_BITS for vo in vos]
        decoded = []
        for i in list(rng.permutation(len(starts))) + list(range(len(starts))):
            r._load_block(starts[i])
            decoded.append((i, bytes(r._buffer)))
        seq = b"".join(b for i, b in decoded[len(starts):])
        assert seq == full
        for i, b in decoded[:len(starts)]:
            assert b == decoded[len(starts) + i][1]
    r.close()


def test_block_cache_consistency(dataset):
    """LRU-cached block loads: identical buffers, correct sequential
    continuation after a cache hit, bounded size."""
    r = Reader(dataset["ref"], max_cache=4)
    dim = ("chr1",)
    full = r.fetch_chunk_bytes(dim)
    r._load_chunk(r.chunk_key2offset[dim], jump=False)
    vos = [vo >> _VO_BITS for vo in r._chunk_block_1st_record_virtual_offsets]
    blocks = []
    for st in vos:
        r._block_start_offset = -1
        r._load_block(st)
        blocks.append(bytes(r._buffer))
    assert b"".join(blocks) == full
    assert len(r._block_cache) <= 4
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(vos), 40):
        r._block_start_offset = -1
        r._load_block(vos[i])
        assert r._buffer == blocks[i]
        if i + 1 < len(vos):
            r._load_block()  # sequential continuation after a (possible) hit
            assert r._buffer == blocks[i + 1]
    # cached random-id lookups equal the direct per-record decode
    ids = np.sort(rng.choice(len(full) // 12, 500, replace=False)) + 1
    got = r.get_records_by_ids_bytes(dim, ids)
    exp = b"".join(full[(i - 1) * 12:i * 12] for i in ids)
    assert got == exp
    assert b"".join(r._getRecordsByIds(dim, ids)) == exp
    r.close()
    assert not r._block_cache


def test_query_regions_file_matches_list(dataset, tmp_path):
    r = Reader(dataset["noref_cells"][0])
    df = pd.read_csv(dataset["regions"], sep="\t", header=None)
    regions = [[(c,), s, e] for c, s, e in df.itertuples(index=False)]
    a = list(r.query(regions=regions, printout=False))
    b = list(r.query(regions=dataset["regions"], printout=False))
    r.close()
    assert a == b and len(a) > 0


def test_query_printout_matches_iter(dataset, tmp_path):
    r = Reader(dataset["noref_cells"][1])
    out = tmp_path / "q.tsv"
    _with_stdout(str(out), lambda: r.query(regions=dataset["regions"],
                                           printout=True))
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == r.header["chunk_dims"] + r.header["columns"]
    assert len(lines) > 10
    r.close()


def test_query_with_reference_matches_numpy(dataset, tmp_path):
    """Multi-region reference-join query == independent numpy join."""
    cell, ref = dataset["cells"][0], dataset["ref"]
    rr, rc = Reader(ref), Reader(cell)
    rdt = [("pos", "<u8"), ("strand", "S1"), ("context", "S3")]
    cdt = [("mc", "u1"), ("cov", "u1")]
    df = pd.read_csv(dataset["regions"], sep="\t", header=None)
    exp = []
    for c, s, e in df.itertuples(index=False):
        ra = np.frombuffer(rr.fetch_chunk_bytes((c,)), dtype=rdt)
        ca = np.frombuffer(rc.fetch_chunk_bytes((c,)), dtype=cdt)
        idx = np.flatnonzero((ra["pos"] >= s) & (ra["pos"] <= e))
        for i in idx:
            exp.append([c, str(ra["pos"][i]), ra["strand"][i].decode(),
                        ra["context"][i].decode(), str(ca["mc"][i]),
                        str(ca["cov"][i])])
    rr.close()
    got = [[str(x) for x in row] for row in
           rc.query(regions=dataset["regions"], reference=ref,
                    printout=False)]
    assert got == exp
    out = tmp_path / "q.tsv"
    _with_stdout(str(out), lambda: rc.query(regions=dataset["regions"],
                                            reference=ref, printout=True))
    lines = out.read_text().splitlines()
    assert [ln.split("\t") for ln in lines[1:]] == exp
    rc.close()


def test_to_bgzip_matches_pandas(dataset, tmp_path):
    """Template serialisation == the previous DataFrame.to_csv output."""
    cell, ref = dataset["cells"][0], dataset["ref"]
    out = str(tmp_path / "c.tsv.gz")
    r = Reader(cell)
    r.to_bgzip(out, reference=ref, allc_format=True, cov_col="cov")
    r.close()
    rr, rc = Reader(ref), Reader(cell)
    rdt = [("pos", "<u8"), ("strand", "S1"), ("context", "S3")]
    cdt = [("mc", "u1"), ("cov", "u1")]
    exp = []
    for dim in rc.chunk_key2offset:
        a = np.frombuffer(rc.fetch_chunk_bytes(dim), dtype=cdt)
        ra = np.frombuffer(rr.fetch_chunk_bytes(dim), dtype=rdt)
        m = a["cov"] > 0
        df = pd.DataFrame({"_chrom": dim[0], "pos": ra["pos"][m],
                           "strand": ra["strand"][m], "context": ra["context"][m],
                           "mc": a["mc"][m], "cov": a["cov"][m], "flag": 1})
        for c in ("strand", "context"):
            df[c] = df[c].str.decode("utf-8")
        exp.append(df.to_csv(None, sep="\t", header=False, index=False))
    rr.close()
    rc.close()
    assert _gz_text(out) == "".join(exp)


def test_array_reference_matches_struct(tmp_path):
    import struct
    from cytozip.array import _write_reference_cz
    man = pd.DataFrame({"chrom": ["c1"] * 5 + ["c2"] * 3,
                        "pos": [1, 5, 9, 70000, 3, 2, 4, 6],
                        "gene": ["a", "bb", None, "dddddd", "e", "", "ff", "g"],
                        "w": np.linspace(0, 1, 8)})
    out = str(tmp_path / "ref.cz")
    cols = ["pos", "gene", "w"]
    _write_reference_cz(out, man, "chrom", cols, man.groupby("chrom", sort=False))
    r = Reader(out)
    fmts = r.header["formats"]
    for chrom, g in man.groupby("chrom", sort=False):
        n = int(fmts[1][:-1])
        exp = b"".join(struct.pack("<" + "".join(fmts), int(p),
                                   str(s).encode()[:n], float(w))
                       for p, s, w in zip(g.pos, g.gene, g.w))
        assert r.fetch_chunk_bytes((chrom,)) == exp
    r.close()


def test_estimate_theta_float32_exact():
    from cytozip.model import estimate_theta
    rng = np.random.default_rng(4)
    cov = rng.integers(0, 70000, 200000)
    mc = rng.integers(0, cov + 1)
    exp = ((mc.astype(np.float64) + 1) / (cov + 2.0)).astype(np.float32)
    assert np.array_equal(estimate_theta(mc, cov), exp)


def test_rms_permutation_reproducible_across_threads():
    """Same seed -> identical p-values regardless of thread count/schedule."""
    from cytozip.dmr_accel import rms_run_sites
    rng = np.random.default_rng(12)
    n_cells, n_sites = 8, 3000
    cov = rng.integers(0, 12, size=(n_cells, n_sites)).astype(np.int64)
    mc = rng.binomial(cov, np.where(np.arange(n_cells) < 4, 0.2, 0.7)[:, None])
    mc = np.ascontiguousarray(mc.astype(np.int64))
    idx = np.arange(n_sites, dtype=np.int64)
    p1, d1 = rms_run_sites(mc, cov, idx, 4, n_permute=300, n_threads=1, seed=7)
    p2, d2 = rms_run_sites(mc, cov, idx, 4, n_permute=300, n_threads=4, seed=7)
    p3, _ = rms_run_sites(mc, cov, idx, 4, n_permute=300, n_threads=4, seed=8)
    assert np.array_equal(p1, p2) and np.array_equal(d1, d2)
    assert not np.array_equal(p1, p3)
    assert p1.max() <= 1.0
    # malformed mc > cov is clamped: |delta| <= 1
    bad = mc.copy()
    bad[0] = cov[0] + 5
    _, d = rms_run_sites(bad, cov, idx, 4, n_permute=50, seed=1)
    assert np.abs(d).max() <= 1.0


def test_peaks_index_rows_are_one_based(dataset, tmp_path):
    """Context-index IDs are 1-based: peaks must select CG sites only."""
    from cytozip.index import index_context
    from cytozip.peaks import _pseudo_reads_chrom_worker
    idx = str(tmp_path / "cgn.cz")
    index_context(dataset["ref"], idx, pattern="CGN", jobs=1)
    rr, ir = Reader(dataset["ref"]), Reader(idx)
    rdt = [("pos", "<u8"), ("strand", "S1"), ("context", "S3")]
    for dim in rr.chunk_key2offset:
        ra = np.frombuffer(rr.fetch_chunk_bytes(dim), dtype=rdt)
        rows = ir.get_index_rows(dim, n_rows=ra.size)
        assert rows.min() >= 0 and rows.max() < ra.size
        assert all(c.startswith(b"CG") for c in ra["context"][rows].tolist())
        # 1-based raw IDs vs 0-based rows
        assert np.array_equal(ir.get_ids_from_index(dim).astype(np.int64) - 1,
                              rows)
    cdt = [("mc", "u1"), ("cov", "u1")]
    rc = Reader(dataset["cells"][0])
    out = str(tmp_path / "pr")
    os.makedirs(out)
    chrom, bed, n, _, _ = _pseudo_reads_chrom_worker(
        ("chr1",), dataset["cells"][0], dataset["ref"], idx, "mc", "cov",
        1, 150, None, out, "x")
    ra = np.frombuffer(rr.fetch_chunk_bytes(("chr1",)), dtype=rdt)
    ca = np.frombuffer(rc.fetch_chunk_bytes(("chr1",)), dtype=cdt)
    rows = ir.get_index_rows(("chr1",))
    exp = int((ca["cov"][rows] - ca["mc"][rows])[ca["cov"][rows] >= 1].sum())
    assert n == exp
    rr.close(); ir.close(); rc.close()


def test_array2cz_missing_probes_become_nan(tmp_path):
    from cytozip.array import array2cz, dequantize_cz
    man = pd.DataFrame({"probe_id": [f"cg{i}" for i in range(6)],
                        "chrom": ["c1"] * 6, "pos": np.arange(1, 7) * 10})
    df = pd.DataFrame({"sample_id": ["s1"] * 4, "probe_id": ["cg0", "cg2", "cg3", "cg5"],
                       "beta": [0.1, 0.2, 0.3, 0.4], "n": [1, 2, 3, 4]})
    paths = array2cz(df, man, str(tmp_path / "o"), output_reference=str(tmp_path / "r.cz"),
                     value_cols=("beta", "n"), overwrite=True)
    r = Reader(paths[0])
    got = np.frombuffer(r.fetch_chunk_bytes(("c1",)), dtype=[("beta", "<f4"), ("n", "u1")])
    r.close()
    assert got.size == 6
    assert np.isnan(got["beta"][[1, 4]]).all() and np.allclose(got["beta"][[0, 2, 3, 5]], [0.1, 0.2, 0.3, 0.4])
    assert got["n"].tolist() == [1, 0, 2, 3, 0, 4]
    with pytest.raises(ValueError, match="not in manifest"):
        array2cz(df.assign(probe_id=["cg0", "zz", "cg3", "cg5"]), man,
                 str(tmp_path / "o2"), value_cols=("beta",), overwrite=True)


def test_index_regions_drops_inverted(dataset, tmp_path):
    from cytozip.index import index_regions
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t100\t900\tA\nchr1\t500\t500\tB\nchr1\t800\t200\tC\nchr2\t10\t50\tD\n")
    out = str(tmp_path / "ridx.cz")
    index_regions(dataset["ref"], out, bed=str(bed), jobs=1)
    r = Reader(out)
    ids = r.get_ids_from_index(("chr1",))
    assert ids.shape == (1, 2) and ids[0, 1] > ids[0, 0]
    r.close()


def test_catcz_skips_empty_source(dataset, tmp_path):
    from cytozip.cz import check_cz
    empty = str(tmp_path / "empty.cz")
    Writer(empty, formats=["B", "B"], columns=["mc", "cov"], chunk_dims=["chrom"]).close()
    assert os.path.getsize(empty) > 0
    ok, reason = check_cz(empty)
    assert not ok and "no chunks" in reason
    assert check_cz(dataset["cells"][0]) == (True, "ok")
    trunc = str(tmp_path / "trunc.cz")
    with open(dataset["cells"][0], "rb") as fh, open(trunc, "wb") as out:
        out.write(fh.read()[:-40])
    assert check_cz(trunc)[0] is False
    out = str(tmp_path / "cat.cz")
    w = Writer(out, formats=["B", "B"], columns=["mc", "cov"], chunk_dims=["chrom"])
    w.catcz(input=[dataset["cells"][0], empty, dataset["cells"][1]], key_added="cell_id")
    r = Reader(out)
    assert len(r.chunk_key2offset) == 2 * len(CHROM_SIZES)
    r.close()


def test_merge_dms_to_dmr_carries_rates():
    from cytozip.dmr import _merge_dms_to_dmr
    df = pd.DataFrame({"chrom": ["c"] * 3, "pos": [1, 10, 500], "p": [.1, .2, .3],
                       "delta": [.5, .4, .3], "rate_a": [1., 2., 3.], "rate_b": [.5, .5, .5]})
    out = _merge_dms_to_dmr(df, max_dist=100, min_dms=1)
    assert out.shape[0] == 2 and out["rate_a"].tolist() == [1.5, 3.0]
    out2 = _merge_dms_to_dmr(df[["chrom", "pos", "p", "delta"]], max_dist=100, min_dms=1)
    assert "rate_a" not in out2.columns


def test_dataloader_index_windows_match_bruteforce(dataset, tmp_path):
    """Context-filtered windows (sparse per-block gather) == numpy gather,
    for both non-delta and delta-encoded (record-aligned) cells."""
    from cytozip.index import index_context
    from cytozip.dataloader import CzWindowLoader
    idx = str(tmp_path / "cgn.cz")
    index_context(dataset["ref"], idx, pattern="CGN", jobs=1)
    # delta-encoded copy of cell1 (same rows, different block geometry)
    r1 = Reader(dataset["cells"][1])
    dcell = str(tmp_path / "cell1_delta.cz")
    w = Writer(dcell, formats=r1.header["formats"], columns=r1.header["columns"],
               chunk_dims=["chrom"], delta_cols=["mc"])
    for dim in r1.chunk_key2offset:
        w.write_chunk(r1.fetch_chunk_bytes(dim), list(dim))
    w.close()
    r1.close()
    cells = [dataset["cells"][0], dcell]
    ld = CzWindowLoader(dataset["ref"], cells, index=idx, threads=2)
    ir = Reader(idx)
    cdt = [("mc", "u1"), ("cov", "u1")]
    rdt = [("pos", "<u8"), ("strand", "S1"), ("context", "S3")]
    rr = Reader(dataset["ref"])
    for chrom in CHROM_SIZES:
        rows = ir.get_index_rows((chrom,))
        pos = np.frombuffer(rr.fetch_chunk_bytes((chrom,)), dtype=rdt)["pos"][rows].astype(np.int64)
        exp = []
        for c in cells:
            rc = Reader(c)
            a = np.frombuffer(rc.fetch_chunk_bytes((chrom,)), dtype=cdt)[rows]
            rc.close()
            exp.append((a["mc"].astype(np.int64), a["cov"].astype(np.int64)))
        got_mc = np.zeros((2, rows.size), np.int64)
        got_cov = np.zeros((2, rows.size), np.int64)
        seen = np.zeros(rows.size, bool)
        for wdw in ld.iter_windows(chrom, binsize=700, max_sites=50):
            p = np.asarray(wdw.pos)
            j = np.searchsorted(pos, p)
            assert np.array_equal(pos[j], p)
            got_mc[:, j] = np.asarray(wdw.mc)
            got_cov[:, j] = np.asarray(wdw.cov)
            seen[j] = True
        assert seen.all()
        for i in range(2):
            assert np.array_equal(got_mc[i], exp[i][0])
            assert np.array_equal(got_cov[i], exp[i][1])
        # single-region path
        lo, hi = int(pos[10]), int(pos[60])
        rpos, rmc, rcov = ld.load_region(chrom, lo, hi + 1)
        m = (pos >= lo) & (pos <= hi)
        assert np.array_equal(np.asarray(rpos), pos[m])
        assert np.array_equal(np.asarray(rmc)[1], exp[1][0][m])
        assert np.array_equal(np.asarray(rcov)[0], exp[0][1][m])
    ld.close(); ir.close(); rr.close()


def test_cz_to_anndata_sites(dataset, tmp_path):
    pytest.importorskip("anndata")
    from cytozip.index import index_context
    from cytozip.features import cz_to_anndata_sites, _compute_beta_params_sparse
    idx = str(tmp_path / "cgn.cz")
    index_context(dataset["ref"], idx, pattern="CGN", jobs=1)
    cells = dataset["cells"]
    rdt = [("pos", "<u8"), ("strand", "S1"), ("context", "S3")]
    cdt = [("mc", "u1"), ("cov", "u1")]
    rr, ir = Reader(dataset["ref"]), Reader(idx)
    # dense truth over CG sites of chr1+chr2 (in that order)
    pos_all, mc_all, cov_all, ctx_all = [], [], [], []
    for chrom in ("chr1", "chr2"):
        rows = ir.get_index_rows((chrom,))
        ra = np.frombuffer(rr.fetch_chunk_bytes((chrom,)), dtype=rdt)[rows]
        pos_all.append(np.array([f"{chrom}:{p}" for p in ra["pos"]]))
        ctx_all.append(ra["context"].astype(str))
        m, c = [], []
        for cell in cells:
            rc = Reader(cell)
            a = np.frombuffer(rc.fetch_chunk_bytes((chrom,)), dtype=cdt)[rows]
            rc.close()
            m.append(a["mc"].astype(np.int64)); c.append(a["cov"].astype(np.int64))
        mc_all.append(np.stack(m)); cov_all.append(np.stack(c))
    rr.close(); ir.close()
    names = np.concatenate(pos_all); MC = np.concatenate(mc_all, 1); COV = np.concatenate(cov_all, 1)
    labels = [os.path.basename(c)[:-3] for c in cells]

    ad = cz_to_anndata_sites(cells, dataset["ref"], index=idx, chroms=["chr1", "chr2"],
                             score="frac", jobs=1)
    assert list(ad.obs_names) == labels and list(ad.var_names) == list(names)
    assert ad.var["context"].str.startswith("CG").all()
    assert np.array_equal(ad.layers["mc"].toarray(), MC)
    assert np.array_equal(ad.layers["cov"].toarray(), COV)
    exp_frac = np.where(COV > 0, MC / np.maximum(COV, 1), 0.0)
    assert np.allclose(ad.X.toarray(), exp_frac, atol=1e-6)

    # posterior, parallel, catcz'd input, min_cov
    cat = str(tmp_path / "cat.cz")
    w = Writer(cat, formats=["B", "B"], columns=["mc", "cov"], chunk_dims=["chrom"])
    w.catcz(input=cells, key_added="cell_id")
    ad2 = cz_to_anndata_sites(cat, dataset["ref"], index=idx, chroms=["chr1", "chr2"],
                              score="posterior_frac", min_cov=2, jobs=2,
                              output=str(tmp_path / "sites.h5ad"))
    assert sorted(ad2.obs_names) == sorted(labels)
    order = [list(ad2.obs_names).index(l) for l in labels]
    COV2 = np.where(COV >= 2, COV, 0); MC2 = np.where(COV >= 2, MC, 0)
    assert np.array_equal(ad2.layers["cov"].toarray()[order], COV2)
    a, b, _ = _compute_beta_params_sparse(ad2.layers["mc"], ad2.layers["cov"])
    X = ad2.X.toarray()[order]
    a = a[order]; b = b[order]
    exp = np.where(COV2 > 0, (MC2 + a[:, None]) / (COV2 + a[:, None] + b[:, None]), 0.0)
    assert np.allclose(X, exp, atol=1e-5)
    assert os.path.exists(str(tmp_path / "sites.h5ad"))
    # no index -> all cytosines, single chrom, pos var_names
    ad3 = cz_to_anndata_sites(cells[:2], dataset["ref"], chroms=["chrM"], score="cov",
                              var_names="pos")
    rc = Reader(cells[0]); n = len(rc.fetch_chunk_bytes(("chrM",))) // 2; rc.close()
    assert ad3.shape == (2, n) and ad3.var_names[0].isdigit()


def test_fisher_1v1_matches_scipy():
    from scipy.stats import fisher_exact
    from cytozip.dmr import _fisher_1v1_sites
    rng = np.random.default_rng(5)
    cov = rng.integers(0, 25, size=(2, 3000))
    mc = rng.binomial(cov, 0.4)
    keep = np.flatnonzero(rng.random(3000) < 0.8)
    p, da = _fisher_1v1_sites(mc, cov, keep)
    for j, i in enumerate(keep):
        ca, cb = cov[0, i], cov[1, i]
        if ca <= 0 or cb <= 0:
            assert p[j] == 1.0 and da[j] == 0.0
            continue
        exp = fisher_exact([[mc[0, i], ca - mc[0, i]],
                            [mc[1, i], cb - mc[1, i]]])[1]
        assert abs(p[j] - exp) <= 1e-6 * max(exp, 1e-300) + 1e-12
        assert da[j] == pytest.approx(mc[0, i] / ca - mc[1, i] / cb)


def test_pseudo_reads_output():
    from cytozip.peaks import _write_pseudo_reads
    rng = np.random.default_rng(2)
    pos = np.sort(rng.choice(1_000_000, 5000, replace=False)) + 1
    sig = rng.integers(0, 5, size=pos.size)
    buf = io.StringIO()
    n = _write_pseudo_reads(buf, "chrX", pos, sig, 100)
    exp = np.repeat(pos[sig > 0], sig[sig > 0])
    lines = buf.getvalue().splitlines()
    assert n == exp.size == len(lines)
    got = np.array([int(x.split("\t")[1]) for x in lines])
    assert np.array_equal(got, np.maximum(0, exp - 100))


def test_bam_fasta_loader(dataset):
    from cytozip.bam import _get_chromosome_sequence_upper, _read_faidx
    fai = _read_faidx(dataset["fasta"] + ".fai")
    fa = pysam.FastaFile(dataset["fasta"])
    for c in CHROM_SIZES:
        assert _get_chromosome_sequence_upper(dataset["fasta"], fai, c) == \
            fa.fetch(c).upper()
    fa.close()


def test_dataloader_windows_match_load_chrom(dataset):
    from cytozip.dataloader import CzWindowLoader
    ld = CzWindowLoader(dataset["ref"], dataset["cells"], threads=2)
    for chrom in CHROM_SIZES:
        pos, mc, cov = ld.load_chrom(chrom)
        tot = 0
        for w in ld.iter_windows(chrom, binsize=500):
            tot += int(np.asarray(w.cov).sum())
        assert tot == int(cov.sum())
    ld.close()


def test_merge_cz_sum(dataset, tmp_path):
    from cytozip.merge import merge_cz
    out = str(tmp_path / "m.cz")
    merge_cz(input=dataset["cells"], output=out, jobs=2, formats=["H", "H"],
             bgzip=False)
    rm = Reader(out)
    for dim in rm.chunk_key2offset:
        tot = None
        for c in dataset["cells"]:
            r = Reader(c)
            a = np.frombuffer(r.fetch_chunk_bytes(dim),
                              dtype=[("mc", "u1"), ("cov", "u1")])
            r.close()
            s = np.stack([a["mc"], a["cov"]]).astype(np.int64)
            tot = s if tot is None else tot + s
        m = np.frombuffer(rm.fetch_chunk_bytes(dim),
                          dtype=[("mc", "<u2"), ("cov", "<u2")])
        assert np.array_equal(m["mc"], tot[0]) and \
            np.array_equal(m["cov"], tot[1])
    rm.close()


def test_pivot_fraction_bgzip(dataset, tmp_path):
    from cytozip.pivot import pivot_fraction
    out = str(tmp_path / "frac.txt")
    pivot_fraction(indir=dataset["root"],
                   cz_paths=[os.path.basename(c) for c in dataset["cells"]],
                   output=out, jobs=2, reference=dataset["ref"], bgzip=True)
    assert os.path.exists(out + ".gz") and os.path.exists(out + ".gz.tbi")
    df = pd.read_csv(out + ".gz", sep="\t")
    names = [os.path.basename(c)[:-3] for c in dataset["cells"]]
    assert list(df.columns[-len(names):]) == names
    r = Reader(dataset["cells"][0])
    mc_cov = np.concatenate([
        np.frombuffer(r.fetch_chunk_bytes((c,)),
                      dtype=[("mc", "u1"), ("cov", "u1")])
        for c in df.iloc[:, 0].unique()])
    r.close()
    frac = np.where(mc_cov["cov"] == 0, 0.0,
                    mc_cov["mc"] / np.maximum(mc_cov["cov"], 1))
    assert np.allclose(df[names[0]].to_numpy(float), frac, atol=5e-3)


def test_pivot_fisher_matches_scalar(dataset):
    pytest.importorskip("fast_fisher")
    from fast_fisher.fast_fisher_cython import test1t, odds_ratio
    from cytozip.pivot import _fisher_worker_fast
    rng = np.random.default_rng(9)
    cov = rng.integers(0, 8, size=(4000, 3))
    mc = rng.binomial(cov, 0.5)
    df = pd.DataFrame({f"c{j}.{k}": (mc if k == "mc" else cov)[:, j]
                       for j in range(3) for k in ("mc", "cov")})
    out = _fisher_worker_fast(df)
    mcs, ucs = mc.sum(1), (cov - mc).sum(1)
    for j in range(3):
        for i in range(0, 4000, 37):
            a, b = int(mc[i, j]), int(cov[i, j] - mc[i, j])
            c, d = int(mcs[i] - a), int(ucs[i] - b)
            assert out[f"c{j}.odd_ratio"].iloc[i] == "%.3g" % odds_ratio(a, b, c, d)
            assert out[f"c{j}.pval"].iloc[i] == "%.3g" % test1t(a, b, c, d)


def test_solve_fractions_simplex():
    from cytozip.model import _solve_fractions
    rng = np.random.default_rng(11)
    for k in range(10):
        th = rng.random((3000, 5))
        f = rng.dirichlet(np.ones(5))
        b = th @ f + rng.normal(0, 0.02, 3000)
        frac, r2 = _solve_fractions(th, b, weight=rng.integers(1, 9, 3000))
        assert frac.min() >= 0 and abs(frac.sum() - 1) < 1e-9
        assert np.abs(frac - f).max() < 0.05
        frac_u, _ = _solve_fractions(th, b * 0.6, allow_unknown=True)
        assert frac_u.sum() <= 1 + 1e-9


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        print(json.dumps(capture(sys.argv[2]), indent=1))
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    else:
        print(__doc__)
