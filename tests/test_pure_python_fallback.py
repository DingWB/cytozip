"""The pure-Python read paths must match the Cython accelerators exactly.

A build without ``cytozip.cz_accel`` (no compiler / libdeflate, or an
unsupported platform) falls back to ``zlib`` + numpy. Whole-chunk reads of
**delta-encoded** files used to skip the per-block delta decode in that
fallback and silently return raw differences as positions; this test pins
the fix by scoring every read path with and without the accelerators.

Run with::

    pytest tests/test_pure_python_fallback.py
"""
import numpy as np
import pytest

from cytozip import Reader, Writer
import cytozip.cz as cz

_ACCEL_NAMES = [n for n in dir(cz) if n.startswith('_c_')]


def _disable_accel(monkeypatch):
    """Disable every Cython accelerator for the duration of a test."""
    cz._ensure_cz_accel()
    monkeypatch.setattr(cz, '_CZ_ACCEL_LOADED', True)
    for name in _ACCEL_NAMES:
        monkeypatch.setattr(cz, name, None)
    # The public block loader is rebound to the Cython one on load.
    monkeypatch.setattr(cz, '_load_bcz_block', cz._py_load_bcz_block)


def _write(path, n=250_000, delta=True, chroms=('chr1', 'chr2')):
    """Multi-block delta file: pos (Q, delta) + mc/cov (B)."""
    w = Writer(path, formats=['Q', 'B', 'B'], columns=['pos', 'mc', 'cov'],
               chunk_dims=['chrom'], sort_col='pos',
               delta_cols=['pos'] if delta else None)
    truth = {}
    for c in chroms:
        pos = 1 + 3 * np.arange(n, dtype=np.int64)
        mc = (np.arange(n) % 2).astype(np.uint8)
        cov = (1 + np.arange(n) % 3).astype(np.uint8)
        rec = np.zeros(n, dtype=[('pos', '<u8'), ('mc', 'u1'), ('cov', 'u1')])
        rec['pos'] = pos; rec['mc'] = mc; rec['cov'] = cov
        w.write_chunk(rec.tobytes(), [c])
        truth[c] = (pos, mc, cov)
    w.close()
    return truth


def _read_all_paths(path, chunk):
    r = Reader(path)
    out = {
        'raw': r.fetch_chunk_bytes(chunk),
        'chunk2numpy': r.chunk2numpy(chunk),
        'fetch_first': [tuple(x) for x, _ in zip(r.fetch(chunk), range(5))],
        'query': list(r.query(chunk_key=chunk, start=301, end=310, printout=False)),
        'query_numpy': r.query_numpy(chunk_key=chunk, start=301, end=310),
        'df': r.chunk2df(chunk),
    }
    r.close()
    return out


@pytest.mark.parametrize('delta', [True, False])
def test_pure_python_matches_accelerated(tmp_path, delta, monkeypatch):
    path = str(tmp_path / 'f.cz')
    truth = _write(path, delta=delta)
    acc = _read_all_paths(path, ('chr2',))
    # sanity vs ground truth with the accelerators on
    pos, mc, cov = truth['chr2']
    assert np.array_equal(acc['chunk2numpy']['f0'], pos)
    assert np.array_equal(acc['chunk2numpy']['f1'], mc)
    assert np.asarray(acc['query_numpy'])['f0'].tolist() == [301, 304, 307, 310]

    # now the same reads without any Cython helper
    _disable_accel(monkeypatch)
    assert cz._c_fetch_chunk is None
    pure = _read_all_paths(path, ('chr2',))

    assert pure['raw'] == acc['raw']
    assert np.array_equal(pure['chunk2numpy'], acc['chunk2numpy'])
    assert pure['fetch_first'] == acc['fetch_first']
    assert pure['query'] == acc['query']
    assert np.array_equal(np.asarray(pure['query_numpy']),
                          np.asarray(acc['query_numpy']))
    assert pure['df'].equals(acc['df'])


def test_pure_python_writer_roundtrip(tmp_path, monkeypatch):
    """A file written without the accelerators reads back identically."""
    _disable_accel(monkeypatch)
    path = str(tmp_path / 'pp.cz')
    truth = _write(path, n=70_000)
    r = Reader(path)
    arr = r.chunk2numpy(('chr1',))
    assert np.array_equal(arr['f0'], truth['chr1'][0])
    assert list(r.query(chunk_key=('chr1',), start=4, end=7, printout=False)) == [
        ('chr1', 4, 1, 2), ('chr1', 7, 0, 3)]
    r.close()
