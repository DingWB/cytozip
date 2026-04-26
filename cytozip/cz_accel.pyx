# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
cz_accel.pyx — Cython-accelerated hot-path functions for the cytozip format.

This module provides C-level implementations of performance-critical
operations that would otherwise be bottlenecks in pure Python:

  - Block I/O:     load_bcz_block, compress_block (raw DEFLATE via zlib C API)
  - Record packing: c_pack_records, c_pack_records_fast
  - Sequential read: c_read, c_readline, c_read_1record
  - Random access:  c_seek_and_read_1record, c_pos2id, c_query_regions
  - Bulk operations: c_fetch_chunk, c_get_records_by_ids, c_block_first_values
  - Genome scanning: c_extract_c_positions, c_write_c_records
  - Utility:       c_write_chunk_tail

All functions are optional: cz.py gracefully falls back to pure-Python
implementations if this module is not compiled or importable.
"""
import os
import struct
import zlib

from bisect import bisect_right

cimport openmp
from cython.parallel cimport prange

# ---------------------------------------------------------------------------
# Constants shared with cz.py (must match exactly).
# ---------------------------------------------------------------------------
_block_magic = b"CB"  # 2-byte magic at the start of each compressed block

# Block format constants — must match cz.py.
cdef unsigned long _BLOCK_MAX_LEN = (1 << 18) - 1  # 256 KiB - 1
cdef int _VO_OFFSET_BITS = 20
cdef unsigned long long _VO_OFFSET_MASK = (1 << 20) - 1  # 0xFFFFF
cdef Py_ssize_t _BLOCK_HEADER_TRAILER_BYTES = 10    # magic(2)+bsize(4)+data_len(4)

# Module-level Struct cache to avoid re-parsing format strings on every call.
_struct_cache = {}

cdef object _get_struct(str fmts):
    """Return a cached struct.Struct for the given format string."""
    st = _struct_cache.get(fmts)
    if st is None:
        st = struct.Struct(f"<{fmts}")
        _struct_cache[fmts] = st
    return st


cdef bytes _parse_blocks_from_buffer(bytes raw_all):
    """Parse and decompress all BCZ blocks from an in-memory buffer.

    Block layout: magic(2B) + bsize(uint32 4B) + deflate(bsize-10) +
    data_len(uint32 4B).

    Two-phase implementation:
      Phase 1 (serial): walk block headers, record (deflate_off, deflate_size).
      Phase 2 (parallel via OpenMP, opt-in): inflate each block into a
        scratch buffer using a per-thread decompressor.
      Phase 3 (serial): concatenate scratch buffers into the result bytes.

    Threading is gated by ``CYTOZIP_INFLATE_THREADS`` (default min(4, omp_max));
    with 0 or 1 threads, or fewer than 2 blocks, the serial path runs.
    """
    cdef Py_ssize_t total = len(raw_all)
    cdef const unsigned char *base = <const unsigned char *> (<const char *> raw_all)
    cdef Py_ssize_t offset = 0
    cdef unsigned long bsize
    cdef Py_ssize_t deflate_size

    # ---- Phase 1: header scan (serial) --------------------------------
    cdef Py_ssize_t cap = 16
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t *def_off = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
    cdef Py_ssize_t *def_sz = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
    cdef Py_ssize_t *new_off
    cdef Py_ssize_t *new_sz
    if def_off == NULL or def_sz == NULL:
        if def_off != NULL: free(def_off)
        if def_sz != NULL: free(def_sz)
        raise MemoryError()
    while offset + 6 <= total:
        if base[offset] != 67 or base[offset + 1] != 66:  # 'C', 'B'
            break
        bsize = ((<unsigned long> base[offset + 2])
                 | ((<unsigned long> base[offset + 3]) << 8)
                 | ((<unsigned long> base[offset + 4]) << 16)
                 | ((<unsigned long> base[offset + 5]) << 24))
        if offset + <Py_ssize_t> bsize > total:
            break
        deflate_size = <Py_ssize_t> bsize - _BLOCK_HEADER_TRAILER_BYTES
        if n == cap:
            cap *= 2
            new_off = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
            new_sz = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
            if new_off == NULL or new_sz == NULL:
                if new_off != NULL: free(new_off)
                if new_sz != NULL: free(new_sz)
                free(def_off); free(def_sz)
                raise MemoryError()
            memcpy(new_off, def_off, n * sizeof(Py_ssize_t))
            memcpy(new_sz, def_sz, n * sizeof(Py_ssize_t))
            free(def_off); free(def_sz)
            def_off = new_off
            def_sz = new_sz
        def_off[n] = offset + 6
        def_sz[n] = deflate_size
        n += 1
        offset += <Py_ssize_t> bsize

    if n == 0:
        free(def_off); free(def_sz)
        return b""

    # ---- Phase 2: parallel inflate ------------------------------------
    cdef int n_threads = _get_inflate_threads_setting()
    if n < 2:
        n_threads = 1
    cdef size_t out_cap = <size_t> _BLOCK_MAX_LEN + 1
    cdef Py_ssize_t *act_sz = <Py_ssize_t *> malloc(n * sizeof(Py_ssize_t))
    cdef unsigned char **out_bufs = <unsigned char **> malloc(n * sizeof(void*))
    cdef int *ret_codes = <int *> malloc(n * sizeof(int))
    if act_sz == NULL or out_bufs == NULL or ret_codes == NULL:
        if act_sz != NULL: free(act_sz)
        if out_bufs != NULL: free(out_bufs)
        if ret_codes != NULL: free(ret_codes)
        free(def_off); free(def_sz)
        raise MemoryError()

    cdef Py_ssize_t i
    for i in range(n):
        out_bufs[i] = NULL
        act_sz[i] = 0
        ret_codes[i] = 0

    cdef libdeflate_decompressor *single_d
    cdef int tid
    cdef size_t actual_local
    cdef int rc

    if n_threads > 1:
        _ensure_thread_decoms(n_threads)
        with nogil:
            for i in prange(n, num_threads=n_threads, schedule='static'):
                tid = openmp.omp_get_thread_num()
                out_bufs[i] = <unsigned char *> malloc(out_cap)
                if out_bufs[i] == NULL:
                    ret_codes[i] = -1
                else:
                    actual_local = 0
                    rc = libdeflate_deflate_decompress(
                        _ld_thread_decoms[tid],
                        base + def_off[i], <size_t> def_sz[i],
                        out_bufs[i], out_cap, &actual_local)
                    ret_codes[i] = rc
                    act_sz[i] = <Py_ssize_t> actual_local
    else:
        single_d = _get_decompressor()
        for i in range(n):
            out_bufs[i] = <unsigned char *> malloc(out_cap)
            if out_bufs[i] == NULL:
                ret_codes[i] = -1
                continue
            actual_local = 0
            rc = libdeflate_deflate_decompress(
                single_d,
                base + def_off[i], <size_t> def_sz[i],
                out_bufs[i], out_cap, &actual_local)
            ret_codes[i] = rc
            act_sz[i] = <Py_ssize_t> actual_local

    # ---- Phase 3: validate + concat -----------------------------------
    cdef Py_ssize_t total_out = 0
    cdef int err_rc = 0
    cdef Py_ssize_t err_idx = -1
    for i in range(n):
        if ret_codes[i] != LIBDEFLATE_SUCCESS:
            err_rc = ret_codes[i]
            err_idx = i
            break
        total_out += act_sz[i]

    if err_rc != 0:
        for i in range(n):
            if out_bufs[i] != NULL:
                free(out_bufs[i])
        free(def_off); free(def_sz); free(act_sz); free(out_bufs); free(ret_codes)
        if err_rc == -1:
            raise MemoryError("alloc per-block output buffer")
        raise RuntimeError(
            f"libdeflate_deflate_decompress failed at block {err_idx}: rc={err_rc}")

    cdef bytes result = PyBytes_FromStringAndSize(NULL, total_out)
    cdef char *dst = PyBytes_AsString(result)  # mutable while refcount==1
    cdef Py_ssize_t off2 = 0
    for i in range(n):
        if act_sz[i] > 0:
            memcpy(dst + off2, out_bufs[i], <size_t> act_sz[i])
            off2 += act_sz[i]
        free(out_bufs[i])
    free(def_off); free(def_sz); free(act_sz); free(out_bufs); free(ret_codes)
    return result


# ---------------------------------------------------------------------------
# DELTA decoding helpers (per-block cumsum for delta-encoded columns).
#
# Delta files store [v0, v1-v0, v2-v1, ...] for each delta column within
# a block. Decoding = np.cumsum along the column. Blocks are record-aligned
# for delta files, so no record ever spans block boundaries.
# ---------------------------------------------------------------------------
# numpy is imported lazily: pulling it in costs ~60 ms of CLI cold-start
# time, but is only needed when querying files that use DELTA encoding.
_np = None

cdef inline object _get_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np

cdef inline bytes _apply_delta_bytes(bytes data, object delta_dtype, object delta_col_names):
    """Decode delta-encoded columns of a single block's byte buffer.

    Returns the original *data* unchanged when delta_dtype is None or
    delta_col_names is empty. Otherwise returns a new bytes object with
    cumsum applied to each named column.

    Fast path: when every delta column has a homogeneous unsigned-int
    field of width 1/2/4/8 bytes, run an inline in-place cumsum on a
    bytearray copy (one allocation) instead of going through numpy
    (frombuffer + copy + np.cumsum + tobytes ≈ four allocations).
    """
    if delta_dtype is None or not delta_col_names:
        return data
    cdef Py_ssize_t itemsize = delta_dtype.itemsize
    cdef Py_ssize_t nrec = len(data) // itemsize
    if nrec == 0:
        return data
    cdef Py_ssize_t used = nrec * itemsize
    # Try the fast inline path first.
    cdef bytes fast = _try_inline_cumsum(data, delta_dtype, delta_col_names)
    if fast is not None:
        if used == len(data):
            return fast
        return fast + data[used:]
    # Fallback: numpy cumsum.
    np = _get_np()
    arr = np.frombuffer(data, dtype=delta_dtype, count=nrec).copy()
    for name in delta_col_names:
        np.cumsum(arr[name], out=arr[name])
    if used == len(data):
        return arr.tobytes()
    return arr.tobytes() + data[used:]


cdef bytes _try_inline_cumsum(bytes data, object delta_dtype, object delta_col_names):
    """Inline in-place cumsum for delta columns when every column is a
    uint8/16/32/64 field of the structured dtype.

    Returns the patched bytes, or None if the dtype is unsupported (the
    caller falls back to the numpy path).
    """
    cdef Py_ssize_t itemsize = delta_dtype.itemsize
    cdef Py_ssize_t nrec = len(data) // itemsize
    cdef Py_ssize_t used = nrec * itemsize
    # Validate every column upfront; bail to numpy on heterogeneous input.
    fields = delta_dtype.fields
    if fields is None:
        return None
    cdef list specs = []  # list of (offset, width)
    for name in delta_col_names:
        if name not in fields:
            return None
        sub = fields[name]
        sub_dtype = sub[0]
        sub_off = sub[1]
        # Reject non-numeric or signed dtypes; cumsum on signed differs.
        if sub_dtype.kind != 'u':
            return None
        if sub_dtype.itemsize not in (1, 2, 4, 8):
            return None
        specs.append((<Py_ssize_t> sub_off, <Py_ssize_t> sub_dtype.itemsize))

    cdef bytearray out = bytearray(data[:used])
    cdef unsigned char *base = <unsigned char *> PyByteArray_AsString(out)
    cdef Py_ssize_t off, w, i
    cdef unsigned long long acc
    cdef unsigned char *p
    for spec in specs:
        off = spec[0]
        w = spec[1]
        p = base + off
        acc = 0
        if w == 1:
            for i in range(nrec):
                acc = (acc + (<unsigned long long> p[0])) & 0xff
                p[0] = <unsigned char> acc
                p += itemsize
        elif w == 2:
            for i in range(nrec):
                acc = (acc + (<unsigned long long> (
                    p[0] | (<unsigned long long> p[1] << 8)))) & 0xffff
                p[0] = <unsigned char> (acc & 0xff)
                p[1] = <unsigned char> ((acc >> 8) & 0xff)
                p += itemsize
        elif w == 4:
            for i in range(nrec):
                acc = (acc + (<unsigned long long> (
                    p[0]
                    | (<unsigned long long> p[1] << 8)
                    | (<unsigned long long> p[2] << 16)
                    | (<unsigned long long> p[3] << 24)))) & 0xffffffffULL
                p[0] = <unsigned char> (acc & 0xff)
                p[1] = <unsigned char> ((acc >> 8) & 0xff)
                p[2] = <unsigned char> ((acc >> 16) & 0xff)
                p[3] = <unsigned char> ((acc >> 24) & 0xff)
                p += itemsize
        else:  # w == 8
            for i in range(nrec):
                acc = (acc + (<unsigned long long> (
                    p[0]
                    | (<unsigned long long> p[1] << 8)
                    | (<unsigned long long> p[2] << 16)
                    | (<unsigned long long> p[3] << 24)
                    | (<unsigned long long> p[4] << 32)
                    | (<unsigned long long> p[5] << 40)
                    | (<unsigned long long> p[6] << 48)
                    | (<unsigned long long> p[7] << 56))))
                p[0] = <unsigned char> (acc & 0xff)
                p[1] = <unsigned char> ((acc >> 8) & 0xff)
                p[2] = <unsigned char> ((acc >> 16) & 0xff)
                p[3] = <unsigned char> ((acc >> 24) & 0xff)
                p[4] = <unsigned char> ((acc >> 32) & 0xff)
                p[5] = <unsigned char> ((acc >> 40) & 0xff)
                p[6] = <unsigned char> ((acc >> 48) & 0xff)
                p[7] = <unsigned char> ((acc >> 56) & 0xff)
                p += itemsize
    return bytes(out)


cdef bytes _try_inline_diff(bytes data, object delta_dtype, object delta_col_names):
    """Symmetric encoder for `_try_inline_cumsum`: in-place finite
    difference (block[i] -= block[i-1]) for delta-encoded uint columns.

    Returns the patched bytes, or None if the dtype is unsupported (the
    caller falls back to the numpy path).
    """
    cdef Py_ssize_t itemsize = delta_dtype.itemsize
    cdef Py_ssize_t nrec = len(data) // itemsize
    cdef Py_ssize_t used = nrec * itemsize
    if nrec < 2:
        # 0- or 1-record blocks: encoded == raw.
        return bytes(data[:used]) if used == len(data) else bytes(data[:used])
    fields = delta_dtype.fields
    if fields is None:
        return None
    cdef list specs = []
    for name in delta_col_names:
        if name not in fields:
            return None
        sub = fields[name]
        sub_dtype = sub[0]
        sub_off = sub[1]
        if sub_dtype.kind != 'u':
            return None
        if sub_dtype.itemsize not in (1, 2, 4, 8):
            return None
        specs.append((<Py_ssize_t> sub_off, <Py_ssize_t> sub_dtype.itemsize))

    cdef bytearray out = bytearray(data[:used])
    cdef unsigned char *base = <unsigned char *> PyByteArray_AsString(out)
    cdef Py_ssize_t off, w, i
    cdef unsigned long long prev, cur, diff
    cdef unsigned char *p
    for spec in specs:
        off = spec[0]
        w = spec[1]
        # Walk records right-to-left so each subtraction uses the
        # original value (not the just-overwritten delta).
        if w == 1:
            for i in range(nrec - 1, 0, -1):
                p = base + off + i * itemsize
                cur = p[0]
                prev = (p - itemsize)[0]
                diff = (cur - prev) & 0xff
                p[0] = <unsigned char> diff
        elif w == 2:
            for i in range(nrec - 1, 0, -1):
                p = base + off + i * itemsize
                cur = p[0] | (<unsigned long long> p[1] << 8)
                prev = (p - itemsize)[0] | (<unsigned long long> (p - itemsize)[1] << 8)
                diff = (cur - prev) & 0xffffULL
                p[0] = <unsigned char> (diff & 0xff)
                p[1] = <unsigned char> ((diff >> 8) & 0xff)
        elif w == 4:
            for i in range(nrec - 1, 0, -1):
                p = base + off + i * itemsize
                cur = (p[0]
                       | (<unsigned long long> p[1] << 8)
                       | (<unsigned long long> p[2] << 16)
                       | (<unsigned long long> p[3] << 24))
                prev = ((p - itemsize)[0]
                        | (<unsigned long long> (p - itemsize)[1] << 8)
                        | (<unsigned long long> (p - itemsize)[2] << 16)
                        | (<unsigned long long> (p - itemsize)[3] << 24))
                diff = (cur - prev) & 0xffffffffULL
                p[0] = <unsigned char> (diff & 0xff)
                p[1] = <unsigned char> ((diff >> 8) & 0xff)
                p[2] = <unsigned char> ((diff >> 16) & 0xff)
                p[3] = <unsigned char> ((diff >> 24) & 0xff)
        else:  # w == 8
            for i in range(nrec - 1, 0, -1):
                p = base + off + i * itemsize
                cur = (p[0]
                       | (<unsigned long long> p[1] << 8)
                       | (<unsigned long long> p[2] << 16)
                       | (<unsigned long long> p[3] << 24)
                       | (<unsigned long long> p[4] << 32)
                       | (<unsigned long long> p[5] << 40)
                       | (<unsigned long long> p[6] << 48)
                       | (<unsigned long long> p[7] << 56))
                prev = ((p - itemsize)[0]
                        | (<unsigned long long> (p - itemsize)[1] << 8)
                        | (<unsigned long long> (p - itemsize)[2] << 16)
                        | (<unsigned long long> (p - itemsize)[3] << 24)
                        | (<unsigned long long> (p - itemsize)[4] << 32)
                        | (<unsigned long long> (p - itemsize)[5] << 40)
                        | (<unsigned long long> (p - itemsize)[6] << 48)
                        | (<unsigned long long> (p - itemsize)[7] << 56))
                diff = cur - prev
                p[0] = <unsigned char> (diff & 0xff)
                p[1] = <unsigned char> ((diff >> 8) & 0xff)
                p[2] = <unsigned char> ((diff >> 16) & 0xff)
                p[3] = <unsigned char> ((diff >> 24) & 0xff)
                p[4] = <unsigned char> ((diff >> 32) & 0xff)
                p[5] = <unsigned char> ((diff >> 40) & 0xff)
                p[6] = <unsigned char> ((diff >> 48) & 0xff)
                p[7] = <unsigned char> ((diff >> 56) & 0xff)
    return bytes(out)


def c_delta_encode_block(bytes data, object delta_dtype, object delta_col_names):
    """Public entry: encode delta-columns in `data` (per-record diff).

    Mirror of the cumsum decoder. Returns bytes; preserves trailing
    bytes beyond the last full record. Falls back to numpy when dtype
    is unsupported.
    """
    if delta_dtype is None or not delta_col_names:
        return data
    cdef Py_ssize_t itemsize = delta_dtype.itemsize
    cdef Py_ssize_t nrec = len(data) // itemsize
    if nrec == 0:
        return data
    cdef Py_ssize_t used = nrec * itemsize
    cdef bytes fast = _try_inline_diff(data, delta_dtype, delta_col_names)
    if fast is not None:
        if used == len(data):
            return fast
        return fast + data[used:]
    np = _get_np()
    arr = np.frombuffer(data, dtype=delta_dtype, count=nrec).copy()
    if nrec > 1:
        for name in delta_col_names:
            arr[name][1:] = np.diff(arr[name])
    if used == len(data):
        return arr.tobytes()
    return arr.tobytes() + data[used:]


cdef bytes _parse_blocks_from_buffer_delta(bytes raw_all, object delta_dtype, object delta_col_names):
    """Delta-aware variant of _parse_blocks_from_buffer.

    Each block is inflated (in parallel when enabled) and delta-decoded
    serially before concatenation, because delta decode runs Python/numpy
    code that requires the GIL.
    """
    if delta_dtype is None or not delta_col_names:
        return _parse_blocks_from_buffer(raw_all)

    cdef Py_ssize_t total = len(raw_all)
    cdef const unsigned char *base = <const unsigned char *> (<const char *> raw_all)
    cdef Py_ssize_t offset = 0
    cdef unsigned long bsize
    cdef Py_ssize_t deflate_size

    # ---- Phase 1: header scan ----------------------------------------
    cdef Py_ssize_t cap = 16
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t *def_off = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
    cdef Py_ssize_t *def_sz = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
    cdef Py_ssize_t *new_off
    cdef Py_ssize_t *new_sz
    if def_off == NULL or def_sz == NULL:
        if def_off != NULL: free(def_off)
        if def_sz != NULL: free(def_sz)
        raise MemoryError()
    while offset + 6 <= total:
        if base[offset] != 67 or base[offset + 1] != 66:
            break
        bsize = ((<unsigned long> base[offset + 2])
                 | ((<unsigned long> base[offset + 3]) << 8)
                 | ((<unsigned long> base[offset + 4]) << 16)
                 | ((<unsigned long> base[offset + 5]) << 24))
        if offset + <Py_ssize_t> bsize > total:
            break
        deflate_size = <Py_ssize_t> bsize - _BLOCK_HEADER_TRAILER_BYTES
        if n == cap:
            cap *= 2
            new_off = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
            new_sz = <Py_ssize_t *> malloc(cap * sizeof(Py_ssize_t))
            if new_off == NULL or new_sz == NULL:
                if new_off != NULL: free(new_off)
                if new_sz != NULL: free(new_sz)
                free(def_off); free(def_sz)
                raise MemoryError()
            memcpy(new_off, def_off, n * sizeof(Py_ssize_t))
            memcpy(new_sz, def_sz, n * sizeof(Py_ssize_t))
            free(def_off); free(def_sz)
            def_off = new_off
            def_sz = new_sz
        def_off[n] = offset + 6
        def_sz[n] = deflate_size
        n += 1
        offset += <Py_ssize_t> bsize

    if n == 0:
        free(def_off); free(def_sz)
        return b""

    # ---- Phase 2: parallel inflate -----------------------------------
    cdef int n_threads = _get_inflate_threads_setting()
    if n < 2:
        n_threads = 1
    cdef size_t out_cap = <size_t> _BLOCK_MAX_LEN + 1
    cdef Py_ssize_t *act_sz = <Py_ssize_t *> malloc(n * sizeof(Py_ssize_t))
    cdef unsigned char **out_bufs = <unsigned char **> malloc(n * sizeof(void*))
    cdef int *ret_codes = <int *> malloc(n * sizeof(int))
    if act_sz == NULL or out_bufs == NULL or ret_codes == NULL:
        if act_sz != NULL: free(act_sz)
        if out_bufs != NULL: free(out_bufs)
        if ret_codes != NULL: free(ret_codes)
        free(def_off); free(def_sz)
        raise MemoryError()
    cdef Py_ssize_t i
    for i in range(n):
        out_bufs[i] = NULL
        act_sz[i] = 0
        ret_codes[i] = 0

    cdef libdeflate_decompressor *single_d
    cdef int tid
    cdef size_t actual_local
    cdef int rc

    if n_threads > 1:
        _ensure_thread_decoms(n_threads)
        with nogil:
            for i in prange(n, num_threads=n_threads, schedule='static'):
                tid = openmp.omp_get_thread_num()
                out_bufs[i] = <unsigned char *> malloc(out_cap)
                if out_bufs[i] == NULL:
                    ret_codes[i] = -1
                else:
                    actual_local = 0
                    rc = libdeflate_deflate_decompress(
                        _ld_thread_decoms[tid],
                        base + def_off[i], <size_t> def_sz[i],
                        out_bufs[i], out_cap, &actual_local)
                    ret_codes[i] = rc
                    act_sz[i] = <Py_ssize_t> actual_local
    else:
        single_d = _get_decompressor()
        for i in range(n):
            out_bufs[i] = <unsigned char *> malloc(out_cap)
            if out_bufs[i] == NULL:
                ret_codes[i] = -1
                continue
            actual_local = 0
            rc = libdeflate_deflate_decompress(
                single_d,
                base + def_off[i], <size_t> def_sz[i],
                out_bufs[i], out_cap, &actual_local)
            ret_codes[i] = rc
            act_sz[i] = <Py_ssize_t> actual_local

    # Validate inflate phase
    cdef int err_rc = 0
    cdef Py_ssize_t err_idx = -1
    for i in range(n):
        if ret_codes[i] != LIBDEFLATE_SUCCESS:
            err_rc = ret_codes[i]
            err_idx = i
            break
    if err_rc != 0:
        for i in range(n):
            if out_bufs[i] != NULL:
                free(out_bufs[i])
        free(def_off); free(def_sz); free(act_sz); free(out_bufs); free(ret_codes)
        if err_rc == -1:
            raise MemoryError("alloc per-block output buffer")
        raise RuntimeError(
            f"libdeflate_deflate_decompress failed at block {err_idx}: rc={err_rc}")

    # ---- Phase 3: delta-decode + concat (serial, GIL) ----------------
    cdef list chunks = []
    cdef bytes blk
    for i in range(n):
        blk = PyBytes_FromStringAndSize(<char *> out_bufs[i], act_sz[i])
        free(out_bufs[i])
        chunks.append(_apply_delta_bytes(blk, delta_dtype, delta_col_names))
    free(def_off); free(def_sz); free(act_sz); free(out_bufs); free(ret_codes)
    if not chunks:
        return b""
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Low-level libdeflate C API declarations.
#
# libdeflate is a modern SIMD-optimized replacement for zlib. It produces
# and consumes the exact same raw DEFLATE bitstream as zlib (RFC 1951),
# so existing .cz files stay 100% compatible. Decompression is typically
# ~2-3x faster than zlib, compression ~1.3-2x faster.
#
# Unlike zlib which has a streaming state-machine API (z_stream +
# inflate/deflate), libdeflate uses a simpler one-shot API that matches
# the .cz block model (each block is <=65535 bytes, known up-front).
#
# A single module-level decompressor / compressor-per-level is cached.
# libdeflate objects are NOT thread-safe; the GIL protects us during
# Cython calls, but if pure Python threads ever share them we'd need
# per-thread instances.
# ---------------------------------------------------------------------------
cdef extern from "libdeflate.h":
    # libdeflate.h declares these as plain C structs (no typedef) so we
    # must tell Cython to emit ``struct libdeflate_{de,}compressor`` in the
    # generated C code via the quoted-name form.
    ctypedef struct libdeflate_decompressor "struct libdeflate_decompressor":
        pass
    ctypedef struct libdeflate_compressor "struct libdeflate_compressor":
        pass

    # libdeflate_result enum values (0 = success)
    int LIBDEFLATE_SUCCESS
    int LIBDEFLATE_BAD_DATA
    int LIBDEFLATE_SHORT_OUTPUT
    int LIBDEFLATE_INSUFFICIENT_SPACE

    libdeflate_decompressor *libdeflate_alloc_decompressor()
    void libdeflate_free_decompressor(libdeflate_decompressor *d)
    int libdeflate_deflate_decompress(libdeflate_decompressor *d,
                                      const void *in_, size_t in_nbytes,
                                      void *out, size_t out_nbytes_avail,
                                      size_t *actual_out_nbytes_ret) nogil

    libdeflate_compressor *libdeflate_alloc_compressor(int compression_level)
    void libdeflate_free_compressor(libdeflate_compressor *c)
    size_t libdeflate_deflate_compress_bound(libdeflate_compressor *c, size_t in_nbytes) nogil
    size_t libdeflate_deflate_compress(libdeflate_compressor *c,
                                       const void *in_, size_t in_nbytes,
                                       void *out, size_t out_nbytes_avail) nogil


from cpython.bytearray cimport PyByteArray_FromStringAndSize, PyByteArray_AsString
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free


# ---------------------------------------------------------------------------
# Cached libdeflate objects. The decompressor is stateless across calls
# (it just holds lookup tables); compressors are per-level.
# ---------------------------------------------------------------------------
cdef libdeflate_decompressor *_ld_decompressor = NULL
cdef libdeflate_compressor *_ld_compressors[13]   # levels 0..12
cdef int _ld_compressor_count = 13


cdef libdeflate_decompressor *_get_decompressor() except NULL:
    global _ld_decompressor
    if _ld_decompressor == NULL:
        _ld_decompressor = libdeflate_alloc_decompressor()
        if _ld_decompressor == NULL:
            raise MemoryError("libdeflate_alloc_decompressor failed")
    return _ld_decompressor


# ---------------------------------------------------------------------------
# Per-thread decompressor pool for multi-threaded inflate.
#
# libdeflate decompressors are NOT thread-safe (each holds a working
# state buffer that gets mutated during decompress). For OpenMP-parallel
# block inflation we keep one decompressor per OpenMP thread slot.
# Allocated lazily; never freed until process exit.
# ---------------------------------------------------------------------------
cdef libdeflate_decompressor **_ld_thread_decoms = NULL
cdef int _ld_thread_decoms_n = 0
cdef int _inflate_threads_cached = 0  # 0 = uninitialized; 1 = serial; >1 = parallel


cdef int _get_inflate_threads_setting():
    """Return desired number of inflate threads (cached after first call)."""
    global _inflate_threads_cached
    if _inflate_threads_cached != 0:
        return _inflate_threads_cached
    cdef int n = 0
    env = os.environ.get("CYTOZIP_INFLATE_THREADS")
    if env is not None:
        try:
            n = int(env)
        except (TypeError, ValueError):
            n = 0
    if n <= 0:
        # Default: cap at 4 to avoid oversubscription on shared nodes.
        omp_max = openmp.omp_get_max_threads()
        n = omp_max if omp_max < 4 else 4
    if n < 1:
        n = 1
    _inflate_threads_cached = n
    return n


cdef int _ensure_thread_decoms(int n_threads) except -1:
    """Ensure the per-thread decompressor pool has at least *n_threads* slots."""
    global _ld_thread_decoms, _ld_thread_decoms_n
    cdef int i
    cdef libdeflate_decompressor **new_arr
    if n_threads <= _ld_thread_decoms_n:
        return 0
    new_arr = <libdeflate_decompressor **> malloc(n_threads * sizeof(void*))
    if new_arr == NULL:
        raise MemoryError("alloc thread decompressor array")
    for i in range(_ld_thread_decoms_n):
        new_arr[i] = _ld_thread_decoms[i]
    for i in range(_ld_thread_decoms_n, n_threads):
        new_arr[i] = libdeflate_alloc_decompressor()
        if new_arr[i] == NULL:
            # Free anything we just allocated, restore old state.
            for j in range(_ld_thread_decoms_n, i):
                libdeflate_free_decompressor(new_arr[j])
            free(new_arr)
            raise MemoryError("libdeflate_alloc_decompressor failed (per-thread)")
    if _ld_thread_decoms != NULL:
        free(_ld_thread_decoms)
    _ld_thread_decoms = new_arr
    _ld_thread_decoms_n = n_threads
    return 0



cdef libdeflate_compressor *_get_compressor(int level) except NULL:
    global _ld_compressors
    if level < 0 or level > 12:
        level = 6
    if _ld_compressors[level] == NULL:
        _ld_compressors[level] = libdeflate_alloc_compressor(level)
        if _ld_compressors[level] == NULL:
            raise MemoryError(f"libdeflate_alloc_compressor(level={level}) failed")
    return _ld_compressors[level]


# Initialize the compressor slot array to NULL.
cdef int _i
for _i in range(13):
    _ld_compressors[_i] = NULL


# ---------------------------------------------------------------------------
# Per-thread compressor pool for multi-threaded deflate.
# Indexed by [level][thread_id]. Lazily allocated.
# ---------------------------------------------------------------------------
cdef libdeflate_compressor **_ld_thread_comps[13]
cdef int _ld_thread_comps_n = 0
cdef int _deflate_threads_cached = 0  # 0 = uninitialized; >=1 = chosen value

# Initialize per-thread compressor slot array.
for _i in range(13):
    _ld_thread_comps[_i] = NULL


cdef int _get_deflate_threads_setting():
    """Return desired number of deflate threads (cached after first call)."""
    global _deflate_threads_cached
    if _deflate_threads_cached != 0:
        return _deflate_threads_cached
    cdef int n = 0
    env = os.environ.get("CYTOZIP_DEFLATE_THREADS")
    if env is not None:
        try:
            n = int(env)
        except (TypeError, ValueError):
            n = 0
    if n <= 0:
        # Default: cap at 4 to avoid oversubscription on shared nodes.
        omp_max = openmp.omp_get_max_threads()
        n = omp_max if omp_max < 4 else 4
    if n < 1:
        n = 1
    _deflate_threads_cached = n
    return n


cdef int _ensure_thread_comps(int level, int n_threads) except -1:
    """Ensure the per-thread compressor pool for *level* has at least
    *n_threads* slots."""
    global _ld_thread_comps, _ld_thread_comps_n
    cdef int i, j
    cdef libdeflate_compressor **new_arr
    if level < 0 or level > 12:
        level = 6
    # First-time grow: allocate slot array up to n_threads for this level.
    if _ld_thread_comps[level] == NULL or n_threads > _ld_thread_comps_n:
        new_arr = <libdeflate_compressor **> malloc(n_threads * sizeof(void*))
        if new_arr == NULL:
            raise MemoryError("alloc thread compressor array")
        # Copy existing pointers (if any) for this level.
        for i in range(_ld_thread_comps_n):
            if _ld_thread_comps[level] != NULL:
                new_arr[i] = _ld_thread_comps[level][i]
            else:
                new_arr[i] = NULL
        # Allocate any new slots.
        for i in range(_ld_thread_comps_n, n_threads):
            new_arr[i] = libdeflate_alloc_compressor(level)
            if new_arr[i] == NULL:
                for j in range(_ld_thread_comps_n, i):
                    libdeflate_free_compressor(new_arr[j])
                free(new_arr)
                raise MemoryError(
                    f"libdeflate_alloc_compressor(level={level}) failed")
        if _ld_thread_comps[level] != NULL:
            free(_ld_thread_comps[level])
        _ld_thread_comps[level] = new_arr
        # Track the largest pool size across levels.
        if n_threads > _ld_thread_comps_n:
            _ld_thread_comps_n = n_threads
    else:
        # Already large enough, but this level's slots may be NULL if a
        # previous call used a different level. Fill in any NULL entries.
        for i in range(n_threads):
            if _ld_thread_comps[level][i] == NULL:
                _ld_thread_comps[level][i] = libdeflate_alloc_compressor(level)
                if _ld_thread_comps[level][i] == NULL:
                    raise MemoryError(
                        f"libdeflate_alloc_compressor(level={level}) failed")
    return 0


def c_compress_blocks_parallel(blocks, level=6):
    """Compress a list of bytes objects in parallel via libdeflate + OpenMP.

    Each input block is compressed independently using a per-thread
    compressor instance (libdeflate compressors are not thread-safe).

    Parameters
    ----------
    blocks : list of bytes
        Block payloads to compress.
    level : int
        Compression level (1-12).

    Returns
    -------
    list of bytes
        Same length as *blocks*; element i is raw DEFLATE of blocks[i].
    """
    cdef int lvl = int(level)
    if lvl < 0 or lvl > 12:
        lvl = 6
    cdef Py_ssize_t n = len(blocks)
    if n == 0:
        return []

    cdef int n_threads = _get_deflate_threads_setting()
    if n < 2:
        n_threads = 1

    # Convert input list to C arrays of (ptr, len). Holding a ref via
    # py_blocks keeps the input bytes objects alive throughout.
    cdef list py_blocks = list(blocks)
    cdef const unsigned char **in_ptrs = <const unsigned char **> malloc(
        n * sizeof(void*))
    cdef Py_ssize_t *in_lens = <Py_ssize_t *> malloc(n * sizeof(Py_ssize_t))
    cdef unsigned char **out_bufs = <unsigned char **> malloc(n * sizeof(void*))
    cdef size_t *out_caps = <size_t *> malloc(n * sizeof(size_t))
    cdef size_t *out_actual = <size_t *> malloc(n * sizeof(size_t))
    cdef int *err_codes = <int *> malloc(n * sizeof(int))
    if (in_ptrs == NULL or in_lens == NULL or out_bufs == NULL
            or out_caps == NULL or out_actual == NULL or err_codes == NULL):
        if in_ptrs != NULL: free(in_ptrs)
        if in_lens != NULL: free(in_lens)
        if out_bufs != NULL: free(out_bufs)
        if out_caps != NULL: free(out_caps)
        if out_actual != NULL: free(out_actual)
        if err_codes != NULL: free(err_codes)
        raise MemoryError()

    cdef Py_ssize_t i
    cdef bytes b
    cdef libdeflate_compressor *single_c
    # Compute bounds and allocate per-block output buffers (serial, GIL).
    # Use the level-0 compressor (or any pre-allocated one) to compute
    # bounds — bound is the same regardless of level.
    single_c = _get_compressor(lvl)
    for i in range(n):
        b = py_blocks[i]
        in_ptrs[i] = <const unsigned char *> (<const char *> b)
        in_lens[i] = len(b)
        out_caps[i] = libdeflate_deflate_compress_bound(single_c, <size_t> in_lens[i])
        out_bufs[i] = <unsigned char *> malloc(out_caps[i])
        out_actual[i] = 0
        err_codes[i] = 0
        if out_bufs[i] == NULL:
            err_codes[i] = -1

    cdef int tid
    cdef size_t actual_local
    cdef libdeflate_compressor **comps_for_level

    if n_threads > 1:
        _ensure_thread_comps(lvl, n_threads)
        comps_for_level = _ld_thread_comps[lvl]
        with nogil:
            for i in prange(n, num_threads=n_threads, schedule='static'):
                if err_codes[i] == 0:
                    actual_local = libdeflate_deflate_compress(
                        comps_for_level[openmp.omp_get_thread_num()],
                        in_ptrs[i], <size_t> in_lens[i],
                        out_bufs[i], out_caps[i])
                    out_actual[i] = actual_local
                    if actual_local == 0:
                        err_codes[i] = -2
    else:
        for i in range(n):
            if err_codes[i] == 0:
                actual_local = libdeflate_deflate_compress(
                    single_c,
                    in_ptrs[i], <size_t> in_lens[i],
                    out_bufs[i], out_caps[i])
                out_actual[i] = actual_local
                if actual_local == 0:
                    err_codes[i] = -2

    # Validate; build result list.
    cdef list result = [None] * n
    cdef int err_rc = 0
    cdef Py_ssize_t err_idx = -1
    for i in range(n):
        if err_codes[i] != 0:
            err_rc = err_codes[i]
            err_idx = i
            break
    if err_rc != 0:
        for i in range(n):
            if out_bufs[i] != NULL:
                free(out_bufs[i])
        free(in_ptrs); free(in_lens); free(out_bufs)
        free(out_caps); free(out_actual); free(err_codes)
        if err_rc == -1:
            raise MemoryError("alloc per-block compress output buffer")
        raise RuntimeError(
            f"libdeflate_deflate_compress failed at block {err_idx}: rc={err_rc}")

    for i in range(n):
        result[i] = PyBytes_FromStringAndSize(
            <char *> out_bufs[i], <Py_ssize_t> out_actual[i])
        free(out_bufs[i])
    free(in_ptrs); free(in_lens); free(out_bufs)
    free(out_caps); free(out_actual); free(err_codes)
    return result


def c_inflate_bytes(data):
    """Decompress raw DEFLATE bytes using libdeflate.

    libdeflate requires the output buffer size to be known in advance.
    We start with a generous estimate and grow on LIBDEFLATE_INSUFFICIENT_SPACE.

    Parameters
    ----------
    data : bytes
        Raw DEFLATE compressed data (no zlib/gzip header).

    Returns
    -------
    bytes : decompressed data.
    """
    cdef libdeflate_decompressor *d = _get_decompressor()
    cdef const unsigned char *in_ptr = <const unsigned char *> (<const char *> data)
    cdef Py_ssize_t in_len = len(data)
    cdef size_t out_size = max(<size_t> (in_len * 4 + 1024), <size_t> 65536)
    cdef unsigned char *out_buf = <unsigned char *> malloc(out_size)
    cdef size_t actual_out = 0
    cdef int ret
    cdef unsigned char *new_buf
    cdef size_t new_size
    if out_buf == NULL:
        raise MemoryError()

    while True:
        ret = libdeflate_deflate_decompress(d, in_ptr, <size_t> in_len,
                                            out_buf, out_size, &actual_out)
        if ret == LIBDEFLATE_SUCCESS:
            break
        if ret == LIBDEFLATE_INSUFFICIENT_SPACE:
            new_size = out_size * 2
            new_buf = <unsigned char *> malloc(new_size)
            if new_buf == NULL:
                free(out_buf)
                raise MemoryError()
            free(out_buf)
            out_buf = new_buf
            out_size = new_size
            continue
        free(out_buf)
        raise RuntimeError(f"libdeflate_deflate_decompress failed: {ret}")

    res = PyBytes_FromStringAndSize(<char *> out_buf, <Py_ssize_t> actual_out)
    free(out_buf)
    return res


def c_deflate_raw(block, level=6):
    """Compress bytes into raw DEFLATE format using libdeflate.

    Output is byte-compatible with zlib's ``compressobj(level, DEFLATED, -15)``
    (both emit raw DEFLATE per RFC 1951), so existing readers work unchanged.

    Parameters
    ----------
    block : bytes
        Data to compress.
    level : int
        Compression level (1-12 for libdeflate; values > 9 yield better
        ratios than zlib can produce).

    Returns
    -------
    bytes : raw DEFLATE compressed data.
    """
    cdef int lvl = int(level)
    cdef libdeflate_compressor *c = _get_compressor(lvl)
    cdef const unsigned char *in_ptr = <const unsigned char *> (<const char *> block)
    cdef Py_ssize_t in_len = len(block)
    cdef size_t bound = libdeflate_deflate_compress_bound(c, <size_t> in_len)
    cdef unsigned char *out_buf = <unsigned char *> malloc(bound)
    cdef size_t actual
    if out_buf == NULL:
        raise MemoryError()

    actual = libdeflate_deflate_compress(c, in_ptr, <size_t> in_len, out_buf, bound)
    if actual == 0:
        free(out_buf)
        raise RuntimeError("libdeflate_deflate_compress failed (insufficient space)")
    res = PyBytes_FromStringAndSize(<char *> out_buf, <Py_ssize_t> actual)
    free(out_buf)
    return res


# ---------------------------------------------------------------------------
# Internal helper: fast block decompression.
# Used by load_bcz_block and other functions that need to decompress
# individual blocks.  Allocates a fixed 64 KB output buffer since
# blocks are guaranteed to decompress to at most _BLOCK_MAX_LEN bytes.
# ---------------------------------------------------------------------------

cdef bytes _c_inflate(bytes data):
    """Internal: decompress raw DEFLATE via libdeflate (no Python overhead)."""
    cdef libdeflate_decompressor *d = _get_decompressor()
    cdef const unsigned char *in_ptr = <const unsigned char *> data
    cdef Py_ssize_t in_len = len(data)
    # Blocks are up to _BLOCK_MAX_LEN bytes uncompressed; allocate one
    # extra byte to make INSUFFICIENT_SPACE detectable distinct from an
    # exact fit.
    cdef size_t out_size = _BLOCK_MAX_LEN + 1
    cdef unsigned char *out_buf = <unsigned char *> malloc(out_size)
    cdef size_t actual_out = 0
    cdef int ret
    if out_buf == NULL:
        raise MemoryError()

    ret = libdeflate_deflate_decompress(d, in_ptr, <size_t> in_len,
                                        out_buf, out_size, &actual_out)
    if ret != LIBDEFLATE_SUCCESS:
        free(out_buf)
        raise RuntimeError(f"libdeflate_deflate_decompress failed: {ret}")
    res = PyBytes_FromStringAndSize(<char *> out_buf, <Py_ssize_t> actual_out)
    free(out_buf)
    return res


def compress_block(block, level=6):
    """Compress a block using raw DEFLATE (no zlib/gzip header).

    This is the counterpart of ``_c_inflate`` used by Writer._write_block().
    """
    return c_deflate_raw(block, level)


def load_bcz_block(handle, decompress=False):
    """Read a single BCZ block from *handle*.

    Block binary layout:
      [magic 2B 'MB'] [block_size 2B] [deflate_data ...] [data_len 2B]

    block_size includes magic(2) + block_size(2) + data_len(2) = 6 extra
    bytes beyond the deflate payload.

    Parameters
    ----------
    handle : file-like
        Binary file handle positioned at the start of a block.
    decompress : bool
        If True, decompress and return the raw bytes; otherwise
        just skip ahead and return the uncompressed length.

    Returns
    -------
    (block_size, data_or_data_len)

    Raises
    ------
    StopIteration  on EOF or non-block magic.
    """
    magic = handle.read(2)
    if not magic or magic != _block_magic:
        raise StopIteration
    # block_size is 4-byte little-endian unsigned int.
    cdef bytes bs_raw = handle.read(4)
    cdef unsigned long block_size = ((<unsigned char>bs_raw[0])
                                      | ((<unsigned long><unsigned char>bs_raw[1]) << 8)
                                      | ((<unsigned long><unsigned char>bs_raw[2]) << 16)
                                      | ((<unsigned long><unsigned char>bs_raw[3]) << 24))
    cdef Py_ssize_t deflate_size = block_size - _BLOCK_HEADER_TRAILER_BYTES
    cdef bytes dl_raw
    cdef unsigned long data_len
    if decompress:
        raw = handle.read(deflate_size)
        data = _c_inflate(raw)
        # skip the trailing uncompressed length field (uint32, 4 bytes)
        handle.read(4)
        return block_size, data
    else:
        # seek forward over deflate data and read trailing uncompressed length
        handle.seek(deflate_size, 1)
        dl_raw = handle.read(4)
        data_len = ((<unsigned char>dl_raw[0])
                    | ((<unsigned long><unsigned char>dl_raw[1]) << 8)
                    | ((<unsigned long><unsigned char>dl_raw[2]) << 16)
                    | ((<unsigned long><unsigned char>dl_raw[3]) << 24))
        return block_size, data_len


def unpack_records(data, fmt):
    """Unpack concatenated binary record data into a list of tuples.

    Uses struct.Struct for efficient repeated unpacking with
    ``unpack_from()`` to avoid repeated format string parsing.
    Pre-allocates the result list for better performance.

    Parameters
    ----------
    data : bytes
        Concatenated packed records.
    fmt : str
        Struct format string (without '<' prefix).

    Returns
    -------
    list of tuples, one per record.
    """
    if not data:
        return []
    cdef object st = _get_struct(fmt)
    cdef Py_ssize_t unit = st.size
    cdef Py_ssize_t n = len(data) // unit
    cdef list res = [None] * n
    cdef Py_ssize_t off = 0
    cdef Py_ssize_t i
    unpack_from = st.unpack_from
    for i in range(n):
        res[i] = unpack_from(data, off)
        off += unit
    return res


def c_read(handle, block_raw_length, buffer, within_block_offset, size,
           delta_dtype=None, delta_col_names=None):
    """Read *size* bytes from a BGZF-like block stream.

    Handles the case where the requested data spans multiple blocks:
    reads from the current buffer, then loads subsequent blocks as
    needed via ``load_bcz_block``.

    When ``delta_dtype`` and ``delta_col_names`` are supplied, each
    freshly loaded block is delta-decoded before being consumed.

    Returns
    -------
    (data_bytes, new_block_raw_length, new_buffer, new_within_block_offset)
    """
    cdef Py_ssize_t need = size
    cdef bytearray out = bytearray()
    cdef bytes buf = buffer if buffer is not None else b""
    cdef Py_ssize_t buflen = len(buf)
    cdef Py_ssize_t within = within_block_offset
    cdef object block_size_data
    while need and block_raw_length:
        buflen = len(buf)
        if within + need <= buflen:
            out.extend(buf[within: within + need])
            within += need
            need = 0
            break
        else:
            # take rest of buffer
            if within < buflen:
                out.extend(buf[within:])
                need -= (buflen - within)
            # load next block
            try:
                block_size_data = load_bcz_block(handle, True)
            except StopIteration:
                buf = b""
                block_raw_length = 0
                within = 0
                break
            block_raw_length, buf = block_size_data
            buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
            within = 0

    return bytes(out), block_raw_length, buf, within


def c_readline(handle, block_raw_length, buffer, within_block_offset, newline=b"\n",
               delta_dtype=None, delta_col_names=None):
    """Read a single line (including the newline character) from a block stream.

    Handles lines that span block boundaries by accumulating bytes
    until the newline is found. Delta-decodes each newly loaded block
    when ``delta_dtype``/``delta_col_names`` are supplied.

    Returns
    -------
    (line_bytes, new_block_raw_length, new_buffer, new_within_block_offset)
    """
    cdef bytearray out = bytearray()
    cdef bytes buf = buffer if buffer is not None else b""
    cdef Py_ssize_t within = within_block_offset
    cdef Py_ssize_t i
    cdef object block_size_data
    cdef Py_ssize_t buflen
    while block_raw_length:
        buflen = len(buf)
        i = buf.find(newline, within)
        if i == -1:
            # append rest and load next block
            if within < buflen:
                out.extend(buf[within:])
            try:
                block_size_data = load_bcz_block(handle, True)
            except StopIteration:
                buf = b""
                block_raw_length = 0
                within = 0
                break
            block_raw_length, buf = block_size_data
            buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
            within = 0
        elif i + 1 == buflen:
            # newline at end of block
            out.extend(buf[within:])
            try:
                block_size_data = load_bcz_block(handle, True)
            except StopIteration:
                buf = b""
                block_raw_length = 0
                within = 0
                break
            block_raw_length, buf = block_size_data
            buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
            within = 0
            break
        else:
            out.extend(buf[within: i + 1])
            within = i + 1
            break

    return bytes(out), block_raw_length, buf, within


def c_pos2id(handle, block_virtual_offsets, fmts, unit_size, positions,
             col_to_query=0, block_first_coords=None,
             delta_dtype=None, delta_col_names=None):
    """Map genomic position ranges to primary record IDs.

    For each (start, end) pair in *positions*, finds the first and last
    record whose ``col_to_query`` value falls within [start, end).
    Returns a list of [id_start, id_end] pairs (or None if not found).

    When ``block_first_coords`` is supplied (the per-block first-record
    values preloaded from the chunk tail), binary search is done purely
    in memory with :func:`bisect.bisect_right` — no inflate probes.
    Otherwise falls back to seek-and-inflate bisect on *block_offsets*.

    Delta-encoded columns are decoded per block when
    ``delta_dtype``/``delta_col_names`` are supplied.
    """
    cdef list results = []
    cdef list block_offsets = list(block_virtual_offsets)
    cdef Py_ssize_t nblocks = len(block_offsets)
    cdef unsigned long vo
    cdef unsigned long block_start
    cdef unsigned long within
    cdef object block_size_data
    cdef bytes data
    cdef Py_ssize_t off
    cdef object rec
    cdef Py_ssize_t start_block_index = 0
    cdef Py_ssize_t primary_id
    cdef Py_ssize_t id_start, id_end
    cdef object fc = block_first_coords
    cdef bint has_fc = (fc is not None) and (len(fc) == nblocks)
    # Record-aligned blocks (delta files) need record-count arithmetic — see
    # explanation in c_query_regions.
    cdef bint record_aligned = bool(delta_col_names)
    cdef Py_ssize_t records_per_block = _BLOCK_MAX_LEN // unit_size

    # Pre-build struct for repeated use
    cdef object st = _get_struct(fmts)
    unpack_from = st.unpack_from

    # iterate positions
    for start, end in positions:
        if has_fc:
            # In-memory O(log N) bisect on preloaded first_coords — no inflate.
            start_block_index = bisect_right(fc, start) - 1
            if start_block_index < 0:
                start_block_index = 0
        else:
            # Fallback: seek + inflate + read first record per probe.
            start_block_index = _bisect_block_index(
                handle, block_offsets, unpack_from, unit_size, col_to_query, start,
                start_block_index, nblocks)
        vo = block_offsets[start_block_index]
        block_start = vo >> _VO_OFFSET_BITS
        within = vo & _VO_OFFSET_MASK
        handle.seek(block_start)
        try:
            block_size_data = load_bcz_block(handle, True)
        except StopIteration:
            results.append(None)
            continue
        _, data = block_size_data
        data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
        off = within
        # move forward until record[col] >= start
        try:
            rec = unpack_from(data, off)
        except Exception:
            results.append(None)
            continue
        while rec[col_to_query] < start:
            off += unit_size
            if record_aligned:
                primary_id = records_per_block * start_block_index + (off // unit_size)
            else:
                primary_id = ((_BLOCK_MAX_LEN * start_block_index) + off) // unit_size
            if off + unit_size > len(data):
                try:
                    block_size_data = load_bcz_block(handle, True)
                    _, data = block_size_data
                    data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                    start_block_index += 1
                    off = 0
                except StopIteration:
                    break
            try:
                rec = unpack_from(data, off)
            except Exception:
                break
        if rec[col_to_query] < start:
            results.append(None)
            continue
        # compute primary id for start
        if record_aligned:
            primary_id = records_per_block * start_block_index + (off // unit_size)
        else:
            primary_id = int((_BLOCK_MAX_LEN * start_block_index + off) / unit_size)
        id_start = primary_id
        # advance until record[col] >= end
        while True:
            try:
                off += unit_size
                primary_id += 1
                if off + unit_size > len(data):
                    try:
                        block_size_data = load_bcz_block(handle, True)
                        _, data = block_size_data
                        data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                        start_block_index += 1
                        off = 0
                    except StopIteration:
                        break
                rec = unpack_from(data, off)
                if rec[col_to_query] >= end:
                    break
            except Exception:
                break
        id_end = primary_id
        results.append([id_start, id_end])

    return results


def c_pack_records(rows, fmt):
    """Pack a sequence of row tuples into bytes using struct.pack.

    Pre-compiles the struct format for repeated use.  This is faster
    than calling struct.pack() in a Python loop because the Struct
    object caches the compiled format.
    """
    cdef object st = _get_struct(fmt)
    cdef bytearray out = bytearray()
    cdef object row
    pack = st.pack
    for row in rows:
        out.extend(pack(*row))
    return bytes(out)


def c_pack_records_fast(rows, fmt):
    """Pack rows into bytes using a pre-allocated buffer.

    Allocates a single bytearray of exactly (n_rows * unit_size) bytes
    and uses ``pack_into()`` to write directly into the buffer without
    intermediate allocations.  ~2x faster than ``c_pack_records`` for
    large row counts.
    """
    cdef object rows_list = list(rows)
    if not rows_list:
        return b""
    cdef object st = _get_struct(fmt)
    cdef Py_ssize_t unit = st.size
    cdef Py_ssize_t n = len(rows_list)
    cdef bytearray out = bytearray(unit * n)
    cdef Py_ssize_t i
    pack_into = st.pack_into
    for i in range(n):
        pack_into(out, i * unit, *rows_list[i])
    return bytes(out)


def _read_block_first_value(handle, unsigned long vo, object unpack_from, Py_ssize_t unit_size, Py_ssize_t s):
    """Read and return column *s* of the first record at virtual offset *vo*.

    Returns None if the block cannot be read.
    """
    cdef unsigned long block_start = vo >> _VO_OFFSET_BITS
    cdef unsigned long within = vo & _VO_OFFSET_MASK
    handle.seek(block_start)
    try:
        block_size_data = load_bcz_block(handle, True)
    except StopIteration:
        return None
    _, data = block_size_data
    if within + unit_size <= len(data):
        return unpack_from(data, within)[s]
    return None


def _bisect_block_index(handle, list block_offsets, object unpack_from,
                        Py_ssize_t unit_size, Py_ssize_t s, object target,
                        Py_ssize_t lo, Py_ssize_t hi):
    """Binary-search blocks to find the last block whose first record <= target.

    Returns the block index (between lo and hi-1 inclusive) such that
    block[idx].first_value <= target < block[idx+1].first_value.
    Only decompresses O(log N) blocks instead of all N.
    """
    cdef Py_ssize_t mid
    cdef object val
    while lo < hi:
        mid = (lo + hi) // 2
        val = _read_block_first_value(handle, block_offsets[mid], unpack_from, unit_size, s)
        if val is None or val <= target:
            lo = mid + 1
        else:
            hi = mid
    # lo is now the first block whose first value > target,
    # so the answer is lo - 1 (clamped to >= 0).
    return max(lo - 1, 0)


def c_block_first_values(handle, block_virtual_offsets, fmts, unit_size, s):
    """Read the first record of each block and return column *s* values.

    Used by query and pos2id to build a sorted array of block-starting
    positions for efficient binary search.
    """
    cdef list out = []
    cdef unsigned long vo
    cdef unsigned long block_start
    cdef unsigned long within
    cdef object block_size_data
    cdef bytes data
    cdef object st = _get_struct(fmts)
    unpack_from = st.unpack_from
    for vo in block_virtual_offsets:
        block_start = vo >> _VO_OFFSET_BITS
        within = vo & _VO_OFFSET_MASK
        handle.seek(block_start)
        try:
            block_size_data = load_bcz_block(handle, True)
        except StopIteration:
            out.append(None)
            continue
        _, data = block_size_data
        if within + unit_size <= len(data):
            rec = unpack_from(data, within)
            out.append(rec[s])
        else:
            out.append(None)
    return out


def c_read_1record(handle, block_raw_length, buffer, within, fmts, unit_size,
                   delta_dtype=None, delta_col_names=None):
    """Read a single record (unpacked tuple) from the stream, updating buffer state.

    When ``delta_dtype``/``delta_col_names`` are supplied and a new block
    is loaded mid-function, that block is delta-decoded before use.
    Note: delta files use record-aligned blocks so the span-path below is
    exercised only for the initial-buffer-exhausted case, never a true split.
    """
    cdef bytes buf = buffer if buffer is not None else b""
    cdef Py_ssize_t buflen = len(buf)
    cdef object st = _get_struct(fmts)
    if within + unit_size <= buflen:
        rec = st.unpack_from(buf, within)
        within += unit_size
        return rec, block_raw_length, buf, within
    # need to load next block or remaining bytes
    out = bytearray()
    if within < buflen:
        out.extend(buf[within:])
    try:
        block_size_data = load_bcz_block(handle, True)
    except StopIteration:
        return None, 0, b"", 0
    block_raw_length, buf = block_size_data
    buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
    needed = unit_size - len(out)
    if needed <= len(buf):
        out.extend(buf[:needed])
        within = needed
        rec = st.unpack(bytes(out))
        return rec, block_raw_length, buf, within
    else:
        return None, block_raw_length, buf, 0


def c_seek_and_read_1record(handle, virtual_offset, fmts, unit_size,
                            delta_dtype=None, delta_col_names=None):
    """Seek to virtual offset and read one record (unpacked tuple).

    Each loaded block is delta-decoded when ``delta_dtype``/``delta_col_names``
    are supplied.
    """
    start = virtual_offset >> _VO_OFFSET_BITS
    within = virtual_offset & _VO_OFFSET_MASK
    handle.seek(start)
    cdef object st = _get_struct(fmts)
    try:
        block_size_data = load_bcz_block(handle, True)
    except StopIteration:
        return None
    _, buf = block_size_data
    buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
    if within + unit_size <= len(buf):
        return st.unpack_from(buf, within)
    # need to assemble across block boundary
    out = bytearray()
    if within < len(buf):
        out.extend(buf[within:])
    try:
        block_size_data = load_bcz_block(handle, True)
    except StopIteration:
        return None
    _, buf2 = block_size_data
    buf2 = _apply_delta_bytes(buf2, delta_dtype, delta_col_names)
    needed = unit_size - len(out)
    if needed <= len(buf2):
        out.extend(buf2[:needed])
        return st.unpack(bytes(out))
    return None


def c_query_regions(handle, block_virtual_offsets, fmts, unit_size, regions, s, e, dim,
                    block_first_coords=None,
                    delta_dtype=None, delta_col_names=None):
    """Accelerated query for a single dim.

    When ``block_first_coords`` is supplied (the per-block first-record
    values preloaded from the chunk tail), binary search uses pure
    in-memory :func:`bisect.bisect_right` — zero inflate probes. This is
    the fast path that matches tabix-class performance for regional
    queries on indexed .cz files (files written with ``sort_col``).

    Otherwise falls back to :func:`_bisect_block_index` which decompresses
    one block per probe (O(log N) inflates). Used for files without a
    first_coords index (e.g. reference-less allc files).

    Delta-encoded columns are decoded per block when
    ``delta_dtype``/``delta_col_names`` are supplied.
    """
    cdef list results = []
    cdef list block_offsets = list(block_virtual_offsets)
    cdef Py_ssize_t nblocks = len(block_offsets)
    cdef Py_ssize_t start_block_index = 0
    cdef unsigned long vo
    cdef unsigned long block_start
    cdef unsigned long within
    cdef object block_size_data
    cdef bytes data
    cdef Py_ssize_t off
    cdef object rec
    cdef Py_ssize_t primary_id
    cdef object fc = block_first_coords
    cdef bint has_fc = (fc is not None) and (len(fc) == nblocks)
    # When delta encoding is enabled, the writer aligns records to block
    # boundaries: every block holds exactly ``records_per_block`` records
    # (= _BLOCK_MAX_LEN // unit_size) and the trailing bytes are unused.
    # The byte-arithmetic formula ``(_BLOCK_MAX_LEN * idx + off) / unit_size``
    # is then wrong by ``(_BLOCK_MAX_LEN % unit_size) * idx / unit_size``
    # records.  Use record-count arithmetic instead.
    cdef bint record_aligned = bool(delta_col_names)
    cdef Py_ssize_t records_per_block = _BLOCK_MAX_LEN // unit_size

    cdef object st = _get_struct(fmts)
    unpack_from = st.unpack_from

    for start, end in regions:
        if has_fc:
            # Real O(log N) in-memory bisect on first_coords array.
            start_block_index = bisect_right(fc, start) - 1
            if start_block_index < 0:
                start_block_index = 0
        else:
            # Fallback: bisect that decompresses one block per probe.
            start_block_index = _bisect_block_index(
                handle, block_offsets, unpack_from, unit_size, s, start,
                start_block_index, nblocks)
        vo = block_offsets[start_block_index]
        block_start = vo >> _VO_OFFSET_BITS
        within = vo & _VO_OFFSET_MASK
        handle.seek(block_start)
        try:
            block_size_data = load_bcz_block(handle, True)
        except StopIteration:
            continue
        _, data = block_size_data
        data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
        off = within
        try:
            rec = unpack_from(data, off)
        except Exception:
            continue
        while rec[s] < start:
            off += unit_size
            if off + unit_size > len(data):
                try:
                    block_size_data = load_bcz_block(handle, True)
                    _, data = block_size_data
                    data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                    start_block_index += 1
                    off = 0
                except StopIteration:
                    break
            try:
                rec = unpack_from(data, off)
            except Exception:
                break
        if rec[s] < start:
            continue
        # +1 to make primary_id 1-based, matching fetchByStartID expectation
        if record_aligned:
            primary_id = records_per_block * start_block_index + (off // unit_size) + 1
        else:
            primary_id = int((_BLOCK_MAX_LEN * start_block_index + off) / unit_size) + 1
        results.append(("primary_id_&_dim:", primary_id, dim))
        while True:
            if rec[e] <= end:
                results.append((dim, rec))
                off += unit_size
                primary_id += 1
                if off + unit_size > len(data):
                    try:
                        block_size_data = load_bcz_block(handle, True)
                        _, data = block_size_data
                        data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                        start_block_index += 1
                        off = 0
                    except StopIteration:
                        break
                try:
                    rec = unpack_from(data, off)
                except Exception:
                    break
            else:
                break
    return results


def c_query_regions_flat(handle, block_virtual_offsets, fmts, unit_size,
                         start, end, s, e, dim,
                         block_first_coords=None,
                         delta_dtype=None, delta_col_names=None):
    """Fast single-region variant of :func:`c_query_regions`.

    Returns a ``list`` of flat tuples ``(dim_0, ..., dim_k, col_0, ..., col_n)``
    ready for the caller — no ``primary_id_&_dim`` marker, no second pass in
    Python. Shaves ~0.4 µs per record (no generator yield, no tuple concat in
    Python) and lets ``Reader.query(..., printout=False)`` materialise the
    full result set in a single call into C code.

    ``dim`` must be a tuple (the chunk's chunk_key values, e.g. ``('chr9',)``).
    Numeric columns only — callers should route records with bytes columns
    through the Python fallback so utf-8 decoding stays out of the hot loop.

    Delta-encoded columns are decoded per block when
    ``delta_dtype``/``delta_col_names`` are supplied.
    """
    cdef list results = []
    cdef list block_offsets = list(block_virtual_offsets)
    cdef Py_ssize_t nblocks = len(block_offsets)
    cdef Py_ssize_t start_block_index = 0
    cdef unsigned long vo
    cdef unsigned long block_start
    cdef unsigned long within
    cdef object block_size_data
    cdef bytes data
    cdef Py_ssize_t off
    cdef object rec
    cdef object fc = block_first_coords
    cdef bint has_fc = (fc is not None) and (len(fc) == nblocks)
    cdef tuple dim_tuple = tuple(dim)

    cdef object st = _get_struct(fmts)
    unpack_from = st.unpack_from

    if has_fc:
        start_block_index = bisect_right(fc, start) - 1
        if start_block_index < 0:
            start_block_index = 0
    else:
        start_block_index = _bisect_block_index(
            handle, block_offsets, unpack_from, unit_size, s, start,
            0, nblocks)
    vo = block_offsets[start_block_index]
    block_start = vo >> _VO_OFFSET_BITS
    within = vo & _VO_OFFSET_MASK
    handle.seek(block_start)
    try:
        block_size_data = load_bcz_block(handle, True)
    except StopIteration:
        return results
    _, data = block_size_data
    data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
    off = within
    try:
        rec = unpack_from(data, off)
    except Exception:
        return results
    while rec[s] < start:
        off += unit_size
        if off + unit_size > len(data):
            try:
                block_size_data = load_bcz_block(handle, True)
                _, data = block_size_data
                data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                off = 0
            except StopIteration:
                return results
        try:
            rec = unpack_from(data, off)
        except Exception:
            return results
    if rec[s] < start:
        return results
    while rec[e] <= end:
        results.append(dim_tuple + rec)
        off += unit_size
        if off + unit_size > len(data):
            try:
                block_size_data = load_bcz_block(handle, True)
                _, data = block_size_data
                data = _apply_delta_bytes(data, delta_dtype, delta_col_names)
                off = 0
            except StopIteration:
                break
        try:
            rec = unpack_from(data, off)
        except Exception:
            break
    return results


def c_write_chunk_tail(handle, chunk_data_len, block_1st_record_virtual_offsets, chunk_dims):
    """Write chunk tail metadata in a single batched I/O call.

    Packs the header (data_len + nblocks) and all block virtual offsets
    into one struct.pack() call to minimize write() syscalls.
    """
    n_blocks = len(block_1st_record_virtual_offsets)
    # Pack header + all block offsets in one call
    cdef bytearray buf = bytearray(
        struct.pack(f"<QQ{n_blocks}Q", chunk_data_len, n_blocks,
                    *block_1st_record_virtual_offsets))
    # Append dims directly to the same buffer
    for dim in chunk_dims:
        dim_b = dim.encode('utf-8') if isinstance(dim, str) else dim
        buf.append(len(dim_b))
        buf.extend(dim_b)
    handle.write(bytes(buf))
    return None


def c_parse_blocks_buffer(bytes raw_all, delta_dtype=None, delta_col_names=None):
    """Public wrapper around the internal block-buffer parser.

    Lets a caller (Reader async prefetch) read the compressed bytes
    itself (e.g. on a background thread) and then hand them in for
    multi-threaded decompression. Mirrors the in-process logic of
    :func:`c_fetch_chunk`'s bulk-I/O path without doing the I/O.
    """
    if not raw_all:
        return b""
    if delta_dtype is not None and delta_col_names:
        return _parse_blocks_from_buffer_delta(raw_all, delta_dtype, delta_col_names)
    return _parse_blocks_from_buffer(raw_all)


def c_fetch_chunk(handle, chunk_start_offset_plus10, block_virtual_offsets, fmts, unit_size,
                  chunk_compressed_size=None,
                  delta_dtype=None, delta_col_names=None):
    """Read and decompress all blocks of a chunk into one bytes object.

    When *chunk_compressed_size* is provided, reads the entire compressed
    region in one I/O call, then parses and decompresses individual
    blocks from the in-memory buffer.  This reduces N seek+read syscalls
    to a single read.

    Delta-encoded columns are decoded per block before concatenation when
    ``delta_dtype``/``delta_col_names`` are supplied.
    """
    cdef list block_offsets = list(block_virtual_offsets)
    if not block_offsets:
        return b""
    cdef unsigned long first_block_start = block_offsets[0] >> _VO_OFFSET_BITS
    handle.seek(first_block_start)
    # Bulk I/O path: read all compressed block data in one syscall
    if chunk_compressed_size is not None and chunk_compressed_size > 0:
        raw_all = handle.read(chunk_compressed_size)
        if delta_dtype is not None and delta_col_names:
            return _parse_blocks_from_buffer_delta(raw_all, delta_dtype, delta_col_names)
        return _parse_blocks_from_buffer(raw_all)
    # Fallback: sequential block reads (when compressed size is unknown)
    cdef list chunks = []
    cdef object block_size_data
    cdef Py_ssize_t nblocks = len(block_offsets)
    cdef Py_ssize_t i
    cdef bytes blk
    for i in range(nblocks):
        try:
            block_size_data = load_bcz_block(handle, True)
        except StopIteration:
            break
        blk = block_size_data[1]
        blk = _apply_delta_bytes(blk, delta_dtype, delta_col_names)
        chunks.append(blk)
    if not chunks:
        return b""
    return b"".join(chunks)


def c_get_records_by_ids(handle, chunk_block_virtual_offsets, unit_size, IDs,
                         delta_dtype=None, delta_col_names=None,
                         block_size=None):
    """Fetch raw record bytes for given 1-based primary IDs.

    Caches the currently decompressed block to avoid redundant I/O
    when consecutive IDs fall in the same block.

    For delta files, ``block_size`` is the record-aligned block length
    (``(_BLOCK_MAX_LEN // unit_size) * unit_size``); default is
    ``_BLOCK_MAX_LEN`` for raw files.  The loaded block is delta-decoded
    when ``delta_dtype``/``delta_col_names`` are supplied.
    """
    cdef list results = []
    cdef unsigned long vo
    cdef unsigned long block_start
    cdef object block_size_data
    cdef bytes buf = b""
    cdef Py_ssize_t current_block_start = -1
    cdef Py_ssize_t effective_block = _BLOCK_MAX_LEN
    if block_size is not None:
        effective_block = block_size
    for ID in IDs:
        idx = ((ID - 1) * unit_size) // effective_block
        within = ((ID - 1) * unit_size) % effective_block
        vo = chunk_block_virtual_offsets[idx]
        block_start = vo >> _VO_OFFSET_BITS
        if block_start != current_block_start:
            handle.seek(block_start)
            try:
                block_size_data = load_bcz_block(handle, True)
            except StopIteration:
                results.append(b"")
                current_block_start = block_start
                buf = b""
                continue
            _, buf = block_size_data
            buf = _apply_delta_bytes(buf, delta_dtype, delta_col_names)
            current_block_start = block_start
        if within + unit_size <= len(buf):
            results.append(buf[within: within + unit_size])
        else:
            results.append(b"")
    return results


# ============================================================
# Optimized WriteC - extract C positions from genome sequence
#
# The pure-Python WriteC in allc.py iterates over every base in a
# chromosome sequence using BioPython's Seq API.  This Cython
# implementation operates directly on the raw byte buffer of the
# sequence, avoiding Python object creation per nucleotide.
# For a typical mammalian genome (~3 GB), this provides a ~10-50x
# speedup.
# ============================================================

# Character code constants for direct byte comparison (avoids ord() calls)
cdef unsigned char _CHAR_A = 65   # ord('A')
cdef unsigned char _CHAR_T = 84   # ord('T')
cdef unsigned char _CHAR_C = 67   # ord('C')
cdef unsigned char _CHAR_G = 71   # ord('G')
cdef unsigned char _CHAR_N = 78   # ord('N')
cdef unsigned char _CHAR_a = 97   # ord('a')
cdef unsigned char _CHAR_t = 116  # ord('t')
cdef unsigned char _CHAR_c = 99   # ord('c')
cdef unsigned char _CHAR_g = 103  # ord('g')
cdef unsigned char _CHAR_n = 110  # ord('n')
cdef unsigned char _CHAR_PLUS = 43   # ord('+')
cdef unsigned char _CHAR_MINUS = 45  # ord('-')

# Complement mapping for nucleotides (256-entry lookup table for O(1) access)
cdef unsigned char[256] _complement_table
cdef bint _complement_table_initialized = False

cdef void _init_complement_table():
    """Initialize the 256-entry nucleotide complement lookup table.

    Called once lazily; maps A<->T, C<->G (both cases), N->N.
    All other bytes map to themselves.
    """
    global _complement_table_initialized
    if _complement_table_initialized:
        return
    cdef int i
    for i in range(256):
        _complement_table[i] = <unsigned char>i  # default: identity
    _complement_table[_CHAR_A] = _CHAR_T
    _complement_table[_CHAR_T] = _CHAR_A
    _complement_table[_CHAR_C] = _CHAR_G
    _complement_table[_CHAR_G] = _CHAR_C
    _complement_table[_CHAR_a] = _CHAR_t
    _complement_table[_CHAR_t] = _CHAR_a
    _complement_table[_CHAR_c] = _CHAR_g
    _complement_table[_CHAR_g] = _CHAR_c
    _complement_table[_CHAR_N] = _CHAR_N
    _complement_table[_CHAR_n] = _CHAR_n
    _complement_table_initialized = True


cdef inline unsigned char _to_upper(unsigned char c) noexcept nogil:
    """Convert char to uppercase."""
    if c >= 97 and c <= 122:  # 'a' to 'z'
        return c - 32
    return c


def c_extract_c_positions(seq_bytes):
    """
    Extract all C positions (forward and reverse strand) from a DNA sequence.
    
    This is the core hot loop of WriteC, optimized in Cython.
    
    Parameters
    ----------
    seq_bytes : bytes
        DNA sequence as bytes (from record.seq.__str__().encode() or str(record.seq).encode())
    
    Returns
    -------
    list of tuples: [(pos, strand_byte, context_bytes), ...]
        pos: 1-based position (int)
        strand_byte: b'+' or b'-'
        context_bytes: 3-byte context (e.g., b'CGA')
    """
    _init_complement_table()
    
    cdef const unsigned char[:] seq_view
    cdef Py_ssize_t N, i
    cdef unsigned char base, base_up, c0, c1, c2
    cdef list results = []
    cdef bytes context
    cdef unsigned char ctx_buf[3]
    
    if not seq_bytes:
        return results
    
    seq_view = seq_bytes
    N = len(seq_view)
    
    for i in range(N):
        base = seq_view[i]
        base_up = _to_upper(base)
        
        if base_up == _CHAR_C:
            # Forward strand C
            # Context is seq[i:i+3] uppercased
            ctx_buf[0] = _CHAR_C
            if i + 1 < N:
                ctx_buf[1] = _to_upper(seq_view[i + 1])
            else:
                ctx_buf[1] = _CHAR_N
            if i + 2 < N:
                ctx_buf[2] = _to_upper(seq_view[i + 2])
            else:
                ctx_buf[2] = _CHAR_N
            context = PyBytes_FromStringAndSize(<char*>ctx_buf, 3)
            results.append((i + 1, b'+', context))
        
        elif base_up == _CHAR_G:
            # Reverse strand: G on forward = C on reverse
            # Context is reverse complement of seq[i-2:i+1]
            # For position i, reverse complement context:
            #   seq[i] -> complement = C (always)
            #   seq[i-1] -> complement and becomes position 1
            #   seq[i-2] -> complement and becomes position 2
            ctx_buf[0] = _CHAR_C  # complement of G
            if i >= 1:
                ctx_buf[1] = _to_upper(_complement_table[seq_view[i - 1]])
            else:
                ctx_buf[1] = _CHAR_N
            if i >= 2:
                ctx_buf[2] = _to_upper(_complement_table[seq_view[i - 2]])
            else:
                ctx_buf[2] = _CHAR_N
            context = PyBytes_FromStringAndSize(<char*>ctx_buf, 3)
            results.append((i + 1, b'-', context))
    
    return results


def c_write_c_records(seq_bytes, batch_size=5000):
    """
    Generator version that yields batches of packed records for WriteC.
    
    This extracts C positions and packs them into binary format suitable for
    direct writing with Writer.write_chunk().
    
    Parameters
    ----------
    seq_bytes : bytes
        DNA sequence as bytes
    batch_size : int
        Number of records per batch
    
    Yields
    ------
    tuple: (packed_data_bytes, count)
        packed_data_bytes: binary data packed as '<Qc3s' format
        count: number of records in this batch
    """
    _init_complement_table()
    
    cdef const unsigned char[:] seq_view
    cdef Py_ssize_t N, i, batch_count
    cdef unsigned char base, base_up
    cdef bytearray batch_buf
    cdef unsigned char ctx_buf[3]
    cdef Py_ssize_t pos
    
    if not seq_bytes:
        return
    
    seq_view = seq_bytes
    N = len(seq_view)
    
    # Format: Q (8 bytes) + c (1 byte) + 3s (3 bytes) = 12 bytes per record
    # represent: position, strand, context
    cdef Py_ssize_t record_size = 12
    batch_buf = bytearray(batch_size * record_size)
    batch_count = 0
    cdef Py_ssize_t buf_offset = 0
    
    for i in range(N):
        base = seq_view[i]
        base_up = _to_upper(base)
        
        if base_up == _CHAR_C:
            # Forward strand C
            pos = i + 1
            ctx_buf[0] = _CHAR_C
            if i + 1 < N:
                ctx_buf[1] = _to_upper(seq_view[i + 1])
            else:
                ctx_buf[1] = _CHAR_N
            if i + 2 < N:
                ctx_buf[2] = _to_upper(seq_view[i + 2])
            else:
                ctx_buf[2] = _CHAR_N
            
            # Pack directly into buffer: struct.pack('<Qc3s', pos, b'+', context)
            # Q = 8 bytes little endian unsigned long long
            batch_buf[buf_offset] = pos & 0xFF
            batch_buf[buf_offset + 1] = (pos >> 8) & 0xFF
            batch_buf[buf_offset + 2] = (pos >> 16) & 0xFF
            batch_buf[buf_offset + 3] = (pos >> 24) & 0xFF
            batch_buf[buf_offset + 4] = (pos >> 32) & 0xFF
            batch_buf[buf_offset + 5] = (pos >> 40) & 0xFF
            batch_buf[buf_offset + 6] = (pos >> 48) & 0xFF
            batch_buf[buf_offset + 7] = (pos >> 56) & 0xFF
            # c = 1 byte char
            batch_buf[buf_offset + 8] = _CHAR_PLUS
            # 3s = 3 bytes
            batch_buf[buf_offset + 9] = ctx_buf[0]
            batch_buf[buf_offset + 10] = ctx_buf[1]
            batch_buf[buf_offset + 11] = ctx_buf[2]
            
            buf_offset += record_size
            batch_count += 1
            
            if batch_count >= batch_size:
                yield bytes(batch_buf[:buf_offset]), batch_count
                batch_buf = bytearray(batch_size * record_size)
                buf_offset = 0
                batch_count = 0
        
        elif base_up == _CHAR_G:
            # Reverse strand C
            pos = i + 1
            ctx_buf[0] = _CHAR_C
            if i >= 1:
                ctx_buf[1] = _to_upper(_complement_table[seq_view[i - 1]])
            else:
                ctx_buf[1] = _CHAR_N
            if i >= 2:
                ctx_buf[2] = _to_upper(_complement_table[seq_view[i - 2]])
            else:
                ctx_buf[2] = _CHAR_N
            
            batch_buf[buf_offset] = pos & 0xFF
            batch_buf[buf_offset + 1] = (pos >> 8) & 0xFF
            batch_buf[buf_offset + 2] = (pos >> 16) & 0xFF
            batch_buf[buf_offset + 3] = (pos >> 24) & 0xFF
            batch_buf[buf_offset + 4] = (pos >> 32) & 0xFF
            batch_buf[buf_offset + 5] = (pos >> 40) & 0xFF
            batch_buf[buf_offset + 6] = (pos >> 48) & 0xFF
            batch_buf[buf_offset + 7] = (pos >> 56) & 0xFF
            batch_buf[buf_offset + 8] = _CHAR_MINUS
            batch_buf[buf_offset + 9] = ctx_buf[0]
            batch_buf[buf_offset + 10] = ctx_buf[1]
            batch_buf[buf_offset + 11] = ctx_buf[2]
            
            buf_offset += record_size
            batch_count += 1
            
            if batch_count >= batch_size:
                yield bytes(batch_buf[:buf_offset]), batch_count
                batch_buf = bytearray(batch_size * record_size)
                buf_offset = 0
                batch_count = 0
    
    # Yield remaining records
    if batch_count > 0:
        yield bytes(batch_buf[:buf_offset]), batch_count


# ---------------------------------------------------------------------------
# CZIX footer fast parser
# ---------------------------------------------------------------------------
def c_parse_czix(bytes buf, int n_chunk_dims):
    """Parse a CZIX chunk index buffer into ``(index_dict, chunk_key2offset)``.

    ``buf`` must be the CZIX payload read from the file tail starting at the
    'CZIX' magic and excluding the 28-byte EOF sentinel.  Returns ``None`` if
    the magic does not match.  The returned ``index_dict`` maps a tuple of
    dim strings to ``{'start', 'size', 'data_len', 'nblocks'}``; the second
    dict mirrors only the ``start`` field for fast query lookups.
    """
    cdef Py_ssize_t blen = len(buf)
    if blen < 12:
        return None
    cdef const unsigned char* p = <const unsigned char*> buf
    # magic check: b'CZIX'
    if p[0] != 0x43 or p[1] != 0x5A or p[2] != 0x49 or p[3] != 0x58:
        return None
    cdef unsigned long long n_chunks = 0
    cdef Py_ssize_t i
    for i in range(8):
        n_chunks |= (<unsigned long long> p[4 + i]) << (8 * i)
    cdef Py_ssize_t off = 12
    cdef dict index = {}
    cdef dict dim2cs = {}
    cdef unsigned long long n_chunks_i
    cdef Py_ssize_t j, dlen
    cdef unsigned long long start, size, data_len, nblocks
    cdef tuple dims
    cdef list dim_list
    for n_chunks_i in range(n_chunks):
        dim_list = [None] * n_chunk_dims
        for j in range(n_chunk_dims):
            if off >= blen:
                return None
            dlen = p[off]
            off += 1
            if off + dlen + 32 > blen:
                return None
            dim_list[j] = buf[off:off + dlen].decode('utf-8')
            off += dlen
        dims = tuple(dim_list)
        # unpack 4 little-endian uint64 fields inline
        start = 0
        for i in range(8):
            start |= (<unsigned long long> p[off + i]) << (8 * i)
        size = 0
        for i in range(8):
            size |= (<unsigned long long> p[off + 8 + i]) << (8 * i)
        data_len = 0
        for i in range(8):
            data_len |= (<unsigned long long> p[off + 16 + i]) << (8 * i)
        nblocks = 0
        for i in range(8):
            nblocks |= (<unsigned long long> p[off + 24 + i]) << (8 * i)
        off += 32
        index[dims] = {
            'start': start, 'size': size,
            'data_len': data_len, 'nblocks': nblocks,
        }
        dim2cs[dims] = start
    return index, dim2cs


# ---------------------------------------------------------------------------
# Tab-separated text parser for allc.tsv-style files (N4)
# ---------------------------------------------------------------------------
def c_parse_tab_lines_int(list lines, list cols, int sep_byte=0x09):
    """Fast parser for tab-separated lines with integer-only target columns.

    Used by `_parse_tabix_lines` when every output column is an integer
    type. Returns a list of int64 numpy arrays (one per requested column),
    in the order of `cols`. Caller is responsible for clipping to the
    final dtype.

    This bypasses pandas.read_csv (which builds a DataFrame and a Python
    string→int loop on a temporary buffer) and is ~2-3× faster on long
    line lists composed of small integers.

    `cols` must be sorted ascending (caller's responsibility).
    """
    np = _get_np()
    cdef Py_ssize_t n_lines = len(lines)
    cdef Py_ssize_t n_cols = len(cols)
    if n_cols == 0:
        return []
    # Allocate output as int64 (caller clips); avoids overflow during parse.
    out_arrays = [np.empty(n_lines, dtype=np.int64) for _ in range(n_cols)]
    cdef Py_ssize_t[:] target_cols = np.asarray(cols, dtype=np.intp)
    cdef Py_ssize_t li, ci
    cdef long long[::1] _v
    cdef long long** ptrs = <long long**> malloc(n_cols * sizeof(long long*))
    if ptrs == NULL:
        raise MemoryError()
    for ci in range(n_cols):
        _v = out_arrays[ci]
        ptrs[ci] = &_v[0]

    cdef bytes line_b
    cdef const unsigned char* lp
    cdef Py_ssize_t llen
    cdef Py_ssize_t pos
    cdef Py_ssize_t cur_col
    cdef Py_ssize_t want_idx
    cdef Py_ssize_t target_col
    cdef long long val
    cdef int neg
    cdef unsigned char c

    try:
        for li in range(n_lines):
            line = lines[li]
            if isinstance(line, str):
                line_b = line.encode('ascii')
            else:
                line_b = line
            lp = <const unsigned char*> (<const char*> line_b)
            llen = len(line_b)
            pos = 0
            cur_col = 0
            want_idx = 0
            target_col = target_cols[0]
            while pos < llen and want_idx < n_cols:
                if cur_col == target_col:
                    # Parse signed integer at pos until tab or EOL.
                    val = 0
                    neg = 0
                    if pos < llen and lp[pos] == 0x2d:  # '-'
                        neg = 1
                        pos += 1
                    while pos < llen:
                        c = lp[pos]
                        if c == sep_byte or c == 0x0a or c == 0x0d:
                            break
                        if 0x30 <= c <= 0x39:
                            val = val * 10 + (c - 0x30)
                            pos += 1
                        else:
                            # Non-numeric content (e.g. context letters): skip rest of field.
                            val = 0
                            while pos < llen and lp[pos] != sep_byte and lp[pos] != 0x0a and lp[pos] != 0x0d:
                                pos += 1
                            break
                    ptrs[want_idx][li] = -val if neg else val
                    want_idx += 1
                    if want_idx < n_cols:
                        target_col = target_cols[want_idx]
                # Skip to next tab or end of line.
                while pos < llen and lp[pos] != sep_byte and lp[pos] != 0x0a and lp[pos] != 0x0d:
                    pos += 1
                if pos < llen and lp[pos] == sep_byte:
                    pos += 1
                    cur_col += 1
            # Fill missing trailing columns with 0.
            while want_idx < n_cols:
                ptrs[want_idx][li] = 0
                want_idx += 1
    finally:
        free(ptrs)
    return out_arrays
