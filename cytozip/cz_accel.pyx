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
import struct
import zlib

from bisect import bisect_right

# ---------------------------------------------------------------------------
# Constants shared with cz.py (must match exactly).
# ---------------------------------------------------------------------------
_block_magic = b"CB"  # 2-byte magic at the start of each compressed block

cdef unsigned long _BLOCK_MAX_LEN = 65535  # max decompressed block size

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

    This avoids N seek+read syscalls by processing the pre-read buffer
    directly. Each block is: magic(2B) + bsize(2B) + deflate_data + data_len(2B).
    """
    cdef list chunks = []
    cdef Py_ssize_t offset = 0
    cdef Py_ssize_t total = len(raw_all)
    cdef unsigned short bsize
    cdef Py_ssize_t deflate_size
    while offset + 4 <= total:
        # Check block magic 'CB' (0x43, 0x42)
        if raw_all[offset] != 67 or raw_all[offset + 1] != 66:
            break
        bsize = (<unsigned char>raw_all[offset + 2]) | ((<unsigned char>raw_all[offset + 3]) << 8)
        if offset + bsize > total:
            break
        deflate_size = bsize - 6
        chunks.append(_c_inflate(raw_all[offset + 4:offset + 4 + deflate_size]))
        offset += bsize
    if not chunks:
        return b""
    return b"".join(chunks)


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
    """
    if delta_dtype is None or not delta_col_names:
        return data
    cdef Py_ssize_t itemsize = delta_dtype.itemsize
    cdef Py_ssize_t nrec = len(data) // itemsize
    if nrec == 0:
        return data
    np = _get_np()
    arr = np.frombuffer(data, dtype=delta_dtype, count=nrec).copy()
    for name in delta_col_names:
        np.cumsum(arr[name], out=arr[name])
    # Preserve any trailing bytes beyond record alignment (shouldn't exist
    # for delta files, but be defensive).
    cdef Py_ssize_t used = nrec * itemsize
    if used == len(data):
        return arr.tobytes()
    return arr.tobytes() + data[used:]


cdef bytes _parse_blocks_from_buffer_delta(bytes raw_all, object delta_dtype, object delta_col_names):
    """Delta-aware variant of _parse_blocks_from_buffer.

    Each block is inflated and immediately delta-decoded before being
    concatenated, because delta is block-local.
    """
    if delta_dtype is None or not delta_col_names:
        return _parse_blocks_from_buffer(raw_all)
    cdef list chunks = []
    cdef Py_ssize_t offset = 0
    cdef Py_ssize_t total = len(raw_all)
    cdef unsigned short bsize
    cdef Py_ssize_t deflate_size
    cdef bytes blk
    while offset + 4 <= total:
        if raw_all[offset] != 67 or raw_all[offset + 1] != 66:
            break
        bsize = (<unsigned char>raw_all[offset + 2]) | ((<unsigned char>raw_all[offset + 3]) << 8)
        if offset + bsize > total:
            break
        deflate_size = bsize - 6
        blk = _c_inflate(raw_all[offset + 4:offset + 4 + deflate_size])
        chunks.append(_apply_delta_bytes(blk, delta_dtype, delta_col_names))
        offset += bsize
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
                                      size_t *actual_out_nbytes_ret)

    libdeflate_compressor *libdeflate_alloc_compressor(int compression_level)
    void libdeflate_free_compressor(libdeflate_compressor *c)
    size_t libdeflate_deflate_compress_bound(libdeflate_compressor *c, size_t in_nbytes)
    size_t libdeflate_deflate_compress(libdeflate_compressor *c,
                                       const void *in_, size_t in_nbytes,
                                       void *out, size_t out_nbytes_avail)


from cpython.bytearray cimport PyByteArray_FromStringAndSize, PyByteArray_AsString
from cpython.bytes cimport PyBytes_FromStringAndSize
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
    cdef size_t out_size = 65536  # blocks are at most 65535 bytes uncompressed
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
    # Read block_size as 2-byte little-endian unsigned short directly,
    # avoiding struct.unpack overhead on this hot path.
    cdef bytes bs_raw = handle.read(2)
    cdef unsigned short block_size = (<unsigned char>bs_raw[0]) | ((<unsigned char>bs_raw[1]) << 8)
    cdef Py_ssize_t deflate_size = block_size - 6
    cdef bytes dl_raw
    cdef unsigned short data_len
    if decompress:
        raw = handle.read(deflate_size)
        data = _c_inflate(raw)
        # skip the trailing uncompressed length field
        handle.read(2)
        return block_size, data
    else:
        # seek forward deflate data and read trailing uncompressed length
        handle.seek(deflate_size, 1) # skip the compressed data.
        dl_raw = handle.read(2)
        data_len = (<unsigned char>dl_raw[0]) | ((<unsigned char>dl_raw[1]) << 8)
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
        block_start = vo >> 16
        within = vo & 0xFFFF
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
    cdef unsigned long block_start = vo >> 16
    cdef unsigned long within = vo & 0xFFFF
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
        block_start = vo >> 16
        within = vo & 0xFFFF
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
    start = virtual_offset >> 16
    within = virtual_offset & 0xFFFF
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
        block_start = vo >> 16
        within = vo & 0xFFFF
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

    ``dim`` must be a tuple (the chunk's dimension values, e.g. ``('chr9',)``).
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
    block_start = vo >> 16
    within = vo & 0xFFFF
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
    cdef unsigned long first_block_start = block_offsets[0] >> 16
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
        block_start = vo >> 16
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


def c_write_c_records(seq_bytes, chunksize=5000):
    """
    Generator version that yields batches of packed records for WriteC.
    
    This extracts C positions and packs them into binary format suitable for
    direct writing with Writer.write_chunk().
    
    Parameters
    ----------
    seq_bytes : bytes
        DNA sequence as bytes
    chunksize : int
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
    batch_buf = bytearray(chunksize * record_size)
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
            
            if batch_count >= chunksize:
                yield bytes(batch_buf[:buf_offset]), batch_count
                batch_buf = bytearray(chunksize * record_size)
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
            
            if batch_count >= chunksize:
                yield bytes(batch_buf[:buf_offset]), batch_count
                batch_buf = bytearray(chunksize * record_size)
                buf_offset = 0
                batch_count = 0
    
    # Yield remaining records
    if batch_count > 0:
        yield bytes(batch_buf[:buf_offset]), batch_count


# ---------------------------------------------------------------------------
# CZIX footer fast parser
# ---------------------------------------------------------------------------
def c_parse_czix(bytes buf, int n_dims):
    """Parse a CZIX chunk index buffer into ``(index_dict, dim2chunk_start)``.

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
        dim_list = [None] * n_dims
        for j in range(n_dims):
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
