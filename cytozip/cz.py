#!/usr/bin/env python
"""
cz.py — Core module for the cytozip (ChunkZIP) binary format.

The .cz file format is a columnar, chunk-based binary format designed for
efficient storage and random access of large tabular data (e.g., DNA
methylation allc tables).  The format is inspired by BGZF (blocked gzip)
but uses a two-level hierarchy:

  File layout:
    [Header] [Chunk 0] [Chunk 1] ... [Chunk N] [ChunkIndex] [EOF marker]

  - Header:  magic, version, total_size, message, column formats/names,
             dimension names.
  - Chunk:   A sequence of independently compressed *blocks* that share
             the same dimension values (e.g., one chromosome).
             Each block is at most 65535 bytes of raw data, compressed
             with raw DEFLATE.
  - Chunk tail:  Appended after the last block of a chunk; stores
             total uncompressed length, number of blocks, per-block
             virtual offsets, and dimension values.
  - Chunk index:  Written at end-of-file before the EOF marker so that
             remote/HTTP readers can locate chunks in O(1) without
             scanning the whole file.
  - EOF marker: 28-byte sentinel (borrowed from BGZF EOF).

Virtual offsets
---------------
A virtual offset encodes both the physical file position of a block and
the byte offset within the decompressed block data, packed into a single
64-bit integer:  (block_start << 16) | within_block_offset.
This allows O(1) random access to any record.

"""
import glob
import os, sys
import struct
import zlib
from builtins import open as _open
import gzip
import math
import multiprocessing
from loguru import logger


# ---------------------------------------------------------------------------
# Lazy imports for numpy and pandas — deferred until first access so that
# ``import cytozip`` and simple CLI commands stay fast (~1 s instead of ~10 s).
# ---------------------------------------------------------------------------
class _LazyModule:
	"""Proxy that defers ``import`` until the first attribute access,
	then replaces itself in *globals()* with the real module."""
	__slots__ = ('_name', '_alias')
	def __init__(self, name, alias):
		self._name = name
		self._alias = alias
	def __getattr__(self, attr):
		import importlib
		mod = importlib.import_module(self._name)
		globals()[self._alias] = mod
		return getattr(mod, attr)

np = _LazyModule('numpy', 'np')
pd = _LazyModule('pandas', 'pd')

# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------
_cz_magic = b'CZIP'              # 4-byte file magic identifying a .cz file
_block_magic = b"CB"              # 2-byte magic at the start of each block
_chunk_magic = b"CC"              # 2-byte magic at the start of each chunk
_BLOCK_MAX_LEN = 65535            # Maximum uncompressed block size (2^16 - 1)
# 28-byte BGZF-style EOF marker written at the end of every .cz file.
_cz_eof = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00CZ\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
_chunk_index_magic = b"CZIX"      # 4-byte magic for the chunk index section

# Pre-created Struct objects for frequently used format strings.
# Using Struct.unpack/pack avoids re-parsing the format string on every call.
_struct_B = struct.Struct("<B")   # unsigned byte
_struct_H = struct.Struct("<H")   # unsigned short
_struct_Q = struct.Struct("<Q")   # unsigned long long
_struct_f = struct.Struct("<f")   # float
_struct_4s = struct.Struct("<4s") # 4-byte string (magic)
_struct_2Q = struct.Struct("<2Q") # two unsigned long longs (chunk tail header)
_struct_4Q = struct.Struct("<4Q") # four unsigned long longs (chunk index entry)

# Dynamically import the package version produced by setuptools_scm.
try:
	from ._version import version as _version
except Exception:
	_version = "0.0.0"

# Try to import accelerated Cython functions. If unavailable, fall back to
# pure-Python implementations below.
try:
	from .cz_accel import load_bcz_block as _c_load_bcz_block, compress_block as _c_compress_block
	from .cz_accel import unpack_records as _c_unpack_records
	from .cz_accel import c_read as _c_read
	from .cz_accel import c_readline as _c_readline
	from .cz_accel import c_pos2id as _c_pos2id
	from .cz_accel import c_read_1record as _c_read_1record
	from .cz_accel import c_seek_and_read_1record as _c_seek_and_read_1record
	from .cz_accel import c_query_regions as _c_query_regions
	from .cz_accel import c_write_chunk_tail as _c_write_chunk_tail
	from .cz_accel import c_pack_records as _c_pack_records
	from .cz_accel import c_pack_records_fast as _c_pack_records_fast
	from .cz_accel import c_fetch_chunk as _c_fetch_chunk
	from .cz_accel import c_get_records_by_ids as _c_get_records_by_ids
	from .cz_accel import c_block_first_values as _c_block_first_values
	from .cz_accel import c_extract_c_positions as _c_extract_c_positions
	from .cz_accel import c_write_c_records as _c_write_c_records
	# logger.debug("Cython accelerated functions loaded successfully")
except Exception:
	logger.warning("Cython accelerated functions not available, falling back to pure-Python implementations")
	_c_load_bcz_block = None
	_c_compress_block = None
	_c_unpack_records = None
	_c_read = None
	_c_readline = None
	_c_pos2id = None
	_c_read_1record = None
	_c_seek_and_read_1record = None
	_c_query_regions = None
	_c_write_chunk_tail = None
	_c_pack_records = None
	_c_pack_records_fast = None
	_c_fetch_chunk = None
	_c_get_records_by_ids = None
	_c_block_first_values = None
	_c_extract_c_positions = None
	_c_write_c_records = None

def dtype_func(fmt_char):
	"""Return a type-casting function for a given struct format character.

	For float formats ('f', 'd') returns ``float``.
	For string formats ('s', 'c') returns a function converting str -> bytes.
	For integer formats returns a function that clamps values to the maximum
	representable value of the format (e.g. 255 for 'B', 65535 for 'H').
	"""
	if fmt_char in ['f', 'd']:
		return float

	def str2byte(x):
		return bytes(str(x), 'utf-8')

	if fmt_char in ['s', 'c']:
		return str2byte
	# Integer types: compute the maximum value this format can hold.
	size = struct.calcsize(fmt_char)
	max_val = 2 ** (size * 8) - 1  # e.g. 'B' -> 255, 'H' -> 65535, 'I' -> 2^32-1

	def int_func(i):
		i = int(i)
		if i <= max_val:
			return i
		else:
			return max_val  # clamp to max

	return int_func

# ==========================================================
def get_dtfuncs(formats, tobytes=True):
	"""Build a list of type-casting functions matching the given struct formats.

	Parameters
	----------
	formats : list of str
		Struct format strings (e.g. ['H', '3s', 'I']).
	tobytes : bool
		If True (default), string formats 's'/'c' produce bytes;
		if False they produce str.

	Returns
	-------
	list of callables, one per format.
	"""
	dtype_map = {f: dtype_func(f[-1]) for f in formats}
	if not tobytes:
		dtype_map['s'] = str
		dtype_map['c'] = str
	return [dtype_map[t] for t in formats]


# ==========================================================
def make_virtual_offset(block_start_offset, within_block_offset):
	"""Encode a (block_start, within_block) pair into a 64-bit virtual offset.

	The upper 48 bits store the physical file offset of the compressed
	block; the lower 16 bits store the byte offset within the *decompressed*
	block data.  This scheme is the same as BGZF virtual offsets.
	"""
	if within_block_offset < 0 or within_block_offset >= _BLOCK_MAX_LEN:
		raise ValueError(
			"Require 0 <= within_block_offset < 2**16, got %i" % within_block_offset
		)
	if block_start_offset < 0 or block_start_offset >= 2 ** 48:
		raise ValueError(
			"Require 0 <= block_start_offset < 2**48, got %i" % block_start_offset
		)
	return (block_start_offset << 16) | within_block_offset


# ==========================================================
def split_virtual_offset(virtual_offset):
	"""Decode a 64-bit virtual offset into (block_start_offset, within_block_offset)."""
	start = virtual_offset >> 16
	return start, virtual_offset ^ (start << 16)


# ==========================================================
def SummaryBczBlocks(handle):
	"""Iterate over all blocks in the file, yielding per-block summary info.

	Yields
	------
	start_offset : int
		Physical file position of the block.
	block_length : int
		Total compressed block size (bytes on disk).
	data_start : int
		Cumulative decompressed data offset (byte position
		within the chunk's decompressed stream).
	data_len : int
		Decompressed data length for this block.
	"""
	if isinstance(handle, Reader):
		raise TypeError("Function BczBlocks expects a binary handle")
	data_start = 0
	while True:
		start_offset = handle.tell()
		try:
			block_length, data_len = _load_bcz_block(handle)
		except StopIteration:
			break
		yield start_offset, block_length, data_start, data_len
		data_start += data_len

# ==========================================================
def _py_load_bcz_block(handle, decompress=False):
	"""Pure-Python fallback for reading a single BCZ block from *handle*.

	Block layout on disk:
	  [magic 2B] [block_size 2B] [deflate_data (block_size-6)B] [data_len 2B]

	Parameters
	----------
	decompress : bool
		If True, decompress the deflate payload and return the raw bytes;
		otherwise, just skip ahead and return the uncompressed length.

	Returns
	-------
	(block_size, data_or_data_len)

	Raises
	------
	StopIteration  when EOF or a non-block magic is encountered.
	"""
	magic = handle.read(2)
	if not magic or magic != _block_magic:  # next chunk or EOF
		raise StopIteration
	block_size = _struct_H.unpack(handle.read(2))[0]
	# equal to struct.unpack("<H", handle.read(2))[0]
	if decompress:
		deflate_size = block_size - 6
		d = zlib.decompressobj(-15)
		data = d.decompress(handle.read(deflate_size)) + d.flush()
		data_len = _struct_H.unpack(handle.read(2))[0]
		return block_size, data
	else:
		handle.seek(block_size - 6, 1)
		data_len = _struct_H.unpack(handle.read(2))[0]
		return block_size, data_len

# Use accelerated loader if available, otherwise use Python fallback.
_load_bcz_block = _c_load_bcz_block if _c_load_bcz_block is not None else _py_load_bcz_block

# ==========================================================
def open1(infile):
	"""Open a text file or gzip file for line-by-line reading.

	If *infile* is already a file-like object (has ``readline``), return
	it as-is.  Otherwise open the path, auto-detecting gzip by the
	``.gz`` extension.
	"""
	if hasattr(infile, 'readline'):
		return infile
	if infile.endswith('.gz'):
		f = gzip.open(infile, 'rb')
	else:
		f = open(infile, 'r')
	return f
# ==========================================================
def _gz_input_parser(infile, formats, sep='\t', usecols=[1,4,5], dim_cols=[0],
					 chunksize=5000):
	"""Parse a gzip-compressed text file into (DataFrame, dims) chunks.

	Similar to ``_text_input_parser`` but handles ``.gz`` byte decoding.
	Each yielded DataFrame contains at most *chunksize* rows sharing
	the same dimension values (e.g., same chromosome).
	"""
	f = open1(infile)
	data, i, prev_dims = [], 0, None
	dtfuncs = get_dtfuncs(formats)
	line = f.readline()
	while line:
		line = line.decode('utf-8')
		values = line.rstrip('\n').split(sep)
		dims = [values[i] for i in dim_cols]
		if dims != prev_dims:  # a new dim (chrom), for example, chr1 -> chr2
			if len(data) > 0:  # write rest data of chr1
				yield pd.DataFrame(data, columns=usecols), prev_dims
				data, i = [], 0
			prev_dims = dims
		if i >= chunksize:  # dims are the same, but reach chunksize
			yield pd.DataFrame(data, columns=usecols), prev_dims
			data, i = [], 0
		data.append([func(v) for v, func in zip([values[i] for i in usecols],
												dtfuncs)])
		line = f.readline()
		i += 1
	f.close()
	if len(data) > 0:
		yield pd.DataFrame(data, columns=usecols), prev_dims

# ==========================================================
def _text_input_parser(infile,formats,sep='\t',usecols=[1,4,5],dim_cols=[0],
					   chunksize=5000):
	"""
	Parse text input file (txt, csv, tsv and so on.) into chunks, every chunk has
	the same dim_cols (for example, chromosomes).

	Parameters
	----------
	infile : path
		input file path or sys.stdin.buffer
	formats : list
		list of formats to pack into .cz file.
	sep :str
	usecols :list
		columns index in input file to be packed into .cz file.
	dim_cols : list
		dimensions column index, default is [0]
	chunksize :int

	Returns
	-------
	a generator, each element is a tuple, first element of tuple is a dataframe
	 (header names is usecols),second element is the dims,
	in each dataframe, the dim are the same.

	"""
	# cdef int i, N
	f = open1(infile)
	data, i, prev_dims = [], 0, None
	dtfuncs = get_dtfuncs(formats)
	line = f.readline()
	while line:
		values = line.rstrip('\n').split(sep)
		dims = [values[i] for i in dim_cols]
		if dims != prev_dims:  # a new dim (chrom), for example, chr1 -> chr2
			if len(data) > 0:  # write rest data of chr1
				yield pd.DataFrame(data, columns=usecols), prev_dims
				data, i = [], 0
			prev_dims = dims
		if i >= chunksize:  # dims are the same, but reach chunksize
			yield pd.DataFrame(data, columns=usecols), prev_dims
			data, i = [], 0
		data.append([func(values[i]) for i, func in zip(usecols, dtfuncs)])
		line = f.readline()
		i += 1
	f.close()
	if len(data) > 0:
		yield pd.DataFrame(data, columns=usecols), prev_dims


# ==========================================================
def _input_parser(infile, formats, sep='\t', usecols=[1, 4, 5], dim_cols=[0],
				  chunksize=5000):
	"""Dispatch to gz or text parser based on whether *infile* is gzip-compressed."""
	if hasattr(infile, 'readline'):
		yield from _text_input_parser(infile, formats, sep, usecols, dim_cols, chunksize)
	elif infile.endswith('.gz'):
		yield from _gz_input_parser(infile, formats, sep, usecols, dim_cols, chunksize)
	else:
		yield from _text_input_parser(infile, formats, sep, usecols, dim_cols, chunksize)


# ==========================================================
# ---------------------------------------------------------------------------
# Filter predicates used by category_ssi to build subset indexes (SSI).
# Each function inspects a raw record tuple and returns True/False.
# ---------------------------------------------------------------------------
def _isCG(record):
	"""Return True if the 3-byte context starts with 'CG' (CpG site)."""
	return record[2][:2] == b'CG'


def _isForwardCG(record):
	"""Return True if the site is a forward-strand (+) CpG."""
	return record[2][:2] == b'CG' and record[1] == b'+'

# ==========================================================
def _isCH(record):
	"""Return True if the site is NOT CpG (i.e., CH context)."""
	return not _isCG(record)


# ==========================================================
class RemoteFile:
	"""File-like object backed by HTTP Range requests with read-ahead caching.

	Implements read(), seek(), tell(), close() so it can be used as a drop-in
	replacement for a local binary file handle in Reader.

	Parameters
	----------
	url : str
		HTTP or HTTPS URL to the file.
	cache_size : int
		Read-ahead cache size in bytes (default 2MB). Each HTTP request
		fetches at least this many bytes to reduce request count.
	"""

	def __init__(self, url, cache_size=2 * 1024 * 1024):
		import urllib.request
		self._urllib = urllib.request
		self.url = url
		self._pos = 0
		self._cache_size = cache_size
		self._cache_start = -1
		self._cache = b""
		self._closed = False
		# HEAD request to get file size
		req = urllib.request.Request(url, method='HEAD')
		with urllib.request.urlopen(req) as resp:
			cl = resp.headers.get('Content-Length')
			if cl is None:
				raise ValueError(
					"Server did not return Content-Length for %s" % url)
			self._size = int(cl)

	def read(self, size=-1):
		if size == 0:
			return b""
		if size < 0:
			size = self._size - self._pos
		if size <= 0:
			return b""
		# Check cache hit
		if self._cache_start >= 0:
			cache_end = self._cache_start + len(self._cache)
			if self._cache_start <= self._pos and self._pos + size <= cache_end:
				offset = self._pos - self._cache_start
				self._pos += size
				return self._cache[offset:offset + size]
		# Cache miss: fetch from remote with read-ahead
		fetch_size = max(size, self._cache_size)
		start = self._pos
		end = min(start + fetch_size - 1, self._size - 1)
		# Near end of file: extend backward to fill cache (so chunk index
		# + index_offset + EOF are fetched in a single request)
		if end - start + 1 < fetch_size:
			start = max(0, end - fetch_size + 1)
		req = self._urllib.Request(self.url)
		req.add_header('Range', 'bytes=%d-%d' % (start, end))
		with self._urllib.urlopen(req) as resp:
			self._cache = resp.read()
			# If server does not support Range (returns 200 not 206),
			# it sent the full file from offset 0.
			cr = resp.headers.get('Content-Range')
			if cr and cr.startswith('bytes '):
				self._cache_start = start
			else:
				self._cache_start = 0
		local_off = self._pos - self._cache_start
		data = self._cache[local_off:local_off + size]
		self._pos += len(data)
		return data

	def seek(self, offset, whence=0):
		if whence == 0:
			self._pos = offset
		elif whence == 1:
			self._pos += offset
		elif whence == 2:
			self._pos = self._size + offset
		return self._pos

	def tell(self):
		return self._pos

	def close(self):
		self._closed = True
		self._cache = b""

	@property
	def size(self):
		return self._size


# ==========================================================
class Reader:
	"""Reader for .cz (ChunkZIP) files.

	Supports both local and remote (HTTP) files.  On open, the header
	and chunk index are parsed so that any chunk can be accessed by its
	dimension key (e.g., chromosome name) in O(1) time.

	Key concepts:
	  - *Chunk*: a group of blocks sharing the same dimension values.
	  - *Block*: a single DEFLATE-compressed payload (max 64 KB uncompressed).
	  - *Virtual offset*: a 64-bit value encoding both the block's file
	    position (upper 48 bits) and the byte offset within the
	    decompressed block (lower 16 bits).
	"""

	def __init__(self, input, mode="rb", fileobj=None, max_cache=100):
		r"""Initialize the class for reading a CZ (cytozip) file.
		"""
		if max_cache < 1:
			raise ValueError("Use max_cache with a minimum of 1")
		if input and fileobj:
			raise ValueError("Supply either filename or fileobj, not both")
		# Want to reject output modes like w, a, x, +
		if mode.lower() not in ("r", "tr", "rt", "rb", "br"):
			raise ValueError(
				"Must use a read mode like 'r' (default), 'rt', or 'rb' for binary"
			)
		# If an open file was passed, make sure it was opened in binary mode.
		if fileobj:
			if fileobj.read(0) != b"":
				raise ValueError("fileobj not opened in binary mode")
			handle = fileobj
		else:
			if isinstance(input, str) and input.startswith(('http://', 'https://')):
				handle = RemoteFile(input)
			else:
				input = os.path.abspath(os.path.expanduser(input))
				handle = _open(input, "rb")
		self._handle = handle
		self._is_remote = isinstance(handle, RemoteFile)
		self.input = input
		self.max_cache = max_cache
		self._block_start_offset = None
		self._block_raw_length = None
		# Cache chunk tail metadata keyed by chunk start_offset.
		# Avoids re-reading the tail on repeated _load_chunk calls
		# for the same chunk (especially beneficial for remote files).
		self._chunk_tail_cache = {}
		self.read_header()

	def read_header(self):
		"""Parse the .cz file header and populate ``self.header``.

		Header binary layout (all little-endian)::

		  magic        : 4 bytes ('CZIP')
		  version      : 4 bytes (float)
		  total_size   : 8 bytes (uint64) - file size written at close time
		  message_len  : 2 bytes (uint16)
		  message      : message_len bytes (utf-8 string)
		  ncols        : 1 byte (uint8)  - number of data columns
		  For each column:
		    fmt_len    : 1 byte -> format string (e.g. 'H', '3s')
		  For each column:
		    name_len   : 1 byte -> column name
		  ndims        : 1 byte - number of dimension names
		  For each dim:
		    dim_len    : 1 byte -> dimension name

		After parsing, ``self.header['header_size']`` points to the byte
		offset where the first chunk begins.
		"""
		self.header = {}
		f = self._handle
		magic = struct.unpack("<4s", f.read(4))[0]
		if magic != _cz_magic:
			raise ValueError("Not a right format?")
		self.header['magic'] = magic
		self.header['version'] = struct.unpack("<f", f.read(4))[0]
		total_size = struct.unpack("<Q", f.read(8))[0]
		if total_size == 0:
			raise ValueError("File not completed !")
		self.header['total_size'] = total_size
		n = struct.unpack("<H", f.read(2))[0]  # message_len
		self.header['message'] = struct.unpack(f"<{n}s", f.read(n))[0].decode()
		format_len = struct.unpack("<B", f.read(1))[0]
		formats = []
		for i in range(format_len):
			n = struct.unpack("<B", f.read(1))[0]
			fmt = struct.unpack(f"<{n}s", f.read(n))[0].decode()
			formats.append(fmt)
		self.header['formats'] = formats
		columns = []
		for i in range(format_len):
			n = struct.unpack("<B", f.read(1))[0]
			name = struct.unpack(f"<{n}s", f.read(n))[0].decode()
			columns.append(name)
		self.header['columns'] = columns
		assert len(formats) == len(columns)
		dimensions = []
		n_dims = struct.unpack("<B", f.read(1))[0]
		for i in range(n_dims):
			n = struct.unpack("<B", f.read(1))[0]
			dim = struct.unpack(f"<{n}s", f.read(n))[0].decode()
			dimensions.append(dim)
		self.header['dimensions'] = dimensions
		self.header['header_size'] = f.tell()  # end of header, begin of 1st chunk
		self.fmts = ''.join(formats)
		self._unit_size = struct.calcsize(self.fmts)
		self._struct_obj = struct.Struct(f"<{self.fmts}")
		# Number of consecutive blocks needed before record boundaries
		# realign with block boundaries.  Equals _unit_size / gcd, which
		# is the same as lcm(_unit_size, _BLOCK_MAX_LEN) / _BLOCK_MAX_LEN.
		# E.g. unit_size=6, _BLOCK_MAX_LEN=65535, gcd=3 → _unit_nblock=2, 每 2 个 block 为一个对齐周期
		# math.gcd(_unit_size, _BLOCK_MAX_LEN)：The greatest common divisor of the two
		self._unit_nblock = int(self._unit_size / (math.gcd(self._unit_size, _BLOCK_MAX_LEN)))
		# _unit_nblock: 每隔多少个 block，record 边界会重新与 block 边界对齐
		if getattr(self, '_is_remote', False):
			self._chunk_rows = None
			self._summary_from_chunk_index()
		else:
			self._scan_chunks()

	def _scan_chunks(self):
		"""Scan all chunks to build dim2chunk_start mapping (no pandas)."""
		r = self._load_chunk(start_offset=self.header['header_size'], jump=True)
		rows = []
		dim2cs = {}
		while r:
			nrow = int(self._chunk_data_len / self._unit_size)
			if self._chunk_dims in dim2cs:
				raise ValueError("Duplicated chunk dimensions detected,"
								 "Would cause conflict for querying, please check.")
			dim2cs[self._chunk_dims] = self._chunk_start_offset
			rows.append([self._chunk_start_offset, self._chunk_size,
						 self._chunk_dims, self._chunk_end_offset,
						 self._chunk_nblocks, nrow])
			r = self._load_chunk(jump=True)
		self.dim2chunk_start = dim2cs
		self._chunk_rows = rows

	@property
	def chunk_info(self):
		"""Lazily build the chunk_info DataFrame on first access."""
		try:
			return self._chunk_info
		except AttributeError:
			pass
		import pandas as _pd
		header = ['chunk_start_offset', 'chunk_size', 'chunk_dims',
				  'chunk_tail_offset', 'chunk_nblocks', 'chunk_nrows']
		if self._chunk_rows is None:
			self._scan_chunks()
		chunk_info = _pd.DataFrame(self._chunk_rows, columns=header)
		for i, dimension in enumerate(self.header['dimensions']):
			chunk_info.insert(i, dimension, chunk_info.chunk_dims.apply(
				lambda x: x[i]
			))
		chunk_info.set_index('chunk_dims', inplace=True)
		self._chunk_info = chunk_info
		return chunk_info

	@chunk_info.setter
	def chunk_info(self, value):
		self._chunk_info = value

	def print_header(self):
		for k in self.header:
			print(k, " : ", self.header[k])

	def _load_chunk(self, start_offset=None, jump=True):
		"""Load metadata for the chunk at *start_offset*.

		This reads the chunk magic ('MC'), its total compressed size, then
		jumps to the chunk *tail* to read:
		  - total uncompressed data length
		  - number of blocks
		  - per-block first-record virtual offsets (only if ``jump=False``)
		  - dimension values for this chunk

		Parameters
		----------
		start_offset : int or None
			Physical file offset of the chunk.  If None, continues from
			the end of the previously loaded chunk.
		jump : bool
			If True, skip reading per-block virtual offsets (faster for
			scanning/summary).  If False, also load the full block offset
			array needed for random access / querying.

		Returns
		-------
		bool – True if a chunk was successfully loaded, False on EOF or
		       when the magic doesn't match.
		"""
		if start_offset is None:  # continue from end of previous chunk
			start_offset = self._chunk_end_offset
		if start_offset >= self.header['total_size']:
			return False
		# Check chunk tail cache first to avoid re-reading from disk/network.
		cached = self._chunk_tail_cache.get(start_offset)
		if cached is not None:
			self._chunk_start_offset = start_offset
			self._chunk_size = cached['chunk_size']
			self._chunk_data_len = cached['data_len']
			self._chunk_nblocks = cached['nblocks']
			self._chunk_dims = cached['dims']
			self._chunk_end_offset = cached['end_offset']
			if jump:
				self._chunk_block_1st_record_virtual_offsets = []
			else:
				self._chunk_block_1st_record_virtual_offsets = list(cached['block_vos'])
			return True
		self._handle.seek(start_offset)
		self._chunk_start_offset = start_offset  # real offset on disk.
		magic = self._handle.read(2)
		if magic != _chunk_magic:
			return False
		self._chunk_size = _struct_Q.unpack(self._handle.read(8))[0]
		# load chunk tail, jump all blocks
		self._handle.seek(self._chunk_start_offset + self._chunk_size)
		self._chunk_data_len, self._chunk_nblocks = _struct_2Q.unpack(
			self._handle.read(16))
		self._chunk_block_1st_record_virtual_offsets = []
		if jump:  # no need to load _chunk_block_1st_record_virtual_offsets
			self._handle.seek(self._chunk_nblocks * 8, 1)
		# _chunk_tail_offset = end position of this chunk = start position of next chunk.
		else:
			# Bulk-read all block virtual offsets in one read + one unpack.
			n = self._chunk_nblocks
			raw = self._handle.read(n * 8)
			self._chunk_block_1st_record_virtual_offsets = list(
				struct.unpack(f"<{n}Q", raw))
		dims = []
		for t in self.header['dimensions']:
			n = _struct_B.unpack(self._handle.read(1))[0]
			dim = struct.unpack(f"<{n}s", self._handle.read(n))[0].decode()
			dims.append(dim)
		self._chunk_dims = tuple(dims)
		self._chunk_end_offset = self._handle.tell()
		# Populate cache if block_vos were read (jump=False).
		# jump=True callers (summary/scan) don't need caching.
		if not jump:
			self._chunk_tail_cache[start_offset] = {
				'chunk_size': self._chunk_size,
				'data_len': self._chunk_data_len,
				'nblocks': self._chunk_nblocks,
				'dims': self._chunk_dims,
				'end_offset': self._chunk_end_offset,
				'block_vos': list(self._chunk_block_1st_record_virtual_offsets),
			}
		return True

	def get_chunks(self):
		r = self._load_chunk(self.header['header_size'], jump=False)
		while r:
			yield [self._chunk_start_offset, self._chunk_size,
				   self._chunk_dims, self._chunk_data_len, self._chunk_end_offset,
				   self._chunk_nblocks, self._chunk_block_1st_record_virtual_offsets
				   ]
			r = self._load_chunk(jump=False)

	def summary_chunks(self, printout=True):
		chunk_info = self.chunk_info  # triggers lazy build if needed
		if printout:
			try:
				sys.stdout.write('\t'.join(chunk_info.columns.tolist()) + '\n')
				for i, row in chunk_info.iterrows():
					sys.stdout.write('\t'.join([str(v) for v in row.tolist()]) + '\n')
				sys.stdout.flush()
			except BrokenPipeError:
				pass
			finally:
				self.close()
		else:
			return chunk_info

	def read_chunk_index(self):
		"""Read chunk index from end of file for O(1) chunk lookup.

		Returns dict mapping dim tuple -> {start, size, data_len, nblocks}
		"""
		f = self._handle
		cur = f.tell()
		# Read index_offset from 36 bytes before end (8B offset + 28B EOF)
		f.seek(0, 2) # whence=2 表示从文件末尾偏移。offset=0 即移到文件最末尾
		file_size = f.tell()
		if file_size < 36:
			f.seek(cur)
			return None
		f.seek(file_size - 28 - 8)
		index_offset = _struct_Q.unpack(f.read(8))[0]
		if index_offset == 0 or index_offset >= file_size:
			f.seek(cur)
			return None
		f.seek(index_offset)
		magic = f.read(4)
		if magic != _chunk_index_magic:
			f.seek(cur)
			return None
		n_chunks = _struct_Q.unpack(f.read(8))[0]
		index = {}
		n_dims = len(self.header['dimensions'])
		for _ in range(n_chunks):
			dims = []
			for _d in range(n_dims):
				n = _struct_B.unpack(f.read(1))[0]
				val = struct.unpack(f"<{n}s", f.read(n))[0].decode()
				dims.append(val)
			start, size, data_len, nblocks = _struct_4Q.unpack(f.read(32))
			index[tuple(dims)] = {
				'start': start, 'size': size, 'data_len': data_len,
				'nblocks': nblocks,
			}
		f.seek(cur)
		return index
	
	def _summary_from_chunk_index(self):
		"""Build dim2chunk_start from chunk index (2-3 HTTP requests for remote).

		Falls back to sequential scanning if no chunk index is found.
		"""
		idx = self.read_chunk_index()
		if idx is None:
			logger.warning("No chunk index found in .cz file, "
							"falling back to sequential scan (slow for remote files)")
			self._scan_chunks()
			return None
		rows = []
		dim2cs = {}
		for dims, info in idx.items():
			nrow = info['data_len'] // self._unit_size
			tail_header_size = (16 + info['nblocks'] * 8
								+ sum(len(d.encode('utf-8')) + 1 for d in dims))
			tail_offset = info['start'] + info['size'] + tail_header_size
			if dims in dim2cs:
				raise ValueError("Duplicated chunk dimensions detected")
			dim2cs[dims] = info['start']
			rows.append([info['start'], info['size'], dims,
						tail_offset, info['nblocks'], nrow])
		self.dim2chunk_start = dim2cs
		self._chunk_rows = rows

	def summary_blocks(self, printout=True):
		r = self._load_chunk(self.header['header_size'], jump=True)
		header = ['chunk_dims'] + ['block_start_offset', 'block_size',
								   'block_data_start', 'block_data_len']
		if printout:
			sys.stdout.write('\t'.join(header) + '\n')
		else:
			rows = []
		try:
			while r:
				self._handle.seek(self._chunk_start_offset + 10)
				chunk_info = [self._chunk_dims]
				for block in SummaryBczBlocks(self._handle):
					block = chunk_info + list(block)
					if printout:
						sys.stdout.write('\t'.join([str(v) for v in block]) + '\n')
					else:
						rows.append(block)
				r = self._load_chunk(jump=True)
			if printout:
				sys.stdout.flush()
		except BrokenPipeError:
			pass
		finally:
			self.close()
		if not printout:
			df = pd.DataFrame(rows, columns=header)
			return df

	def chunk2df(self, dims, reformat=False, chunksize=None):
		"""Read an entire chunk into a pandas DataFrame.

		Decompresses all blocks for the chunk identified by *dims*,
		unpacks the binary records, and returns a DataFrame with columns
		matching the file header.

		Parameters
		----------
		dims : tuple
			dimension key identifying the chunk (e.g., ('chr1',)).
		reformat : bool
			If True, decode bytes-type columns (s/c formats) into strings.
		chunksize : int or None
			If set, yield DataFrames of at most *chunksize* blocks instead
			of returning a single DataFrame.
		"""
		r = self._load_chunk(self.dim2chunk_start[dims], jump=False)
		# Fast path: use Cython chunk fetcher to read all blocks at once
		if _c_fetch_chunk is not None and chunksize is None:
			chunk_bytes = _c_fetch_chunk(self._handle, self._chunk_start_offset + 10,
								self._chunk_block_1st_record_virtual_offsets,
								self.fmts, self._unit_size)
			if chunk_bytes:
				if _c_unpack_records is not None:
					rows = _c_unpack_records(chunk_bytes, self.fmts)
				else:
					rows = list(self._struct_obj.iter_unpack(chunk_bytes))
				df = pd.DataFrame(rows, columns=self.header['columns'])
				if not reformat:
					return df
				for col, fmt in zip(df.columns.tolist(), self.header['formats']):
					if fmt[-1] in ['c', 's']:
						df[col] = df[col].apply(lambda x: str(x, 'utf-8'))
				return df
		# Fallback: block-by-block with record alignment
		self._cached_data = b''
		self._load_block(start_offset=self._chunk_start_offset + 10)  #
		rows = []
		i = 0
		unpack_fn = _c_unpack_records if _c_unpack_records is not None else None
		while self._block_raw_length > 0:
			# deal with such case: unit_size is 10, but data(_buffer) size is 18,
			self._cached_data += self._buffer
			end_index = len(self._cached_data) - (len(self._cached_data) % self._unit_size)
			chunk_bytes = self._cached_data[:end_index]
			if unpack_fn is not None:
				rows.extend(unpack_fn(chunk_bytes, self.fmts))
			else:
				rows.extend(self._struct_obj.iter_unpack(chunk_bytes))
			if not chunksize is None and i >= chunksize:
				df = pd.DataFrame(rows, columns=self.header['columns'])
				rows = []
				i = 0
				yield df
			self._cached_data = self._cached_data[end_index:]
			self._load_block()
			i += 1
		if chunksize is None:
			df = pd.DataFrame(rows, columns=self.header['columns'])
			if not reformat:
				return df
			for col, fmt in zip(df.columns.tolist(), self.header['formats']):
				# print(col,fmt)
				if fmt[-1] in ['c', 's']:
					df[col] = df[col].apply(lambda x: str(x, 'utf-8'))
			return df
		else:
			if len(rows) > 0:
				df = pd.DataFrame(rows, columns=self.header['columns'])
				yield df

	def _load_block(self, start_offset=None):
		"""Load and decompress a single block into ``self._buffer``.

		If *start_offset* is None, the next sequential block is loaded.
		Otherwise, seek to the given physical file position first.
		After loading, ``self._buffer`` holds the decompressed data and
		``self._within_block_offset`` is reset to 0.
		"""
		if start_offset is None:
			# If the file is being read sequentially, then _handle.tell()
			# should be pointing at the start of the next block.
			# However, if seek has been used, we can't assume that.
			start_offset = self._block_start_offset + self._block_raw_length
		elif start_offset == self._block_start_offset:
			self._within_block_offset = 0
			return
		# Now load the block
		self._handle.seek(start_offset)
		self._block_start_offset = start_offset
		try:
			block_size, self._buffer = _load_bcz_block(self._handle, True)
		except StopIteration:  # EOF
			block_size = 0
			self._buffer = b""
		self._within_block_offset = 0
		self._block_raw_length = block_size

	def _byte2str(self, values):
		"""Convert a record tuple to a list of printable strings.

		Bytes-type fields (s/c formats) are decoded to UTF-8 strings;
		numeric fields are converted via ``str()``.
		"""
		return [str(v, 'utf-8') if f[-1] in ['s', 'c'] else str(v)
				for v, f in zip(values, self.header['formats'])]

	def _byte2real(self, values):
		"""Convert a record tuple, decoding bytes fields to strings but
		leaving numeric fields as their native Python types.
		"""
		return [str(v, 'utf-8') if f[-1] in ['s', 'c'] else v
				for v, f in zip(values, self.header['formats'])]

	def _empty_generator(self):
		while True:
			yield []

	def view(self, show_dim=None, header=True, dimension=None,
			 reference=None):
		"""
		View .cz file.

		Parameters
		----------
		show_dim : str
			index of dims given to writer.write_chunk, separated by comma,
			default is None, dims[show_dim] will be shown in each row (such as sampleID
			and chrom)
		header : bool
			whether to print the header.
		dimension: None, bool, list or file path
			If None (default): use the default chunk order in .cz file; \
			If list: use this dimension (dims) as order and print records.\
			If path: use the first len(dimension) columns as dim order, there should be\
				no header in file path and use \t as separator.\
			If dimension is dictionary,such as -D "{'sampleID':'cell1','chrom':'chr1'}",\
			will filter chunk using sampleID=cell1 and chrom=chr1.

		Returns
		-------

		"""
		if isinstance(show_dim, int):
			show_dim = [show_dim]
		elif isinstance(show_dim, str):
			show_dim = [int(i) for i in show_dim.split(',')]

		if dimension is None or isinstance(dimension, dict):
			chunk_info = self.chunk_info.copy()
			if isinstance(dimension, dict):
				selected_dim = dimension.copy()
				for d, v in selected_dim.items():
					chunk_info = chunk_info.loc[chunk_info[d] == v]
			dimension = chunk_info.index.tolist()
		else:
			# equal to query chromosome if self.header['dimensions'][0]==chrom
			if isinstance(dimension, str):
				order_path = os.path.abspath(os.path.expanduser(dimension))
				if os.path.exists(order_path):
					dimension = pd.read_csv(order_path, sep='\t', header=None,
											usecols=show_dim)[show_dim].apply(lambda x: tuple(x.tolist()),
																			  axis=1).tolist()
				else:
					dimension = dimension.split(',')
			if isinstance(dimension, (list, tuple)) and isinstance(dimension[0], str):
				dimension = [tuple([o]) for o in dimension]
			if not isinstance(dimension, (list, tuple, np.ndarray)):  # dim is a list
				raise ValueError("input of dim_order is not corrected !")

		if not reference is None:
			reference = os.path.abspath(os.path.expanduser(reference))
			ref_reader = Reader(reference)
		if not show_dim is None:
			header_columns = [self.header['dimensions'][t] for t in show_dim]
			dim_header = "\t".join(header_columns) + '\t'
		else:
			dim_header = ''
		if header:  # show header
			if not reference is None:
				columns = ref_reader.header['columns'] + self.header['columns']
			else:
				columns = self.header['columns']
			line = "\t".join(columns)
			sys.stdout.write(dim_header + line + '\n')

		for d in dimension:  # dimension is a list of tuple ([(d1,d2),(d1,d2)])
			if d not in self.dim2chunk_start:
				continue
			if not show_dim is None:
				dim_stdout = "\t".join([d[t] for t in show_dim]) + '\t'
			else:
				dim_stdout = ''
			records = self.fetch(d)
			if not reference is None:
				ref_records = ref_reader.fetch(d)
			else:
				ref_records = self._empty_generator()
			try:
				for record, ref_record in zip(records, ref_records):
					line = '\t'.join([str(i) for i in ref_record + record])
					try:
						sys.stdout.write(dim_stdout + line + '\n')
					except:
						sys.stdout.close()
						if not reference is None:
							ref_reader.close()
						self.close()
						return
			except:
				raise ValueError(f"reference {reference} not matched.")
		sys.stdout.close()
		self.close()
		if not reference is None:
			ref_reader.close()

	@staticmethod
	def regions_ssi_worker(input, output, dim, df1, formats, columns, dimensions,
						   chunksize):
		print(dim)
		reader = Reader(input)
		positions = df1.loc[:, ['start', 'end']].values.tolist()
		records = reader.pos2id(dim, positions, col_to_query=0)

		writer = Writer(output, formats=formats,
						columns=columns, dimensions=dimensions,
						message=os.path.basename(input))
		data_parts, i = [], 0
		dtfuncs = get_dtfuncs(writer.formats)

		for record, name in zip(records, df1.Name.tolist()):
			if record is None:
				continue
			id_start, id_end = record
			# print(id_start,id_end,name)
			data_parts.append(struct.pack(f"<{writer.fmts}",
							*[func(v) for v, func in zip([id_start, id_end, name],
														 dtfuncs)]))
			i += 1
			if (i % chunksize) == 0:
				writer.write_chunk(b''.join(data_parts), dim)
				data_parts = []
				i = 0
		if len(data_parts) > 0:
			writer.write_chunk(b''.join(data_parts), dim)
		writer.close()
		reader.close()

	def regions_ssi(self, output, formats=['I', 'I'],
					columns=['ID_start', 'ID_end'],
					dimensions=['chrom'], bed=None,
					chunksize=2000, n_jobs=4):
		n_dim = len(dimensions)
		df = pd.read_csv(bed, sep='\t', header=None, usecols=list(range(n_dim + 3)),
						 names=['chrom', 'start', 'end', 'Name'])
		max_name_len = df.Name.apply(lambda x: len(x)).max()
		formats = formats + [f'{max_name_len}s']
		columns = columns + ['Name']
		dimensions = dimensions
		pool = multiprocessing.Pool(n_jobs)
		jobs = []
		outdir = output + '.tmp'
		if not os.path.exists(outdir):
			os.mkdir(outdir)
		for chrom, df1 in df.groupby('chrom'):
			dim = tuple([chrom])
			if dim not in self.dim2chunk_start:
				continue
			outfile = os.path.join(outdir, chrom + '.cz')
			job = pool.apply_async(self.regions_ssi_worker,
								   (self.input, outfile, dim, df1, formats, columns,
									dimensions, chunksize))
			jobs.append(job)
		for job in jobs:
			r = job.get()
		pool.close()
		pool.join()
		# merge
		writer = Writer(output=output, formats=formats,
						columns=columns, dimensions=dimensions,
						message=os.path.basename(bed))
		writer.catcz(input=f"{outdir}/*.cz")
		os.system(f"rm -rf {outdir}")

	def category_ssi(self, output=None, formats=['I'], columns=['ID'],
					 dimensions=['chrom'], match_func=_isForwardCG,
					 chunksize=2000):
		if output is None:
			output = self.input + '.' + match_func.__name__ + '.ssi'
		else:
			output = os.path.abspath(os.path.expanduser(output))
		writer = Writer(output, formats=formats, columns=columns,
						dimensions=dimensions, fileobj=None,
						message=os.path.basename(self.input))
		data_parts = []
		_ssi_pack = struct.Struct(f"<{writer.fmts}").pack
		for dim in self.dim2chunk_start:
			print(dim)
			for i, record in enumerate(self.__fetch__(dim)):
				if match_func(record):
					data_parts.append(_ssi_pack(i + 1))
				if (i % chunksize) == 0 and len(data_parts) > 0:
					writer.write_chunk(b''.join(data_parts), dim)
					data_parts = []
			if len(data_parts) > 0:
				writer.write_chunk(b''.join(data_parts), dim)
				data_parts = []
		writer.close()

	def get_ids_from_ssi(self, dim):
		if len(self.header['columns']) == 1:
			s, e = 0, 1  # only one columns, ID
			return np.array([record[0] for record in self.__fetch__(dim, s=s, e=e)])
		else:  # two columns: ID_start, ID_end
			s, e = 0, 2
			return np.array([record for record in self.__fetch__(dim, s=s, e=e)])

	def _getRecordsByIds(self, dim=None, IDs=None):
		"""

		Parameters
		----------
		dim : tuple
		reference : path
		IDs : np.array

		Returns
		-------

		"""
		self._load_chunk(self.dim2chunk_start[dim], jump=False)
		# Fast path: delegate entirely to Cython if available.
		if _c_get_records_by_ids is not None:
			for rec in _c_get_records_by_ids(
					self._handle, self._chunk_block_1st_record_virtual_offsets,
					self._unit_size, IDs):
				yield rec
			return
		# For each ID, compute the block index and within-block byte offset.
		# block_index = (ID-1) * unit_size // BLOCK_MAX_LEN
		# The virtual offset array stored in the chunk tail maps block index
		# to (block_file_position << 16) | first_record_within_block_offset.
		block_index = ((IDs - 1) * self._unit_size) // (_BLOCK_MAX_LEN)
		block_start_offsets = np.array([self._chunk_block_1st_record_virtual_offsets[
											idx] >> 16 for idx in block_index])
		within_block_offsets = ((IDs - 1) * self._unit_size) % (_BLOCK_MAX_LEN)
		prev_block_start_offset = None
		for block_start_offset, within_block_offset in zip(
			block_start_offsets, within_block_offsets):
			if block_start_offset != prev_block_start_offset:
				self._load_block(block_start_offset)
				prev_block_start_offset = block_start_offset
			self._within_block_offset = within_block_offset
			yield self.read(self._unit_size)

	def getRecordsByIds(self, dim=None, reference=None, IDs=None):
		if reference is None:
			for record in self._getRecordsByIds(dim, IDs):
				yield self._byte2real(self._struct_obj.unpack(record))
		else:
			ref_reader = Reader(os.path.abspath(os.path.expanduser(reference)))
			ref_records = ref_reader._getRecordsByIds(dim=dim, IDs=IDs)
			records = self._getRecordsByIds(dim=dim, IDs=IDs)
			for ref_record, record in zip(ref_records, records):
				yield ref_reader._byte2real(ref_reader._struct_obj.unpack(
					ref_record
				)) + self._byte2real(self._struct_obj.unpack(
					record
				))
			ref_reader.close()

	def _getRecordsByIdRegions(self, dim=None, IDs=None):
		"""Yield raw record bytes for ranges of primary IDs.

		Each element of *IDs* is a (id_start, id_end) pair (inclusive,
		1-based).  For each range, yields a list of raw record bytes.
		Blocks are cached to avoid redundant decompression when
		consecutive ranges fall in the same block.
		"""
		self._load_chunk(self.dim2chunk_start[dim], jump=False)
		prev_id_end = float("inf")  # track previous range end to detect block reuse
		for id_start, id_end in IDs:
			if id_start < prev_id_end:
				# New range starts before previous end; must re-seek
				prev_block_start_offset = None
			# Compute which block this ID falls in
			block_index = ((id_start - 1) * self._unit_size) // (_BLOCK_MAX_LEN)
			block_start_offset = self._chunk_block_1st_record_virtual_offsets[
									 block_index] >> 16
			if block_start_offset != prev_block_start_offset:
				self._load_block(block_start_offset)
				prev_block_start_offset = block_start_offset
			# Set the within-block offset to the start of the target record
			self._within_block_offset = ((id_start - 1) * self._unit_size
										 ) % _BLOCK_MAX_LEN
			# Read all records in this range (inclusive)
			yield [self.read(self._unit_size) for i in range(id_start, id_end + 1)]
			prev_id_end = id_end

	def getRecordsByIdRegions(self, dim=None, reference=None, IDs=None):
		"""
		Get .cz content for a given dim and IDs
		Parameters
		----------
		dim : tuple
		reference : path
		IDs : list or np.ndarray
			every element of IDs is a list or tuple with length=2, id_start and id_end

		Returns
		-------
		generator
		"""
		self._load_chunk(self.dim2chunk_start[dim], jump=False)
		if reference is None:
			for records in self._getRecordsByIdRegions(dim, IDs):
				yield np.array([self._byte2real(self._struct_obj.unpack(
					record)) for record in records])
		else:
			ref_reader = Reader(os.path.abspath(os.path.expanduser(reference)))
			ref_records = ref_reader._getRecordsByIdRegions(dim=dim, IDs=IDs)
			records = self._getRecordsByIdRegions(dim=dim, IDs=IDs)
			for ref_records, records in zip(ref_records, records):
				ref_record = np.array([ref_reader._byte2real(ref_reader._struct_obj.unpack(
					record)) for record in ref_records])
				record = np.array([self._byte2real(self._struct_obj.unpack(
					record)) for record in records])
				yield np.hstack((ref_record, record))
			ref_reader.close()

	def subset(self, dim, ssi=None, IDs=None, reference=None, printout=True):
		if isinstance(dim, str):
			dim = tuple([dim])
		if ssi is None and IDs is None:
			raise ValueError("Please provide either ssi or IDs")
		if not ssi is None:
			ssi_reader = Reader(ssi)
			IDs = ssi_reader.get_ids_from_ssi(dim)
			ssi_reader.close()
		if not reference is None:
			ref_reader = Reader(os.path.abspath(os.path.expanduser(reference)))
			ref_header = ref_reader.header['columns']
			ref_reader.close()
		else:
			ref_header = []
		header = self.header['dimensions'] + ref_header + self.header['columns']
		if len(IDs.shape) == 1:
			if printout:
				sys.stdout.write('\t'.join(header) + '\n')
				try:
					for record in self.getRecordsByIds(dim, reference, IDs):
						sys.stdout.write('\t'.join(list(dim) + list(map(str, record))) + '\n')
					# print(list(dim)+list(map(str,record)))
				except:
					sys.stdout.close()
					self.close()
					return
				sys.stdout.close()
			else:  # each element is one record
				yield from self.getRecordsByIds(dim, reference, IDs)
		else:
			if printout:
				sys.stdout.write('\t'.join(header) + '\n')
				try:
					for data in self.getRecordsByIdRegions(dim, reference, IDs):
						for row in data:
							sys.stdout.write('\t'.join(list(dim) + list(map(str, row))) + '\n')
				except:
					sys.stdout.close()
					self.close()
					return
				sys.stdout.close()
			else:  # each element is a data array (multiple records)
				yield from self.getRecordsByIdRegions(dim, reference, IDs)

	def __fetch_deprecated__(self, dims, s=None, e=None):
		"""
		Generator for a given dims

		Parameters
		----------
		dims : tuple
			element length of dims should match the dimensions in header['dimensions']

		Returns
		-------

		"""
		s = 0 if s is None else s  # 0-based
		e = len(self.header['columns']) if e is None else e  # 1-based
		r = self._load_chunk(self.dim2chunk_start[dims], jump=True)
		self._cached_data = b''
		# unit_nblock = int(self._unit_size / (math.gcd(self._unit_size, _BLOCK_MAX_LEN)))
		self._load_block(start_offset=self._chunk_start_offset + 10)  #
		while self._block_raw_length > 0:
			self._cached_data += self._buffer
			end_index = len(self._cached_data) - (len(self._cached_data) % self._unit_size)
			chunk_bytes = self._cached_data[:end_index]
			if _c_unpack_records is not None:
				for result in _c_unpack_records(chunk_bytes, self.fmts):
					yield result[s:e]
			else:
				for result in self._struct_obj.iter_unpack(chunk_bytes):
					yield result[s:e]  # a tuple
			# print(result)
			self._cached_data = self._cached_data[end_index:]
			self._load_block()

	def __fetch__(self, dim, s=None, e=None):
		"""
		Generator for a given dims

		Parameters
		----------
		dims : tuple
			element length of dims should match the dimensions in header['dimensions']

		Returns
		-------

		"""
		s = 0 if s is None else s  # 0-based
		e = len(self.header['columns']) if e is None else e  # 1-based
		r = self._load_chunk(self.dim2chunk_start[dim], jump=(_c_fetch_chunk is None))
		# Fast path: if Cython chunk fetcher is available, read and decompress
		# the entire chunk at once without decompressing blocks individually.
		if _c_fetch_chunk is not None:
			chunk_bytes = _c_fetch_chunk(self._handle, self._chunk_start_offset + 10,
								self._chunk_block_1st_record_virtual_offsets,
								self.fmts, self._unit_size)
			if chunk_bytes:
				if _c_unpack_records is not None:
					for result in _c_unpack_records(chunk_bytes, self.fmts):
						yield result[s:e]
				else:
					for result in self._struct_obj.iter_unpack(chunk_bytes):
						yield result[s:e]
			return
		# Fallback: block-by-block decompression with record alignment
		unit_nblock = self._unit_nblock
		self._load_block(start_offset=self._chunk_start_offset + 10)
		i, _cached_data = 1, self._buffer
		while self._block_raw_length > 0:
			self._load_block()
			if i < unit_nblock:
				_cached_data += self._buffer
				i += 1
			else:
				# unpack the whole buffer
				if _c_unpack_records is not None:
					for result in _c_unpack_records(_cached_data, self.fmts):
						yield result[s:e]
				else:
					for result in self._struct_obj.iter_unpack(_cached_data):
						yield result[s:e]
				i, _cached_data = 1, self._buffer
		if len(_cached_data) > 0:
			if _c_unpack_records is not None:
				for result in _c_unpack_records(_cached_data, self.fmts):
					yield result[s:e]
			else:
				for result in self._struct_obj.iter_unpack(_cached_data):
					yield result[s:e]

	def fetch_chunk_bytes(self, dim):
		"""
		Return all decompressed bytes for a given chunk as a single bytes object.
		This allows efficient numpy-based processing via np.frombuffer.

		Parameters
		----------
		dim : tuple
			Chunk dimension key.

		Returns
		-------
		bytes
		"""
		r = self._load_chunk(self.dim2chunk_start[dim], jump=False)
		if not r:
			return b""
		if _c_fetch_chunk is not None:
			raw = _c_fetch_chunk(
				self._handle, self._chunk_start_offset + 10,
				self._chunk_block_1st_record_virtual_offsets,
				self.fmts, self._unit_size
			)
		else:
			# Fallback: read block by block
			chunks = []
			self._load_block(start_offset=self._chunk_start_offset + 10)
			chunks.append(self._buffer)
			while self._block_raw_length > 0:
				self._load_block()
				chunks.append(self._buffer)
			raw = b"".join(chunks)
		return raw

	@classmethod
	def from_url(cls, url, cache_size=2 * 1024 * 1024):
		"""Open a remote .cz file via HTTP Range requests.

		Uses chunk index for O(1) chunk lookup, requiring only 2-3 HTTP
		requests to initialise (header + chunk index).

		Parameters
		----------
		url : str
			HTTP/HTTPS URL to the .cz file.
		cache_size : int
			Read-ahead cache size per HTTP request (default 2MB).

		Returns
		-------
		Reader
		"""
		remote = RemoteFile(url, cache_size=cache_size)
		reader = cls.__new__(cls)
		reader._handle = remote
		reader._is_remote = True
		reader.input = url
		reader.max_cache = 100
		reader._block_start_offset = None
		reader._block_raw_length = None
		reader.read_header()
		return reader

	def fetch(self, dim):
		for record in self.__fetch__(dim):
			yield self._byte2real(record)  # all columns of each row

	def batch_fetch(self, dims, chunksize=5000):
		i = 0
		data = []
		for row in self.fetch(dims):
			data.append(row)
			if i >= chunksize:
				yield data
				data, i = [], 0
			i += 1
		if len(data) > 0:
			yield data

	def fetchByStartID(self, dims, n=None):  # n is 1-based, n >=1
		if isinstance(dims, str):
			dims = tuple([dims])
		r = self._load_chunk(self.dim2chunk_start[dims], jump=False)
		if not n is None:  # seek to the position of the n rows
			block_index = ((n - 1) * self._unit_size) // (_BLOCK_MAX_LEN)
			first_record_vos = self._chunk_block_1st_record_virtual_offsets[block_index]
			within_block_offset = ((n - 1) * self._unit_size) % (_BLOCK_MAX_LEN)
			block_start_offset = (first_record_vos >> 16)
			virtual_start_offset = (block_start_offset << 16) | within_block_offset
			self.seek(virtual_start_offset)  # load_block is inside sek
		while True:
			# yield self._read_1record()
			yield self._struct_obj.unpack(self.read(self._unit_size))

	def _read_1record(self):
		# accelerated single-record read
		if _c_read_1record is not None:
			rec, self._block_raw_length, self._buffer, self._within_block_offset = _c_read_1record(
				self._handle, self._block_raw_length, self._buffer,
				self._within_block_offset, self.fmts, self._unit_size
			)
			return rec
		# fallback
		tmp = self._buffer[
			  self._within_block_offset:self._within_block_offset + self._unit_size]
		self._within_block_offset += self._unit_size
		return self._struct_obj.unpack(tmp)

	def _query_regions(self, regions, s, e):
		prev_dim = None
		block_1st_starts = None
		for dim, start, end in regions:
			if dim != prev_dim:
				start_block_index = 0
				prev_dim = dim
				r = self._load_chunk(self.dim2chunk_start[dim], jump=False)
				if _c_query_regions is None:
					# fallback: compute block first starts for Python path
					block_1st_starts = np.array([
						self._seek_and_read_1record(offset)[s]
						for offset in self._chunk_block_1st_record_virtual_offsets
					])
			# Fast path: Cython handles each (start, end) individually
			if _c_query_regions is not None:
				res = _c_query_regions(self._handle,
							self._chunk_block_1st_record_virtual_offsets,
							self.fmts, self._unit_size, [(start, end)], s, e, dim)
				for item in res:
					yield item
				continue
			start_block_index = max(0, int(np.searchsorted(block_1st_starts, start, side='right')) - 1)
			virtual_offset = self._chunk_block_1st_record_virtual_offsets[start_block_index]
			self.seek(virtual_offset)  # seek to the target block, self._buffer
			block_start_offset = self._block_start_offset
			record = self._struct_obj.unpack(self.read(self._unit_size))
			while record[s] < start:
				record = self._struct_obj.unpack(self.read(self._unit_size))
			# here we got the 1st record, return the id (row number) of 1st record.
			# size of each block is 65535 (2**16-1=_BLOCK_MAX_LEN)
			# primary_id = int((_BLOCK_MAX_LEN * block_index +
			#                   self._within_block_offset) / self._unit_size)
			if self._block_start_offset > block_start_offset:
				start_block_index += 1
			primary_id = int((_BLOCK_MAX_LEN * start_block_index +
							  self._within_block_offset) / self._unit_size)
			# primary_id is the current record (record[s] == start),
			# with_block_offset is the end of current record
			# 1-based, primary_id >=1, cause _within_block_offset >= self._unit_size
			yield "primary_id_&_dim:", primary_id, dim
			while record[e] <= end:
				yield dim, record
				record = self._struct_obj.unpack(self.read(self._unit_size))
		# start_block_index_tmp = start_block_index

	def pos2id(self, dim, positions, col_to_query=0):  # return IDs (primary_id)
		self._load_chunk(self.dim2chunk_start[dim], jump=False)
		# Try accelerated implementation if available
		if _c_pos2id is not None:
			# pass handle and relevant chunk metadata
			res = _c_pos2id(self._handle,
			                 self._chunk_block_1st_record_virtual_offsets,
			                 self.fmts, self._unit_size, positions, col_to_query)
			for r in res:
				yield r
			return
		# fallback python implementation
		block_1st_starts = [
			self._seek_and_read_1record(offset)[col_to_query]
			for offset in self._chunk_block_1st_record_virtual_offsets
		]
		start_block_index = 0
		for start, end in positions:
			while start_block_index < self._chunk_nblocks - 1:
				if block_1st_starts[start_block_index + 1] > start:
					break
				start_block_index += 1
			virtual_offset = self._chunk_block_1st_record_virtual_offsets[start_block_index]
			self.seek(virtual_offset)  # seek to the target block, self._buffer
			block_start_offset = self._block_start_offset
			record = self._struct_obj.unpack(self.read(self._unit_size))
			while record[col_to_query] < start:
				try:
					record = self._struct_obj.unpack(self.read(self._unit_size))
				except:
					break
			if record[col_to_query] < start:
				yield None
				continue
			if self._block_start_offset > block_start_offset:
				start_block_index += 1
			primary_id = int((_BLOCK_MAX_LEN * start_block_index +
				                  self._within_block_offset) / self._unit_size)
			id_start = primary_id
			while record[col_to_query] < end:
				try:
					record = self._struct_obj.unpack(self.read(self._unit_size))
					primary_id += 1
				except:
					break
			id_end = primary_id  # ID for end position,should be included
			yield [id_start, id_end]

	def query(self, dimension=None, start=None, end=None, regions=None,
			  query_col=[0], reference=None, printout=True):
		"""
		query .cz file by dimension, start and end, if regions provided, dimension, start and
		end should be None, regions should be a list, each element of regions
		is a list, for example, regions=[[('cell1','chr1'),1,10],
		[('cell10','chr22'),100,200]],and so on.

		Parameters
		----------
		dimension : str
			dimension, such as chr1 (if dimensions 'chrom' is inlcuded in
			header['dimensions']), or sample1 (if something like 'sampleID' is
			included in header['dimensions'] and chunk contains such dimension)
		start : int
			start position, if None, the entire dimension would be returned.
		end : int
			end position
		regions : list or file path
			regions=[
			[dimension,start,end], [dimension,start,end]
			],
			dimension is a tuple, for example, dimension=('chr1');
			if regions is a file path (separated by tab and no header),
			the first len(header['dimensions']) columns would be
			used as dimension, and next two columns would be used as start and end.
		query_col : list
			index of columns (header['columns']) to be queried,for example,
			if header['columns']=['pos','mv','cov'], then pos is what we want to
			query, so query_col should be [0], but if
			header['columns']=['start','end','peak'], then start and end are what
			we would like to query, query_col should be [0,1]

		Returns
		-------

		"""
		if dimension is None and regions is None:
			raise ValueError("Please provide either dimension,start,end or regions")
		if (not dimension is None) and (not regions is None):
			raise ValueError("Please query by dimension,start,end or by regions, "
							 "not both !")
		if not dimension is None:
			assert not start is None, "To query the whole chromosome, please use view, instead of query"
			assert not end is None, "To query the whole chromosome, please use view, instead of query"
			if isinstance(dimension, str):
				dimension = tuple([dimension])
				regions = [[dimension, start, end]]
			elif isinstance(dimension, dict):
				chunk_info = self.chunk_info.copy()
				selected_dim = dimension.copy()
				for d, v in selected_dim.items():
					chunk_info = chunk_info.loc[chunk_info[d] == v]
				dimension = chunk_info.index.tolist()
				regions = [[dim, start, end] for dim in dimension]
			elif isinstance(dimension, tuple):
				regions = [[dimension, start, end]]
			else:
				raise ValueError("Unknown types of dimension")
		else:
			if isinstance(regions, str):
				region_path = os.path.abspath(os.path.expanduser(regions))
				if os.path.exists(region_path):
					n_dim = len(self.header['dimensions'])
					usecols = list(range(n_dim + 2))
					df = pd.read_csv(region_path, sep='\t', header=None,
									 usecols=usecols)
					regions = df.apply(lambda x: [
						tuple(x[:n_dim]),
						x[n_dim],
						x[n_dim + 1]
					], axis=1).tolist()
				else:
					raise ValueError(f"path {region_path} not existed.")
			else:  # check format of regions
				if not isinstance(regions[0][0], tuple):
					raise ValueError("The first element of first region is not a tuple")

		if len(query_col) == 1:
			s = e = query_col[0]
		elif len(query_col) == 2:
			s, e = query_col
		else:
			raise ValueError("length of query_col can not be greater than 2.")

		if reference is None:  # query position in this .cz file.
			for i in set([s, e]):
				if self.header['formats'][i][-1] in ['s', 'c']:
					raise ValueError("The query_col of columns is not int or float",
									 "Can not perform querying without reference!",
									 f"columns: {self.header['columns']}",
									 f"formats: {self.header['formats']}")
			header = self.header['dimensions'] + self.header['columns']
			if printout:
				sys.stdout.write('\t'.join(header) + '\n')
				records = self._query_regions(regions, s, e)
				record = records.__next__()
				while True:
					try:
						record = records.__next__()
					except:
						break
					try:
						dims, row = record
						try:
							sys.stdout.write('\t'.join([str(d) for d in dims] +
													   self._byte2str(row)) + '\n')
						except:
							sys.stdout.close()
							self.close()
							return
					except ValueError:  # len(record) ==3: #"primary_id_&_dim:", primary_id, dim
						flag, primary_id, dim = record
				sys.stdout.close()
				self.close()
			else:
				return self._query_iter(regions, s, e)
		else:
			reference = os.path.abspath(os.path.expanduser(reference))
			ref_reader = Reader(reference)
			header = (self.header['dimensions'] + ref_reader.header['columns']
					  + self.header['columns'])
			if printout:
				sys.stdout.write('\t'.join(header) + '\n')
				# query reference first
				ref_records = ref_reader._query_regions(regions, s, e)
				flag, primary_id, dim = ref_records.__next__()  # 1st should contain primary_id
				records = self.fetchByStartID(dim, n=primary_id)
				while True:
					try:
						ref_record = next(ref_records)
						record = self._byte2str(next(records))
					except:
						break
					try:
						dims, row = ref_record
						rows = ref_reader._byte2str(row) + record
						try:
							sys.stdout.write('\t'.join([str(d) for d in dims] + rows) + '\n')
						except:
							sys.stdout.close()
							self.close()
							return
					except ValueError:  # len(record) ==3: #"primary_id_&_dim:", primary_id, dim
						flag, primary_id, dim = ref_record  # next block or next dim
						records = self.fetchByStartID(dim, n=primary_id)
				sys.stdout.close()
				self.close()
			else:
				return self._query_iter_ref(regions, s, e, ref_reader)
			ref_reader.close()

	def _query_iter(self, regions, s, e):
		records = self._query_regions(regions, s, e)
		record = records.__next__()
		while True:
			try:
				record = records.__next__()
			except:
				break
			try:
				dims, row = record
				yield list(dims) + list(self._byte2real(row))
			except ValueError:
				flag, primary_id, dim = record

	def _query_iter_ref(self, regions, s, e, ref_reader):
		ref_records = ref_reader._query_regions(regions, s, e)
		ref_record = ref_records.__next__()  # 1st should contain primary_id
		flag, primary_id, dim = ref_record
		records = self.fetchByStartID(dim, n=primary_id)
		while True:
			try:
				ref_record = ref_records.__next__()
				record = self._byte2str(records.__next__())
			except:
				break
			try:
				dims, row = ref_record
				rows = ref_reader._byte2str(row) + record
				yield list(dims) + rows
			except ValueError:  # len(record) ==3: #"primary_id_&_dim:", primary_id, dim
				flag, primary_id, dim = ref_record  # next block or next dim
				records = self.fetchByStartID(dim, n=primary_id)
		ref_reader.close()

	def _seek_and_read_1record(self, virtual_offset):
		if _c_seek_and_read_1record is not None:
			return _c_seek_and_read_1record(self._handle, virtual_offset, self.fmts, self._unit_size)
		self.seek(virtual_offset)
		return self._struct_obj.unpack(self.read(self._unit_size))

	def tell(self):
		"""Return a 64-bit unsigned BGZF virtual offset."""
		if 0 < self._within_block_offset and self._within_block_offset == len(
			self._buffer):
			# Special case where we're right at the end of a (non empty) block.
			# For non-maximal blocks could give two possible virtual offsets,
			# but for a maximal block can't use _BLOCK_MAX_LEN as the within block
			# offset. Therefore for consistency, use the next block and a
			# within block offset of zero.
			return (self._block_start_offset + self._block_raw_length) << 16
		else:
			return (self._block_start_offset << 16) | self._within_block_offset

	def seek(self, virtual_offset):
		"""Seek to a 64-bit unsigned BGZF virtual offset."""
		# Do this inline to avoid a function call,
		# start_offset, within_block = split_virtual_offset(virtual_offset)
		start_offset = virtual_offset >> 16
		within_block = virtual_offset ^ (start_offset << 16)
		if start_offset != self._block_start_offset:
			# Don't need to load the block if already there
			self._load_block(start_offset)
			if start_offset != self._block_start_offset:
				raise ValueError("start_offset not loaded correctly")
		if within_block > len(self._buffer):
			if not (within_block == 0 and len(self._buffer) == 0):
				raise ValueError(
					"Within offset %i but block size only %i"
					% (within_block, len(self._buffer))
				)
		self._within_block_offset = within_block
		return virtual_offset

	def read(self, size=1):
		"""Read method for the BGZF module."""
		# accelerated path
		if _c_read is not None:
			data, self._block_raw_length, self._buffer, self._within_block_offset = _c_read(
				self._handle, self._block_raw_length, self._buffer,
				self._within_block_offset, size
			)
			return data
		# fallback
		result = b""
		while size and self._block_raw_length:
			if self._within_block_offset + size <= len(self._buffer):
				data = self._buffer[
					self._within_block_offset: self._within_block_offset + size
					]
				self._within_block_offset += size
				result += data
				break
			else:  # need to load the enxt block
				data = self._buffer[self._within_block_offset:]
				size -= len(data)
				self._load_block()
				result += data
		return result

	def readline(self):
		"""Read a single line for the BGZF file."""
		# accelerated path
		if _c_readline is not None:
			data, self._block_raw_length, self._buffer, self._within_block_offset = _c_readline(
				self._handle, self._block_raw_length, self._buffer,
				self._within_block_offset, getattr(self, '_newline', b'\n')
			)
			return data
		# fallback
		result = b""
		while self._block_raw_length:
			i = self._buffer.find(self._newline, self._within_block_offset)
			# Three cases to consider,
			if i == -1:
				# No newline, need to read in more data
				data = self._buffer[self._within_block_offset:]
				self._load_block()  # will reset offsets
				result += data
			elif i + 1 == len(self._buffer):
				# Found new line, but right at end of block (SPECIAL)
				data = self._buffer[self._within_block_offset:]
				# Must now load the next block to ensure tell() works
				self._load_block()  # will reset offsets
				if not data:
					raise ValueError("Must be at least 1 byte")
				result += data
				break
			else:
				# Found new line, not at end of block (easy case, no IO)
				data = self._buffer[self._within_block_offset: i + 1]
				self._within_block_offset = i + 1
				# assert data.endswith(self._newline)
				result += data
				break

		return result

	def close(self):
		"""Close BGZF file."""
		self._handle.close()
		self._buffer = None
		self._block_start_offset = None

	def seekable(self):
		"""Return True indicating the BGZF supports random access."""
		return True

	def isatty(self):
		"""Return True if connected to a TTY device."""
		return False

	def fileno(self):
		"""Return integer file descriptor."""
		return self._handle.fileno()

	def __enter__(self):
		"""Open a file operable with WITH statement."""
		return self

	def __exit__(self, type, value, traceback):
		"""Close a file with WITH statement."""
		self.close()


# ==========================================================
def extract(input=None, outfile=None, ssi=None, chunksize=5000):
	ssi_reader = Reader(os.path.abspath(os.path.expanduser(ssi)))
	reader = Reader(os.path.abspath(os.path.expanduser(input)))
	writer = Writer(outfile, formats=reader.header['formats'],
					columns=reader.header['columns'],
					dimensions=reader.header['dimensions'],
					message=ssi)
	# dtfuncs = get_dtfuncs(writer.formats)
	for dim in reader.dim2chunk_start.keys():
		print(dim)
		IDs = ssi_reader.get_ids_from_ssi(dim)
		if len(IDs.shape) != 1:
			raise ValueError("Only support 1D ssi now!")
		records = reader._getRecordsByIds(dim, IDs)
		data_parts, i = [], 0
		for record in records:  # unpacked bytes
			data_parts.append(record)
			i += 1
			if i > chunksize:
				writer.write_chunk(b''.join(data_parts), dim)
				data_parts, i = [], 0
		if len(data_parts) > 0:
			writer.write_chunk(b''.join(data_parts), dim)
	writer.close()
	reader.close()
	ssi_reader.close()


# ==========================================================
class Writer:
	def __init__(self, output=None, mode="wb", formats=['B', 'B'],
				 columns=['mc', 'cov'], dimensions=['chrom'], 
				 fileobj=None,message='', level=6, verbose=0):
		"""
		cytozip .cz writer.

		Parameters
		----------
		output : path or None
			if None (default) bytes stream would be written to stdout.
		mode : str
			'wb', 'ab'
		formats : list
			format for each column, see https://docs.python.org/3/library/struct.html#format-characters for detail format.
		columns : list
			columns names, length should be the same as formats.
		dimensions : list
			dimensions to be included for each chunk, for example, if you would like
			to include sampleID and chromosomes as dimension title, then set
			Dimsnsion=['sampleID','chrom'], then give each chunk a dims,
			for example, dims for chunk1 is ['cell1','chr1'], dims for chunk2 is
			['cell1','chr2'], ['cell2','chr1'],....['celln','chr22']...
		fileobj : object
			an openned file object
		message : str
			a message to be included in the header, a genome assemble version
			is highly recommended to be set as message.
		level : int
			compress level (default is 6)
		verbose : int
			whether to print debug information
		"""
		if output and fileobj:
			raise ValueError("Supply either output or fileobj, not both")
		if fileobj:  # write to an existed openned file object
			if fileobj.read(0) != b"":
				raise ValueError("fileobj not opened in binary mode")
			handle = fileobj
		else:
			if not output is None:  # write output to a file
				if "w" not in mode.lower() and "a" not in mode.lower():
					raise ValueError(f"Must use write or append mode")
				handle = _open(output, mode)
				self.output = output
			else:  # write to stdout buffer
				handle = sys.stdout.buffer
		self._handle = handle
		self._buffer = bytearray()
		self._chunk_start_offset = None
		self._chunk_dims = None
		if isinstance(formats, str) and ',' not in formats:
			self.formats = [formats]
		elif isinstance(formats, str):
			self.formats = formats.split(',')
		else:
			self.formats = list(formats)
		self.level = level
		if isinstance(columns, str):
			self.columns = columns.split(',')
		else:
			self.columns = list(columns)
		if len(self.formats) != len(self.columns):
			raise ValueError(
				f"formats and columns must have the same length, "
				f"got {len(self.formats)} formats and {len(self.columns)} columns"
			)
		if isinstance(dimensions, str):
			self.dimensions = dimensions.split(',')
		else:
			self.dimensions = list(dimensions)
		self._magic_size = len(_cz_magic)
		self.verbose = verbose
		self.message = message
		self._chunk_index_entries = []  # for writing chunk index at close
		# Pre-select compress function to avoid per-block None check
		if _c_compress_block is not None:
			# Use Cython-accelerated compressor if available, otherwise fall back
			# to the pure-Python zlib.compressobj route.
			self._compress = _c_compress_block
		else:
			def _py_compress(block, level):
				c = zlib.compressobj(level, zlib.DEFLATED, -15, zlib.DEF_MEM_LEVEL, 0)
				return c.compress(block) + c.flush()
			self._compress = _py_compress
		# Pre-select records packer (fast > normal > pure-python)
		if _c_pack_records_fast is not None:
			self._pack_records = _c_pack_records_fast
		elif _c_pack_records is not None:
			self._pack_records = _c_pack_records
		else:
			self._pack_records = None
		if 'a' not in mode:
			self.write_header()

	def write_header(self):
		"""Write the .cz file header.

		See Reader.read_header() for the binary layout description.
		Notable details:

		- ``total_size`` is written as 0 here (placeholder) and
		  updated by ``_write_total_size_eof_close()`` at file close.
		- ``_n_dim_offset`` records where the dimension count byte is,
		  so ``catcz()`` can go back and increment it when merging
		  files that add a new dimension.
		- ``_header_size`` marks the end of the header = start of the
		  first chunk.
		"""
		f = self._handle
		f.write(struct.pack(f"<{self._magic_size}s", _cz_magic))  # 4 bytes
		# Convert version string (e.g. "0.5.2") to float (0.52) for binary format
		try:
			_ver_parts = _version.split('.')
			_ver_float = float(f"{_ver_parts[0]}.{''.join(_ver_parts[1:])}")
		except (ValueError, IndexError, AttributeError):
			_ver_float = 0.0
		f.write(struct.pack("<f", _ver_float))  # 4 bytes, float
		f.write(struct.pack("<Q", 0))  # 8 bytes, total size place holder, including magic.
		f.write(struct.pack("<H", len(self.message)))  # length of message 2bytes
		f.write(struct.pack(f"<{len(self.message)}s",
							bytes(self.message, 'utf-8')))
		f.write(struct.pack("<B", len(self.formats)))  # 1byte, ncols
		assert len(self.columns) == len(self.formats)
		for fmt in self.formats:
			fmt_len = len(fmt)
			f.write(struct.pack("<B", fmt_len))  # length of each format, 1 byte
			f.write(struct.pack(f"<{fmt_len}s", bytes(fmt, 'utf-8')))
		for name in self.columns:
			name_len = len(name)
			f.write(struct.pack("<B", name_len))  # length of each name, 1 byte
			f.write(struct.pack(f"<{name_len}s", bytes(name, 'utf-8')))
		self._n_dim_offset = f.tell()
		# when a new dim is added, go back here and rewrite the n_dim (1byte)
		f.write(struct.pack("<B", len(self.dimensions)))  # 1byte
		for dim in self.dimensions:
			dname_len = len(dim)
			f.write(struct.pack("<B", dname_len))  # length of each name, 1 byte
			f.write(struct.pack(f"<{dname_len}s", bytes(dim, 'utf-8')))
		# when multiple .cz are cat into one .cz and a new dimension is added,
		# such as sampleID, seek to this position (_header_size) and write another
		# two element: new_dname_len (B) and new_dim, then go to
		# _n_dim_offset,rewrite the n_dim (n_dim = n_dim + 1) and new dimension name
		self._header_size = f.tell()
		self.fmts = ''.join(list(self.formats))
		self._unit_size = struct.calcsize(self.fmts)

	def _write_block(self, block):
		"""Compress and write a single block of raw record data.

		Steps:
		  1. Compress with raw DEFLATE (via Cython or Python zlib).
		  2. Write block header: magic (2B) + block_size (2B).
		  3. Write compressed payload.
		  4. Write uncompressed data length (2B) as trailer.
		  5. Record the virtual offset of this block's first record in
		     ``self._block_1st_record_virtual_offsets`` for the chunk index.
		"""
		compressed = self._compress(block, self.level)
		bsize = _struct_H.pack(len(compressed) + 6)
		# block size: magic (2 btyes) + block_size (2bytes) + compressed data +
		# block_data_len (2 bytes)
		data_len = len(block)
		uncompressed_length = _struct_H.pack(data_len)  # 2 bytes
		data = _block_magic + bsize + compressed + uncompressed_length
		# Physical file offset where this compressed block starts on disk.
		block_start_offset = self._handle.tell()
		# Compute the byte offset of the first *complete* record within this
		# block's decompressed data.  Records may span block boundaries: if
		# the previous block ended in the middle of a record, the beginning
		# of this block contains the tail of that partial record.
		# Example: unit_size=12, chunk_data_len=65535 (not a multiple of 12).
		# 65535 % 12 = 3, so 3 bytes of the last record spill into this block,
		# and within_block_offset = 12 - 3 = 9.
		# If chunk_data_len is already aligned, within_block_offset = 0.
		# old formula: within_block_offset=int(np.ceil(self._chunk_data_len / self._unit_size) * self._unit_size - self._chunk_data_len)
		within_block_offset = (-self._chunk_data_len) % self._unit_size
		# d=65535, u=12
		# d % u  =  65535 % 12  =  3 # How many bytes have been exceeded from the previous alignment point?
		# (-d) % u = -65535 % 12  =  9 # How many more bytes are needed to reach the next alignment point?
		# Encode as a 64-bit BGZF virtual offset: upper 48 bits hold the
		# physical block position, lower 16 bits hold the within-block
		# offset of the first complete record.
		virtual_offset = (block_start_offset << 16) | within_block_offset
		# Store for the chunk tail / chunk index so readers can do O(1)
		# random access to any block's first record.
		self._block_1st_record_virtual_offsets.append(virtual_offset)
		# block_start_offset are real position on disk, not a virtual offset
		self._handle.write(data)
		# how many bytes (uncompressed) have been writen.
		self._chunk_data_len += data_len

	def _write_chunk_tail(self):
		"""Write the chunk tail after all blocks.

		Chunk tail layout:
		  chunk_data_len   : 8 bytes (Q) – total uncompressed data bytes
		  n_blocks          : 8 bytes (Q)
		  block_vos[]       : n_blocks * 8 bytes – virtual offset of each
		                      block's first record
		  dims[]            : for each dimension: 1-byte length + string
		"""
		if _c_write_chunk_tail is not None:
			_c_write_chunk_tail(self._handle, self._chunk_data_len,
							self._block_1st_record_virtual_offsets, self._chunk_dims)
			return
		# fallback: batch all writes into a single buffer
		n = len(self._block_1st_record_virtual_offsets)
		buf = _struct_2Q.pack(self._chunk_data_len, n)
		if n > 0:
			buf += struct.pack(f"<{n}Q", *self._block_1st_record_virtual_offsets)
		# dims
		for dim in self._chunk_dims: # type: ignore
			dim_bytes = dim.encode('utf-8')
			buf += _struct_B.pack(len(dim_bytes)) + dim_bytes
		self._handle.write(buf)

	def _chunk_finished(self):
		"""Finalize the current chunk.

		Seeks back to the chunk header to write the now-known total
		chunk size, then writes the chunk tail (data_len, block offsets,
		dims).  Also records this chunk's metadata for the chunk index
		that will be written at file close time.
		"""
		# current position is the end of chunk.
		cur_offset = self._handle.tell()  # before chunk tail, not a virtual offset
		# go back and write the total chunk size (tail not included)
		self._handle.seek(self._chunk_start_offset + 2)  # 2 bytes for magic
		chunk_size = cur_offset - self._chunk_start_offset
		# chunk_size including the chunk_size itself, but not including chunk tail.
		self._handle.write(_struct_Q.pack(chunk_size)) # write to the place holder
		# go back the current position
		self._handle.seek(cur_offset)  # not a virtual offset
		self._write_chunk_tail()
		# Record chunk info for chunk index
		self._chunk_index_entries.append({
			'dims': list(self._chunk_dims), # type: ignore
			'start': self._chunk_start_offset,
			'size': chunk_size,
			'data_len': self._chunk_data_len,
			'nblocks': len(self._block_1st_record_virtual_offsets),
		})

	def write_chunk(self, data, dims):  # dims is a list.
		"""Write data into one chunk with the given dimension values.

		If *dims* differs from the previously written chunk's dims, the
		previous chunk is finalized (flushed) and a new chunk header is
		started.  Data is buffered internally and split into blocks of
		at most ``_BLOCK_MAX_LEN``.

		Parameters
		----------
		data : bytes
			Packed binary record data.
		dims : list or tuple
			dimension values for this chunk (e.g., ['chr1']).
		"""
		# assert len(dims)==len(self.dimensions)
		if self._chunk_dims != dims:
			# the first chunk or another new chunk
			if self._chunk_dims is not None:
				# this is another new chunk, current position is the end of chunk.
				self.flush()  # finish_chunk in flush
			# else: #the first chunk, no chunk was writen previously.
			self._chunk_dims = dims
			self._chunk_start_offset = self._handle.tell()
			self._handle.write(_chunk_magic)
			# chunk total size place holder: 0
			self._handle.write(struct.pack("<Q", 0))  # 8 bytes; including this chunk_size
			self._chunk_data_len = 0
			self._block_1st_record_virtual_offsets = []

		if len(self._buffer) + len(data) < _BLOCK_MAX_LEN:
			self._buffer.extend(data)
		else:
			self._buffer.extend(data)
			while len(self._buffer) >= _BLOCK_MAX_LEN:
				self._write_block(bytes(self._buffer[:_BLOCK_MAX_LEN]))
				del self._buffer[:_BLOCK_MAX_LEN]

	def _parse_input_no_ref(self, input_handle):
		"""Generator that parses input data (DataFrame or file) into
		(DataFrame_chunk, dimension_values) pairs.

		Handles both pandas DataFrames and file paths.  For DataFrames,
		groups by dim_cols; for files, delegates to ``_input_parser()``.
		"""
		if isinstance(input_handle, pd.DataFrame):
			# usecols and dim_cols should be in the columns of this dataframe.
			if self.chunksize is None:
				for dim, df1 in input_handle.groupby(self.dim_cols)[self.usecols]:
					if not isinstance(dim, list):
						dim = [dim]
					yield df1, dim
			else:
				while input_handle.shape[0] > 0:
					df = input_handle.iloc[:self.chunksize]
					for dim, df1 in df.groupby(self.dim_cols)[self.usecols]:
						if not isinstance(dim, list):
							dim = [dim]
						yield df1, dim
					input_handle = input_handle.iloc[self.chunksize:]
		else:
			yield from _input_parser(input_handle, self.formats, self.sep,
									 self.usecols, self.dim_cols,
									 self.chunksize)

	def tocz(self, input=None, usecols=[4, 5], dim_cols=[0],
			 sep='\t', chunksize=5000, header=None, skiprows=0):
		"""
		Pack dataframe, stdin or a file path into .cz file with or without reference
		coordinates file. For example::

			cytozip Writer -O genes_flank2k.cz -F Q,Q,21s,c,20s,35s -C begin,end,EnsemblID,strand,gene_name,gene_type -D chrom tocz -I genes_flank2k.bed.gz -u 1,2,3,5,6,7

		Parameters
		----------
		input : input
			list, tuple, np.ndarray or dataframe, stdin (input is None, "stdin" or "-"),
			or a file path (need to specify sep,header and skiprows).
		usecols : list
			usecols is the index of columns to be packed into .cz columns.
		dim_cols : list
			index of columns to be set as dimensions of Writer, such as chrom
			columns.
		sep : str
			defauls is "\t", used as separator for the input file path.
		chunksize : int
			Chunk input dataframe of file path.
		header : bool
			whether the input file path contain header.
		skiprows : int
			number of rows to skip for input file path.
		pr: int or str
			If reference was given, use the `pr` column as coordinates, which
			would be used to match the `p` column in the input file or df.
			If input is a file path, then `pr` should be int, if input is a dataframe with
			custom columns names, then `pr` should be string matching the column
			name in dataframe.
		p: int or str
			Similar as `pr`, `p` is the column in input to match the coordinate in
			the reference file. When input is a file path or stdin, p should be int,
			but if input is a dataframe, p should be the name appear in the dataframe
			header columns.

		Returns
		-------

		"""
		self.sep = sep
		if not isinstance(input, pd.DataFrame):
			if isinstance(usecols, int):
				self.usecols = [int(usecols)]
			elif isinstance(usecols, str):
				self.usecols = [int(i) for i in usecols.split(',')]
			else:
				self.usecols = [int(i) for i in usecols]
			if isinstance(dim_cols, int):
				self.dim_cols = [int(dim_cols)]
			elif isinstance(dim_cols, str):
				self.dim_cols = [int(i) for i in dim_cols.split(',')]
			else:
				self.dim_cols = [int(i) for i in dim_cols]
		else:
			self.usecols = usecols
			self.dim_cols = dim_cols

		assert len(self.usecols) == len(self.formats)
		assert len(self.dim_cols) == len(self.dimensions)
		self.chunksize = chunksize
		self.header = header
		self.skiprows = skiprows
		# if self.verbose > 0:
		#     print("input: ", type(input), input)

		if input is None or input == 'stdin':
			data_generator = self._parse_input_no_ref(sys.stdin)
		elif isinstance(input, str):
			input_path = os.path.abspath(os.path.expanduser(input))
			if not os.path.exists(input_path):
				raise ValueError("Unknown format for input")
			if os.path.exists(input_path + '.tbi'):
				print("Please use allc2cz command to convert input to .cz.")
				return
			else:
				data_generator = self._parse_input_no_ref(input_path)
		elif isinstance(input, (list, tuple, np.ndarray)):
			input = pd.DataFrame(input)
			data_generator = self._parse_input_no_ref(input)
		elif isinstance(input, pd.DataFrame):
			data_generator = self._parse_input_no_ref(input)
		else:
			raise ValueError(f"Unknow input type")

		# if self.verbose > 0:
		#     print(self.usecols, self.dim_cols)
		#     print(type(self.usecols), type(self.dim_cols))
		for df, dim in data_generator:
			if self._pack_records is not None:
				rows = df.values.tolist()
				data = self._pack_records(rows, self.fmts)
			else:
				st = struct.Struct(f"<{self.fmts}")
				data = b''.join(st.pack(*row) for row in df.values)
			self.write_chunk(data, dim)
		self.close()

	@staticmethod
	def create_new_dim(basename):
		if basename.endswith('.cz'):
			return basename[:-3]
		return basename

	def catcz(self, input=None, dim_order=None, add_dim=False,
			  title="filename"):
		"""
		Cat multiple .cz files into one .cz file.

		Parameters
		----------
		input : str or list
			Either a str (including ``*``, as input for glob, should be inside the
			double quotation marks if using fire) or a list.
		dim_order : None, list or path
			If dim_order=None, input will be sorted using python sorted.
			If dim_order is a list, tuple or array of basename.rstrip(.cz), sorted as dim_order.
			If dim_order is a file path (for example, chrom size path to dim_order chroms
			or only use selected chroms) will be sorted as
			the 1st column of the input file path (without header, tab-separated).
			default is None
		add_dim: bool or function
			whether to add .cz file names as an new dimension to the merged
			.cz file. For example, we have multiple .cz files for many cells, in
			each .cz file, the dimensions are ['chrom'], after merged, we would
			like to add file name of .cz as a new dimension ['cell_id']. In this case,
			the dimensions in the merged header would be ["chrom","cell_id"], and
			in each chunk, in addition to the previous dim ['chr1'] or ['chr22'].., a
			new dim would also be append to the previous dim, like ['chr1','cell_1'],
			['chr22','cell_100'].
			However, if add_dim is a function, the input to this function is the .cz
			file basename, the returned value from this funcion would be used
			as new dim and added into the chunk_dims. The default function to
			convert filename to dim name is self.create_new_dim.
		title: str
			if add_dim is True or a python function, title would be append to
			the header['dimensions'] of the merged .cz file's header. If the title of
			new dimension had already given in Writer dimensions,
			title can be None, otherwise, title should be provided.

		Returns
		-------

		"""
		if isinstance(input, str) and '*' in input:
			input = glob.glob(input)
		if not isinstance(input, list):
			raise ValueError("input should be either a list of a string including *.")
		if dim_order is None:
			input = sorted(input)
		else:
			# creating a dict, keys are file base name, value are full path
			path_map = {os.path.basename(inp)[:-3]: inp for inp in input}
			if self.verbose > 0:
				print(path_map)
			if isinstance(dim_order, str):
				# read each file from path containing file basename
				dim_order = pd.read_csv(os.path.abspath(os.path.expanduser(dim_order)),
										sep='\t', header=None, usecols=[0])[0].tolist()
			if isinstance(dim_order, (list, tuple, np.ndarray)):  # dim_order is a list
				# input=[str(i)+'.cz' for i in dim_order]
				input = [path_map[str(i)] for i in dim_order]
			else:
				raise ValueError("input of dim_order is not corrected !")
		if self.verbose > 0:
			print(input)
		self.new_dim_creator = None
		if add_dim != False:  # add filename as another dimension.
			if add_dim == True:
				self.new_dim_creator = self.create_new_dim
			elif callable(add_dim):  # is a function
				self.new_dim_creator = add_dim
			else:
				raise ValueError("add_dim should either be True,False or a function")

		for file_path in input:
			reader = Reader(file_path)
			# data_size = reader.header['total_size'] - reader.header['header_size']
			chunks = reader.get_chunks()
			(start_offset, chunk_size, dims, data_len,
			 end_offset, nblocks, b1str_virtual_offsets) = next(chunks)
			# check whether new dim has already been added to header
			if not self.new_dim_creator is None:
				new_dim = [self.new_dim_creator(os.path.basename(file_path))]
			else:
				new_dim = []
			if len(dims + tuple(new_dim)) > len(self.dimensions):
				# new_dim title should be also added to header dimensions
				self.dimensions = self.dimensions + [title]
				self._handle.seek(self._n_dim_offset)
				# when a new dim is added, go back here and rewrite the n_dim (1byte)
				self._handle.write(struct.pack("<B", len(self.dimensions)))  # 1byte
				self._handle.seek(self._header_size)
				# write new dim to the end of header
				dname_len = len(title)
				self._handle.write(struct.pack("<B", dname_len))  # length of each name, 1 byte
				self._handle.write(struct.pack(f"<{dname_len}s", bytes(title, 'utf-8')))
				self._header_size = self._handle.tell()
			while True:  # process each file (chunk)
				reader._handle.seek(start_offset)  # chunk start offset
				# write the enrire chunk
				new_chunk_start_offset = self._handle.tell()
				delta_offset = new_chunk_start_offset - start_offset
				# self._handle.write(reader._handle.read(end_offset - start_offset))
				self._handle.write(reader._handle.read(chunk_size + 16))
				# modify the chunk_black_1st_record_virtual_offsets
				# Vectorized: shift all block physical offsets by delta_offset
				offsets = np.array(b1str_virtual_offsets, dtype=np.uint64)
				block_starts = offsets >> 16
				within_offsets = offsets & 0xFFFF
				new_offsets = ((block_starts + delta_offset) << 16) | within_offsets
				self._handle.write(new_offsets.tobytes())
				# rewrite the new dim onto the tail of chunk
				for dim in dims + tuple(new_dim):
					dim_bytes = dim.encode('utf-8')
					self._handle.write(_struct_B.pack(len(dim_bytes)))
					self._handle.write(dim_bytes)
				# print(dim_len,dim)
				# read next chunk
				try:
					(start_offset, chunk_size, dims, data_len,
					 end_offset, nblocks, b1str_virtual_offsets) = chunks.__next__()
				except:
					break
			reader.close()
		self._write_total_size_eof_close()

	def flush(self):
		"""Flush data explicitally."""
		while len(self._buffer) >= _BLOCK_MAX_LEN:
			self._write_block(bytes(self._buffer[:_BLOCK_MAX_LEN]))
			del self._buffer[:_BLOCK_MAX_LEN]
		self._write_block(bytes(self._buffer))
		self._chunk_finished()
		self._buffer = bytearray()
		self._handle.flush()

	def _write_chunk_index(self):
		"""Write the chunk index at end-of-file for O(1) chunk lookup.

		The chunk index enables remote/HTTP readers to locate any chunk
		without scanning the entire file.

		Index layout:
		  magic          : 4 bytes ('CZIX')
		  n_chunks       : 8 bytes (Q)
		  For each chunk:
		    dim strings  : 1B length + string, repeated for each dimension
		    start        : 8B  – physical file offset of chunk start
		    size         : 8B  – compressed chunk size
		    data_len     : 8B  – total uncompressed data bytes
		    nblocks      : 8B  – number of blocks
		  index_offset   : 8B  – written after the index so readers can
		                   find the index by reading 8 bytes before EOF

		Block virtual offsets are NOT stored here; they are already in
		each chunk's tail and are read on demand via _load_chunk().
		"""
		f = self._handle
		index_offset = f.tell()
		f.write(_chunk_index_magic)  # 4 bytes: "CZIX"
		f.write(_struct_Q.pack(len(self._chunk_index_entries)))
		for entry in self._chunk_index_entries:
			# dimension values
			for dim_val in entry['dims']:
				dim_bytes = bytes(dim_val, 'utf-8')
				f.write(_struct_B.pack(len(dim_bytes)))
				f.write(dim_bytes)
			# chunk info: start, size, data_len, nblocks
			f.write(_struct_4Q.pack(entry['start'], entry['size'],
									entry['data_len'], entry['nblocks']))
		# Write index offset so readers can find the index from end of file
		f.write(_struct_Q.pack(index_offset))

	def _write_total_size_eof_close(self):
		"""Finalize the file: write total size into the header, append EOF marker.

		Seeks back to the header's total_size field (at offset magic_size + 4)
		and writes the current file size, then writes the 28-byte EOF sentinel.
		"""
		cur_offset = self._handle.tell() # total size
		self._handle.seek(self._magic_size + 4)  # magic and version
		self._handle.write(_struct_Q.pack(cur_offset))  # real offset: total size
		self._handle.seek(cur_offset)
		self._handle.write(_cz_eof)
		self._handle.flush()
		self._handle.close()

	def close(self):
		"""Flush remaining data, write chunk index and EOF marker, then close.

		This is the main entry point for finalizing a .cz file.
		Order of operations:

		1. Flush any remaining buffered data as a final block.
		2. Finalize the current chunk (write chunk tail).
		3. Write the chunk index (for O(1) lookup by remote readers).
		4. Write total file size into the header + 28-byte EOF.
		5. Close the underlying file handle.
		"""
		if self._buffer:
			self.flush()
		else:
			self._chunk_finished()
		self._write_chunk_index()
		self._write_total_size_eof_close()

	def tell(self):
		"""Return a BGZF 64-bit virtual offset."""
		return make_virtual_offset(self._handle.tell(), len(self._buffer))

	def seekable(self):
		return False

	def isatty(self):
		"""Return True if connected to a TTY device."""
		return False

	def fileno(self):
		"""Return integer file descriptor."""
		return self._handle.fileno()

	def __enter__(self):
		"""Open a file operable with WITH statement."""
		return self

	def __exit__(self, type, value, traceback):
		"""Close a file with WITH statement."""
		self.close()

# ==========================================================
if __name__=="__main__":
	from cytozip import main
	main()
