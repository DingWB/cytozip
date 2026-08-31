"""
Coordinate-index builders for cytozip reference .cz files.

Two flavours of index:

* :func:`index_context` — 1-D index of per-site primary IDs whose
  ``context`` (and optional ``strand``) match a user pattern (CGN, CHN,
  +CGN, CAC, CAG, CAA, CAT, CHG, CWG, ...). Consumed by
  :func:`cytozip.dmr.call_dmr` / :func:`cytozip.dmr.call_dmr_ch`
  ``--index``, :func:`cytozip.cz.extractCG`, :func:`cytozip.allc.allc2cz`.

* :func:`index_regions` — 2-D index of ``(ID_start, ID_end, Name)``
  ranges, one row per BED region. Consumed by
  :func:`cytozip.cz.aggregate` ``--index`` and
  :func:`cytozip.features.cz_to_anndata`.

Both functions accept a ``chunk_keys`` argument (Python iterable,
comma-separated string, or a path to a chrom-sizes-like file) to
restrict indexing to a subset of chunk keys (typically chromosomes,
but more generally the first column of the .cz ``chunk_dims``).
"""

import os
import struct
import shutil
import multiprocessing

import numpy as np
import pandas as pd
from loguru import logger

from .cz import Reader, Writer, get_dtfuncs, _make_np_dtype


# ---------------------------------------------------------------------------
# IUPAC pattern matching for build_context_index.
#
# A reference .cz stores a fixed-width ``context`` field (e.g. 3 bytes for
# the standard CGN / CHN / CAG / ... triplet) and a 1-byte ``strand``.
# Given a user pattern we build a boolean mask over the whole chunk in a
# single numpy pass (~30-100x faster than per-record Python).
#
# Pattern syntax::
#
#     [+|-]<MOTIF>
#
#   * Optional leading ``+`` / ``-`` restricts to that strand.
#   * MOTIF is a sequence of IUPAC codes:
#         A C G T  - exact base
#         N        - any base (A/C/G/T)
#         H        - not G        (A/C/T)        D - not C   (A/G/T)
#         B        - not A        (C/G/T)        V - not T   (A/C/G)
#         R = A/G  Y = C/T  S = G/C  W = A/T  K = G/T  M = A/C
#
# Examples::
#
#     CGN, CHN, CAN, CTN, CCN
#     CAC, CAG, CAA, CAT, CHG, CHH, CCG, CWG
#     +CGN, -CHN, +CAC
# ---------------------------------------------------------------------------
_IUPAC = {
	'A': b'A', 'C': b'C', 'G': b'G', 'T': b'T',
	'N': b'ACGT',
	'R': b'AG', 'Y': b'CT', 'S': b'GC', 'W': b'AT',
	'K': b'GT', 'M': b'AC',
	'B': b'CGT', 'D': b'AGT', 'H': b'ACT', 'V': b'ACG',
}


def _expand_iupac(motif):
	"""Expand an IUPAC motif into the list of all matching exact byte strings."""
	seqs = [b'']
	for ch in motif.upper():
		if ch not in _IUPAC:
			raise ValueError(f"invalid IUPAC code {ch!r} in pattern {motif!r}")
		bases = _IUPAC[ch]
		seqs = [s + bytes([b]) for s in seqs for b in bases]
	return seqs


def _parse_context_pattern(pattern):
	"""Split user pattern into ``(strand_filter, motif)``.

	``strand_filter`` is ``b'+'`` / ``b'-'`` / ``None``; ``motif`` is the
	uppercase IUPAC string with the strand prefix stripped.
	"""
	pattern = pattern.strip()
	if not pattern:
		raise ValueError("empty context pattern")
	strand = None
	if pattern[0] in '+-':
		strand = pattern[0].encode()
		pattern = pattern[1:]
	motif = pattern.upper()
	if not motif:
		raise ValueError(f"context pattern {pattern!r} has no motif")
	return strand, motif


def _locate_context_column(columns, formats):
	"""Return ``(index, byte_width)`` of the ``context`` column."""
	if 'context' in columns:
		ci = columns.index('context')
	else:
		ci = next((i for i, f in enumerate(formats) if f.endswith('s')), None)
		if ci is None:
			raise ValueError("no context column found in reference header")
	return ci, struct.calcsize(formats[ci])


def _locate_strand_column(columns, formats):
	"""Return ``(index, byte_width)`` of the ``strand`` column."""
	if 'strand' in columns:
		si = columns.index('strand')
	else:
		si = next((i for i, f in enumerate(formats)
		           if f.endswith('c') or f == '1s'), None)
		if si is None:
			raise ValueError("no strand column found in reference header")
	return si, struct.calcsize(formats[si])


def _vectorised_context_mask(arr, columns, formats, pattern):
	"""Return a bool ndarray marking rows of ``arr`` matching ``pattern``.

	``arr`` is a structured ndarray as returned by
	:meth:`cytozip.cz.Reader.chunk2numpy` (positional fields ``f0, f1, ...``).
	"""
	strand_filter, motif = _parse_context_pattern(pattern)
	ci, ctx_n = _locate_context_column(columns, formats)
	if len(motif) > ctx_n:
		raise ValueError(
			f"pattern motif {motif!r} ({len(motif)} bases) is longer than "
			f"the reference context width ({ctx_n} bases)")
	ctx = arr[f'f{ci}'].view(f'|S{ctx_n}')

	# Strip trailing N's: matching them is a no-op and reduces a 3-base
	# motif like 'CGN' to a 2-base prefix 'CG' that np.char.startswith
	# can dispatch directly. Lets the caller write 'CG' or 'CGN'
	# interchangeably.
	core = motif.rstrip('N')
	if not core:
		mask = np.ones(arr.shape[0], dtype=bool)
	elif all(c in 'ACGT' for c in core):
		# Pure ACGT prefix: single C-level startswith.
		mask = np.char.startswith(ctx, core.encode())
	else:
		# Has IUPAC ambiguity code(s): OR a handful of exact prefixes.
		# Number of expansions is small for typical motifs
		# (CHN -> 3, CHG -> 3, CWG -> 2, CHH -> 9, ...).
		seqs = _expand_iupac(core)
		mask = np.char.startswith(ctx, seqs[0])
		for seq in seqs[1:]:
			mask |= np.char.startswith(ctx, seq)

	if strand_filter is not None:
		si, sn = _locate_strand_column(columns, formats)
		strand = arr[f'f{si}'].view(f'|S{sn}')
		mask &= (strand == strand_filter)
	return mask


def _resolve_chunk_keys(chunk_keys):
	"""Normalise the user ``chunk_keys`` argument to a ``set`` (or ``None``).

	A chunk key is the first element of the .cz ``chunk_dims`` tuple
	(typically a chromosome name for reference / methylation .cz files).

	Accepts:

	* ``None`` / empty / empty list -> ``None`` (no filter, use all keys).
	* ``list`` / ``tuple`` / ``set`` of names.
	* Comma-separated string (``"chr1,chr2,chrX"``).
	* Path to a file whose first whitespace-separated column lists the
	  keys (e.g. ``mm10.chrom.sizes``); blank / ``#`` lines are skipped.
	"""
	if chunk_keys is None:
		return None
	if isinstance(chunk_keys, (list, tuple, set)):
		out = {str(c) for c in chunk_keys if str(c).strip()}
		return out or None
	s = str(chunk_keys).strip()
	if not s:
		return None
	p = os.path.abspath(os.path.expanduser(s))
	if os.path.isfile(p):
		out = set()
		with open(p) as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				out.add(line.split()[0])
		return out or None
	return {c.strip() for c in s.split(',') if c.strip()} or None


# ===========================================================================
# 1-D context index
# ===========================================================================
def _build_context_index_worker(input, output, dim, formats, columns,
								chunk_dims, pattern, batch_size, message):
	"""Per-chrom vectorised worker for :func:`index_context`."""
	# Reads one chrom's whole chunk front-to-back -> sequential.
	reader = Reader(input, mmap_advise="sequential")
	try:
		arr = reader.chunk2numpy(dim, reformat=False)
		writer = Writer(output, formats=formats, columns=columns,
						chunk_dims=chunk_dims, message=message,
						delta_cols=['ID'])
		try:
			if arr.size:
				mask = _vectorised_context_mask(
					arr, reader.header['columns'],
					reader.header['formats'], pattern)
				ids = (np.flatnonzero(mask) + 1).astype('<u4')
				for s in range(0, ids.size, batch_size):
					writer.write_chunk(ids[s:s + batch_size].tobytes(), dim)
		finally:
			writer.close()
	finally:
		reader.close()


# ===========================================================================
# 2-D region index
# ===========================================================================
def _build_region_index_worker(input, output, dim, df1, formats, columns,
							   chunk_dims, batch_size):
	"""Per-chrom vectorised worker for :func:`index_regions`.

	Loads the chrom's ``pos`` column once and resolves all region
	``(start, end)`` ranges with two ``np.searchsorted`` calls
	(``side='left'``):

	* ``id_start`` = first record with ``pos >= start``
	* ``id_end``   = first record with ``pos >= end``

	Regions whose start lies past the last reference position are dropped
	(matches the pre-vectorised ``pos2id`` ``yield None`` branch).
	"""
	logger.debug(dim)
	# Reads one chrom's whole chunk front-to-back -> sequential.
	reader = Reader(input, mmap_advise="sequential")
	try:
		arr = reader.chunk2numpy(dim, reformat=False)
		header_columns = reader.header['columns']
		if 'pos' in header_columns:
			pi = header_columns.index('pos')
		else:
			pi = next(i for i, f in enumerate(reader.header['formats'])
			          if f[-1] in 'BHIQbhiq')
		pos = arr[f'f{pi}']

		starts = df1['start'].to_numpy()
		ends = df1['end'].to_numpy()
		names = df1['Name'].to_numpy()

		id_start = np.searchsorted(pos, starts, side='left')
		id_end = np.searchsorted(pos, ends, side='left')
		valid = id_start < pos.size
		id_start = id_start[valid]
		id_end = id_end[valid]
		names = names[valid]

		writer = Writer(output, formats=formats, columns=columns,
						chunk_dims=chunk_dims,
						message=os.path.basename(input),
						delta_cols=['ID_start', 'ID_end'])
		try:
			if id_start.size:
				# Vectorized pack: build the structured record array in one
				# shot instead of a per-region struct.pack loop. Integer
				# columns are clipped to their dtype range (matching the old
				# ``int_func`` clamp for unsigned formats); the Name column is
				# encoded to fixed-width bytes (struct '<Ns' == numpy 'SN',
				# both null-padded/truncated to N).
				rec_dt = _make_np_dtype(formats, columns)
				out = np.empty(id_start.size, dtype=rec_dt)
				col_data = (id_start, id_end, names)
				for ci, (col, fmt) in enumerate(zip(columns, formats)):
					npdt = rec_dt[col]
					if fmt[-1] in 'BHILQbhiq':
						info = np.iinfo(npdt)
						out[col] = np.clip(
							np.asarray(col_data[ci]).astype(np.int64),
							info.min, info.max).astype(npdt)
					else:
						out[col] = np.asarray(col_data[ci]).astype(npdt)
				buf = out.tobytes()
				rec_size = rec_dt.itemsize
				step = batch_size * rec_size
				for s in range(0, len(buf), step):
					writer.write_chunk(buf[s:s + step], dim)
		finally:
			writer.close()
	finally:
		reader.close()


# ===========================================================================
# Module-level convenience wrappers (CLI entry points)
# ===========================================================================
def index_context(input, output=None, pattern="CGN", jobs=4, chunk_keys=None):
	"""
	Build a 1-D *context* coordinate index (.cz) over a reference .cz.

	A reference .cz built by :class:`cytozip.allc.AllC` /
	``czip build_ref`` contains every cytosine in the genome (pattern
	``C``). Most methylation analyses however only care about a specific
	sequence context:

	* ``CGN`` — every CpG dinucleotide on either strand. Used for **5mC /
	  5hmC** analyses (CG DMR calling, gene-body mCG, CG bigwig tracks).
	* ``CHN`` (H = A/C/T) — every non-CpG cytosine. Used for **mCH**
	  analyses (e.g. neuronal mCH DMR calling, mCH gene scores).
	* ``+CGN`` — forward-strand CpGs only (one row per CpG dinucleotide,
	  useful when collapsing CG strands with ``extractCG --merge_cg``).
	* Sub-context triplets such as ``CAC``, ``CAG``, ``CAA``, ``CAT``,
	  ``CCG``, ``CWG`` (W = A/T) for fine-grained mCH analyses.
	* Any IUPAC pattern of length up to the reference context width
	  (``A C G T N R Y S W K M B D H V``), optionally prefixed with
	  ``+`` / ``-`` to restrict to a single strand.

	The output .cz stores, per chromosome, the integer ``primary_id`` of
	every row in the reference whose context matches ``pattern``. It is
	tiny (usually < 1% of the reference) and is the standard way to
	restrict downstream commands to a single context. Typical consumers:

	* :func:`cytozip.dmr.call_dmr` / :func:`cytozip.dmr.call_dmr_ch`
	  (``-s/--index``) — only test CG (resp. CH / CAC / CAG / ...) sites
	  when calling DMRs.
	* :func:`cytozip.cz.extractCG` — produce CG-only per-cell .cz files.
	* :func:`cytozip.allc.allc2cz` and :func:`cytozip.cz.aggregate` — use a
	  CG/CH index as a ``reference`` to keep only context-relevant rows.

	Parameters
	----------
	input : path
		Reference .cz file produced by ``czip build_ref`` (must contain
		a ``context`` column, e.g. ``CGN``).
	output : path, optional
		Output index .cz path. Defaults to ``<input>.<pattern>.index``.
	pattern : str
		Context pattern. Built-ins: ``CGN``, ``CHN``, ``+CGN``. Also
		accepts any IUPAC motif (``CAC``, ``CAG``, ``CAA``, ``CAT``,
		``CHG``, ``CWG``, ``+CAC``, ...).
	jobs : int
		Number of parallel worker processes (one shard per chunk key,
		merged via ``catcz``). Use ``jobs >= 8`` for large mammalian
		references — typical 5-10x wall-clock speedup over serial.
	chunk_keys : list / str / path, optional
		Restrict to a subset of chunk keys (typically chromosomes for a
		reference .cz). Accepts a Python iterable, a comma-separated
		string (``"chr1,chr2,chrX"``) or a path to a file whose first
		whitespace-separated column lists the keys (e.g.
		``~/Ref/mm10/mm10_ucsc.nochrM.sizes``).

	Examples
	--------
	CLI::

		czip index context -I mm10_with_chrL.allc.cz \\
			-p CGN -O mm10_with_chrL.CGN.cz
		czip index context -I mm10_with_chrL.allc.cz \\
			-p CAC -O mm10_with_chrL.CAC.cz \\
			-k ~/Ref/mm10/mm10_ucsc.nochrM.sizes -j 8

	Python::

		import cytozip
		cytozip.index_context('mm10_with_chrL.allc.cz',
		                      output='mm10.CGN.cz', pattern='CGN')

	See Also
	--------
	index_regions : Build a 2-D *region* index from a BED file.
	"""
	input = os.path.abspath(os.path.expanduser(input))
	if output is None:
		output = input + '.' + pattern + '.index'
	else:
		output = os.path.abspath(os.path.expanduser(output))

	jobs = int(jobs)
	batch_size = 2000
	formats = ['I']
	columns = ['ID']
	chunk_dims = ['chrom']

	# Validate pattern up front (raises early on bad input).
	_parse_context_pattern(pattern)

	reader = Reader(input, mmap_advise="random")
	try:
		key_filter = _resolve_chunk_keys(chunk_keys)
		dims = [d for d in reader.chunk_key2offset
		        if key_filter is None
		        or (d[0] if isinstance(d, tuple) else d) in key_filter]
		if not dims:
			raise ValueError(
				f"no chunk keys selected (chunk_keys={chunk_keys!r}); "
				f"reference contains: "
				f"{[d[0] if isinstance(d, tuple) else d for d in reader.chunk_key2offset][:10]}...")

		# Multi-process per-chunk-key path.
		if jobs > 1 and len(dims) > 1:
			outdir = output + '.tmp'
			os.makedirs(outdir, exist_ok=True)
			pool = multiprocessing.Pool(jobs)
			tasks = []
			for dim in dims:
				chrom = dim[0] if isinstance(dim, tuple) else dim
				shard = os.path.join(outdir, f"{chrom}.cz")
				tasks.append(pool.apply_async(
					_build_context_index_worker,
					(input, shard, dim, formats, columns, chunk_dims,
					 pattern, batch_size, os.path.basename(input))))
			for t in tasks:
				t.get()
			pool.close()
			pool.join()
			writer = Writer(output=output, formats=formats,
							columns=columns, chunk_dims=chunk_dims,
							message=os.path.basename(input),
							delta_cols=['ID'])
			writer.catcz(input=f"{outdir}/*.cz", key_added=None)
			shutil.rmtree(outdir, ignore_errors=True)
			return

		# Single-process vectorised path.
		writer = Writer(output, formats=formats, columns=columns,
						chunk_dims=chunk_dims, fileobj=None,
						message=os.path.basename(input),
						delta_cols=['ID'])
		header_columns = reader.header['columns']
		header_formats = reader.header['formats']
		for dim in dims:
			logger.debug(dim)
			arr = reader.chunk2numpy(dim, reformat=False)
			if arr.size == 0:
				continue
			mask = _vectorised_context_mask(
				arr, header_columns, header_formats, pattern)
			ids = (np.flatnonzero(mask) + 1).astype('<u4')
			if ids.size == 0:
				continue
			for s in range(0, ids.size, batch_size):
				writer.write_chunk(ids[s:s + batch_size].tobytes(), dim)
		writer.close()
	finally:
		reader.close()


def index_regions(input, output=None, bed=None, jobs=4, chunk_keys=None):
	"""
	Build a 2-D *region* coordinate index (.cz) over a reference .cz from
	a BED file.

	Whereas :func:`index_context` records *individual* site IDs (1-D),
	``index_regions`` records, per region in the BED, the
	``(ID_start, ID_end)`` half-open range of reference rows that fall
	inside it, plus the region ``Name``. This makes it the natural input
	for on-disk **per-region aggregation** — e.g. summing per-cell mc/cov
	over gene bodies (± flanks), promoters, CGIs, peaks, or DMRs.

	Typical use cases:

	* :func:`cytozip.cz.aggregate` (``--index``) — collapse a per-cell
	  .cz into one row per region (e.g. one row per gene), summing
	  mc/cov within.
	* :func:`cytozip.features.cz_to_anndata` — build a cell × region
	  AnnData over a feature BED.
	* As a region mask for ``query`` / ``view`` workflows.

	The BED must be tab-separated with at least 4 columns:
	``chrom  start  end  name`` (BED-style 0-based half-open coordinates).
	Work is parallelized per chromosome via ``jobs`` worker processes.

	Parameters
	----------
	input : path
		Reference .cz file (e.g. produced by ``czip build_ref``).
	output : path, optional
		Output index .cz path. Defaults to
		``<input>.<bed_basename>.index``.
	bed : path
		BED file with regions to index (chrom/start/end/name; extra
		columns are ignored).
	jobs : int
		Number of parallel worker processes (one per chunk-key shard).
	chunk_keys : list / str / path, optional
		Restrict to a subset of chunk keys (typically chromosomes for a
		reference .cz). Accepts a Python iterable, a comma-separated
		string, or a path to a file whose first whitespace-separated
		column lists the keys (e.g. ``~/Ref/mm10/mm10_ucsc.nochrM.sizes``).

	Examples
	--------
	CLI::

		czip index regions -I mm10_with_chrL.allc.cz \\
			-O mm10_with_chrL.genes_flank2k.cz \\
			-b genes_flank2k.bed.gz -j 4

		# Then aggregate per-cell mC over those regions:
		czip aggregate -I cell.cz -O cell_gene.cz \\
			--index mm10_with_chrL.genes_flank2k.cz

	See Also
	--------
	index_context : Build a 1-D context (CGN / CHN / +CGN) index.
	cytozip.cz.aggregate : Sum mc/cov per region using a region index.
	"""
	bed = os.path.abspath(os.path.expanduser(bed))
	input = os.path.abspath(os.path.expanduser(input))
	if output is None:
		output = input + '.' + os.path.basename(bed) + '.index'
	else:
		output = os.path.abspath(os.path.expanduser(output))

	jobs = int(jobs)
	batch_size = 2000
	formats = ['I', 'I']
	columns = ['ID_start', 'ID_end']
	chunk_dims = ['chrom']

	n_chunk_dims = len(chunk_dims)
	df = pd.read_csv(bed, sep='\t', header=None,
	                 usecols=list(range(n_chunk_dims + 3)),
	                 names=['chrom', 'start', 'end', 'Name'])
	max_name_len = int(df.Name.apply(lambda x: len(x)).max())
	formats = formats + [f'{max_name_len}s']
	columns = columns + ['Name']

	key_filter = _resolve_chunk_keys(chunk_keys)

	reader = Reader(input, mmap_advise="random")
	try:
		outdir = output + '.tmp'
		os.makedirs(outdir, exist_ok=True)
		pool = multiprocessing.Pool(jobs)
		tasks = []
		for chrom, df1 in df.groupby('chrom'):
			dim = (chrom,)
			if dim not in reader.chunk_key2offset:
				continue
			if key_filter is not None and chrom not in key_filter:
				continue
			shard = os.path.join(outdir, chrom + '.cz')
			tasks.append(pool.apply_async(
				_build_region_index_worker,
				(input, shard, dim, df1, formats, columns,
				 chunk_dims, batch_size)))
		for t in tasks:
			t.get()
		pool.close()
		pool.join()
		writer = Writer(output=output, formats=formats,
						columns=columns, chunk_dims=chunk_dims,
						message=os.path.basename(bed),
						delta_cols=['ID_start', 'ID_end'])
		writer.catcz(input=f"{outdir}/*.cz", key_added=None)
		shutil.rmtree(outdir, ignore_errors=True)
	finally:
		reader.close()
