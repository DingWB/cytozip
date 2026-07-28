#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
allc.py — DNA methylation allc-file I/O built on top of the cytozip format.

This module provides:
  - :class:`AllC`: Extract all C (cytosine) positions from a reference
           genome and store them as a .cz coordinate file.
  - :func:`allc2cz`: Convert allc.tsv.gz (tabix-indexed) to .cz format,
             optionally using a reference .cz for coordinate alignment.
  - :func:`extractCG`: Extract CG-context records from a full .cz file.

See :func:`cytozip.index.index_context` for building CGN / CHN / +CGN
coordinate indexes from a reference .cz.

Sibling modules:
  - :mod:`cytozip.cz`:     generic .cz format (Reader / Writer / extract
                           / index_regions / aggregate).
  - :mod:`cytozip.merge`:  per-cell .cz merging (``merge_cz``,
                           ``merge_cell_type``, Fisher-test mode).
  - :mod:`cytozip.dmr`:    peak calling (``call_peaks``, ``to_bedgraph``)
                           and DMR analysis (``call_dmr``, ``call_dmr_ch``,
                           ``call_dmr_array``, ``annot_dmr``).

@author: DingWB
"""
import os
import struct
import multiprocessing
from loguru import logger
from .cz import (Reader, Writer, get_dtfuncs,
                 _fmt_to_np_dtype, _chrom_axis,
                 _all_numeric_formats, _pack_chunk_data,
                 _write_np_chunks, _parse_tabix_lines,
                 np, pd)
# Lazily access Cython accelerators via the cz module namespace so that
# ``import cytozip.allc`` does not force cz_accel to load (~65 ms).
from . import cz as _cz

# ==========================================================
def WriteC(record, outdir, batch_size=5000, delta_cols=None):
    """
    Extract C positions from a BioPython SeqRecord and write to .cz file.
    
    Uses Cython-accelerated implementation when available for ~10-50x speedup.
    
    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        A BioPython sequence record (chromosome)
    outdir : str
        output directory path
    batch_size : int
        Number of records per chunk (default: 5000)
    delta_cols : None, int, str, or list, optional
        Columns to store with in-block delta encoding (typically ``'pos'``
        for tighter compression of monotonic positions). Forwarded to
        :class:`cytozip.cz.Writer`. ``None`` (default) disables delta
        encoding.
    """
    chrom = record.id
    output = os.path.join(outdir, chrom + ".cz")
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return None
    logger.debug(chrom)
    writer = Writer(output, formats=['Q', 'c', '3s'],
                    columns=['pos', 'strand', 'context'],
                    chunk_dims=['chrom'], sort_col='pos',
                    delta_cols=delta_cols)
    
    # Use Cython-accelerated version if available
    _cz._ensure_cz_accel()
    if _cz._c_write_c_records is not None: # 10 times faster than pure Python
        # Convert sequence to bytes once
        seq_bytes = str(record.seq).encode('ascii')
        for data, count in _cz._c_write_c_records(seq_bytes, batch_size):
            if data:
                writer.write_chunk(data, [chrom])
        writer.close()
        return
    
    # Fallback to pure Python implementation
    dtfuncs = get_dtfuncs(writer.formats)
    seq_length = len(record.seq)
    rows_buf = []
    data = b''
    for i in range(seq_length):  # 0-based
        base = record.seq[i:i + 1].upper()
        if base.__str__() == 'C':  # forward strand
            context = record.seq[i: i + 3].upper().__str__()  # pos, left l1 base pair and right l2 base pair
            strand = '+'
        elif base.reverse_complement().__str__() == 'C':  # reverse strand
            context = record.seq[i - 2:i + 1].reverse_complement().upper().__str__()
            strand = '-'
        else:
            continue
        context_len = len(context)
        if context_len < 3:
            if context_len == 0:
                context = "CNN"
            else:
                context = context + 'N' * (3 - context_len)

        # f.write(f"{chrom}\t{i}\t{i + 1}\t{context}\t{strand}\n")
        values = [func(v) for v, func in zip([i + 1, strand, context], dtfuncs)]
        rows_buf.append(values)
        # position is 0-based (start) 1-based (end position, i+1)
        if (i % batch_size == 0 and len(rows_buf) > 0):
            if writer._pack_records is not None:
                data = writer._pack_records(rows_buf, writer.fmts)
            else:
                st = struct.Struct(writer.fmts)
                data = b''.join(st.pack(*r) for r in rows_buf)
            writer.write_chunk(data, [chrom])
            rows_buf = []
    if len(rows_buf) > 0:
        if writer._pack_records is not None:
            data = writer._pack_records(rows_buf, writer.fmts)
        else:
            st = struct.Struct(writer.fmts)
            data = b''.join(st.pack(*r) for r in rows_buf)
        writer.write_chunk(data, [chrom])
    writer.close()


# ==========================================================
class AllC:
    def __init__(self, genome=None, output="hg38_allc.cz",
                 pattern="C", jobs=12, keep_temp=False, delta=True,
                 chroms=None):
        """
        Extract position of specific pattern in the reference genome, for example C.
            Example: python ~/Scripts/python/tbmate.py AllC -g ~/genome/hg38/hg38.fa --jobs 10 run
            Or call within python: ac=AllC(genome="/gale/netapp/home2/wding/genome/hg38/hg38.fa")
        Parameters
        ----------
        genome: path
            reference genome (FASTA). Any format readable by
            ``Bio.SeqIO.parse(..., "fasta")``.
        output: path
            path for the output reference ``.cz`` file
            (default: ``"hg38_allc.cz"``).
        pattern: str
            nucleotide pattern to extract. Only ``'C'`` (every cytosine,
            written by :func:`WriteC`) is currently implemented (default: ``'C'``).
        jobs: int
            number of CPU (parallel processes) used for the Pool. ``None``
            falls back to ``os.cpu_count()`` (default: 12).
        keep_temp: bool
            if True, keep the per-chromosome temp directory
            (``<output>.tmp``) after merging; otherwise it is removed at the
            end of :meth:`run` (default: False).
        delta: bool
            if True (default), DELTA-encode the strictly-monotonic ``pos``
            column. Positions in a reference ``.cz`` are sorted and closely
            spaced (~3-10 bp for CGN/CHN), so per-block deltas compress
            ~4-5x tighter than raw 8-byte ``Q`` values after DEFLATE. Set
            to False for the fastest query path at the cost of ~2-3x larger
            files.
        chroms: path, optional
            Path to a ``.fai`` index file or a plain text file whose first
            (tab-separated, no header) column lists chromosome names. When
            provided, only these chromosomes are extracted, and the merged
            reference ``.cz`` stores chunks in exactly this order. ``None``
            (default) processes every sequence in the genome fasta.
        """
        self.genome=os.path.abspath(os.path.expanduser(genome))
        self.output=os.path.abspath(os.path.expanduser(output))
        self.outdir=self.output+'.tmp'
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.pattern=pattern
        from Bio import SeqIO
        self.records = SeqIO.parse(self.genome, "fasta")
        self.jobs = jobs if not jobs is None else os.cpu_count()
        self.keep_temp = keep_temp
        # Optional chromosome whitelist / ordering. Works for both a
        # samtools ``.fai`` index and a plain single-column text file since
        # in both cases the chromosome name is the first tab-separated field.
        if chroms is not None:
            chroms = os.path.abspath(os.path.expanduser(chroms))
            chrom_df = pd.read_csv(chroms, sep='\t', header=None, usecols=[0])
            self.chroms = chrom_df.iloc[:, 0].astype(str).tolist()
            self.chrom_set = set(self.chroms)
        else:
            self.chroms = None
            self.chrom_set = None
        self.chroms_path = chroms
        # DELTA-encode the strictly-monotonic ``pos`` column by default.
        # Positions in a reference .cz are sorted and closely spaced (~3-10 bp
        # for CGN/CHN), so per-block deltas compress ~4-5x tighter than raw
        # 8-byte Q values after DEFLATE. Disable with ``delta=False`` to get
        # the fastest query path at the cost of ~2-3x larger files.
        self.delta_cols = ['pos'] if delta else None
        if pattern=='C':
            self.func=WriteC

    def writePattern(self):
        """Extract the pattern from every (selected) chromosome in parallel.

        Dispatches one :func:`WriteC` task per fasta record to a
        multiprocessing pool, writing a per-chromosome ``.cz`` file into
        ``self.outdir``. Records absent from ``self.chrom_set`` (when a
        ``chroms`` whitelist was given) are skipped.
        """
        pool = multiprocessing.Pool(self.jobs)
        tasks = []
        for record in self.records:
            # Skip sequences not in the requested chromosome list (if any).
            if self.chrom_set is not None and record.id not in self.chrom_set:
                continue
            task = pool.apply_async(self.func, (record, self.outdir, 5000, self.delta_cols))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()

    def merge(self):
        """Concatenate the per-chromosome temp ``.cz`` files into ``self.output``.

        When a ``chroms`` list was provided, chunks are merged in that
        exact order; otherwise :meth:`cytozip.cz.Writer.catcz` falls back to
        its default ``sorted()`` ordering.
        """
        writer = Writer(output=self.output, formats=['Q', 'c', '3s'],
                        columns=['pos', 'strand', 'context'],
                        chunk_dims=['chrom'], message=self.genome,
                        sort_col='pos', delta_cols=self.delta_cols)
        # When a chromosome list is given, merge chunks in that exact order;
        # otherwise fall back to catcz's default sorted() ordering.
        writer.catcz(input=f"{self.outdir}/*.cz", chunk_order=self.chroms,
                     key_added=None)

    def run(self):
        """Run the full pipeline: extract, merge, and clean up temp files.

        Calls :meth:`writePattern` then :meth:`merge`, and removes the
        temp directory ``<output>.tmp`` unless ``keep_temp=True``.
        """
        self.writePattern()
        self.merge()
        if not self.keep_temp:
            os.system(f"rm -rf {self.outdir}")


def allc2cz(input, output, reference=None, missing_value=[0, 0],
           formats=['B', 'B'], columns=['mc', 'cov'], chunk_dims=['chrom'],
           usecols=[4, 5], ref_pos_col=0, allc_pos_col=1, sep='\t', chroms=None,
           batch_size=5000, sort_col=None, delta_cols=None,
           jobs=1, pattern='*.allc.tsv.gz', skip_existing=True,
           _ref_pos_dict=None):
    """
    convert allc.tsv.gz to .cz file.

    When ``input`` is a directory, ALL matching allc files in the directory
    are converted in parallel. The reference .cz (if given) is decoded once
    in the parent process and shared with workers via fork-based copy-on-write,
    so the reference memory cost is paid only once regardless of ``jobs``.

    Parameters
    ----------
    input : path
        One of: (a) a single ``allc.tsv.gz`` (must have a ``.tbi`` index);
        (b) a directory containing many ``allc.tsv.gz`` files (batch mode);
        (c) an *allc_path table* — a headerless text file whose first column
        is the cell ID and second column is the path to that cell's
        ``allc.tsv.gz`` (batch mode; each output is named ``<cell_id>.cz``).
    output : path
        Output .cz file (single-file mode), or output directory (batch mode:
        directory or allc_path table input).
    reference : path
        path to reference coordinates.
    jobs : int
        Number of parallel worker processes for batch mode (default: 1).
        Ignored when ``input`` is a single file.
    pattern : str
        Glob pattern used to discover allc files when ``input`` is a directory
        (default: ``'*.allc.tsv.gz'``). Ignored for allc_path table input.
    skip_existing : bool
        In batch mode, skip files whose output .cz already exists (default: True).
    formats: list
        When reference is provided, we only need to pack mc and cov,
        ['H', 'H'] is suggested for pseudobulk data (H is unsigned short integer, only 2 bytes),
        and ['B', 'B'] is suggested for single cell data (B is unsigned char, only 1 byte).
        if reference is not provided, we also need to pack position (Q is
        recommanded), in this case, formats should be ['Q','H','H'].
    columns: list
        columns names, in default is ['mc','cov'] (reference is provided), if no
        referene provided, one should use ['pos','mc','cov'].
    chunk_dims: list
        chunk_dims passed to cytozip.Writer, chunk_key name, for allc file, chunk_key
        is chrom.
    usecols: list
        default is [4, 5], for a typical .allc.tsv.gz, if no reference is provided,
        the columns to be packed should be [1,4,5] (pos, mv and cov).
        If reference is provided, then we only need to pack [4,5] (mc and cov).
    ref_pos_col: int
        index of position column in reference .cz header columns [0]
    allc_pos_col: int
        index of position column in input input or bed column.
    batch_size : int
        default is 5000
    chroms : path
        path to chrom_size path or similar file containing chromosomes order,
        the first columns should be chromosomes, tab separated and no header.
    missing_value : list, optional
        Values to fill (mc, cov) at reference positions absent from the
        input allc. Default ``[0, 0]``. Only used when ``reference`` is
        provided.
    sep : str, optional
        Column separator for the input allc file (default ``'\t'``).
    sort_col : None / int / str / False, optional
        Forwarded to :class:`cytozip.cz.Writer`. Selects the column whose
        per-block first values are recorded for O(log N) coordinate
        binary search. ``None`` auto-detects (the integer ``pos`` column
        when present); pass ``False`` to disable.
    delta_cols : None / int / str / list, optional
        Forwarded to :class:`cytozip.cz.Writer`. Columns to store with
        in-block delta encoding (typically ``'pos'`` for tighter
        compression of monotonic positions). ``None`` (default) disables
        delta encoding.
    _ref_pos_dict : dict, optional
        Internal: pre-decoded ``{chrom: pos_array}`` dict shared from the
        parent process in batch mode so reference decoding is paid once.
        Not intended to be set by end users.

    Returns
    -------
    None
        The result is written to ``output`` (a single ``.cz`` file in
        single-file mode, or a directory of ``.cz`` files in batch mode).
        Nothing is returned; existing outputs are skipped.
    """
    # ---- Batch mode: input is a directory --------------------------------
    if isinstance(input, str) and os.path.isdir(os.path.expanduser(input)):
        return _allc2cz_batch(
            input_dir=input, output_dir=output, reference=reference,
            missing_value=missing_value, formats=formats, columns=columns,
            chunk_dims=chunk_dims, usecols=usecols,
            ref_pos_col=ref_pos_col, allc_pos_col=allc_pos_col, sep=sep,
            chroms=chroms, batch_size=batch_size,
            sort_col=sort_col, delta_cols=delta_cols,
            jobs=jobs, pattern=pattern, skip_existing=skip_existing,
        )
    # ---- Batch mode: input is an allc_path table (cell_id, allc_path) -----
    if isinstance(input, str) and os.path.isfile(os.path.expanduser(input)) \
            and _looks_like_allc_path_table(os.path.expanduser(input)):
        return _allc2cz_batch(
            input_dir=None, output_dir=output, reference=reference,
            files_map=_read_allc_path_table(input),
            missing_value=missing_value, formats=formats, columns=columns,
            chunk_dims=chunk_dims, usecols=usecols,
            ref_pos_col=ref_pos_col, allc_pos_col=allc_pos_col, sep=sep,
            chroms=chroms, batch_size=batch_size,
            sort_col=sort_col, delta_cols=delta_cols,
            jobs=jobs, pattern=pattern, skip_existing=skip_existing,
        )
    if os.path.exists(output):
        logger.info(f"{output} existed, skip.")
        return
    allc_path = os.path.abspath(os.path.expanduser(input))
    if not os.path.exists(allc_path + '.tbi'):
        raise ValueError("index file .tbi not existed, please create index first.")
    logger.info(allc_path)
    import pysam
    tbi = pysam.TabixFile(allc_path)
    contigs = tbi.contigs
    if not chroms is None:
        chroms = os.path.abspath(os.path.expanduser(chroms))
        df = pd.read_csv(chroms, sep='\t', header=None, usecols=[0])
        chrom_list = df.iloc[:, 0].tolist()
        all_chroms = [c for c in chrom_list if c in contigs]
    else:
        all_chroms = contigs
    if not reference is None:
        reference = os.path.abspath(os.path.expanduser(reference))
        message = os.path.basename(reference)
    else:
        message = ''
    # When the .cz stores coordinates itself (no reference), auto-enable
    # the first_coords index on the 'pos' column so region queries use
    # true in-memory O(log N) bisect. User can override via sort_col=.
    if sort_col is None and reference is None and 'pos' in columns:
        sort_col = 'pos'
    # Auto-enable DELTA on the 'pos' column when the file stores its own
    # coordinates. Sorted positions compress ~4-5x tighter as in-block
    # deltas after DEFLATE. User can override via delta_cols=.
    if delta_cols is None and reference is None and 'pos' in columns:
        delta_cols = ['pos']
    writer = Writer(output, formats=formats, columns=columns,
                    chunk_dims=chunk_dims, message=message,
                    sort_col=sort_col, delta_cols=delta_cols)
    unit_size = writer._unit_size
    use_numpy = _all_numeric_formats(formats)  # use vectorized numpy path if all columns are numeric

    if not reference is None:
        # When a pre-decoded ref_pos dict is supplied (batch mode), we can
        # skip opening the reference Reader entirely — every worker shares
        # the same numpy arrays via fork copy-on-write.
        need_ref_reader = _ref_pos_dict is None
        ref_reader = None
        ref_record_dtype = None
        if need_ref_reader:
            ref_reader = Reader(reference)
            # Hint the kernel to evict pages we've already walked. Without
            # this, sequentially reading every chunk of a multi-GB ref
            # (e.g. mm10 at ~1.3 GB) would pin the whole file in our RSS.
            ref_reader.advise_sequential()
            # Build a numpy structured dtype from the reference file's column
            # formats so we can bulk-read reference positions via np.frombuffer
            # instead of iterating record-by-record in Python.
            ref_fmts = ref_reader.header['formats']
            ref_record_dtype = np.dtype(
                [(f'c{i}', _fmt_to_np_dtype(f[-1]) if _fmt_to_np_dtype(f[-1]) else f'S{struct.calcsize(f)}')
                 for i, f in enumerate(ref_fmts)]
            )
        if use_numpy:
            np_dtypes = [_fmt_to_np_dtype(f[-1]) for f in formats]
            # Build a structured dtype matching the Writer's struct layout
            # so packed output bytes are directly compatible.
            struct_dtype = np.dtype([(f'f{i}', dt) for i, dt in enumerate(np_dtypes)])
            mv_arr = np.array(tuple(missing_value), dtype=struct_dtype)  # template for missing values
            parse_cols = [allc_pos_col] + list(usecols)  # position col + data cols
            parse_dtypes = ['<i8'] + np_dtypes  # int64 for pos, user dtypes for data
            for chrom in all_chroms:
                # FAST PATH: bulk-read all reference positions for this chrom
                # as a numpy array, then use searchsorted for O(n log n)
                # alignment of query positions against reference positions.
                if _ref_pos_dict is not None:
                    ref_pos_arr = _ref_pos_dict.get(chrom)
                    if ref_pos_arr is None or ref_pos_arr.size == 0:
                        continue
                else:
                    raw = ref_reader.fetch_chunk_bytes(tuple([chrom]))
                    if not raw:
                        continue
                    ref_records = np.frombuffer(raw, dtype=ref_record_dtype)
                    ref_pos_arr = ref_records[f'c{ref_pos_col}'].astype(np.int64)
                    if ref_pos_arr.size == 0:
                        continue
                # Bulk-read allc query data
                lines = list(tbi.fetch(chrom))
                if not lines:
                    # No query data: write all missing values
                    out = np.full(ref_pos_arr.size, mv_arr, dtype=struct_dtype)
                    _write_np_chunks(writer, out, chrom, batch_size, unit_size)
                    continue
                # Vectorized line parsing via pd.read_csv
                parsed = _parse_tabix_lines(lines, parse_cols, parse_dtypes, sep)
                query_pos = parsed[0].astype(np.int64)
                query_cols = parsed[1:]
                # Vectorized matching: use searchsorted to find where each
                # query position falls in the sorted reference position array.
                # `valid` mask identifies which query positions have an exact
                # match in the reference.
                indices = np.searchsorted(ref_pos_arr, query_pos)
                indices_clipped = np.minimum(indices, ref_pos_arr.size - 1)
                valid = (indices < ref_pos_arr.size) & (ref_pos_arr[indices_clipped] == query_pos)
                # Build output array initialized to missing_value
                out = np.full(ref_pos_arr.size, mv_arr, dtype=struct_dtype)
                matched_ref_idx = indices_clipped[valid]
                for ci in range(len(usecols)):
                    out[f'f{ci}'][matched_ref_idx] = query_cols[ci][valid]
                _write_np_chunks(writer, out, chrom, batch_size, unit_size)
                # Done with this chrom's ref pages: hand them back to
                # the kernel so they don't accumulate in RSS.
                if need_ref_reader:
                    ref_reader.release_chunk(tuple([chrom]))
        else:
            # Fallback: non-numeric formats, use original per-row logic
            if not need_ref_reader:
                raise ValueError(
                    "Pre-loaded ref_pos_dict only supports numeric formats; "
                    "got non-numeric formats=%r" % (formats,))
            dtfuncs = get_dtfuncs(formats, tobytes=False)
            for chrom in all_chroms:
                ref_positions = ref_reader.__fetch__(tuple([chrom]), s=ref_pos_col, e=ref_pos_col + 1)
                records = tbi.fetch(chrom)
                rows_buf = []
                i = 0
                try:
                    row_query = next(records).rstrip('\n').split(sep)
                    row_query_pos = int(row_query[allc_pos_col])
                except StopIteration:
                    row_query = None
                    row_query_pos = None
                for ref_pos in ref_positions:
                    if row_query_pos is None or ref_pos[0] < row_query_pos:
                        rows_buf.append(tuple(missing_value))
                        i += 1
                    else:
                        if ref_pos[0] == row_query_pos:
                            vals = tuple(func(row_query[j]) for j, func in zip(usecols, dtfuncs))
                            rows_buf.append(vals)
                            i += 1
                        try:
                            row_query = next(records).rstrip('\n').split(sep)
                            row_query_pos = int(row_query[allc_pos_col])
                        except (StopIteration, ValueError, IndexError):
                            row_query_pos = None
                            break
                    if i > batch_size:
                        writer.write_chunk(_pack_chunk_data(rows_buf, writer), [chrom])
                        rows_buf, i = [], 0
                if row_query_pos is None:
                    for ref_pos in ref_positions:
                        rows_buf.append(tuple(missing_value))
                        i += 1
                        if i > batch_size:
                            writer.write_chunk(_pack_chunk_data(rows_buf, writer), [chrom])
                            rows_buf, i = [], 0
                if len(rows_buf) > 0:
                    writer.write_chunk(_pack_chunk_data(rows_buf, writer), [chrom])
        if need_ref_reader and ref_reader is not None:
            ref_reader.close()
    else:
        if use_numpy:
            np_dtypes = [_fmt_to_np_dtype(f[-1]) for f in formats]
            struct_dtype = np.dtype([(f'f{i}', dt) for i, dt in enumerate(np_dtypes)])
            for chrom in all_chroms:
                lines = list(tbi.fetch(chrom))
                if not lines:
                    continue
                # Vectorized line parsing via pd.read_csv
                parsed = _parse_tabix_lines(lines, list(usecols), np_dtypes, sep)
                n = len(lines)
                out = np.empty(n, dtype=struct_dtype)
                for ci in range(len(usecols)):
                    out[f'f{ci}'] = parsed[ci]
                _write_np_chunks(writer, out, chrom, batch_size, unit_size)
        else:
            dtfuncs = get_dtfuncs(formats, tobytes=False)
            for chrom in all_chroms:
                rows_buf = []
                i = 0
                for line in tbi.fetch(chrom):
                    values = line.rstrip('\n').split(sep)
                    vals = tuple(func(values[j]) for j, func in zip(usecols, dtfuncs))
                    rows_buf.append(vals)
                    i += 1
                    if i >= batch_size:
                        writer.write_chunk(_pack_chunk_data(rows_buf, writer), [chrom])
                        rows_buf, i = [], 0
                if len(rows_buf) > 0:
                    writer.write_chunk(_pack_chunk_data(rows_buf, writer), [chrom])
    writer.close()
    tbi.close()


# ==========================================================
# Batch allc2cz: process a directory in parallel with a shared reference.
# The reference is decoded once in the parent and inherited by workers via
# fork copy-on-write, so the per-process memory cost is paid only once.
# ==========================================================
_BATCH_REF_POS_DICT = None  # populated in parent before fork; inherited by workers


def _load_ref_pos_dict(reference, ref_pos_col=0):
    """Decode reference .cz once and return ``{chrom: int64 pos array}``.

    The arrays are sorted ascending (which is the on-disk layout for any
    reference produced by :class:`AllC`). Workers can then run
    ``np.searchsorted`` directly without re-opening the reference file.
    """
    ref_path = os.path.abspath(os.path.expanduser(reference))
    ref_reader = Reader(ref_path)
    ref_fmts = ref_reader.header['formats']
    ref_record_dtype = np.dtype(
        [(f'c{i}', _fmt_to_np_dtype(f[-1]) if _fmt_to_np_dtype(f[-1]) else f'S{struct.calcsize(f)}')
         for i, f in enumerate(ref_fmts)]
    )
    chroms = []
    seen = set()
    for dims_key in ref_reader._raw_chunk_index.keys():
        c = dims_key[0]
        if c not in seen:
            seen.add(c)
            chroms.append(c)
    ref_pos = {}
    for chrom in chroms:
        raw = ref_reader.fetch_chunk_bytes(tuple([chrom]))
        if not raw:
            continue
        recs = np.frombuffer(raw, dtype=ref_record_dtype)
        if recs.size == 0:
            continue
        ref_pos[chrom] = recs[f'c{ref_pos_col}'].astype(np.int64)
        ref_reader.release_chunk(tuple([chrom]))
    ref_reader.close()
    return ref_pos


def _allc2cz_worker(args):
    """Pool worker: convert a single allc file using the inherited ref dict."""
    inp, outp, kwargs = args
    try:
        allc2cz(inp, outp, _ref_pos_dict=_BATCH_REF_POS_DICT, **kwargs)
        return (inp, True, None)
    except Exception:
        import traceback
        return (inp, False, traceback.format_exc())


def _strip_allc_suffix(basename):
    """Strip a common allc/tsv suffix from ``basename`` to build the output stem.

    Tries the known allc extensions (``.allc.tsv.gz``, ``.allc.tsv.bgz``,
    ``.tsv.gz``, ``.allc.gz``) in order and falls back to
    :func:`os.path.splitext` when none match. Used in batch mode to name
    each ``<stem>.cz`` output.
    """
    for suf in ('.allc.tsv.gz', '.allc.tsv.bgz', '.tsv.gz', '.allc.gz'):
        if basename.endswith(suf):
            return basename[:-len(suf)]
    return os.path.splitext(basename)[0]


def _read_allc_path_table(path):
    """Parse an *allc_path table* into ``[(cell_id, allc_path), ...]``.

    The table is a headerless text file whose first column is the cell ID
    and second column is the path to that cell's ``allc.tsv.gz``. Columns
    are tab-separated (falls back to any whitespace). ``#`` comment lines
    are ignored.
    """
    path = os.path.abspath(os.path.expanduser(path))
    df = pd.read_csv(path, sep='\t', header=None, comment='#')
    if df.shape[1] < 2:
        df = pd.read_csv(path, sep=r'\s+', header=None, comment='#',
                         engine='python')
    if df.shape[1] < 2:
        raise ValueError(
            f"allc_path table {path!r} must have >=2 columns "
            f"(cell_id, allc_path); got {df.shape[1]}")
    return [(str(row[0]), str(row[1]))
            for row in df.iloc[:, :2].itertuples(index=False)]


def _looks_like_allc_path_table(path):
    """Heuristically decide whether ``path`` is an allc_path table.

    True when ``path`` is a plain-text (non-bgzipped) file with no ``.tbi``
    index whose first non-comment line has >=2 columns and whose second
    column points to an existing file. This distinguishes a
    ``(cell_id, allc_path)`` table from a single ``allc.tsv.gz``.
    """
    if path.endswith(('.gz', '.bgz', '.cz')):
        return False
    if os.path.exists(path + '.tbi'):
        return False
    try:
        with open(path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) < 2:
                    return False
                return os.path.exists(os.path.expanduser(parts[1]))
    except (UnicodeDecodeError, OSError):
        return False
    return False


def _allc2cz_batch(input_dir, output_dir, reference=None, jobs=1,
                   pattern='*.allc.tsv.gz', skip_existing=True,
                   files_map=None, **kwargs):
    """Parallel allc -> cz over a directory (or an explicit table), sharing
    the decoded reference.

    Workers are forked from the parent so the pre-decoded reference dict is
    shared via copy-on-write (zero extra RSS per worker on Linux).

    ``files_map`` (optional) is a list of ``(cell_id, allc_path)`` pairs from
    an *allc_path table*; when given the output for each cell is named
    ``<output_dir>/<cell_id>.cz`` and directory globbing is skipped.
    """
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    if files_map is not None:
        # Explicit (cell_id, allc_path) list from an allc_path table.
        out_by_inp = []
        for cell_id, inp in files_map:
            inp = os.path.abspath(os.path.expanduser(str(inp)))
            if not os.path.exists(inp + '.tbi'):
                logger.warning(f"skip {cell_id}: no .tbi index for {inp}")
                continue
            out_by_inp.append(
                (inp, os.path.join(output_dir, str(cell_id) + '.cz')))
        if not out_by_inp:
            logger.warning("No allc files (with .tbi) found in the provided "
                           "allc_path table.")
            return
    else:
        import glob
        input_dir = os.path.abspath(os.path.expanduser(input_dir))
        files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        files = [f for f in files if os.path.exists(f + '.tbi')]
        if not files:
            logger.warning(f"No allc files matching {pattern!r} (with .tbi) found in {input_dir}")
            return
        out_by_inp = [
            (inp, os.path.join(output_dir,
                               _strip_allc_suffix(os.path.basename(inp)) + '.cz'))
            for inp in files]

    files = [inp for inp, _ in out_by_inp]
    job_args = []
    for inp, outp in out_by_inp:
        if skip_existing and os.path.exists(outp):
            logger.info(f"{outp} existed, skip.")
            continue
        job_args.append((inp, outp, dict(reference=reference, **kwargs)))

    if not job_args:
        logger.info("Nothing to do (all outputs already exist).")
        return

    logger.info(f"Found {len(files)} allc files; {len(job_args)} to convert; jobs={jobs}")

    # Pre-load reference once in the parent so workers share it via fork COW.
    global _BATCH_REF_POS_DICT
    _BATCH_REF_POS_DICT = None
    if reference is not None:
        ref_pos_col = kwargs.get('ref_pos_col', 0)
        logger.info(f"Pre-loading reference into shared memory: {reference}")
        _BATCH_REF_POS_DICT = _load_ref_pos_dict(reference, ref_pos_col=ref_pos_col)
        n_sites = sum(a.size for a in _BATCH_REF_POS_DICT.values())
        logger.info(
            f"Reference loaded: {len(_BATCH_REF_POS_DICT)} chroms, "
            f"{n_sites:,} sites (~{n_sites * 8 / 1e9:.2f} GB int64, shared via COW)"
        )

    try:
        if jobs <= 1 or len(job_args) == 1:
            for args in job_args:
                inp, ok, err = _allc2cz_worker(args)
                if not ok:
                    logger.error(f"failed: {inp}\n{err}")
            return

        try:
            ctx = multiprocessing.get_context('fork')
        except ValueError:
            logger.warning("fork start method unavailable; falling back to default "
                           "(reference will not be shared across workers).")
            ctx = multiprocessing.get_context()

        n_done = 0
        n_fail = 0
        with ctx.Pool(processes=int(jobs)) as pool:
            for inp, ok, err in pool.imap_unordered(_allc2cz_worker, job_args):
                if ok:
                    n_done += 1
                    logger.info(f"[{n_done}/{len(job_args)}] done: {os.path.basename(inp)}")
                else:
                    n_fail += 1
                    logger.error(f"failed: {inp}\n{err}")
        logger.info(f"Batch finished: {n_done} ok, {n_fail} failed.")
    finally:
        # Drop reference from parent so subsequent calls start fresh.
        _BATCH_REF_POS_DICT = None


# ==========================================================
def _extractcg_chunk_worker(args):
    """Extract (and optionally CG-merge) one chunk's CG records.

    Returns ``(dim, data_bytes)``. ``dim`` is the FULL input chunk key so a
    catted file's ``cell_id`` axis is preserved; the CGN index is looked up
    by chromosome only.
    """
    (input_path, index_path, dim, chrom_axis, merge_cg, vec_merge) = args
    reader = _cz._worker_reader(input_path)
    index_reader = _cz._worker_reader(index_path)
    idx_key = (dim[chrom_axis],)
    if idx_key not in index_reader.chunk_key2offset:
        return dim, None
    IDs = index_reader.get_ids_from_index(idx_key)
    if len(IDs.shape) != 1:
        raise ValueError("Only support 1D index now!")
    formats = reader.header['formats']
    fmts = reader.fmts
    # for CG, if pos is forward (+), then pos+1 is reverse strand (-)
    if vec_merge:
        if IDs.size == 0:
            return dim, b''
        # Gather the indexed rows once, then sum each forward/reverse pair
        # column-wise with a single vectorized clip.
        full = reader.chunk2numpy(dim)
        arr = full[IDs - 1]
        m = arr.shape[0] // 2  # number of (forward, reverse) pairs
        if m == 0:
            return dim, b''
        _rec_dtype = np.dtype([
            (f'f{i}', _fmt_to_np_dtype(f[-1]))
            for i, f in enumerate(formats)])
        fwd = arr[:2 * m:2]
        rev = arr[1:2 * m:2]
        out = np.empty(m, dtype=_rec_dtype)
        for i in range(len(formats)):
            fn = f'f{i}'
            dt = _rec_dtype[fn]
            summed = fwd[fn].astype(np.int64) + rev[fn].astype(np.int64)
            info = np.iinfo(dt)
            out[fn] = np.clip(summed, info.min, info.max).astype(dt)
        return dim, out.tobytes()
    records = reader._getRecordsByIds(dim, IDs)
    data_parts = []
    if merge_cg:
        dtfuncs = get_dtfuncs(formats)
        v0 = None
        for i, record in enumerate(records):  # unpacked bytes
            if i % 2 == 0:
                v0 = struct.unpack(f"<{fmts}", record)
            else:
                v1 = struct.unpack(f"<{fmts}", record)
                values = [r1 + r2 for r1, r2 in zip(v0, v1)]
                data_parts.append(struct.pack(fmts,
                                    *[func(v) for v, func in zip(values, dtfuncs)]))
    else:
        for record in records:  # unpacked bytes
            data_parts.append(record)
    return dim, b''.join(data_parts)


def extractCG(input=None, output=None, index=None, batch_size=5000,
              merge_cg=False, jobs=1):
    """
    Extract CG context from .cz file

    Supports both a single per-cell ``.cz`` (``chunk_dims=['chrom']``) and a
    ``catcz``'d multi-cell ``.cz`` (``chunk_dims=['chrom', 'cell_id']``). The
    CGN ``index`` is keyed by chromosome only; for a catted input each
    ``(chrom, cell_id)`` chunk is extracted against the chromosome's index
    and written back under its full key, preserving the per-cell structure.

    Parameters
    ----------
    input : path
        path to the .cz file.
    output : path
        output file path.
    index : path
        index should be index to mm10_with_chrL.allc.cz.CGN.index, not forward
        strand index, but after merge (if merge_cg is True), forward index
        mm10_with_chrL.allc.cz.+CGN.index should be used to generate
        reference, one can
        run: ``cytozip extract -m mm10_with_chrL.allc.cz
        -o mm10_with_chrL.allCG.forward.cz
        -b mm10_with_chrL.allc.cz.+CGN.index`` and use
        mm10_with_chrL.allCG.forward.cz as new reference.
    batch_size : int
    merge_cg : bool
        after merging, only forward strand would be kept, reverse strand values
        would be added to the corresponding forward strand.
    jobs : int
        Number of parallel worker processes across chunks (default 1). A
        catted file has ``n_chroms * n_cells`` chunks, so ``jobs > 1`` gives
        a near-linear speed-up on multi-cell inputs.

    Returns
    -------
    None
        The extracted (and optionally CG-merged) records are written to
        ``output`` as a new ``.cz`` file. Nothing is returned.
    """
    cz_path = os.path.abspath(os.path.expanduser(input))
    index_path = os.path.abspath(os.path.expanduser(index))
    reader = Reader(cz_path)
    chrom_axis = _chrom_axis(reader)
    # Vectorized merge_cg fast path is only exact when every column is an
    # unsigned integer (np.iinfo(dtype).max then equals the clamp used by
    # ``int_func``). methylation count columns (mc/cov) are unsigned, so this
    # covers the real use case; anything else falls back to the per-record loop.
    _vec_merge = merge_cg and all(
        f[-1] in 'BHILQ' for f in reader.header['formats'])
    writer = Writer(output, formats=reader.header['formats'],
                    columns=reader.header['columns'],
                    chunk_dims=reader.header['chunk_dims'],
                    message=index_path)
    dims = list(reader.chunk_key2offset.keys())
    reader.close()
    tasks = [(cz_path, index_path, dim, chrom_axis, merge_cg, _vec_merge)
             for dim in dims]
    if jobs and int(jobs) > 1 and len(tasks) > 1:
        with multiprocessing.Pool(int(jobs)) as pool:
            for dim, data in pool.imap_unordered(_extractcg_chunk_worker, tasks):
                if data:
                    writer.write_chunk(data, dim)
    else:
        try:
            for t in tasks:
                dim, data = _extractcg_chunk_worker(t)
                if data:
                    writer.write_chunk(data, dim)
        finally:
            _cz._close_worker_readers()
    writer.close()

# ==========================================================
_ALLC_COLS = ['chrom', 'pos', 'strand', 'context', 'mc', 'cov', 'methylated']


def _read_allc_table(path, sep='\t'):
    """Read an allc.tsv[.gz] file into a DataFrame with standard columns."""
    path = os.path.abspath(os.path.expanduser(path))
    return pd.read_csv(path, sep=sep, header=None, names=_ALLC_COLS,
                       dtype={'chrom': str, 'pos': 'int64',
                              'strand': str, 'context': str,
                              'mc': 'int64', 'cov': 'int64',
                              'methylated': 'int8'})


# Per-format saturation caps for unsigned-integer struct formats.
_DTYPE_MAX = {'B': 2 ** 8 - 1, 'H': 2 ** 16 - 1, 'I': 2 ** 32 - 1, 'Q': 2 ** 64 - 1}


def compare_allc(allc1, allc2, drop_zero_cov=True, dtype='B',
                 output=None, sep='\t'):
    """
    Compare two allc.tsv[.gz] files and report per-position differences.

    Records are joined on ``(chrom, pos)``. Optionally, records whose
    coverage is 0 are dropped, and ``mc`` / ``cov`` values are truncated
    (clamped) to the maximum value representable by ``dtype`` before
    comparison — this mimics the saturation applied when packing allc
    values into fixed-width unsigned-integer .cz columns.

    Parameters
    ----------
    allc1 : path
        Path to the first allc.tsv[.gz] file.
    allc2 : path
        Path to the second allc.tsv[.gz] file.
    drop_zero_cov : bool, optional
        If True (default), drop records with ``cov == 0`` from both files
        before comparison.
    dtype : {'B', 'H', 'I', 'Q'} or None, optional
        Unsigned-integer struct format used to derive the clamp bound for
        ``mc`` / ``cov``: ``'B'`` -> 255, ``'H'`` -> 65535,
        ``'I'`` -> 4294967295, ``'Q'`` -> 2**64-1 (default ``'B'``).
        Values greater than the derived maximum are clamped to it. Pass
        ``None`` to disable clamping.

        With the default ``'B'``, any ``mc`` or ``cov`` value greater than
        255 in the input allc file is truncated (saturated) to 255 before
        comparison. This is intentional: cytozip stores single-cell ``mc`` /
        ``cov`` as one-byte unsigned integers (``B``) to save space, so the
        packed .cz file itself caps these values at 255. In single-cell
        data, an ``mc`` or ``cov`` above 255 is almost always an artifact
        (e.g. reads piling up in a repeat region) rather than a genuine
        signal. Downstream tools reinforce this: ALLCools' DMR analysis
        further clips coverage to 50, so pre-truncating to 255 here has no
        effect on downstream results. Applying the same 255 cap to both
        files therefore lets ``compare_allc`` faithfully reproduce what the
        packed .cz would contain, so it reports only differences that
        actually survive the format's saturation. Use a wider format
        (``'H'`` / ``'I'`` / ``'Q'``) or ``None`` if you want to compare the
        raw, un-truncated values instead.
    output : path, optional
        If given, write the table of differing rows to this path
        (tab-separated). Default None (do not write).
    sep : str, optional
        Column separator of the input allc files (default ``'\\t'``).

    Returns
    -------
    pandas.DataFrame
        Rows where the two files differ, joined on ``(chrom, pos)`` with
        ``_1`` / ``_2`` suffixes. A row is considered different when it is
        present in only one file, or when ``mc`` / ``cov`` differ.
    """
    max_value = None
    if dtype is not None:
        if dtype not in _DTYPE_MAX:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_MAX)}, got {dtype!r}")
        max_value = _DTYPE_MAX[dtype]

    df1 = _read_allc_table(allc1, sep=sep)
    df2 = _read_allc_table(allc2, sep=sep)

    if drop_zero_cov:
        df1 = df1[df1['cov'] > 0].copy()
        df2 = df2[df2['cov'] > 0].copy()

    if max_value is not None:
        for df in (df1, df2):
            df['mc'] = df['mc'].clip(upper=max_value)
            df['cov'] = df['cov'].clip(upper=max_value)

    merged = df1.merge(df2, on=['chrom', 'pos'], how='outer',
                       suffixes=('_1', '_2'), indicator=True)

    only_in_1 = merged['_merge'] == 'left_only'
    only_in_2 = merged['_merge'] == 'right_only'
    both = merged['_merge'] == 'both'
    mc_diff = both & (merged['mc_1'] != merged['mc_2'])
    cov_diff = both & (merged['cov_1'] != merged['cov_2'])

    diff_mask = only_in_1 | only_in_2 | mc_diff | cov_diff
    diff = merged[diff_mask].copy()

    logger.info(
        f"compare_allc: {len(df1)} records in allc1, {len(df2)} in allc2; "
        f"only_in_1={int(only_in_1.sum())}, only_in_2={int(only_in_2.sum())}, "
        f"mc_diff={int(mc_diff.sum())}, cov_diff={int(cov_diff.sum())}, "
        f"total_diff={len(diff)}")

    if output is not None:
        output = os.path.abspath(os.path.expanduser(output))
        diff.to_csv(output, sep='\t', index=False)
        logger.info(f"Differences written to {output}")

    return diff

if __name__ == "__main__":
    from cytozip import main
    main()
