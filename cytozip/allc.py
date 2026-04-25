#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
allc.py — DNA methylation allc-file I/O built on top of the cytozip format.

This module provides:
  - :class:`AllC`: Extract all C (cytosine) positions from a reference
           genome and store them as a .cz coordinate file.
  - :func:`allc2cz`: Convert allc.tsv.gz (tabix-indexed) to .cz format,
             optionally using a reference .cz for coordinate alignment.
  - :func:`index_context`: Build a context-based coordinate index
             (CGN / CHN / +CGN) for a reference .cz by pattern matching.
  - :func:`extractCG`: Extract CG-context records from a full .cz file.

Sibling modules:
  - :mod:`cytozip.cz`:     generic .cz format (Reader / Writer / extract
                           / index_regions / aggregate).
  - :mod:`cytozip.merge`:  per-cell .cz merging (``merge_cz``,
                           ``merge_cell_type``, Fisher-test mode).
  - :mod:`cytozip.dmr`:    peak calling (``call_peaks``, ``to_bedgraph``)
                           and DMR analysis (``combp``, ``annot_dmr``).

@author: DingWB
"""
import itertools
import os
import sys
import struct
import multiprocessing
from .cz import (Reader, Writer, get_dtfuncs,
                 _STRUCT_TO_NP_DTYPE, _fmt_to_np_dtype,
                 _all_numeric_formats, _pack_chunk_data,
                 _write_np_chunks, _parse_tabix_lines,
                 _isCG, _isForwardCG, _isCH,
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
    """
    chrom = record.id
    output = os.path.join(outdir, chrom + ".cz")
    if os.path.exists(output):
        print(f"{output} existed, skip.")
        return None
    print(chrom)
    writer = Writer(output, formats=['Q', 'c', '3s'],
                    columns=['pos', 'strand', 'context'],
                    chunk_keys=['chrom'], sort_col='pos',
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
                 pattern="C", threads=12, keep_temp=False, delta=True):
        """
        Extract position of specific pattern in the reference genome, for example C.
            Example: python ~/Scripts/python/tbmate.py AllC -g ~/genome/hg38/hg38.fa --threads 10 run
            Or call within python: ac=AllC(genome="/gale/netapp/home2/wding/genome/hg38/hg38.fa")
        Parameters
        ----------
        genome: path
            reference genome.
        out: path
            path for output
        pattern: str
            pattern [C]
        threads: int
            number of CPU used for Pool.
        """
        self.genome=os.path.abspath(os.path.expanduser(genome))
        self.output=os.path.abspath(os.path.expanduser(output))
        self.outdir=self.output+'.tmp'
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.pattern=pattern
        from Bio import SeqIO
        self.records = SeqIO.parse(self.genome, "fasta")
        self.threads = threads if not threads is None else os.cpu_count()
        self.keep_temp = keep_temp
        # DELTA-encode the strictly-monotonic ``pos`` column by default.
        # Positions in a reference .cz are sorted and closely spaced (~3-10 bp
        # for CGN/CHN), so per-block deltas compress ~4-5x tighter than raw
        # 8-byte Q values after DEFLATE. Disable with ``delta=False`` to get
        # the fastest query path at the cost of ~2-3x larger files.
        self.delta_cols = ['pos'] if delta else None
        if pattern=='C':
            self.func=WriteC

    def writePattern(self):
        pool = multiprocessing.Pool(self.threads)
        jobs = []
        for record in self.records:
            job = pool.apply_async(self.func, (record, self.outdir, 5000, self.delta_cols))
            jobs.append(job)
        for job in jobs:
            job.get()
        pool.close()
        pool.join()

    def merge(self):
        writer = Writer(output=self.output, formats=['Q', 'c', '3s'],
                        columns=['pos', 'strand', 'context'],
                        chunk_keys=['chrom'], message=self.genome,
                        sort_col='pos', delta_cols=self.delta_cols)
        writer.catcz(input=f"{self.outdir}/*.cz")

    def run(self):
        self.writePattern()
        self.merge()
        if not self.keep_temp:
            os.system(f"rm -rf {self.outdir}")


def allc2cz(input, output, reference=None, missing_value=[0, 0],
           formats=['B', 'B'], columns=['mc', 'cov'], chunk_keys=['chrom'],
           usecols=[4, 5], ref_pos_col=0, allc_pos_col=1, sep='\t', chrom_order=None,
           batch_size=5000, sort_col=None, delta_cols=None):
    """
    convert allc.tsv.gz to .cz file.

    Parameters
    ----------
    input : path
        path to allc.tsv.gz, should has .tbi index.
    output : path
        output .cz file
    reference : path
        path to reference coordinates.
    formats: list
        When reference is provided, we only need to pack mc and cov,
        ['H', 'H'] is suggested (H is unsigned short integer, only 2 bytes),
        if reference is not provided, we also need to pack position (Q is
        recommanded), in this case, formats should be ['Q','H','H'].
    columns: list
        columns names, in default is ['mc','cov'] (reference is provided), if no
        referene provided, one should use ['pos','mc','cov'].
    chunk_keys: list
        chunk_keys passed to cytozip.Writer, chunk_key name, for allc file, chunk_key
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
    chrom_order : path
        path to chrom_size path or similar file containing chromosomes order,
        the first columns should be chromosomes, tab separated and no header.

    Returns
    -------

    """
    if os.path.exists(output):
        print(f"{output} existed, skip.")
        return
    allc_path = os.path.abspath(os.path.expanduser(input))
    if not os.path.exists(allc_path + '.tbi'):
        raise ValueError(f"index file .tbi not existed, please create index first.")
    print(allc_path)
    import pysam
    tbi = pysam.TabixFile(allc_path)
    contigs = tbi.contigs
    if not chrom_order is None:
        chrom_order = os.path.abspath(os.path.expanduser(chrom_order))
        df = pd.read_csv(chrom_order, sep='\t', header=None, usecols=[0])
        chroms = df.iloc[:, 0].tolist()
        all_chroms = [c for c in chroms if c in contigs]
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
                    chunk_keys=chunk_keys, message=message,
                    sort_col=sort_col, delta_cols=delta_cols)
    unit_size = writer._unit_size
    use_numpy = _all_numeric_formats(formats)  # use vectorized numpy path if all columns are numeric

    if not reference is None:
        ref_reader = Reader(reference)
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
        else:
            # Fallback: non-numeric formats, use original per-row logic
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
def index_context(input, output=None, pattern="CGN"):
    """
    Build a context-based coordinate index (1D) for a given input reference
    .cz file. The output file lists, per chromosome, the ``primary_id`` of
    every site whose context matches ``pattern`` (e.g. CGN / CHN / +CGN).

    Parameters
    ----------
    input : .cz
    output : .index
    pattern : CGN, CHN, +CGN, -CGN

    Returns
    -------

    """
    if pattern == 'CGN':
        judge_func = _isCG
    elif pattern == 'CHN':  # CH
        judge_func = _isCH
    elif pattern == '+CGN':
        judge_func = _isForwardCG
    else:
        raise ValueError("Currently, only CGN, CHN, +CGN supported")
    if output is None:
        output = input + '.' + pattern + '.index'
    else:
        output = os.path.abspath(os.path.expanduser(output))
    reader = Reader(input)
    reader.build_context_index(output=output, formats=['I'], columns=['ID'],
                        chunk_keys=['chrom'], match_func=judge_func,
                        batch_size=2000)
    reader.close()


# ==========================================================
def extractCG(input=None, output=None, index=None, batch_size=5000,
              merge_cg=False):
    """
    Extract CG context from .cz file

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

    Returns
    -------

    """
    cz_path = os.path.abspath(os.path.expanduser(input))
    index_path = os.path.abspath(os.path.expanduser(index))
    index_reader = Reader(index_path)
    reader = Reader(cz_path)
    writer = Writer(output, formats=reader.header['formats'],
                    columns=reader.header['columns'],
                    chunk_keys=reader.header['chunk_keys'],
                    message=index_path)
    dtfuncs = get_dtfuncs(writer.formats)
    for dim in reader.chunk_key2offset.keys():
        # print(dim)
        IDs = index_reader.get_ids_from_index(dim)
        if len(IDs.shape) != 1:
            raise ValueError("Only support 1D index now!")
        records = reader._getRecordsByIds(dim, IDs)
        data_parts, count = [], 0
        # for CG, if pos is forward (+), then pos+1 is reverse strand (-)
        if merge_cg:
            for i, record in enumerate(records):  # unpacked bytes
                if i % 2 == 0:
                    v0 = struct.unpack(f"<{reader.fmts}", record)
                else:
                    v1 = struct.unpack(f"<{reader.fmts}", record)
                    values = [r1 + r2 for r1, r2 in zip(v0, v1)]
                    data_parts.append(struct.pack(writer.fmts,
                                        *[func(v) for v, func in zip(values, dtfuncs)]))
                    count += 1
                if count > batch_size:
                    writer.write_chunk(b''.join(data_parts), dim)
                    data_parts, count = [], 0
        else:
            for record in records:  # unpacked bytes
                data_parts.append(record)
                count += 1
                if count > batch_size:
                    writer.write_chunk(b''.join(data_parts), dim)
                    data_parts, count = [], 0
        if len(data_parts) > 0:
            writer.write_chunk(b''.join(data_parts), dim)
    writer.close()
    reader.close()
    index_reader.close()

if __name__ == "__main__":
    from cytozip import main
    main()
