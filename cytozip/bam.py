#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bam.py - Convert BAM files directly to .cz format (skipping ALLC text).

This module ports the core pileup-based methylation-call extraction from
ALLCools (`_bam_to_allc.py`, written by Yupeng He; itself derived from
methylpy) and pipes the results straight into a cytozip ``Writer`` instead
of writing ALLC tsv.gz + running tabix.

Storage layout options (``mode`` parameter)
-------------------------------------------
``mode="full"`` (default)
    Store ``[pos, strand, context, mc, cov]`` - fully self-contained file.
    formats = ``['Q', 'c', '3s', 'H', 'H']``.
``mode="pos_mc_cov"``
    Store ``[pos, mc, cov]`` - drops strand/context but keeps coordinates.
    formats = ``['Q', 'H', 'H']``. Downstream pipelines can join contexts
    from a reference .cz. Matches the "slim" layout used by ``allc2cz``.
``mode="mc_cov"``
    Store ``[mc, cov]`` only. **Requires ``reference``**: output records
    are aligned one-to-one with the reference .cz's positions; missing
    sites are filled with (0, 0). Smallest on-disk footprint (~4 B / site)
    and matches the reference-driven ``allc2cz`` layout.

Streaming layout of the produced ``.cz``:

* ``chunk_dims = ['chrom']``
* ``sort_col = 'pos'`` (enables O(log N) region query), only when pos is stored
* ``delta_cols = ['pos']`` (positions are monotonic within each chrom),
  only when pos is stored

@author: DingWB (port), original bam->pileup logic by Yupeng He (ALLCools).
"""
from __future__ import annotations

import os
import shlex
import struct
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

from . import cz as _cz_mod
from .cz import (
    Writer, Reader, _all_numeric_formats, _fmt_to_np_dtype,
    _ensure_cz_accel,
    _write_np_chunks,
)
# Trigger Cython accel load so ``_cz_mod._load_bcz_block`` is the C
# implementation (the symbol imported at module import time would be
# pinned to the pure-Python fallback).
_ensure_cz_accel()


# ---------------------------------------------------------------------------
# Helpers (ported / adapted from ALLCools._bam_to_allc)
# ---------------------------------------------------------------------------
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
# C-level translate table for reverse-complement of an ASCII string.
# ``seq.translate(_RC_TABLE)[::-1]`` is ~30x faster than the Python
# ``"".join(_COMPLEMENT[b] for b in reversed(seq))`` generator on the
# small (3-9 base) context windows used here, because the inner loop
# stays entirely in the CPython str fast path.
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")
_MC_SITES = frozenset({"C", "G"})

_VALID_MODES = ("full", "pos_mc_cov", "mc_cov")


def _read_faidx(faidx_path):
    return pd.read_csv(
        faidx_path, index_col=0, header=None, sep="\t",
        names=["NAME", "LENGTH", "OFFSET", "LINEBASES", "LINEWIDTH"],
    )


def _get_chromosome_sequence_upper(fasta_path, fai_df, query_chrom):
    """Load one chromosome's sequence from a fasta file using its .fai index."""
    chrom_pointer = fai_df.loc[query_chrom, "OFFSET"]
    tail = fai_df.loc[query_chrom, "LINEBASES"] - fai_df.loc[query_chrom, "LINEWIDTH"]
    seq_parts = []
    with open(fasta_path) as f:
        f.seek(chrom_pointer)
        for line in f:
            if line[0] == ">":
                break
            seq_parts.append(line[:tail])
    return "".join(seq_parts).upper()


def _strip_indels(read_bases: str) -> str:
    """Remove insertion/deletion operators from an mpileup bases string."""
    if ("+" not in read_bases) and ("-" not in read_bases):
        return read_bases
    out = []
    i = 0
    n = len(read_bases)
    while i < n:
        ch = read_bases[i]
        if ch == "+" or ch == "-":
            j = i + 1
            num_start = j
            while j < n and read_bases[j].isdigit():
                j += 1
            if j == num_start:
                i += 1
                continue
            size = int(read_bases[num_start:j])
            i = j + size
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _convert_bam_strandness(in_bam_path: str, out_bam_path: str) -> None:
    """Rewrite a bismark/hisat-3n BAM so `read.is_forward` matches the
    conversion type (XG/YZ tag). Required for hisat-3n PE / Biskarp PE.
    """
    import pysam
    with pysam.AlignmentFile(in_bam_path) as in_bam, \
            pysam.AlignmentFile(out_bam_path, header=in_bam.header, mode="wb") as out_bam:
        is_ct_func = None
        for read in in_bam:
            if is_ct_func is None:
                if read.has_tag("YZ"):
                    is_ct_func = lambda r: r.get_tag("YZ") == "+"
                elif read.has_tag("XG"):
                    is_ct_func = lambda r: r.get_tag("XG") == "CT"
                else:
                    raise ValueError(
                        "BAM reads lack conversion-type tag (XG/YZ). "
                        "Only bismark/hisat-3n BAMs are supported."
                    )
            ct = is_ct_func(read)
            read.is_forward = ct
            if read.is_paired:
                read.mate_is_forward = ct
            out_bam.write(read)


# ---------------------------------------------------------------------------
# Mode -> Writer layout
# ---------------------------------------------------------------------------
def _resolve_mode(mode):
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    return mode


_VALID_COUNT_FMTS = ("B", "H", "I", "Q")
_COUNT_FMT_MAX = {"B": 0xFF, "H": 0xFFFF, "I": 0xFFFFFFFF,
                  "Q": 0xFFFFFFFFFFFFFFFF}


def _layout_for_mode(mode, count_fmt="H"):
    if count_fmt not in _VALID_COUNT_FMTS:
        raise ValueError(
            f"count_fmt must be one of {_VALID_COUNT_FMTS}, got {count_fmt!r}"
        )
    cf = count_fmt
    if mode == "full":
        return (["Q", "c", "3s", cf, cf],
                ["pos", "strand", "context", "mc", "cov"],
                "pos", ["pos"])
    if mode == "pos_mc_cov":
        return (["Q", cf, cf], ["pos", "mc", "cov"], "pos", ["pos"])
    if mode == "mc_cov":
        return ([cf, cf], ["mc", "cov"], None, None)
    raise ValueError(mode)


class _LazyRefPositions:
    """On-demand loader for per-chrom position arrays from a reference .cz.

    Avoids preloading the entire genome (mm10 ≈ 1.1e9 C sites ≈ 9 GB as
    int64). Positions are decoded as ``uint32`` (fits any vertebrate
    chromosome) and cached per chrom; call :meth:`drop` after the chrom
    is flushed to release memory.
    """

    def __init__(self, reference):
        self._reader = Reader(reference)
        # Tell the kernel to evict pages we've already read; without
        # this hint the entire ref file (~1.3 GB for mm10) ends up
        # counted against our RSS as we walk every chrom.
        self._reader.advise_sequential()
        cols = self._reader.header["columns"]
        fmts = self._reader.header["formats"]
        if "pos" not in cols:
            raise ValueError(
                f"reference {reference} has no 'pos' column "
                f"(columns={cols}); cannot use mode='mc_cov'."
            )
        self._pos_i = cols.index("pos")
        self._record_dtype = np.dtype([
            (f"c{i}",
             _fmt_to_np_dtype(f[-1]) if _fmt_to_np_dtype(f[-1])
             else f"S{struct.calcsize(f)}")
            for i, f in enumerate(fmts)
        ])
        chunk_dims = self._reader.header["chunk_dims"]
        self._chrom_idx = len(chunk_dims) - 1
        # Map chrom -> chunk_dim_tuple for fast lookup.
        self._chrom2dim = {
            dim[self._chrom_idx]: dim
            for dim in self._reader.chunk_key2offset.keys()
        }
        self._cache: dict = {}

    def __contains__(self, chrom):
        return chrom in self._chrom2dim

    def get(self, chrom):
        """Return uint32 position array for *chrom* (loads on first call).

        Streams blocks from the chunk one at a time, extracting only the
        ``pos`` column into a pre-allocated uint32 array.  This avoids
        materialising the full decompressed chunk (which for mm10 chr1
        is ~12 B/record × 1.3e8 = ~1.5 GB of struct bytes that would
        only need ~520 MB once narrowed to uint32).
        """
        arr = self._cache.get(chrom)
        if arr is not None:
            return arr
        dim = self._chrom2dim.get(chrom)
        if dim is None:
            return None
        reader = self._reader
        if not reader._load_chunk(reader.chunk_key2offset[dim], jump=False):
            arr = np.empty(0, dtype=np.uint32)
            self._cache[chrom] = arr
            return arr
        n_records = reader._chunk_data_len // reader._unit_size
        if n_records == 0:
            arr = np.empty(0, dtype=np.uint32)
            self._cache[chrom] = arr
            return arr
        out = np.empty(n_records, dtype=np.uint32)
        pos_field = f"c{self._pos_i}"
        delta_pos_field = f"f{self._pos_i}"
        record_dtype = self._record_dtype
        delta_cols = reader._delta_cols
        delta_col_names = reader._delta_col_names if delta_cols else ()
        delta_np_dtype = reader._delta_np_dtype if delta_cols else None
        unit_size = reader._unit_size
        handle = reader._handle
        # Skip chunk magic (2B) + chunk_size (8B) — blocks start at +10.
        handle.seek(reader._chunk_start_offset + 10)
        write_idx = 0
        load_block = _cz_mod._load_bcz_block
        for _ in range(reader._chunk_nblocks):
            try:
                _bsize, blk = load_block(handle, True)
            except StopIteration:
                break
            if not blk:
                continue
            n_rec = len(blk) // unit_size
            if n_rec == 0:
                continue
            if delta_cols:
                # See note in iter_blocks(): only copy the pos column
                # (uint32) to avoid a per-block full-record copy.
                arr_blk = np.frombuffer(blk, dtype=delta_np_dtype, count=n_rec)
                delta_pos = arr_blk[delta_pos_field].astype(out.dtype, copy=True)
                del arr_blk
                if n_rec > 1:
                    np.cumsum(delta_pos, out=delta_pos)
                out[write_idx:write_idx + n_rec] = delta_pos
            else:
                rec_view = np.frombuffer(blk, dtype=record_dtype, count=n_rec)
                out[write_idx:write_idx + n_rec] = rec_view[pos_field]
            write_idx += n_rec
        if write_idx < n_records:
            out = out[:write_idx]
        self._cache[chrom] = out
        return out

    def drop(self, chrom):
        self._cache.pop(chrom, None)
        # Release the ref pages for this chunk back to the kernel.
        # Without this, walking ~1.3 GB of mmap'd ref keeps every
        # touched page in our RSS until the process exits.
        dim = self._chrom2dim.get(chrom)
        if dim is not None:
            self._reader.release_chunk(dim)

    def iter_blocks(self, chrom):
        """Yield successive uint32 position arrays, one per stored block.

        Avoids materialising the full chrom position array (which for
        mm10 chr1 is ~316 MB).  The blocks are yielded in genomic order,
        each block's positions are themselves sorted.

        Yields
        ------
        np.ndarray of uint32
        """
        dim = self._chrom2dim.get(chrom)
        if dim is None:
            return
        reader = self._reader
        if not reader._load_chunk(reader.chunk_key2offset[dim], jump=False):
            return
        n_records = reader._chunk_data_len // reader._unit_size
        if n_records == 0:
            return
        delta_pos_field = f"f{self._pos_i}"
        pos_field = f"c{self._pos_i}"
        record_dtype = self._record_dtype
        delta_cols = reader._delta_cols
        delta_col_names = reader._delta_col_names if delta_cols else ()
        delta_np_dtype = reader._delta_np_dtype if delta_cols else None
        unit_size = reader._unit_size
        handle = reader._handle
        handle.seek(reader._chunk_start_offset + 10)
        load_block = _cz_mod._load_bcz_block
        for _ in range(reader._chunk_nblocks):
            try:
                _bsize, blk = load_block(handle, True)
            except StopIteration:
                break
            if not blk:
                continue
            n_rec = len(blk) // unit_size
            if n_rec == 0:
                continue
            if delta_cols:
                # Read-only view over the block bytes; copy ONLY the
                # position column as uint32 to avoid a per-block
                # full-record copy (record dtype is ~3x larger).
                # cumsum runs in-place on the small uint32 array.
                arr_blk = np.frombuffer(blk, dtype=delta_np_dtype, count=n_rec)
                delta_pos = arr_blk[delta_pos_field].astype(np.uint32, copy=True)
                del arr_blk
                if n_rec > 1:
                    np.cumsum(delta_pos, out=delta_pos)
                yield delta_pos
            else:
                rec_view = np.frombuffer(blk, dtype=record_dtype, count=n_rec)
                yield rec_view[pos_field].astype(np.uint32, copy=True)

    def has(self, chrom):
        return chrom in self._chrom2dim

    def close(self):
        self._cache.clear()
        self._reader.close()


def _load_reference_positions(reference):
    """Return a lazy per-chrom position loader from a reference .cz.

    The previous implementation preloaded the entire genome as an
    ``{chrom: int64-array}`` dict, which costs ~9 GB on mm10. This
    function now returns a :class:`_LazyRefPositions` that loads each
    chromosome on first access (uint32) and lets the caller :meth:`drop`
    it after flushing.
    """
    return _LazyRefPositions(reference)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def bam_to_cz(
    bam_path: str,
    genome: str,
    output: Optional[str] = None,
    mode: str = "mc_cov",
    count_fmt: str = "B",
    reference: Optional[str] = None,
    num_upstr_bases: int = 0,
    num_downstr_bases: int = 2,
    min_mapq: int = 10,
    min_base_quality: int = 20,
    batch_size: int = 5000,
    convert_bam_strandness: bool = False,
    save_count_df: bool = False,
) -> Optional[pd.DataFrame]:
    """Convert a position-sorted BAM to a ``.cz`` methylation file.

    Parameters
    ----------
    bam_path : str
        Position-sorted BAM (requires ``.bai``; we will build it if missing).
    genome : str
        Indexed reference fasta (``.fai`` required).
    output : str, optional
        Output ``.cz`` path. Defaults to ``<bam_stem>.cz`` next to the BAM.
    mode : {"full", "pos_mc_cov", "mc_cov"}
        Storage layout. See module docstring.
    count_fmt : {"B", "H", "I", "L", "Q"}
        struct code used for the ``mc`` and ``cov`` columns. ``'B'`` (uint8,
        1 byte; max 255) is the most compact and is sufficient for typical
        single-cell bisulfite data where per-site coverage rarely exceeds
        a few tens. Values that exceed the chosen dtype max are clipped
        (with a one-time warning). Defaults to ``'H'`` (uint16, 2 bytes;
        max 65535) for safety.
    reference : str, optional
        Reference .cz (containing a ``pos`` column per chrom). **Required
        when ``mode='mc_cov'``** - output records are aligned one-to-one
        with this reference's positions; missing sites are filled with
        ``(0, 0)``.
    num_upstr_bases / num_downstr_bases : int
        Context window around each C. Typical: BS-seq (0, 2), NOMe-seq (1, 2).
    min_mapq, min_base_quality : int
        Passed straight to ``samtools mpileup``.
    batch_size : int
        Records per on-disk chunk (same semantics as ``allc2cz``).
    convert_bam_strandness : bool
        If True, rewrite the BAM so ``read.is_forward`` matches the
        conversion type (XG/YZ tag).
    save_count_df : bool
        If True, write a ``<output>.count.csv`` with total mC / cov per context.

    Returns
    -------
    pd.DataFrame or None
        Per-context mC / cov summary unless ``save_count_df=True``.
    """
    mode = _resolve_mode(mode)
    count_max = _COUNT_FMT_MAX[count_fmt] if count_fmt in _COUNT_FMT_MAX else 0xFFFF
    if mode == "mc_cov" and reference is None:
        raise ValueError(
            "mode='mc_cov' requires reference (positions are not stored "
            "in the output and must be recovered from the reference)."
        )

    if not os.path.exists(genome):
        raise FileNotFoundError(f"Reference fasta not found: {genome}")
    fai_path = genome + ".fai"
    if not os.path.exists(fai_path):
        raise FileNotFoundError(
            f"Reference fasta not indexed. Run `samtools faidx {genome}` first."
        )
    fai_df = _read_faidx(fai_path)

    if convert_bam_strandness:
        temp_bam = f"{bam_path}.strand.tmp.bam"
        _convert_bam_strandness(bam_path, temp_bam)
        bam_path = temp_bam

    if not os.path.exists(bam_path + ".bai"):
        subprocess.check_call(["samtools", "index", bam_path])

    if output is None:
        stem = os.path.basename(bam_path).split(".")[0]
        output = os.path.join(os.path.dirname(os.path.abspath(bam_path)),
                              stem + ".cz")

    # Writer layout
    formats, columns, sort_col, delta_cols = _layout_for_mode(mode, count_fmt)
    ref_pos_map = None
    if mode == "mc_cov":
        ref_pos_map = _load_reference_positions(reference)
        writer_message = os.path.basename(reference)
    else:
        writer_message = os.path.basename(genome)

    writer = Writer(
        output,
        formats=formats,
        columns=columns,
        chunk_dims=["chrom"],
        sort_col=sort_col,
        delta_cols=delta_cols,
        message=writer_message,
    )
    unit_size = writer._unit_size
    _ = _all_numeric_formats(formats)  # sanity check
    fmt_struct = struct.Struct("<" + "".join(formats))

    mpileup_cmd = (
        f"samtools mpileup -Q {min_base_quality} -q {min_mapq} -B "
        f"-f {genome} {bam_path}"
    )
    pipes = subprocess.Popen(
        shlex.split(mpileup_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    cur_chrom = ""
    seq = None
    context_len = num_upstr_bases + 1 + num_downstr_bases
    cov_dict: dict = {}
    mc_dict: dict = {}
    total_line = 0

    # Per-chrom buffers. For full / pos_mc_cov we flush incrementally in
    # batch_size chunks; for mc_cov we must buffer the whole chrom (we need
    # all observed positions before aligning against reference).
    #
    # Use ``array.array`` (packed C ints, 4 B / 2 B per element) instead of
    # Python lists, which would otherwise hold ~36 B per int object and
    # dominate the working-set on a per-cell run (a typical cell has
    # ~1.4e7 mC sites → ~1.5 GB just for the three buffers).
    import array as _array
    buf_records: list = []
    chrom_pos_buf = _array.array('I')          # uint32 positions
    chrom_mc_buf = _array.array('H')           # uint16 mc counts (already clipped)
    chrom_cov_buf = _array.array('H')          # uint16 cov counts

    _np_count_dtype = _fmt_to_np_dtype(count_fmt) or "<u2"
    mc_cov_struct_dtype = np.dtype([("mc", _np_count_dtype), ("cov", _np_count_dtype)])
    _overflow_warned = [False]

    def _flush_records(chrom: str) -> None:
        if not buf_records:
            return
        writer.write_chunk(b"".join(buf_records), [chrom])
        buf_records.clear()

    def _flush_mc_cov(chrom: str) -> None:
        if not ref_pos_map.has(chrom):
            del chrom_pos_buf[:]
            del chrom_mc_buf[:]
            del chrom_cov_buf[:]
            return

        # Stream the reference positions block-by-block and merge them
        # against the sorted observed positions. This avoids materialising
        # a single full uint32 ref_pos array (~316 MB on mm10 chr1).
        if len(chrom_pos_buf) > 0:
            q_pos = np.frombuffer(chrom_pos_buf, dtype=np.uint32)
            q_mc = np.frombuffer(chrom_mc_buf, dtype=np.uint16)
            q_cov = np.frombuffer(chrom_cov_buf, dtype=np.uint16)
            n_q = int(q_pos.size)
        else:
            q_pos = np.empty(0, dtype=np.uint32)
            q_mc = np.empty(0, dtype=np.uint16)
            q_cov = np.empty(0, dtype=np.uint16)
            n_q = 0

        q_ptr = 0
        any_block = False
        for ref_block in ref_pos_map.iter_blocks(chrom):
            any_block = True
            n_b = int(ref_block.size)
            if n_b == 0:
                continue
            # Advance q_ptr to first observed position that could match
            # this block (>= ref_block[0]).
            if q_ptr < n_q:
                q_ptr += int(np.searchsorted(q_pos[q_ptr:], ref_block[0],
                                              side="left"))
            # Find observed positions strictly less than the block end.
            block_end = ref_block[-1]
            q_end = q_ptr
            if q_ptr < n_q:
                q_end = q_ptr + int(np.searchsorted(
                    q_pos[q_ptr:], block_end, side="right"))
            out_batch = np.zeros(n_b, dtype=mc_cov_struct_dtype)
            if q_end > q_ptr:
                idx = np.searchsorted(ref_block, q_pos[q_ptr:q_end])
                # All idx are < n_b because q_pos values are <= block_end
                # which equals ref_block[-1].
                idx_clip = np.minimum(idx, n_b - 1)
                valid = ref_block[idx_clip] == q_pos[q_ptr:q_end]
                matched = idx_clip[valid]
                if matched.size:
                    out_batch["mc"][matched] = q_mc[q_ptr:q_end][valid].astype(
                        _np_count_dtype, copy=False)
                    out_batch["cov"][matched] = q_cov[q_ptr:q_end][valid].astype(
                        _np_count_dtype, copy=False)
                q_ptr = q_end
            writer.write_chunk(out_batch.tobytes(), [chrom])

        if not any_block:
            # Chrom exists in chrom2dim but had zero records; nothing to write.
            pass

        del q_pos, q_mc, q_cov
        del chrom_pos_buf[:]
        del chrom_mc_buf[:]
        del chrom_cov_buf[:]

    # Glibc keeps freed allocations in per-thread arenas / free-lists and
    # only releases them to the OS at MALLOC_TRIM_THRESHOLD_ (default 128 KB
    # for top-of-heap, but large mmap'd blocks may stay reserved).  After
    # each chrom flush we drop ref_pos (~hundreds of MB) and the per-chrom
    # buffers; calling ``malloc_trim(0)`` lets the OS reclaim those pages.
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
        _malloc_trim = _libc.malloc_trim
        _malloc_trim.argtypes = [ctypes.c_size_t]
        _malloc_trim.restype = ctypes.c_int
    except Exception:  # pragma: no cover - non-glibc / Windows
        _malloc_trim = None

    def _flush(chrom: str) -> None:
        if mode == "mc_cov":
            _flush_mc_cov(chrom)
            # Tell the kernel we're done with this chrom's slice of the
            # mmap'd ref file; without this the file pages keep
            # accumulating in our RSS as we walk every chrom.
            if ref_pos_map is not None:
                ref_pos_map.drop(chrom)
        else:
            _flush_records(chrom)
        # Force glibc to return free'd pages to the OS (best-effort).
        if _malloc_trim is not None:
            _malloc_trim(0)

    try:
        for line in pipes.stdout:
            total_line += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            ref_base = fields[2].upper()

            if fields[0] != cur_chrom:
                if cur_chrom:
                    # Release the previous chromosome's sequence string
                    # (≈100-200 MB for large chroms) BEFORE running its
                    # flush — otherwise the flush's transient allocations
                    # (ref_pos, searchsorted intermediates, out_batch)
                    # stack on top of it and inflate MaxRSS.
                    seq = None
                    _flush(cur_chrom)
                cur_chrom = fields[0]
                seq = _get_chromosome_sequence_upper(genome, fai_df, cur_chrom)

            if ref_base not in _MC_SITES:
                continue

            read_bases = fields[4]
            if ("+" in read_bases) or ("-" in read_bases):
                read_bases = _strip_indels(read_bases)

            pos0 = int(fields[1]) - 1  # mpileup is 1-based; 0-based for seq

            if ref_base == "C":
                lo = pos0 - num_upstr_bases
                hi = pos0 + num_downstr_bases + 1
                if lo < 0 or hi > len(seq):
                    continue
                context = seq[lo:hi]
                strand = b"+"
                unconverted = read_bases.count(".")
                converted = read_bases.count("T")
            else:  # ref_base == 'G'
                lo = pos0 - num_downstr_bases
                hi = pos0 + num_upstr_bases + 1
                if lo < 0 or hi > len(seq):
                    continue
                # Reverse-complement via C-level translate + slice reverse.
                context = seq[lo:hi].translate(_RC_TABLE)[::-1]
                strand = b"-"
                unconverted = read_bases.count(",")
                converted = read_bases.count("a")

            cov = unconverted + converted
            if cov == 0 or len(context) != context_len:
                continue

            # Context counters use raw (unclipped) values.
            cov_dict[context] = cov_dict.get(context, 0) + cov
            mc_dict[context] = mc_dict.get(context, 0) + unconverted

            # Clip to count_fmt range so struct.pack does not raise.
            if unconverted > count_max or cov > count_max:
                if not _overflow_warned[0]:
                    import warnings
                    warnings.warn(
                        f"mc/cov value exceeds count_fmt={count_fmt!r} max "
                        f"({count_max}); clipping. Consider count_fmt='H' "
                        "for bulk/high-coverage data.",
                        stacklevel=2,
                    )
                    _overflow_warned[0] = True
                if unconverted > count_max:
                    unconverted = count_max
                if cov > count_max:
                    cov = count_max

            pos1 = pos0 + 1
            if mode == "full":
                ctx_bytes = context.encode("ascii")[:3].ljust(3, b"N")
                buf_records.append(fmt_struct.pack(pos1, strand, ctx_bytes,
                                                   unconverted, cov))
                if len(buf_records) >= batch_size:
                    _flush_records(cur_chrom)
            elif mode == "pos_mc_cov":
                buf_records.append(fmt_struct.pack(pos1, unconverted, cov))
                if len(buf_records) >= batch_size:
                    _flush_records(cur_chrom)
            else:  # mc_cov: buffer whole chrom
                chrom_pos_buf.append(pos1)
                chrom_mc_buf.append(unconverted)
                chrom_cov_buf.append(cov)

        if cur_chrom:
            _flush(cur_chrom)
    finally:
        pipes.stdout.close()
        writer.close()
        if ref_pos_map is not None:
            ref_pos_map.close()
        if convert_bam_strandness:
            try:
                os.remove(bam_path)
                os.remove(bam_path + ".bai")
            except OSError:
                pass

    count_df = pd.DataFrame({"mc": mc_dict, "cov": cov_dict})
    if not count_df.empty:
        count_df["mc_rate"] = count_df["mc"] / count_df["cov"]
        total_genome_length = int(fai_df["LENGTH"].sum())
        count_df["genome_cov"] = total_line / max(total_genome_length, 1)

    if save_count_df:
        count_df.to_csv(output + ".count.csv")
        return None
    return count_df
