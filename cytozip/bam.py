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
    Store ``[mc, cov]`` only. **Requires ``reference_cz``**: output records
    are aligned one-to-one with the reference .cz's positions; missing
    sites are filled with (0, 0). Smallest on-disk footprint (~4 B / site)
    and matches the reference-driven ``allc2cz`` layout.

Streaming layout of the produced ``.cz``:

* ``chunk_keys = ['chrom']``
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

from .cz import (
    Writer, Reader, _all_numeric_formats, _fmt_to_np_dtype,
    _write_np_chunks,
)


# ---------------------------------------------------------------------------
# Helpers (ported / adapted from ALLCools._bam_to_allc)
# ---------------------------------------------------------------------------
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
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


def _load_reference_positions(reference_cz):
    """Return ``{chrom: np.ndarray(int64 positions)}`` from a reference .cz."""
    r = Reader(reference_cz)
    try:
        cols = r.header["columns"]
        fmts = r.header["formats"]
        if "pos" not in cols:
            raise ValueError(
                f"reference_cz {reference_cz} has no 'pos' column "
                f"(columns={cols}); cannot use mode='mc_cov'."
            )
        pos_i = cols.index("pos")
        ref_record_dtype = np.dtype([
            (f"c{i}",
             _fmt_to_np_dtype(f[-1]) if _fmt_to_np_dtype(f[-1])
             else f"S{struct.calcsize(f)}")
            for i, f in enumerate(fmts)
        ])
        out = {}
        chunk_keys = r.header["chunk_keys"]
        chrom_idx = len(chunk_keys) - 1
        for dim in r.chunk_key2offset.keys():
            chrom = dim[chrom_idx]
            raw = r.fetch_chunk_bytes(dim)
            if not raw:
                out[chrom] = np.empty(0, dtype=np.int64)
                continue
            arr = np.frombuffer(raw, dtype=ref_record_dtype)
            out[chrom] = arr[f"c{pos_i}"].astype(np.int64)
        return out
    finally:
        r.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def bam_to_cz(
    bam_path: str,
    reference_fasta: str,
    output: Optional[str] = None,
    mode: str = "mc_cov",
    count_fmt: str = "B",
    reference_cz: Optional[str] = None,
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
    reference_fasta : str
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
    reference_cz : str, optional
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
    if mode == "mc_cov" and reference_cz is None:
        raise ValueError(
            "mode='mc_cov' requires reference_cz (positions are not stored "
            "in the output and must be recovered from the reference)."
        )

    if not os.path.exists(reference_fasta):
        raise FileNotFoundError(f"Reference fasta not found: {reference_fasta}")
    fai_path = reference_fasta + ".fai"
    if not os.path.exists(fai_path):
        raise FileNotFoundError(
            f"Reference fasta not indexed. Run `samtools faidx {reference_fasta}` first."
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
        ref_pos_map = _load_reference_positions(reference_cz)
        writer_message = os.path.basename(reference_cz)
    else:
        writer_message = os.path.basename(reference_fasta)

    writer = Writer(
        output,
        formats=formats,
        columns=columns,
        chunk_keys=["chrom"],
        sort_col=sort_col,
        delta_cols=delta_cols,
        message=writer_message,
    )
    unit_size = writer._unit_size
    _ = _all_numeric_formats(formats)  # sanity check
    fmt_struct = struct.Struct("<" + "".join(formats))

    mpileup_cmd = (
        f"samtools mpileup -Q {min_base_quality} -q {min_mapq} -B "
        f"-f {reference_fasta} {bam_path}"
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
    buf_records: list = []
    chrom_pos_buf: list = []
    chrom_mc_buf: list = []
    chrom_cov_buf: list = []

    _np_count_dtype = _fmt_to_np_dtype(count_fmt) or "<u2"
    mc_cov_struct_dtype = np.dtype([("mc", _np_count_dtype), ("cov", _np_count_dtype)])
    _overflow_warned = [False]

    def _flush_records(chrom: str) -> None:
        if not buf_records:
            return
        writer.write_chunk(b"".join(buf_records), [chrom])
        buf_records.clear()

    def _flush_mc_cov(chrom: str) -> None:
        ref_pos = ref_pos_map.get(chrom)
        if ref_pos is None or ref_pos.size == 0:
            chrom_pos_buf.clear(); chrom_mc_buf.clear(); chrom_cov_buf.clear()
            return
        out = np.zeros(ref_pos.size, dtype=mc_cov_struct_dtype)
        if chrom_pos_buf:
            q_pos = np.asarray(chrom_pos_buf, dtype=np.int64)
            q_mc = np.asarray(chrom_mc_buf, dtype=np.uint16)
            q_cov = np.asarray(chrom_cov_buf, dtype=np.uint16)
            idx = np.searchsorted(ref_pos, q_pos)
            idx_clip = np.minimum(idx, ref_pos.size - 1)
            valid = (idx < ref_pos.size) & (ref_pos[idx_clip] == q_pos)
            matched = idx_clip[valid]
            out["mc"][matched] = q_mc[valid]
            out["cov"][matched] = q_cov[valid]
        _write_np_chunks(writer, out, chrom, batch_size, unit_size)
        chrom_pos_buf.clear(); chrom_mc_buf.clear(); chrom_cov_buf.clear()

    def _flush(chrom: str) -> None:
        if mode == "mc_cov":
            _flush_mc_cov(chrom)
        else:
            _flush_records(chrom)

    try:
        for line in pipes.stdout:
            total_line += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            ref_base = fields[2].upper()

            if fields[0] != cur_chrom:
                if cur_chrom:
                    _flush(cur_chrom)
                cur_chrom = fields[0]
                seq = _get_chromosome_sequence_upper(reference_fasta, fai_df, cur_chrom)

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
                context = "".join(_COMPLEMENT[b] for b in reversed(seq[lo:hi]))
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
