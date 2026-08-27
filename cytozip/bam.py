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
import struct
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from . import cz as _cz_mod
from .cz import (
    Writer, Reader, _fmt_to_np_dtype,
    _ensure_cz_accel,
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
    """Read a ``.fai`` index into a DataFrame keyed by chromosome name."""
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


def _is_read_ct_conversion_hisat3n(read):
    """hisat-3n: YZ tag ``+`` marks a C->T (Watson/OT) conversion read."""
    return read.get_tag("YZ") == "+"


def _is_read_ct_conversion_bismark(read):
    """bismark: XG tag ``CT`` marks a C->T (Watson/OT) conversion read."""
    return read.get_tag("XG") == "CT"


def _determine_ct_func(bam_path):
    """Pick the conversion-type test from the first read's tags.

    hisat-3n BAMs carry a ``YZ`` tag; bismark BAMs carry an ``XG`` tag.
    A BAM lacking both cannot be strand-corrected (it is not a
    bismark/hisat-3n bisulfite BAM).
    """
    import pysam
    with pysam.AlignmentFile(bam_path) as f:
        read = next(iter(f))
    if read.has_tag("YZ"):
        return _is_read_ct_conversion_hisat3n
    if read.has_tag("XG"):
        return _is_read_ct_conversion_bismark
    raise ValueError(
        "BAM reads lack a conversion-type tag (XG by bismark or YZ by "
        "hisat-3n). convert_bam_strandness can only process bismark or "
        "hisat-3n bisulfite BAMs."
    )


def _convert_bam_strandness(in_bam_path: str, out_bam_path: str,
                            index: bool = True) -> None:
    """Rewrite a bismark/hisat-3n BAM so ``read.is_forward`` matches the
    bisulfite conversion type (XG/YZ tag), not the alignment orientation.

    Ported from ALLCools ``_bam_to_allc._convert_bam_strandness``.

    Why this is needed
    ------------------
    Pileup-based methylation calling (both the htslib and ``samtools
    mpileup`` backends) infers strand from the read FLAG:

    * ref ``C`` sites are counted from forward reads (``.`` / ``T``),
    * ref ``G`` sites are counted from reverse reads (``,`` / ``a``).

    For **bismark SE**, the aligner already orients C->T reads to the
    forward strand and G->A reads to the reverse strand, so the FLAG is
    correct. For **hisat-3n PE / bismark PE**, R1 and R2 keep their
    original orientation, so both C->T and G->A reads appear on either
    strand and the FLAG no longer encodes the conversion type. Counting
    then assigns coverage to the wrong-strand cytosines (e.g. the lambda
    spike-in shows ~50% methylation instead of the conversion-error rate).

    This function forces, per read (and its mate):

    * C->T conversion (``YZ == '+'`` / ``XG == 'CT'``) -> ``is_forward = True``
    * G->A conversion (``YZ == '-'`` / ``XG == 'GA'``) -> ``is_forward = False``

    so downstream base counting stays unchanged and strand-correct.

    Parameters
    ----------
    in_bam_path, out_bam_path : str
        Input / output BAM paths. ``read.is_forward`` is only flipped, so
        coordinate sort order is preserved.
    index : bool, default True
        Build the ``.bai`` for ``out_bam_path`` via pysam (avoids a hard
        dependency on ``samtools`` being on ``$PATH``).
    """
    import pysam
    is_ct_func = _determine_ct_func(in_bam_path)
    with pysam.AlignmentFile(in_bam_path) as in_bam, \
            pysam.AlignmentFile(out_bam_path, header=in_bam.header, mode="wb") as out_bam:
        for read in in_bam:
            if is_ct_func(read):
                read.is_forward = True
                if read.is_paired:
                    read.mate_is_forward = True
            else:
                read.is_forward = False
                if read.is_paired:
                    read.mate_is_forward = False
            out_bam.write(read)
    if index:
        pysam.index(out_bam_path)


def _start_mpileup(samtools_exe, genome, min_base_quality, min_mapq,
                   bam_path, strand_flip=False):
    """Start ``samtools mpileup`` and return ``(proc, text_stdout, feeder)``.

    When ``strand_flip`` is True the input BAM is **streamed** to mpileup's
    stdin with each read's FLAG rewritten to match its bisulfite conversion
    tag (XG/YZ) — the same strand correction as
    :func:`_convert_bam_strandness`, but via a pipe so **no temporary BAM is
    written** (and no ``samtools index`` is needed). Otherwise mpileup reads
    ``bam_path`` directly.

    The returned stdout is a latin-1 text stream: mpileup pileup strings can
    contain bytes > 127 (the char after a ``^`` read-start marker encodes
    MAPQ as ``chr(mapq + 33)``), which would crash a utf-8/locale decoder.
    Base counting only inspects ASCII ``.`` / ``,`` / ``T`` / ``a``.
    """
    import io
    cmd = [samtools_exe, "mpileup",
           "-Q", str(min_base_quality), "-q", str(min_mapq),
           "-B", "-f", genome]
    feeder = None
    if strand_flip:
        import threading
        import pysam
        is_ct_func = _determine_ct_func(bam_path)
        proc = subprocess.Popen(
            cmd + ["-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        exc_holder: dict = {}

        def _feed():
            try:
                with pysam.AlignmentFile(bam_path) as in_bam:
                    out = pysam.AlignmentFile(proc.stdin, "wb",
                                              template=in_bam)
                    for read in in_bam:
                        if is_ct_func(read):
                            read.is_forward = True
                            if read.is_paired:
                                read.mate_is_forward = True
                        else:
                            read.is_forward = False
                            if read.is_paired:
                                read.mate_is_forward = False
                        out.write(read)
                    out.close()
            except Exception as exc:  # pragma: no cover - surfaced by caller
                exc_holder["exc"] = exc
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.cz_exc = exc_holder  # type: ignore[attr-defined]
        feeder.start()
    else:
        proc = subprocess.Popen(
            cmd + [bam_path], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
    text = io.TextIOWrapper(proc.stdout, encoding="latin-1", newline="")
    return proc, text, feeder


# ---------------------------------------------------------------------------
# Mode -> Writer layout
# ---------------------------------------------------------------------------
def _resolve_mode(mode):
    """Validate ``mode`` against the supported output layouts and return it."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    return mode


_VALID_COUNT_FMTS = ("B", "H", "I", "Q")
_COUNT_FMT_MAX = {"B": 0xFF, "H": 0xFFFF, "I": 0xFFFFFFFF,
                  "Q": 0xFFFFFFFFFFFFFFFF}


def _layout_for_mode(mode, count_fmt="H"):
    """Return ``(formats, columns, sort_col, delta_cols)`` for an output ``mode``."""
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
        """True if ``chrom`` is present in the reference .cz."""
        return chrom in self._chrom2dim

    @property
    def chroms(self):
        """Set of chromosome names present in the reference .cz."""
        return set(self._chrom2dim)

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
        """Evict a chromosome's cached positions and release its mmap pages."""
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
        """True if ``chrom`` exists in the reference .cz."""
        return chrom in self._chrom2dim

    def close(self):
        """Clear the position cache and close the underlying reference reader."""
        self._cache.clear()
        self._reader.close()


# ---------------------------------------------------------------------------
# Pre-processing: name-sorted BAM -> position-sorted + deduplicated BAM
# ---------------------------------------------------------------------------
def _resolve_env_bin(exe: str, env: Optional[str]) -> str:
    """Resolve an executable path, optionally from a specific conda env.

    Some tools (e.g. ``picard``) are not installed in every conda env. This
    helper lets callers pin the env that provides ``exe``.

    Parameters
    ----------
    exe : str
        Executable name, e.g. ``"picard"`` or ``"samtools"``.
    env : str, optional
        * ``None`` - return ``exe`` unchanged (resolved on ``$PATH`` at run
          time).
        * a directory path - treated as a conda env *prefix*; the executable
          is looked up at ``<env>/bin/<exe>``.
        * a bare env name (e.g. ``"yap"``) - resolved against the sibling
          env dirs of the current interpreter (e.g.
          ``/home/user/conda/m3c`` -> ``/home/user/conda/yap``), the
          standard ``envs/`` layout, and ``$CONDA_EXE``'s ``envs/``.

    Returns
    -------
    str
        Absolute path to the executable (or ``exe`` unchanged if
        ``env is None``).
    """
    if env is None:
        return exe
    candidates = []
    if os.path.sep in env or os.path.isdir(env):
        candidates.append(env)
    else:
        # Sibling of the current interpreter's env prefix, e.g.
        # /home/.../conda/m3c/bin/python -> /home/.../conda/yap
        cur_prefix = os.path.dirname(
            os.path.dirname(os.path.abspath(sys.executable)))
        parent = os.path.dirname(cur_prefix)
        candidates.append(os.path.join(parent, env))
        candidates.append(os.path.join(parent, "envs", env))
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            conda_root = os.path.dirname(os.path.dirname(conda_exe))
            candidates.append(os.path.join(conda_root, "envs", env))
    for prefix in candidates:
        cand = os.path.join(prefix, "bin", exe)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find {exe!r} in conda env {env!r}. Tried: "
        + ", ".join(os.path.join(p, "bin", exe) for p in candidates)
    )


def name_sort_bam_to_deduped(
    bam_path: str,
    output: Optional[str] = None,
    stats: Optional[str] = None,
    remove_duplicates: bool = True,
    tmp_dir: Optional[str] = None,
    sort_threads: int = 1,
    sort_mem_mb: int = 1000,
    index: bool = True,
    keep_pos_sort: bool = False,
    env: Optional[str] = None,
) -> str:
    """Turn a name-sorted BAM into a position-sorted, deduplicated BAM.

    ``bam_to_cz`` requires a **position-sorted** BAM with a ``.bai`` index.
    The hisat-3n / snmC pipeline (see cemba_data ``hisat3n.smk``) produces a
    name-sorted BAM (``*.all_reads.name_sort.bam``), which must first be
    coordinate-sorted and PCR-deduplicated before it can be fed to
    ``bam_to_cz``. This helper reproduces those two Snakemake rules
    (``sort_bam_by_pos`` + ``dedup``):

    1. ``samtools sort -O BAM`` (name order -> coordinate order)
    2. ``picard MarkDuplicates -REMOVE_DUPLICATES true`` (drop PCR dups)
    3. ``samtools index`` (optional, produces the ``.bai`` needed by
       ``bam_to_cz``)

    Parameters
    ----------
    bam_path : str
        Input name-sorted BAM (e.g. ``*.hisat3n_dna.all_reads.name_sort.bam``).
    output : str, optional
        Output deduplicated BAM path. Defaults to
        ``<stem>.deduped.bam`` next to the input, where ``<stem>`` strips a
        trailing ``.name_sort`` if present.
    stats : str, optional
        Path for the picard MarkDuplicates metrics file. Defaults to
        ``<output>.matrix.txt``.
    remove_duplicates : bool
        If True (default), pass ``-REMOVE_DUPLICATES true`` so duplicates are
        physically removed. If False, duplicates are only flagged.
    tmp_dir : str, optional
        Temp directory for picard. Defaults to ``<output_dir>/temp``
        (created if missing).
    sort_threads : int
        Threads for ``samtools sort`` (``-@``).
    sort_mem_mb : int
        Per-thread memory for ``samtools sort`` in MB (``-m``).
    index : bool
        If True (default), build the ``.bai`` index for the deduped BAM.
    keep_pos_sort : bool
        If True, keep the intermediate coordinate-sorted BAM. By default it
        is deleted after deduplication.
    env : str, optional
        Conda env that provides ``samtools`` / ``picard``. Useful when the
        current env lacks ``picard`` (e.g. run cytozip from ``m3c`` but pull
        ``picard`` from ``yap`` via ``env="yap"``). May be a bare env name
        (resolved against sibling env dirs of the current interpreter and
        the standard ``envs/`` layout) or a full env prefix path. ``None``
        (default) resolves both tools on ``$PATH``.

    Returns
    -------
    str
        Path to the deduplicated BAM.
    """
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"Input BAM not found: {bam_path}")

    samtools_exe = _resolve_env_bin("samtools", env)
    picard_exe = _resolve_env_bin("picard", env)

    bam_dir = os.path.dirname(os.path.abspath(bam_path))
    base = os.path.basename(bam_path)
    if base.endswith(".bam"):
        base = base[: -len(".bam")]
    # Strip a trailing ".name_sort" so the output stem matches the pipeline
    # convention (``*.all_reads.deduped.bam``).
    if base.endswith(".name_sort"):
        stem = base[: -len(".name_sort")]
    else:
        stem = base

    if output is None:
        output = os.path.join(bam_dir, stem + ".deduped.bam")
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if stats is None:
        stats = output + ".matrix.txt"

    if tmp_dir is None:
        tmp_dir = os.path.join(out_dir, "temp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Coordinate-sort (rule sort_bam_by_pos).
    pos_sort_bam = os.path.join(out_dir, stem + ".pos_sort.bam")
    subprocess.check_call([
        samtools_exe, "sort", "-O", "BAM",
        "-@", str(sort_threads),
        "-m", f"{sort_mem_mb}M",
        "-o", pos_sort_bam, bam_path,
    ])

    # 2. Mark / remove PCR duplicates (rule dedup).
    picard_cmd = [
        picard_exe, "MarkDuplicates",
        "-I", pos_sort_bam,
        "-O", output,
        "-M", stats,
        "-REMOVE_DUPLICATES", "true" if remove_duplicates else "false",
        "-TMP_DIR", tmp_dir,
    ]
    try:
        subprocess.check_call(picard_cmd)
    finally:
        if not keep_pos_sort and os.path.exists(pos_sort_bam):
            try:
                os.remove(pos_sort_bam)
            except OSError:
                pass

    # 3. Index (rule index_bam).
    if index:
        subprocess.check_call([samtools_exe, "index", output])

    return output


def _parse_chrom_whitelist(chroms):
    """Normalize a ``chroms`` argument into a set of chromosome names or None.

    Accepts:

    * ``None`` -> ``None`` (no restriction; every contig is piled up).
    * list / tuple / set of names -> a set of ``str``.
    * a path to a chrom-size / ``.fai`` file -> the first
      (tab/whitespace-separated) column of each non-empty line.
    * a comma-separated string ``'chr1,chr2'`` -> the listed names.
    * a single chromosome name -> ``{name}``.
    """
    if chroms is None:
        return None
    if isinstance(chroms, (list, tuple, set)):
        return {str(c) for c in chroms}
    if isinstance(chroms, str):
        path = os.path.abspath(os.path.expanduser(chroms))
        if os.path.exists(path):
            names = set()
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        names.add(line.split()[0])
            return names
        if "," in chroms:
            return {c for c in chroms.split(",") if c}
        return {chroms}
    raise TypeError(
        f"chroms must be None, a list/tuple/set, or a str; got {type(chroms)}")


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
    convert_bam_strandness: bool = True,
    save_count_df: bool = False,
    name_sorted: bool = False,
    env: Optional[str] = None,
    chroms=None,
) -> Optional[pd.DataFrame]:
    """Convert a position-sorted BAM to a ``.cz`` methylation file.

    Parameters
    ----------
    bam_path : str
        Position-sorted BAM (requires ``.bai``; we will build it if missing).
        If ``name_sorted=True``, a name-sorted BAM is accepted instead and
        is first coordinate-sorted + PCR-deduplicated (see ``name_sorted``).
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
        If True (**default**, matching ALLCools ``bam_to_allc``), count
        methylation using the strand implied by the bisulfite conversion
        tag (XG for bismark, YZ for hisat-3n) instead of the alignment
        orientation. **Required for hisat-3n PE / bismark PE data**, where
        R1/R2 keep their original orientation so the FLAG strand no longer
        encodes the conversion type; without it ~half of the reads are
        counted on the wrong-strand cytosines (e.g. the lambda spike-in
        reads ~50% methylation instead of the conversion-error rate). This
        never writes a temporary BAM: the htslib backend derives the
        strand in-process from each read's XG/YZ tag, and the ``samtools
        mpileup`` fallback backend **streams** strand-flipped reads to
        mpileup via a pipe (see :func:`_start_mpileup`). The input must be
        a bismark/hisat-3n BAM carrying an ``XG``/``YZ`` tag, otherwise a
        ``ValueError`` is raised; pass ``convert_bam_strandness=False`` for
        plain (already strand-correct) BAMs such as bismark SE.
    save_count_df : bool
        If True, write a ``<output>.count.csv`` with total mC / cov per context.

    name_sorted : bool
        If True, ``bam_path`` is a **name-sorted** BAM (e.g. the hisat-3n
        ``*.all_reads.name_sort.bam``). It is first passed through
        :func:`name_sort_bam_to_deduped` (coordinate-sort + picard
        MarkDuplicates) to produce the position-sorted, deduplicated BAM
        that ``bam_to_cz`` requires. The generated deduped BAM is used as
        the actual input.
    env : str, optional
        Conda env that provides ``picard`` / ``samtools`` for the
        ``name_sorted`` pre-processing step (e.g. ``env="yap"`` when running
        from an env without picard). Ignored unless ``name_sorted=True``.
        See :func:`name_sort_bam_to_deduped`.
    chroms : None, list, or str, optional
        Restrict the pileup to a whitelist of chromosomes. Accepts a list of
        names, a comma-separated string (``'chr1,chr2'``), or a path to a
        chrom-size / ``.fai`` file (first column). This mirrors ALLCools
        ``bam_to_allc(..., chrom_size=...)``, which limits ``samtools
        mpileup`` to the listed chroms; without it every contig in the
        genome fasta (including alt/decoy/unplaced) is piled up and counted,
        inflating the returned ``count_df``. In ``mode='mc_cov'`` the pileup
        is **always** limited to the reference's chromosomes regardless of
        this argument (sites on other chroms are discarded at write time
        anyway); passing ``chroms`` then further narrows that set.

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

    if name_sorted:
        # Coordinate-sort + PCR-dedup the name-sorted BAM first; the
        # resulting deduped BAM is what the pileup backends consume.
        bam_path = name_sort_bam_to_deduped(bam_path, env=env)

    # Resolve samtools once (optionally from a specific conda env) so both
    # the .bai indexing and the mpileup fallback use the same binary.
    samtools_exe = _resolve_env_bin("samtools", env)
    genome=os.path.expanduser(genome)
    if not os.path.exists(genome):
        raise FileNotFoundError(f"Reference fasta not found: {genome}")
    fai_path = genome + ".fai"
    if not os.path.exists(fai_path):
        raise FileNotFoundError(
            f"Reference fasta not indexed. Run `samtools faidx {genome}` first."
        )
    fai_df = _read_faidx(fai_path)

    # ---- backend selection (needed before strand handling) ----------------
    #   1. htslib (in-process cytozip._bam_pileup, ~10-20x faster than the
    #      mpileup subprocess, byte-equivalent to ALLCools). Default when
    #      the optional extension was built.
    #   2. ``samtools mpileup`` subprocess. Fallback when htslib is
    #      unavailable, and forced via ``CYTOZIP_BAM_BACKEND_MPILEUP=1``.
    force_mpileup = bool(int(os.environ.get(
        "CYTOZIP_BAM_BACKEND_MPILEUP", "0")))
    _PileupCounter = None
    if not force_mpileup:
        try:
            from ._bam_pileup import PileupCounter as _PileupCounter
        except ImportError:
            _PileupCounter = None
    use_htslib = (_PileupCounter is not None) and not force_mpileup
    use_mpileup = not use_htslib

    # ---- strand correction (hisat-3n PE / bismark PE) ---------------------
    # No temporary BAM is ever written. The htslib backend derives the
    # effective strand from the XG/YZ conversion tag in-process; the
    # mpileup subprocess backend streams strand-flipped reads via a pipe.
    strand_via_ext = convert_bam_strandness and use_htslib
    strand_via_pipe = convert_bam_strandness and use_mpileup

    # Only the htslib backend needs a .bai (it does indexed per-chrom
    # queries). The mpileup backend reads sequentially / from a pipe.
    if use_htslib and not os.path.exists(bam_path + ".bai"):
        subprocess.check_call([samtools_exe, "index", bam_path])

    if output is None:
        stem = os.path.basename(bam_path).split(".")[0]
        output = os.path.join(os.path.dirname(os.path.abspath(bam_path)),
                              stem + ".cz")

    # Writer layout
    formats, columns, sort_col, delta_cols = _layout_for_mode(mode, count_fmt)
    ref_pos_map = None
    if mode == "mc_cov":
        ref_pos_map = _LazyRefPositions(reference)
        writer_message = os.path.abspath(os.path.expanduser(reference))
    else:
        writer_message = os.path.abspath(os.path.expanduser(genome))

    # Chromosome whitelist for the pileup. ALLCools restricts mpileup to the
    # chroms in its chrom_size file; without a restriction we also pile up
    # alt/decoy/unplaced contigs, which inflates the returned count_df.
    allowed_chroms = _parse_chrom_whitelist(chroms)
    if mode == "mc_cov" and ref_pos_map is not None:
        # Sites on chroms absent from the reference are discarded at write
        # time anyway, so skip them during pileup too (faster, and keeps the
        # returned count_df consistent with what is actually written).
        ref_chroms = ref_pos_map.chroms
        allowed_chroms = (ref_chroms if allowed_chroms is None
                          else (allowed_chroms & ref_chroms))

    writer = Writer(
        output,
        formats=formats,
        columns=columns,
        chunk_dims=["chrom"],
        sort_col=sort_col,
        delta_cols=delta_cols,
        message=writer_message,
    )
    fmt_struct = struct.Struct("<" + "".join(formats))

    # Backend was selected above (use_htslib / use_mpileup). Only launch the
    # ``samtools mpileup`` subprocess when the htslib backend is not used;
    # when strand-correcting, strand-flipped reads are streamed to mpileup
    # through a pipe (no temp BAM).
    mpileup_proc = None
    mpileup_out = None      # latin-1 text stream over mpileup stdout
    mpileup_feeder = None   # background thread streaming strand-flipped reads
    if not use_htslib:
        reason = ("CYTOZIP_BAM_BACKEND_MPILEUP=1 set" if force_mpileup
                  else "htslib extension (cytozip._bam_pileup) unavailable")
        logger.info(
            f"Using `samtools mpileup` subprocess backend ({reason}). This "
            f"is ~10-20x slower than the in-process htslib backend; build "
            f"the optional `_bam_pileup` extension for speed."
        )
        mpileup_proc, mpileup_out, mpileup_feeder = _start_mpileup(
            samtools_exe, genome, min_base_quality, min_mapq,
            bam_path, strand_flip=strand_via_pipe)
    cov_dict: dict = {}
    mc_dict: dict = {}
    total_line = 0  # set from nonlocal_state after the consume loop
    context_len = num_upstr_bases + 1 + num_downstr_bases

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

    # ----- Per-site consumer (shared by both backends) -----
    # Captures cur_chrom / seq / total_line in the enclosing scope.
    # Returns nothing; mutates the various buffers and counters.
    nonlocal_state = {"cur_chrom": "", "seq": None, "total_line": 0}

    def _handle_site(chrom: str, pos0: int, ref_base: str,
                      unconverted: int, converted: int) -> None:
        nonlocal_state["total_line"] += 1
        if chrom != nonlocal_state["cur_chrom"]:
            if nonlocal_state["cur_chrom"]:
                # Release the previous chromosome's sequence string
                # (≈100-200 MB for large chroms) BEFORE running its
                # flush — otherwise the flush's transient allocations
                # stack on top of it and inflate MaxRSS.
                nonlocal_state["seq"] = None
                _flush(nonlocal_state["cur_chrom"])
            nonlocal_state["cur_chrom"] = chrom
            nonlocal_state["seq"] = _get_chromosome_sequence_upper(
                genome, fai_df, chrom)
        seq_local = nonlocal_state["seq"]
        if ref_base == "C":
            lo = pos0 - num_upstr_bases
            hi = pos0 + num_downstr_bases + 1
            if lo < 0 or hi > len(seq_local):
                return
            context = seq_local[lo:hi]
            strand = b"+"
        else:  # 'G'
            lo = pos0 - num_downstr_bases
            hi = pos0 + num_upstr_bases + 1
            if lo < 0 or hi > len(seq_local):
                return
            context = seq_local[lo:hi].translate(_RC_TABLE)[::-1]
            strand = b"-"
        cov = unconverted + converted
        if cov == 0 or len(context) != context_len:
            return
        cov_dict[context] = cov_dict.get(context, 0) + cov
        mc_dict[context] = mc_dict.get(context, 0) + unconverted
        if unconverted > count_max or cov > count_max:
            if not _overflow_warned[0]:
                logger.warning(
                    f"mc/cov value exceeds count_fmt={count_fmt!r} max "
                    f"({count_max}); clipping. Consider count_fmt='H' for "
                    f"bulk/high-coverage data."
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
                _flush_records(chrom)
        elif mode == "pos_mc_cov":
            buf_records.append(fmt_struct.pack(pos1, unconverted, cov))
            if len(buf_records) >= batch_size:
                _flush_records(chrom)
        else:  # mc_cov: buffer whole chrom
            chrom_pos_buf.append(pos1)
            chrom_mc_buf.append(unconverted)
            chrom_cov_buf.append(cov)

    try:
        if use_htslib:
            # In-process htslib mpileup wrapper. Byte-equivalent to
            # ``samtools mpileup -Q -q -B -f`` (and therefore to
            # ALLCools ``bam_to_allc``), but ~10-20x faster because we
            # avoid the subprocess + text parsing roundtrip.
            pc = _PileupCounter(
                bam_path.encode() if isinstance(bam_path, str) else bam_path,
                genome.encode() if isinstance(genome, str) else genome,
                min_mapq=min_mapq,
                min_base_quality=min_base_quality,
                convert_bam_strandness=strand_via_ext,
            )
            for chrom in pc.references:
                if chrom not in fai_df.index:
                    continue
                if allowed_chroms is not None and chrom not in allowed_chroms:
                    continue
                if chrom != nonlocal_state["cur_chrom"]:
                    if nonlocal_state["cur_chrom"]:
                        nonlocal_state["seq"] = None
                        _flush(nonlocal_state["cur_chrom"])
                    nonlocal_state["cur_chrom"] = chrom
                    nonlocal_state["seq"] = _get_chromosome_sequence_upper(
                        genome, fai_df, chrom)
                pos_arr, ref_arr, mc_arr, cov_arr = pc.iter_chrom(chrom)
                # ref_arr is uint8 (ascii). Vectorize the dispatch:
                # ``_handle_site`` expects (chrom, pos0, ref_base,
                # unconverted=mc, converted=cov-mc).
                for i in range(pos_arr.shape[0]):
                    cov_i = int(cov_arr[i])
                    if cov_i == 0:
                        continue
                    mc_i = int(mc_arr[i])
                    _handle_site(
                        chrom,
                        int(pos_arr[i]) - 1,
                        chr(int(ref_arr[i])),
                        mc_i,
                        cov_i - mc_i,
                    )
        elif use_mpileup:
            for line in mpileup_out:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                if allowed_chroms is not None and fields[0] not in allowed_chroms:
                    continue
                ref_base = fields[2].upper()
                if ref_base not in _MC_SITES:
                    # Still need to register chrom change so seq gets
                    # reloaded for the next C/G site. Do it cheaply by
                    # passing through _handle_site only on relevant sites.
                    if fields[0] != nonlocal_state["cur_chrom"]:
                        # Lightweight chrom switch (no-op for non-C/G).
                        if nonlocal_state["cur_chrom"]:
                            nonlocal_state["seq"] = None
                            _flush(nonlocal_state["cur_chrom"])
                        nonlocal_state["cur_chrom"] = fields[0]
                        nonlocal_state["seq"] = _get_chromosome_sequence_upper(
                            genome, fai_df, fields[0])
                    continue
                read_bases = fields[4]
                if ("+" in read_bases) or ("-" in read_bases):
                    read_bases = _strip_indels(read_bases)
                pos0 = int(fields[1]) - 1
                if ref_base == "C":
                    unconverted = read_bases.count(".")
                    converted = read_bases.count("T")
                else:
                    unconverted = read_bases.count(",")
                    converted = read_bases.count("a")
                _handle_site(fields[0], pos0, ref_base, unconverted, converted)

        if nonlocal_state["cur_chrom"]:
            _flush(nonlocal_state["cur_chrom"])
        total_line = nonlocal_state["total_line"]
    finally:
        if mpileup_proc is not None:
            try:
                mpileup_out.close()   # closes mpileup stdout
            except Exception:
                pass
            if mpileup_feeder is not None:
                mpileup_feeder.join()
                _feeder_exc = getattr(
                    mpileup_feeder, "cz_exc", {}).get("exc")
                if _feeder_exc is not None:
                    logger.warning(
                        f"strand-flip feeder thread failed: {_feeder_exc!r}")
            mpileup_proc.wait()
        writer.close()
        if ref_pos_map is not None:
            ref_pos_map.close()

    count_df = pd.DataFrame({"mc": mc_dict, "cov": cov_dict})
    if not count_df.empty:
        count_df["mc_rate"] = count_df["mc"] / count_df["cov"]
        total_genome_length = int(fai_df["LENGTH"].sum())
        count_df["genome_cov"] = total_line / max(total_genome_length, 1)

    if save_count_df:
        count_df.to_csv(output + ".count.csv")
        return None
    return count_df
