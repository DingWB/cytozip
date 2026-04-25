"""Methylation-array → .cz conversion.

This module provides :func:`array2cz` for converting a per-sample
methylation-array beta (and optionally detection p-value / intensity)
matrix into the cytozip ``.cz`` columnar format.

Input shape conventions
-----------------------
The function accepts a *long-format* DataFrame plus a probe *manifest*
DataFrame:

* ``df`` (long): one row per (sample, probe). Required columns:
  ``sample_id``, ``probe_id``, plus one or more value columns
  (e.g. ``beta``, ``detection_p``).
* ``manifest``: probe-level metadata, one row per probe. Required
  columns: ``probe_id`` and ``chrom`` (used as the chunk key). Any
  additional columns (``pos``, ``strand``, ``context``, ...) are written
  to a single per-array reference ``.cz`` whose rows align 1:1 with
  every per-sample data ``.cz``.

Output layout
-------------
For each sample, a per-sample ``<sample_id><ext>`` is written under
``output_dir`` containing only the value columns. Probes are sorted
within each ``chrom`` chunk according to ``manifest`` order, so all
per-sample files share the exact same row order and can be merged via
:func:`cytozip.merge.merge_cz` (typically with ``agg='mean'``) and
read back via :func:`cytozip.features.cz_to_anndata`.

Optionally, ``output_reference`` writes the manifest as a row-aligned
reference ``.cz``, equivalent to the ``mm10_with_chrL.allc.cz`` used
by BS-seq pipelines, so :meth:`cytozip.cz.Reader.to_bgzip` can splice
chrom/pos columns at export time.

Float quantization
------------------
Methylation-array beta is a float in ``[0, 1]``. Stored as raw float32
it compresses poorly. Set ``quantize_bits=8`` (or ``16``) to losslessly
*linearly* quantize each value column to an integer with a stored
``min``/``max``; the round-trip error is bounded by ``(max-min)/2^bits``
(≈0.2% for 8 bits, ≈0.0008% for 16 bits over [0,1]). The scale
parameters are recorded in the .cz header ``message`` field as a small
JSON blob, e.g. ``{"quant":{"beta":["B",0.0,1.0]}}``.

Use :func:`dequantize_cz` to read a quantized .cz back to floats.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import pandas as pd

from .cz import Reader, Writer


# Per-bit-width storage format and integer max for linear quantization.
_QUANT_FMT = {8: ('B', 0xFF), 16: ('H', 0xFFFF)}


def quantize_uniform(values, lo, hi, bits=8):
    """Linearly quantize a float array to uint8/uint16.

    Maps ``[lo, hi]`` → ``[0, 2**bits - 1]``, clipping out-of-range
    inputs. Returns a numpy array of the matching unsigned dtype.
    """
    if bits not in _QUANT_FMT:
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    fmt, vmax = _QUANT_FMT[bits]
    arr = np.asarray(values, dtype=np.float64)
    # Clip then linearly scale.
    np.clip(arr, lo, hi, out=arr)
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo} hi={hi}")
    scaled = (arr - lo) * (vmax / (hi - lo))
    np.rint(scaled, out=scaled)
    np_dtype = '<u1' if bits == 8 else '<u2'
    return scaled.astype(np_dtype)


def dequantize_uniform(values, lo, hi, bits=8):
    """Reverse of :func:`quantize_uniform` — uint → float64 in [lo, hi]."""
    if bits not in _QUANT_FMT:
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    _, vmax = _QUANT_FMT[bits]
    arr = np.asarray(values, dtype=np.float64)
    return arr * ((hi - lo) / vmax) + lo


def _parse_quant_message(message):
    """Extract ``{col: (storage_fmt, lo, hi)}`` from a Reader's header
    ``message`` field, or ``{}`` if no quant metadata is present.
    """
    if not message:
        return {}
    try:
        meta = json.loads(message)
    except (ValueError, TypeError):
        return {}
    quant = meta.get('quant') if isinstance(meta, dict) else None
    if not isinstance(quant, dict):
        return {}
    out = {}
    for col, spec in quant.items():
        # spec is [storage_fmt, lo, hi]
        if isinstance(spec, (list, tuple)) and len(spec) == 3:
            out[col] = (str(spec[0]), float(spec[1]), float(spec[2]))
    return out


def dequantize_cz(cz_path, dim):
    """Read a quantized .cz chunk back as a pandas DataFrame of floats.

    Convenience helper: opens ``cz_path``, looks up the quant table in
    its header ``message``, and dequantizes any value columns whose
    integer-storage format was recorded there. Non-quantized columns
    are returned as-is.
    """
    reader = Reader(cz_path)
    quant = _parse_quant_message(reader.header.get('message', ''))
    df = reader.chunk2df(dim, reformat=True)
    reader.close()
    for col, (_fmt, lo, hi) in quant.items():
        if col in df.columns:
            df[col] = dequantize_uniform(df[col].values, lo, hi,
                                         bits=8 if _fmt == 'B' else 16)
    return df


# ---------------------------------------------------------------------------


def _pick_storage_format(series):
    """Pick a struct format for a numeric pandas Series."""
    if pd.api.types.is_float_dtype(series):
        return 'f'
    if pd.api.types.is_integer_dtype(series):
        vmax = int(series.abs().max() if len(series) else 0)
        if vmax <= 0xFF:
            return 'B'
        if vmax <= 0xFFFF:
            return 'H'
        if vmax <= 0xFFFFFFFF:
            return 'I'
        return 'Q'
    raise TypeError(
        f"unsupported dtype {series.dtype} for value column {series.name!r}; "
        "only numeric (int/float) value columns are supported")


def _pack_struct_records(df, fmts):
    """Pack rows of ``df`` into a contiguous bytes buffer matching
    ``''.join(fmts)``. Uses numpy structured dtype for speed when
    every field is a simple numeric scalar.
    """
    from .cz import _NP_FMT_MAP
    cols = df.columns.tolist()
    # All-numeric fast path via structured numpy.
    np_dtype = []
    for fmt, col in zip(fmts, cols):
        np_dtype.append((col, _NP_FMT_MAP[fmt[-1]]))
    arr = np.empty(len(df), dtype=np.dtype(np_dtype))
    for col in cols:
        arr[col] = df[col].values
    return arr.tobytes()


def array2cz(
    df,
    manifest,
    output_dir,
    output_reference=None,
    sample_col='sample_id',
    probe_col='probe_id',
    chrom_col='chrom',
    value_cols=('beta',),
    quantize_bits=None,
    quantize_range=(0.0, 1.0),
    ext='.cz',
    sort_col=None,
    overwrite=False,
):
    """Convert a per-sample methylation-array matrix to .cz files.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format data with columns ``sample_col``, ``probe_col``,
        and every name in ``value_cols``.
    manifest : pandas.DataFrame
        Probe metadata with at least ``probe_col`` and ``chrom_col``.
        All other columns are written to the per-array reference .cz
        (when ``output_reference`` is given) and may include ``pos``,
        ``strand``, ``context``, etc.
    output_dir : str
        Directory in which per-sample ``<sample_id><ext>`` files are
        written. Created if it does not exist.
    output_reference : str, optional
        Path to a per-array reference ``.cz`` (probe → chrom + extra
        manifest columns). When set, all extra manifest columns
        (everything except ``probe_col`` and ``chrom_col``) are written
        here, row-aligned with every per-sample output.
    sample_col, probe_col, chrom_col : str
        Column names in ``df`` / ``manifest``.
    value_cols : tuple of str
        Numeric columns from ``df`` to store. Order is preserved in the
        per-sample .cz header.
    quantize_bits : int or None
        ``None`` (default): store float columns as float32 (raw).
        ``8`` or ``16``: linearly quantize float columns over
        ``quantize_range`` to uint8/uint16. Quant parameters are
        recorded in the .cz header ``message``.
    quantize_range : (float, float)
        ``(lo, hi)`` mapped to the integer range. Default ``(0, 1)``
        for beta. Ignored when ``quantize_bits=None``.
    ext : str
        Output extension (default ``'.cz'``).
    sort_col : str or None
        Optional manifest column to sort probes within each chrom by
        (typically ``'pos'``). When set, both per-sample and reference
        outputs use this order.
    overwrite : bool
        If True, existing per-sample files are overwritten.

    Returns
    -------
    list of str
        Paths of the per-sample .cz files written.
    """
    # ---- Validate inputs --------------------------------------------------
    if quantize_bits is not None and quantize_bits not in _QUANT_FMT:
        raise ValueError(f"quantize_bits must be 8 or 16, got {quantize_bits}")
    for col in (sample_col, probe_col):
        if col not in df.columns:
            raise ValueError(f"df missing required column {col!r}")
    for col in value_cols:
        if col not in df.columns:
            raise ValueError(f"df missing value column {col!r}")
    for col in (probe_col, chrom_col):
        if col not in manifest.columns:
            raise ValueError(f"manifest missing required column {col!r}")
    if sort_col is not None and sort_col not in manifest.columns:
        raise ValueError(f"manifest missing sort column {sort_col!r}")

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # ---- Order probes (per-chrom) ----------------------------------------
    # All per-sample outputs and the optional reference share this order
    # so chunk-aligned operations (catcz, merge_cz, fetch_chunk_bytes) work.
    sort_keys = [chrom_col]
    if sort_col is not None:
        sort_keys.append(sort_col)
    manifest = manifest.sort_values(sort_keys, kind='stable').reset_index(drop=True)
    chrom_groups = list(manifest.groupby(chrom_col, sort=False))

    # Build a probe → row-position lookup so we can scatter df values
    # back into manifest order in one vectorised step per sample.
    manifest_probe_to_pos = pd.Series(
        np.arange(len(manifest)), index=manifest[probe_col].values)

    # ---- Choose per-column storage formats -------------------------------
    # If quantizing, force chosen integer fmt; otherwise infer from dtype.
    quant_meta = {}  # col -> (storage_fmt, lo, hi)
    formats = []
    for col in value_cols:
        if quantize_bits is not None and pd.api.types.is_float_dtype(df[col]):
            fmt, _ = _QUANT_FMT[quantize_bits]
            lo, hi = float(quantize_range[0]), float(quantize_range[1])
            quant_meta[col] = (fmt, lo, hi)
            formats.append(fmt)
        else:
            formats.append(_pick_storage_format(df[col]))
    msg_obj = {'quant': {c: list(v) for c, v in quant_meta.items()}} if quant_meta else {}
    msg_str = json.dumps(msg_obj) if msg_obj else 'array2cz'

    # ---- Optional reference .cz (row-aligned probe metadata) -------------
    ref_extra_cols = [c for c in manifest.columns
                      if c not in (probe_col, chrom_col)]
    if output_reference is not None and ref_extra_cols:
        _write_reference_cz(output_reference, manifest, chrom_col,
                            ref_extra_cols, chrom_groups)

    # ---- One pass per sample → per-sample .cz ----------------------------
    out_paths = []
    for sample_id, sub in df.groupby(sample_col, sort=False):
        out_path = os.path.join(output_dir, f"{sample_id}{ext}")
        if os.path.exists(out_path) and not overwrite:
            out_paths.append(out_path)
            continue
        # Scatter sub into manifest order. Probes missing for this sample
        # become NaN (float) / 0 (int) — surfaced via ``fillna``.
        sub = sub.set_index(probe_col)
        try:
            sub = sub.loc[manifest[probe_col].values, list(value_cols)]
        except KeyError as e:
            raise ValueError(
                f"sample {sample_id!r} references probe(s) not in manifest: {e}"
            )

        # Apply quantization where configured. Always produce the storage
        # numpy dtype so later struct packing matches ``formats`` exactly.
        out_df = pd.DataFrame(index=sub.index)
        for col, fmt in zip(value_cols, formats):
            vals = sub[col].values
            if col in quant_meta:
                _, lo, hi = quant_meta[col]
                out_df[col] = quantize_uniform(np.nan_to_num(vals, nan=lo),
                                               lo, hi, bits=quantize_bits)
            else:
                out_df[col] = vals

        writer = Writer(out_path, formats=formats, columns=list(value_cols),
                        chunk_dims=[chrom_col], message=msg_str)
        for chrom, mf_chrom in chrom_groups:
            chrom_idx = manifest_probe_to_pos.loc[mf_chrom[probe_col].values].values
            chrom_df = out_df.iloc[chrom_idx]
            data = _pack_struct_records(chrom_df, formats)
            if data:
                writer.write_chunk(data, [chrom])
        writer.close()
        out_paths.append(out_path)
    return out_paths


def _write_reference_cz(output, manifest, chrom_col, extra_cols, chrom_groups):
    """Write the manifest as a row-aligned reference .cz."""
    output = os.path.abspath(os.path.expanduser(output))
    # Pick a struct format per extra column. Strings → '<N>s' with the
    # column-wide max byte length so we can pack with numpy structured.
    formats = []
    for col in extra_cols:
        s = manifest[col]
        if pd.api.types.is_numeric_dtype(s):
            formats.append(_pick_storage_format(s))
        else:
            # Treat as bytes; max length sets fixed width.
            max_len = int(s.astype(str).map(len).max() or 1)
            formats.append(f"{max_len}s")

    writer = Writer(output, formats=formats, columns=list(extra_cols),
                    chunk_dims=[chrom_col], message='array2cz_ref')
    for chrom, mf_chrom in chrom_groups:
        # Pack one chrom's manifest rows into struct bytes.
        rows = []
        struct_obj = struct.Struct('<' + ''.join(formats))
        for _, row in mf_chrom.iterrows():
            packed_row = []
            for col, fmt in zip(extra_cols, formats):
                v = row[col]
                if fmt.endswith('s'):
                    packed_row.append(
                        str(v).encode('utf-8')[:int(fmt[:-1])])
                else:
                    packed_row.append(v)
            rows.append(struct_obj.pack(*packed_row))
        if rows:
            writer.write_chunk(b''.join(rows), [chrom])
    writer.close()
