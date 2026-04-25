#!/usr/bin/env python
"""Compare a per-cell .allc.tsv.gz against its corresponding .cz file.

The check round-trips the .cz back to .allc.tsv.gz via
:meth:`cytozip.cz.Reader.to_allc` (using the reference .cz for coordinate
lookup) and diff-joins the two allc tables on ``(chrom, pos)``.

Discrepancies are categorised as:

  1. ``cap_explained``  — cz side equals ``cap`` AND original allc value
    exceeds ``cap`` (expected uint8/uint16 saturation).
  2. ``strand_context`` — chrom+pos match but strand / context differ.
  3. ``only_in_allc``   — row present in original allc, missing from cz
    output (e.g. chunk-key filtered out, or reference mismatch).
  4. ``only_in_cz``     — row present in cz output, missing from allc
    (should be zero for a clean round-trip).
  5. ``other_mc_cov``   — mc / cov differ for reasons *other* than cap
    saturation.

Usage
-----
    python tests/compare_allc_vs_cz.py \\
        --allc cytozip_example_data/output/allc/<cell>.allc.tsv.gz \\
        --cz   cytozip_example_data/output/cz/<cell>.cz \\
        --reference cytozip_example_data/output/mm10_with_chrL.allc.cz \\
        [--out report.tsv] [--max-samples 20]

Exits non-zero if any ``other_mc_cov``, ``strand_context``, ``only_in_cz``,
or unexplained ``only_in_allc`` rows are found.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import pandas as pd

from cytozip.cz import Reader


_ALLC_COLS = ['chrom', 'pos', 'strand', 'context', 'mc', 'cov', 'methylated']

# Per-format saturation caps (unsigned integer formats only).
_CAP_MAP = {'B': 2**8 - 1, 'H': 2**16 - 1, 'I': 2**32 - 1, 'Q': 2**64 - 1}


def _load_allc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, names=_ALLC_COLS,
                     dtype={'chrom': str, 'pos': 'int64',
                            'strand': str, 'context': str,
                            'mc': 'int64', 'cov': 'int64',
                            'methylated': 'int8'})
    return df


def _cz_caps(cz_path: str) -> dict[str, int]:
    """Return {column: cap} for unsigned-int mc/cov columns in the .cz."""
    with Reader(cz_path) as r:
        cols = r.header['columns']
        fmts = r.header['formats']
    caps: dict[str, int] = {}
    for col, fmt in zip(cols, fmts):
        char = fmt[-1]
        if char in _CAP_MAP:
            caps[col] = _CAP_MAP[char]
    return caps


def _roundtrip_cz_to_allc(cz_path: str, reference: str, out_dir: str) -> str:
    """Use Reader.to_allc to materialise the cz as allc.tsv.gz."""
    out = os.path.join(out_dir, 'roundtrip.allc.tsv.gz')
    with Reader(cz_path) as r:
        r.to_allc(out, reference=reference, tabix=False)
    return out


def compare(allc_path: str, cz_path: str, reference: str,
            max_samples: int = 20) -> dict:
    caps = _cz_caps(cz_path)
    mc_cap = caps.get('mc')
    cov_cap = caps.get('cov')

    with tempfile.TemporaryDirectory() as td:
        rt_path = _roundtrip_cz_to_allc(cz_path, reference, td)
        left = _load_allc(allc_path)
        right = _load_allc(rt_path)

    # Full outer join on chrom+pos to classify every row.
    merged = left.merge(right, on=['chrom', 'pos'], how='outer',
                        suffixes=('_allc', '_cz'), indicator=True)

    only_in_allc = merged[merged['_merge'] == 'left_only']
    only_in_cz = merged[merged['_merge'] == 'right_only']
    both = merged[merged['_merge'] == 'both'].copy()

    # Column-level diffs on matched rows.
    mc_diff = both['mc_allc'] != both['mc_cz']
    cov_diff = both['cov_allc'] != both['cov_cz']
    strand_diff = both['strand_allc'] != both['strand_cz']
    ctx_diff = both['context_allc'] != both['context_cz']

    # Cap-explained: cz side saturated AND original exceeded the cap.
    def _cap_explained(orig, cz, cap):
        if cap is None:
            return pd.Series(False, index=orig.index)
        return (cz == cap) & (orig > cap)

    mc_cap_exp = _cap_explained(both['mc_allc'], both['mc_cz'], mc_cap)
    cov_cap_exp = _cap_explained(both['cov_allc'], both['cov_cz'], cov_cap)

    mc_other = mc_diff & ~mc_cap_exp
    cov_other = cov_diff & ~cov_cap_exp

    # strand/context mismatch is unusual (indicates reference drift).
    strand_ctx_mask = strand_diff | ctx_diff

    other_mask = mc_other | cov_other
    cap_only_mask = (mc_diff | cov_diff) & ~other_mask

    summary = {
        'n_allc_rows': int(len(left)),
        'n_cz_rows': int(len(right)),
        'n_matched_pos': int(len(both)),
        'n_identical': int(((~mc_diff) & (~cov_diff) &
                           (~strand_diff) & (~ctx_diff)).sum()),
        'n_cap_explained': int(cap_only_mask.sum()),
        'n_mc_other': int(mc_other.sum()),
        'n_cov_other': int(cov_other.sum()),
        'n_strand_context_diff': int(strand_ctx_mask.sum()),
        'n_only_in_allc': int(len(only_in_allc)),
        'n_only_in_cz': int(len(only_in_cz)),
        'mc_cap': mc_cap,
        'cov_cap': cov_cap,
    }

    samples = {
        'only_in_allc': only_in_allc.head(max_samples),
        'only_in_cz': only_in_cz.head(max_samples),
        'cap_explained': both[cap_only_mask].head(max_samples),
        'other_mc_cov': both[other_mask].head(max_samples),
        'strand_context_diff': both[strand_ctx_mask].head(max_samples),
    }

    return {'summary': summary, 'samples': samples}


def _format_report(result: dict) -> str:
    s = result['summary']
    lines = ['=== allc vs cz comparison ===']
    for k, v in s.items():
        lines.append(f'  {k:<24s} {v}')
    lines.append('')
    for name, df in result['samples'].items():
        if len(df) == 0:
            continue
        lines.append(f'--- {name} (showing {len(df)}) ---')
        lines.append(df.to_string(index=False))
        lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--allc', required=True, help='original .allc.tsv.gz')
    ap.add_argument('--cz', required=True, help='per-cell .cz to check')
    ap.add_argument('--reference', required=True,
                    help='reference .cz for coordinate lookup')
    ap.add_argument('--max-samples', type=int, default=20,
                    help='sample rows to print per category')
    ap.add_argument('--out', default=None,
                    help='optional path to write the full report '
                         '(also printed to stdout)')
    args = ap.parse_args(argv)

    result = compare(args.allc, args.cz, args.reference,
                     max_samples=args.max_samples)
    text = _format_report(result)
    print(text)
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(text + '\n')

    s = result['summary']
    has_unexplained = (s['n_mc_other'] + s['n_cov_other']
                       + s['n_strand_context_diff'] + s['n_only_in_cz']) > 0
    return 1 if has_unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
