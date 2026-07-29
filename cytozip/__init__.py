"""cytozip (ChunkZIP) — Chunk-based columnar binary format for fast random access.

Public API:
  - Reader:  Read and query .cz files (local or remote via HTTP).
  - Writer:  Create .cz files from tabular data.
  - AllC:    Extract cytosine positions from a reference genome.
  - allc2cz:  Convert tabix-indexed allc.tsv.gz to .cz format.
  - merge_cz: Merge multiple per-cell .cz files.
  - extract / extractCG / aggregate:  Subset and aggregate .cz data.
  - call_dmr / call_dmr_ch:  BS-seq / single-cell DMR calling.
  - call_dmr_array:  methylation-array (450K/EPIC) DMR calling via comb-p.
  - annot_dmr:  Annotate DMR tables with hypo/hyper sample assignments.

CLI entry point: ``czip <command> [options]``
"""
from ._version import version as __version__

# ---------------------------------------------------------------------------
# Lazy public API — heavy imports (numpy, pandas, pysam, Bio) are deferred
# until a symbol is actually accessed, keeping ``import cytozip`` fast.
# ---------------------------------------------------------------------------
_LAZY_EXPORTS = {
    # cz.py — generic .cz format layer
    'Reader': 'cz', 'Writer': 'cz', 'RemoteFile': 'cz', 'extract': 'cz',
    'aggregate': 'cz', 'align_cz': 'cz', 'open': 'cz',
    # index.py — context / region coordinate index builders
    'index_context': 'index', 'index_regions': 'index',
    # allc.py — methylation allc-file I/O
    'AllC': 'allc', 'allc2cz': 'allc',
    'extractCG': 'allc',
    'compare_allc': 'allc',
    # bam.py — BAM → .cz
    'bam_to_cz': 'bam',
    'name_sort_bam_to_deduped': 'bam',
    # features.py — feature aggregation / anndata
    'cz_to_anndata': 'features', 'parse_features': 'features',
    'parse_gtf': 'features', 'make_genome_bins': 'features',
    # merge.py — per-cell merging pipeline
    'merge_cz': 'merge', 'merge_cell_type': 'merge',
    # pivot.py — per-cell pivot matrices (fraction / fisher)
    'pivot_fraction': 'pivot', 'pivot_fisher': 'pivot',
    # dmr.py — DMR analysis
    'call_dmr_array': 'dmr', 'annot_dmr': 'dmr', 'call_dmr': 'dmr',
    'call_dmr_ch': 'dmr',
    'call_dmr_one_vs_rest': 'dmr',
    'merge_dmr_results': 'dmr',
    'consensus_dmr': 'dmr',
    # peaks.py — ATAC-style peak calling from methylation
    'call_peaks': 'peaks', 'call_peaks_bdg': 'peaks', 'to_bedgraph': 'peaks',
    # model.py — site-likelihood cell-type classifier
    'CellTypeClassifier': 'model', 'predict_cell_type': 'model',
    'deconvolve_bulk': 'model',
    'estimate_theta': 'model', 'top_cytosine_fasta': 'model',
}

# Submodules that can be accessed as cytozip.cz / cytozip.allc
# _SUBMODULES = {'cz', 'allc'}


def __getattr__(name):
    """Module-level __getattr__ (PEP 562).

    When ``cytozip.X`` is accessed and ``X`` is not already in the module
    namespace, Python calls this function instead of raising AttributeError.
    This lets us defer heavy imports (numpy, pandas, pysam …) until the
    user actually needs a specific symbol, keeping ``import cytozip`` fast.
    """
    # if name in _SUBMODULES:
    #     import importlib
    #     return importlib.import_module(f'.{name}', __name__)
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(f'.{mod_name}', __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'cytozip' has no attribute {name!r}")


def __dir__():
    """Make lazy exports visible to tab-completion and ``dir(cytozip)``."""
    return list(globals()) + list(_LAZY_EXPORTS) #+ list(_SUBMODULES)


# ---- helpers for comma-separated list arguments ----------------------------
def _csv_int(s):
    """Parse '4,5' → [4, 5]."""
    return [int(x) for x in s.split(',')]

def _csv_str(s):
    """Parse 'H,H' → ['H', 'H']."""
    return s.split(',')


def _dtype_or_none(s):
    """Parse 'None' → None, otherwise return the string unchanged."""
    return None if s == 'None' else s


def _int_or_float_or_none(s):
    """Parse a top-site selector: 'None' → None, '20000' → int, '0.1' → float."""
    if s is None or str(s).lower() in ('none', ''):
        return None
    try:
        return int(s)
    except ValueError:
        return float(s)


def _float_or_none(s):
    """Parse 'None' → None, otherwise a float."""
    if s is None or str(s).lower() in ('none', ''):
        return None
    return float(s)


def _int_or_none(s):
    """Parse 'None' → None, otherwise an int."""
    if s is None or str(s).lower() in ('none', ''):
        return None
    return int(s)



def _str2bool(s):
    """Parse 'true'/'false' (and common synonyms) → bool."""
    if isinstance(s, bool):
        return s
    v = str(s).strip().lower()
    if v in ('true', 't', 'yes', 'y', '1'):
        return True
    if v in ('false', 'f', 'no', 'n', '0'):
        return False
    import argparse
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {s!r}")


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog='czip',
        description='cytozip — chunk-based columnar binary format CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = parser.add_subparsers(dest='command', help='available commands')

    _fmt = argparse.ArgumentDefaultsHelpFormatter

    # ---- tocz (Writer + tocz) -----------------------------------------------
    p = sub.add_parser('tocz', help='Convert text/stdin to .cz', formatter_class=_fmt)
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-I', '--input', default=None, help='input file (stdin if omitted)')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'],
                   help='column formats, comma-separated (struct chars). '
                        'Unsigned ints cap values: B=255, H=65535, '
                        'I=2^32-1, Q=2^64-1; larger values are truncated '
                        '(saturated) to the max. Default B for single-cell '
                        'mc/cov saves space and is safe: counts >255 are '
                        'usually repeat-region artifacts and downstream '
                        'ALLCools DMR clips coverage to 50 anyway')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names, comma-separated')
    p.add_argument('-D', '--chunk_dims', type=_csv_str, default=['chrom'], help='chunk-key (dimension) names, comma-separated')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack, comma-separated')
    p.add_argument('-d', '--key_cols', type=_csv_int, default=[0], help='chunk-key column indices')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('--header', default=None, help='header row')
    p.add_argument('--skiprows', type=int, default=0, help='rows to skip')
    p.add_argument('-m', '--message', default='', help='message stored in header')
    p.add_argument('-l', '--level', type=int, default=6, help='compression level')
    p.add_argument('--delta_cols', type=_csv_str, default=None,
                   help='comma-separated column names or indices (from parameter --columns) to store '
                        'as in-block deltas (shrinks strictly-monotonic '
                        'columns like pos; trades some query speed for size)')

    # ---- catcz --------------------------------------------------------------
    p = sub.add_parser('catcz', help='Concatenate multiple .cz files into one', formatter_class=_fmt)
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-I', '--input', required=True, help='input pattern or comma-separated .cz paths')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats, comma-separated')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names, comma-separated')
    p.add_argument('-D', '--chunk_dims', type=_csv_str, default=['chrom'], help='chunk-key (dimension) names, comma-separated')
    p.add_argument('--chunk_order', default=None, help='chunk-key order file or comma-separated')
    p.add_argument('--key_added', default='cell_id',
                   help="name of an extra chunk_dim derived from each input's basename "
                        "(default: 'cell_id'); pass empty string to disable")
    p.add_argument('-m', '--message', default='', help='message stored in header')

    # ---- view ----------------------------------------------------------------
    p = sub.add_parser('view', help='View .cz file contents', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('--show_dims', type=_csv_int, default=None, help='chunk-key (dimension) indices to show')
    p.add_argument('--no_header', action='store_true', help='suppress header line')
    p.add_argument('-K', '--chunk_order', default=None, help='filter/order by chunk-key value (e.g. chr1)')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')

    # ---- header --------------------------------------------------------------
    p = sub.add_parser('header', help='Print header of a .cz file', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')

    # ---- check ---------------------------------------------------------------
    p = sub.add_parser('check', help='Check whether a local .cz file is complete',
                       formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, nargs='+',
                   help='input .cz file(s) to check')
    p.add_argument('-q', '--quiet', action='store_true',
                   help='suppress per-file output; exit code 0 = all complete')

    # ---- query ---------------------------------------------------------------
    p = sub.add_parser('query',
                       help='Query .cz file by chunk-key and position range '
                            '(bounds INCLUSIVE: [start, end])',
                       description=(
                           'Query a .cz file by chunk-key and position '
                           'range.\n\n'
                           'Coordinate semantics: --start/--end are '
                           'INCLUSIVE on the value stored in the sort '
                           'column, i.e. [start, end] (NOT BED-style '
                           '0-based half-open [start, end)). The '
                           'coordinate base follows the file: ALLC-derived '
                           '.cz (allc_to_cz / bam_to_cz) is 1-based; '
                           'BED-derived .cz keeps the source base as-is. '
                           'This differs from tabix, which is 0-based '
                           'half-open.'),
                       formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-K', '--chunk_key', default=None, help='chunk-key value to query (e.g. chr1)')
    p.add_argument('-s', '--start', type=int, default=None,
                   help='start of inclusive query interval (1-based for '
                        'ALLC-derived .cz)')
    p.add_argument('-e', '--end', type=int, default=None,
                   help='end of inclusive query interval; the record at '
                        'exactly --end IS included (1-based for '
                        'ALLC-derived .cz)')
    p.add_argument('--regions', default=None, help='regions file (tab-separated, no header)')
    p.add_argument('-q', '--query_col', type=_csv_int, default=[0], help='column indices to query on')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')

    # ---- to_bgzip ------------------------------------------------------------
    p = sub.add_parser('to_bgzip', help='Convert .cz to bgzip-compressed allc.tsv.gz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .allc.tsv.gz file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')
    p.add_argument('-K', '--chunk_order', default=None, help='filter/order by chunk-key value (e.g. chr1)')
    p.add_argument('--cov_col', default=None,
                   help="Name of a column from cz header['columns'] (such as cov). When given, rows where the column "
                        "is zero are dropped before writing (allc convention). Default "
                        "``None`` keeps all rows (suitable for array data, where 0 is "
                        "a valid beta value). if given, drop rows where this column is 0 (allc convention)")
    p.add_argument('--allc_format', action='store_true',
                   help='append a 7th mc_flag=1 column to produce the '
                        'standard ALLCools allc.tsv.gz 7-column layout')
    p.add_argument('--no_tabix', action='store_true', help='skip tabix indexing')

    # ---- summary_chunks / summary_blocks ------------------------------------
    p = sub.add_parser('summary', help='Print chunk summary of a .cz file', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('--blocks', action='store_true', help='show block-level detail instead of chunk-level')

    # ---- extract -------------------------------------------------------------
    p = sub.add_parser('extract', help='Extract subset of .cz using index', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('--index', required=True, help='subset index file')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel worker processes across chunks; a catcz\'d '
                        'multi-cell input has many chunks so jobs>1 gives a '
                        'near-linear speed-up')

    # ---- allc2cz --------------------------------------------------------------
    p = sub.add_parser('allc2cz', help='Convert tabix-indexed allc.tsv.gz to .cz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True,
                   help='input allc.tsv.gz, OR a directory containing many allc.tsv.gz '
                        '(batch mode: --output must be a directory)')
    p.add_argument('-O', '--output', required=True,
                   help='output .cz file (single-file), or output directory (batch mode)')
    p.add_argument('-r', '--reference', default=None, help='reference .cz file')
    p.add_argument('--missing_value', type=_csv_int, default=[0, 0], help='missing value fill')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'],
                   help="comma-separated column formats (struct chars). "
                        "When reference is provided, we only need to pack mc and cov,"
                        "['H', 'H'] is suggested for pseudobulk data (H is unsigned short integer, only 2 bytes),"
                        "and ['B', 'B'] is suggested for single cell data (B is unsigned char, only 1 byte)."
                        "if reference is not provided, we also need to pack position (Q is"
                        "recommanded), in this case, formats should be ['Q','H','H'].")
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names, comma-separated')
    p.add_argument('-D', '--chunk_dims', type=_csv_str, default=['chrom'], help='chunk-key names, comma-separated')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack')
    p.add_argument('--ref_pos_col', type=int, default=0, help='position column index in reference')
    p.add_argument('--allc_pos_col', type=int, default=1, help='position column index in input')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('--chroms', default=None, help='chrom order file')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('--sort_col', default=None,
                   help='column name or index to index via per-block '
                        'first_coords (enables in-memory bisect for region '
                        'queries). Auto-enabled on "pos" when no reference is used.')
    p.add_argument('--delta_cols', type=_csv_str, default=None,
                   help='comma-separated integer column names/indices to '
                        'store as in-block deltas')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='number of parallel workers in batch mode (input is a '
                        'directory). Reference is decoded once and shared via '
                        'fork copy-on-write, so memory cost is paid only once.')
    p.add_argument('--pattern', default='*.allc.tsv.gz',
                   help='glob pattern for batch mode file discovery')
    p.add_argument('--no_skip_existing', action='store_true',
                   help='in batch mode, do NOT skip files whose output already exists')

    # ---- build_ref (AllC) ----------------------------------------------------
    p = sub.add_parser('build_ref', help='Extract C positions from reference genome', formatter_class=_fmt)
    p.add_argument('-g', '--genome', required=True, help='reference genome FASTA')
    p.add_argument('-O', '--output', default='hg38_allc.cz', help='output .cz file')
    p.add_argument('-p', '--pattern', default='C', help='nucleotide pattern')
    p.add_argument('-j', '--jobs', type=int, default=12, help='number of parallel processes (CPUs)')
    p.add_argument('--keep_temp', action='store_true', help='keep temp directory')
    p.add_argument('-s', '--chroms', default=None,
                   help="Path to a `.fai` index file or a plain text file whose first "
                        "(tab-separated, no header) column lists chromosome names. When "
                        "provided, only these chromosomes are extracted, and the merged "
                        "reference `.cz` stores chunks in exactly this order. `None` "
                        "(default) processes every sequence in the genome fasta.")
    p.add_argument('--no_delta', action='store_true',
                   help='disable DELTA encoding on the pos column (default: on, '
                        'gives ~3x smaller reference files with mild query overhead)')

    # ---- index ---------------------------------------------------------------
    # Nested subcommand: `czip index <kind> ...` — produces a subset
    # coordinate index (.cz file) over a reference allc .cz.
    p = sub.add_parser('index', help='Build a coordinate index (context / regions / probes)',
                       formatter_class=_fmt)
    idx_sub = p.add_subparsers(dest='index_kind',
                               help='index kind: context | regions | probes')

    # --- index context (motif / CGN / CHN context pattern) --------------------
    sp = idx_sub.add_parser('context',
                            help='Index sites by sequence context (CGN/CHN/+CGN)',
                            formatter_class=_fmt)
    sp.add_argument('-I', '--input', required=True, help='input reference .cz file')
    sp.add_argument('-O', '--output', default=None, help='output index .cz file')
    sp.add_argument('-p', '--pattern', default='CGN',
                    help='IUPAC context pattern, optional +/- strand prefix '
                         '(e.g. CGN, CHN, +CGN, CAC, CAG, CHG, CWG)')
    sp.add_argument('-j', '--jobs', type=int, default=4,
                    help='number of parallel processes (one shard per chunk key)')
    sp.add_argument('-k', '--chunk-keys', dest='chunk_keys', default=None,
                    help='restrict to chunk keys (typically chromosomes): '
                         'comma-separated list (chr1,chr2,...) or path to a '
                         'chrom-sizes-like file (uses the first '
                         'whitespace-separated column)')

    # --- index regions (BED-based region subset) ------------------------------
    sp = idx_sub.add_parser('regions',
                            help='Index sites by genomic regions from a BED file',
                            formatter_class=_fmt)
    sp.add_argument('-I', '--input', required=True, help='input reference .cz file')
    sp.add_argument('-O', '--output', default=None, help='output index .cz file')
    sp.add_argument('-b', '--bed', required=True, help='BED file with regions')
    sp.add_argument('-j', '--jobs', type=int, default=4, help='number of parallel processes (CPUs)')
    sp.add_argument('-k', '--chunk-keys', dest='chunk_keys', default=None,
                    help='restrict to chunk keys (typically chromosomes): '
                         'comma-separated list (chr1,chr2,...) or path to a '
                         'chrom-sizes-like file (uses the first '
                         'whitespace-separated column)')

    # --- index probes (methylation array probe manifest) ----------------------
    # Planned: maps illumina EPIC / 450K probe IDs to ref primary_id + pos.
    # Placeholder CLI is in place; implementation will follow.
    sp = idx_sub.add_parser('probes',
                            help='Index methylation array probes (EPIC / 450K) — NOT YET IMPLEMENTED',
                            formatter_class=_fmt)
    sp.add_argument('-I', '--input', required=True, help='input reference .cz file (full-C allc)')
    sp.add_argument('-O', '--output', default=None, help='output probe index .cz file')
    sp.add_argument('--manifest', required=True, help='illumina manifest CSV')

    # ---- merge_cz ------------------------------------------------------------
    p = sub.add_parser(
        'merge_cz',
        help='Sum-merge per-cell .cz files (or a single pre-catcz\'d .cz)',
        formatter_class=_fmt,
    )
    p.add_argument('-i', '--input', default=None,
                   help='Unified input. Accepts ANY of: '
                        '(a) directory of per-cell .cz files; '
                        '(b) single per-cell .cz path; '
                        '(c) single pre-catcz\'d .cz path (chunk_dims >= 2; '
                        'catcz step is then skipped); '
                        '(d) comma-separated list of .cz paths.')
    p.add_argument('--class_table', default=None,
                   help='cell class table; requires --input to be a directory')
    p.add_argument('-O', '--output', default=None, help='output .cz path')
    p.add_argument('--prefix', default=None,
                   help='output filename prefix (used when -O/--output is unset)')
    p.add_argument('-j', '--jobs', type=int, default=12,
                   help='number of parallel worker processes')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'],
                   help='per-column struct formats for the output .cz, '
                        'comma-separated (default H,H = uint16 mc,cov)')
    p.add_argument('--chroms', default=None,
                   help='chrom-size file; output chunks are emitted in the '
                        'order of its first column when set')
    p.add_argument('-r', '--reference', default=None,
                   help='unused for sum mode (kept for API compatibility)')
    p.add_argument('--keep_cat', action='store_true',
                   help='keep intermediate output.cat.cz; no effect when '
                        'input is a pre-catcz\'d .cz (always kept)')
    p.add_argument('--blocks_per_batch', type=int, default=None,
                   help='number of batches the LARGEST chrom is split into '
                        '(default = jobs). Smaller chroms get 1 batch each '
                        'via the single-shard rename fast-path.')
    p.add_argument('--temp', action='store_true',
                   help='keep per-shard tmp directory')
    p.add_argument('--no_bgzip', action='store_true',
                   help='skip final bgzip + tabix (only applies if output '
                        'does not already end with --ext)')
    p.add_argument('-c', '--batch_size', type=int, default=50000,
                   help='per-worker buffer pack size (rows); does not affect '
                        'output, only peak worker memory')
    p.add_argument('--ext', default='.cz', help='input file extension')
    p.add_argument('-l', '--level', type=int, default=6,
                   help='DEFLATE compression level for output blocks '
                        '(1=fastest, 6=default, 9=smallest). Level=1 is ~2x '
                        'faster at ~12%% larger output.')
    p.add_argument('--agg', default='sum',
                   help='aggregation across cells/samples: "sum" (default, '
                        'BS-seq mc/cov) or "mean" (e.g. methylation-array '
                        'beta). Comma-separated values give per-column agg.')

    # ---- merge_cell_type -----------------------------------------------------
    p = sub.add_parser('merge_cell_type', help='Merge by cell type', formatter_class=_fmt)
    p.add_argument('-i', '--indir', default=None, help='input directory')
    p.add_argument('--cell_table', default=None, help='cell-type table')
    p.add_argument('-O', '--outdir', default=None, help='output directory')
    p.add_argument('-j', '--jobs', type=int, default=64, help='number of parallel processes (CPUs)')
    p.add_argument('--chroms', default=None, help='chrom order file')
    p.add_argument('--ext', default='.CGN.merged.cz', help='input file extension')

    # ---- pivot_fraction ------------------------------------------------------
    for _name, _help in (
        ('pivot_fraction', 'Pivot per-cell .cz into a wide mc/cov fraction TSV'),
        ('pivot_fisher', 'Pivot per-cell .cz into a one-vs-rest Fisher TSV'),
    ):
        p = sub.add_parser(_name, help=_help, formatter_class=_fmt)
        p.add_argument('-i', '--indir', default=None, help='input directory')
        p.add_argument('--cz_paths', default=None, help='file listing .cz paths')
        p.add_argument('-O', '--output', default=None, help='output .txt file')
        p.add_argument('--prefix', default=None, help='output prefix')
        p.add_argument('-j', '--jobs', type=int, default=12, help='number of parallel processes (CPUs)')
        p.add_argument('--chroms', default=None, help='chrom order file')
        p.add_argument('-r', '--reference', default=None, help='reference .cz file (adds chrom/start/pos columns)')
        p.add_argument('--keep_cat', action='store_true', help='keep intermediate cat file')
        p.add_argument('--blocks_per_batch', type=int, default=None, help='blocks per batch (auto if unset)')
        p.add_argument('--temp', action='store_true', help='keep temp directory')
        p.add_argument('--no_bgzip', action='store_true', help='skip bgzip compression')
        p.add_argument('-c', '--batch_size', type=int, default=50000, help='rows per chunk')
        p.add_argument('--ext', default='.cz', help='input file extension')

    # ---- extractCG -----------------------------------------------------------
    p = sub.add_parser('extractCG', help='Extract CG-context records', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('--index', required=True, help='CGN subset index file')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('--merge_cg', action='store_true', help='merge forward/reverse CG')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel worker processes across chunks; a catcz\'d '
                        'multi-cell input has many chunks so jobs>1 gives a '
                        'near-linear speed-up')

    # ---- compare_allc --------------------------------------------------------
    p = sub.add_parser('compare_allc', help='Compare two allc.tsv[.gz] files', formatter_class=_fmt)
    p.add_argument('-a', '--allc1', required=True, help='first allc.tsv[.gz] file')
    p.add_argument('-b', '--allc2', required=True, help='second allc.tsv[.gz] file')
    p.add_argument('--keep_zero_cov', action='store_true',
                   help='keep records with cov==0 (default: drop them)')
    p.add_argument('--dtype', default='B', type=_dtype_or_none,
                   choices=['B', 'H', 'I', 'Q', None],
                   help='unsigned-int format to derive mc/cov clamp bound '
                        '(B=255, H=65535, I=2^32-1, Q=2^64-1; '
                        'None=disable clamping; default: B). With B, mc/cov '
                        '>255 are truncated to 255 to mirror how cytozip '
                        'packs single-cell mc/cov as 1-byte B; counts >255 '
                        'are usually repeat-region artifacts and downstream '
                        'ALLCools DMR clips coverage to 50, so this does not '
                        'affect downstream analysis')
    p.add_argument('-O', '--output', default=None, help='output TSV of differing rows')
    p.add_argument('--sep', default='\t', help='column separator of input files')

    # ---- aggregate -----------------------------------------------------------
    p = sub.add_parser('aggregate', help='Aggregate records within regions', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('--index', required=True, help='region subset index file')
    p.add_argument('--intersect', default=None, help='intersect filter')
    p.add_argument('--exclude', default=None, help='exclude filter')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats, comma-separated')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel worker processes across chunks; a catcz\'d '
                        'multi-cell input has many chunks so jobs>1 gives a '
                        'near-linear speed-up')

    # ---- call_dmr_array ------------------------------------------------------
    p = sub.add_parser('call_dmr_array',
                       help='Call DMRs on methylation array .cz files (450K / '
                            'EPIC / MSA) via per-probe Welch t (or MW) on '
                            '\u03b2 + comb-p spatial merge',
                       formatter_class=_fmt)
    p.add_argument('-a', '--group_a', required=True,
                   help='comma-separated array .cz paths or a text file listing them')
    p.add_argument('-b', '--group_b', required=True,
                   help='comma-separated array .cz paths or a text file listing them')
    p.add_argument('-r', '--reference', required=True,
                   help='reference .cz with probe coordinates (must have a "pos" column)')
    p.add_argument('-O', '--output', required=True, help='output DMR TSV path')
    p.add_argument('--beta_col', default='beta',
                   help='\u03b2-value column name in each per-sample .cz')
    p.add_argument('--test', choices=['t', 'mw'], default='t',
                   help='per-probe test (t = Welch t on M-values, '
                        'mw = Mann-Whitney U on \u03b2)')
    p.add_argument('--group_names', type=_csv_str, default=['A', 'B'],
                   help='comma-separated group labels (header / log only)')
    p.add_argument('--sidak_p_cutoff', type=float, default=0.05)
    p.add_argument('--delta_beta_cutoff', type=float, default=0.05)
    p.add_argument('--min_samples_per_group', type=int, default=2)
    p.add_argument('--max_dist', type=int, default=1000,
                   help='comb-p region max gap (bp); array probes are sparse, '
                        'default 1000')
    p.add_argument('--acf_dist', type=int, default=None,
                   help='ACF distance (default: round(max_dist/3, -1))')
    p.add_argument('--keep_temp', action='store_true',
                   help='keep <output>.tmp/ with per-chrom BED + cpv outputs')
    p.add_argument('--chroms', default=None,
                   help='restrict to these chromosomes: a path to a '
                        'chrom-size / .fai file (or any text file whose first '
                        'column is the chromosome), OR a comma-separated list '
                        '(chr1,chr2,...)')
    p.add_argument('--probe_pvalues_output', default=None,
                   help='also dump the per-probe (chrom, pos, p, delta_beta) TSV here')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel processes used for the per-chrom comb-p step')

    # ---- annot_dmr -----------------------------------------------------------
    p = sub.add_parser('annot_dmr', help='Annotate DMRs', formatter_class=_fmt)
    p.add_argument('-I', '--input', default='merged_dmr.txt', help='merged DMR file')
    p.add_argument('--matrix', default='merged_dmr.cell_class.beta.txt', help='beta matrix file')
    p.add_argument('-O', '--output', default='dmr.annotated.txt', help='output file')
    p.add_argument('--delta_cutoff', type=float, default=None, help='min delta-beta cutoff')

    # ---- call_dmr ------------------------------------------------------------
    p = sub.add_parser('call_dmr',
                       help='Call DMRs between two groups of single-cell .cz '
                            'files (permutation RMS test, ALLCools/Methylpy style)',
                       formatter_class=_fmt)
    p.add_argument('-a', '--group_a', required=True,
                   help='comma-separated .cz paths or a text file listing them (one per line)')
    p.add_argument('-b', '--group_b', required=True,
                   help='comma-separated .cz paths or a text file listing them (one per line)')
    p.add_argument('-r', '--reference', required=True,
                   help='reference .cz file (pos column)')
    p.add_argument('-O', '--output', required=True,
                   help='output DMR TSV path')
    p.add_argument('-s', '--index', default=None,
                   help='context index .cz (e.g., CGN-only) to restrict sites')
    p.add_argument('--dms_output', default=None,
                   help='also write the per-site DMS TSV here')
    p.add_argument('--group_names', default='A,B',
                   help='comma-separated group names (header / log only)')
    p.add_argument('--p_value_cutoff', type=float, default=0.001)
    p.add_argument('--frac_delta_cutoff', type=float, default=0.2)
    p.add_argument('--min_cov', type=int, default=1,
                   help='per-cell min coverage at a site to contribute')
    p.add_argument('--min_samples_per_group', type=int, default=1)
    p.add_argument('--max_dist', type=int, default=250,
                   help='max gap (bp) between adjacent DMS for merging')
    p.add_argument('--min_dms', type=int, default=1)
    p.add_argument('--n_permute', type=int, default=3000)
    p.add_argument('--min_pvalue', type=float, default=0.01,
                   help='permutation early-stopping threshold (ALLCools default)')
    p.add_argument('--max_row_count', type=int, default=50)
    p.add_argument('--max_total_count', type=int, default=3000)
    p.add_argument('--mc_col', default=None,
                   help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov_col', default=None,
                   help='cov column name or 0-based index (default: last column)')
    p.add_argument('--chroms', default=None,
                   help='restrict to these chromosomes: a path to a '
                        'chrom-size / .fai file (or any text file whose first '
                        'column is the chromosome), OR a comma-separated list '
                        '(chr1,chr2,...)')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='total CPU cores to use (auto-split into '
                        'processes across chunks and OpenMP threads)')
    p.add_argument('--no-delta_prefilter', dest='delta_prefilter',
                   action='store_false', default=True,
                   help='disable the |Delta-frac| pre-filter (run RMS on '
                        'every site)')
    p.add_argument('--no-fisher_1v1', dest='use_fisher_1v1',
                   action='store_false', default=True,
                   help='disable Fisher-exact fallback for 1-vs-1 comparisons')

    # ---- call_dmr_ch ---------------------------------------------------------
    p = sub.add_parser('call_dmr_ch',
                       help='Call non-CG (CH/CA/CT) DMRs with bin aggregation '
                            '+ global mCH normalization + log2fc filter',
                       formatter_class=_fmt)
    p.add_argument('-a', '--group_a', required=True,
                   help='comma-separated .cz paths or a text file listing them (one per line)')
    p.add_argument('-b', '--group_b', required=True,
                   help='comma-separated .cz paths or a text file listing them (one per line)')
    p.add_argument('-r', '--reference', required=True,
                   help='reference .cz file (pos column)')
    p.add_argument('-O', '--output', required=True,
                   help='output DMR TSV path')
    p.add_argument('-s', '--index', default=None,
                   help='context index .cz (e.g., CAN-only) to restrict sites')
    p.add_argument('--dms_output', default=None,
                   help='also write the per-bin DMS TSV here')
    p.add_argument('--bin_size', type=int, default=500,
                   help='bin width (bp) for pooling per-cell mc/cov counts')
    p.add_argument('--context', default='CHN',
                   help='label used in log messages (CHN/CAN/CTN/...)')
    p.add_argument('--group_names', default='A,B')
    p.add_argument('--p_value_cutoff', type=float, default=0.001)
    p.add_argument('--log2fc_cutoff', type=float, default=1.0,
                   help='min |log2(rate_a/rate_b)| after normalization')
    p.add_argument('--abs_delta_cutoff', type=float, default=0.005,
                   help='min |rate_a - rate_b| after normalization')
    p.add_argument('--no_normalize', action='store_true',
                   help='disable per-cell global mCH rescaling pre-pass')
    p.add_argument('--min_cov', type=int, default=3)
    p.add_argument('--min_samples_per_group', type=int, default=2)
    p.add_argument('--max_dist', type=int, default=2000)
    p.add_argument('--min_dms', type=int, default=2)
    p.add_argument('--n_permute', type=int, default=10000)
    p.add_argument('--min_pvalue', type=float, default=0.001)
    p.add_argument('--max_row_count', type=int, default=200)
    p.add_argument('--max_total_count', type=int, default=10000)
    p.add_argument('--mc_col', default=None)
    p.add_argument('--cov_col', default=None)
    p.add_argument('--chroms', default=None,
                   help='restrict to these chromosomes: a path to a '
                        'chrom-size / .fai file (or any text file whose first '
                        'column is the chromosome), OR a comma-separated list '
                        '(chr1,chr2,...)')
    p.add_argument('-j', '--jobs', type=int, default=1)
    p.add_argument('--no-delta_prefilter', dest='delta_prefilter',
                   action='store_false', default=True,
                   help='disable the bin-level pre-filter (run RMS on every bin)')

    # ---- call_dmr_one_vs_rest ------------------------------------------------
    p = sub.add_parser('call_dmr_one_vs_rest',
                       help='Batch one-vs-rest DMR over a folder of pseudobulk '
                            '.cz, optionally stratified by a sample-class TSV',
                       formatter_class=_fmt)
    p.add_argument('-d', '--indir', required=True,
                   help='directory of pseudobulk .cz files')
    p.add_argument('-r', '--reference', required=True,
                   help='reference .cz file')
    p.add_argument('-O', '--outdir', required=True,
                   help='output directory (per-sample DMR TSVs are written '
                        'here, or under <outdir>/<class>/ when stratified)')
    p.add_argument('--ext', default='.cz',
                   help="suffix used to discover pseudobulks; "
                        "sname = filename[:-len(ext)]")
    p.add_argument('--method', default='cg', choices=['cg', 'ch'],
                   help='cg -> call_dmr; ch -> call_dmr_ch')
    p.add_argument('-c', '--class_table', default=None,
                   help='optional 2-col TSV (sname<TAB>class); restricts each '
                        'one-vs-rest comparison to within the same class')
    p.add_argument('--min_class_size', type=int, default=2)
    p.add_argument('-s', '--index', default=None,
                   help='context index .cz forwarded to the DMR caller')
    p.add_argument('--dms_output_dir', default=None,
                   help='if given, also write per-sample DMS TSVs here')
    p.add_argument('--overwrite', action='store_true',
                   help='re-run even if the output TSV exists')
    p.add_argument('-j', '--jobs', type=int, default=1)
    # forwarded knobs (kept minimal; users wanting full control can use the
    # Python API and **dmr_kwargs)
    p.add_argument('--p_value_cutoff', type=float, default=None)
    p.add_argument('--frac_delta_cutoff', type=float, default=None,
                   help='CG only')
    p.add_argument('--log2fc_cutoff', type=float, default=None,
                   help='CH only')
    p.add_argument('--abs_delta_cutoff', type=float, default=None,
                   help='CH only')
    p.add_argument('--bin_size', type=int, default=None,
                   help='CH only')
    p.add_argument('--no_normalize', action='store_true',
                   help='CH only: disable per-cell global mCH rescaling')
    p.add_argument('--min_cov', type=int, default=None)
    p.add_argument('--min_samples_per_group', type=int, default=None)
    p.add_argument('--max_dist', type=int, default=None)
    p.add_argument('--min_dms', type=int, default=None)
    p.add_argument('--n_permute', type=int, default=None)
    p.add_argument('--chroms', default=None,
                   help='restrict to these chromosomes: a path to a '
                        'chrom-size / .fai file (or any text file whose first '
                        'column is the chromosome), OR a comma-separated list '
                        '(chr1,chr2,...)')
    p.add_argument('--no-delta_prefilter', dest='delta_prefilter',
                   action='store_false', default=True,
                   help='forwarded to the underlying caller')
    p.add_argument('--no-fisher_1v1', dest='use_fisher_1v1',
                   action='store_false', default=True,
                   help='CG only: disable Fisher-exact fallback for 1-vs-1')
    p.add_argument('--no-merge', dest='auto_merge', action='store_false',
                   default=True,
                   help='skip the auto merge_dmr_results step')
    p.add_argument('--no-fdr', dest='add_fdr', action='store_false',
                   default=True,
                   help='skip BH-FDR q_min computation in the merge step')
    p.add_argument('--output_format', default=None,
                   choices=[None, 'tsv', 'parquet'],
                   help='format of merged_dmr output (default: tsv)')

    # ---- call_peaks ----------------------------------------------------------
    p = sub.add_parser('call_peaks', help='Call peaks from methylation .cz using MACS3', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file (pos/strand/context)')
    p.add_argument('-O', '--output', default=None, help='output directory for MACS3 results')
    p.add_argument('-n', '--name', default='peaks', help='name prefix for output files')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth'], help='signal type: unmeth=(cov-mc), meth=mc')
    p.add_argument('--control', default=None, choices=['cov', 'mc'], help='coverage-bias control track for macs3 -c (cov recommended); default none')
    p.add_argument('--index', default=None, help='index file for context filtering (e.g., CpG-only)')
    p.add_argument('--genome_size', default='mm', help='genome size for MACS3 (hs/mm/integer)')
    p.add_argument('--fragment_size', type=int, default=300, help='pseudo-read fragment size (bp)')
    p.add_argument('--qvalue', type=float, default=0.05, help='MACS3 q-value cutoff')
    p.add_argument('--broad', action='store_true', help='call broad peaks')
    p.add_argument('--min_cov', type=int, default=1, help='minimum coverage to include a site')
    p.add_argument('--keep_bed', action='store_true', help='keep intermediate pseudo-reads BED')
    p.add_argument('--macs3_args', default='', help='additional MACS3 arguments (quoted string)')
    p.add_argument('--mc_col', default=None, help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov_col', default=None, help='cov column name or 0-based index (default: last column)')

    # ---- call_peaks_bdg ------------------------------------------------------
    p = sub.add_parser('call_peaks_bdg', help='Call peaks from methylation .cz via MACS3 bedGraph back-end (memory-efficient, coverage-controlled)', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file (pos/strand/context)')
    p.add_argument('-O', '--output', default=None, help='output directory for peak results')
    p.add_argument('-n', '--name', default='peaks', help='name prefix for output files')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth'], help='treatment signal: unmeth=(cov-mc), meth=mc')
    p.add_argument('--control', default='cov', choices=['cov', 'mc'], help='coverage-bias control (lambda) track (default cov)')
    p.add_argument('--index', default=None, help='index file for context filtering (e.g., CpG-only)')
    p.add_argument('--ext', type=int, default=300, help='bp each site count is spread over (pileup extension)')
    p.add_argument('--method', default='ppois', help='macs3 bdgcmp score method (ppois/qpois/FE/logLR/...)')
    p.add_argument('--cutoff', type=float, default=2.0, help='bdgpeakcall score cutoff (-log10 p for ppois)')
    p.add_argument('--min_len', type=int, default=None, help='minimum peak length (default ext)')
    p.add_argument('--max_gap', type=int, default=None, help='max gap to merge peaks (default ext//2)')
    p.add_argument('--min_cov', type=int, default=1, help='minimum coverage to include a site')
    p.add_argument('--keep_bdg', action='store_true', help='keep intermediate bedGraph tracks')
    p.add_argument('--mc_col', default=None, help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov_col', default=None, help='cov column name or 0-based index (default: last column)')

    # ---- to_bedgraph ---------------------------------------------------------
    p = sub.add_parser('to_bedgraph', help='Export methylation signal as bedGraph', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file')
    p.add_argument('-O', '--output', default=None, help='output bedGraph file')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth', 'frac_unmeth'], help='signal type')
    p.add_argument('--index', default=None, help='index file for context filtering')
    p.add_argument('--min_cov', type=int, default=1, help='minimum coverage to include a site')
    p.add_argument('--mc_col', default=None, help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov_col', default=None, help='cov column name or 0-based index (default: last column)')

    # ---- bam_to_cz -----------------------------------------------------------
    p = sub.add_parser('bam_to_cz', help='Convert position-sorted BAM directly to .cz (skip ALLC text)', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input position-sorted BAM (bismark/hisat-3n)')
    p.add_argument('-g', '--genome', required=True, help='indexed reference fasta (.fai required)')
    p.add_argument('-O', '--output', default=None, help='output .cz path (default: <bam_stem>.cz)')
    p.add_argument('--num_upstr_bases', type=int, default=0, help='bases upstream of C in context (0 for BS-seq, 1 for NOMe)')
    p.add_argument('--num_downstr_bases', type=int, default=2, help='bases downstream of C in context')
    p.add_argument('--min_mapq', type=int, default=10, help='min read MAPQ (applied by the htslib backend, or passed to samtools mpileup)')
    p.add_argument('--min_base_quality', type=int, default=20, help='min base quality (applied by the htslib backend, or passed to samtools mpileup)')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per batch (one on-disk chunk)')
    p.add_argument('--convert_bam_strandness', type=_str2bool, default=True,
                   metavar='BOOL',
                   help='count methylation on the strand implied by the '
                        'bisulfite-conversion tag (XG for bismark, YZ for '
                        'hisat-3n) instead of the alignment orientation. '
                        'Required for hisat-3n / bismark PE data; no '
                        'temporary BAM is written (strand is derived '
                        'in-process by the htslib backend, or streamed via a '
                        'pipe to the mpileup fallback). Default True; pass '
                        '"false" for plain strand-correct BAMs (e.g. bismark SE)')
    p.add_argument('--save_count_df', action='store_true', help='write <output>.count.csv context summary')
    p.add_argument('--mode', choices=['full', 'pos_mc_cov', 'mc_cov'], default='mc_cov',
                   help='storage layout: full=[pos,strand,context,mc,cov]; '
                        'pos_mc_cov=[pos,mc,cov]; mc_cov=[mc,cov] (requires --reference). '
                        'Default mc_cov is the most compact.')
    p.add_argument('--count_fmt', choices=['B', 'H', 'I', 'Q'], default='B',
                   help='struct code for mc/cov columns: B=uint8 (1 B/max 255, clipped), '
                        'H=uint16 (2 B/max 65535). B is the most compact and suits '
                        'typical single-cell data.')
    p.add_argument('-r', '--reference', default=None,
                   help='reference .cz with pos column; required when --mode mc_cov')
    p.add_argument('--name_sorted', action='store_true',
                   help='input BAM is name-sorted; coordinate-sort + picard '
                        'MarkDuplicates it first (via name_sort_bam_to_deduped)')
    p.add_argument('--env', default=None,
                   help='conda env providing picard/samtools for --name_sorted '
                        '(e.g. yap); bare name or full env prefix path')
    p.add_argument('--chroms', default=None,
                   help='restrict the pileup to these chromosomes: '
                        'comma-separated list (chr1,chr2,...) or a path to a '
                        'chrom-size / .fai file (first column). Mirrors '
                        'ALLCools bam_to_allc --chrom_size; without it every '
                        'contig in the genome fasta (alt/decoy/unplaced) is '
                        'piled up and counted. In --mode mc_cov the pileup is '
                        'always limited to the reference chroms regardless.')

    # ---- cz_to_anndata -------------------------------------------------------
    p = sub.add_parser('cz_to_anndata', help='Aggregate many single-cell .cz files over a feature BED into AnnData h5ad', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, nargs='+',
                   help='input .cz file(s) or a directory; may also be one catcz-merged .cz with cell dim')
    p.add_argument('-f', '--features', required=True,
                   help='BED / BED.gz / BED.bgz path, GTF / GTF.gz path, or an int bin size in bp (e.g. 5000) for genome-wide tiling (then --chrom_size is required)')
    p.add_argument('-O', '--output', default=None, help='output .h5ad path')
    p.add_argument('--use_samples', type=_csv_str, default=None,
                   help='comma-separated whitelist of sample names to '
                        'include (default: all)')
    p.add_argument('--ext', default='.cz',
                   help='filename suffix stripped from per-file basenames '
                        'to derive sample names (default: .cz)')
    p.add_argument('--pos_col', default='pos', help='name of position column in .cz header')
    p.add_argument('--mc_col', default='mc', help='name of mc column')
    p.add_argument('--cov_col', default='cov', help='name of cov column')
    p.add_argument('--obs', default=None, help='optional TSV with cell metadata (index column = cell id)')
    p.add_argument('-r', '--reference', default=None,
                   help='reference .cz supplying pos coords for mc_cov-only cells')
    p.add_argument('--chrom_size', default=None,
                   help='chrom-size / .fai file (required when --features is an int bin size)')
    p.add_argument('--exclude_chroms', type=_csv_str, default=['chrL'],
                   help='comma-separated chroms to drop (genome-bin tiling only)')
    p.add_argument('--blacklist', default=None,
                   help='BED / bed.gz of regions to exclude before aggregation')
    p.add_argument('--flank_bp', type=int, default=2000,
                   help='bp to extend each side of GTF gene intervals (GTF input only)')
    p.add_argument('--gtf_id_col', choices=['gene_name', 'gene_id'],
                   default='gene_name',
                   help='which GTF attribute becomes var_names (GTF input only)')
    p.add_argument('--score', choices=['frac', 'hypo-score', 'hyper-score',
                                       'mc', 'cov', 'umc'],
                   default='frac',
                   help='what to store in .X (mc/cov/umc place raw counts in .X)')
    p.add_argument('--score_cutoff', type=float, default=0.9,
                   help='sparsification threshold for hypo/hyper scores')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='number of parallel processes (CPUs)')

    # ---- model.py: predict_cell_type ---------------------------------------
    p = sub.add_parser('predict_cell_type',
                       help='Classify query cell(s) against cell-type pseudobulk references',
                       formatter_class=_fmt)
    p.add_argument('-q', '--query', required=True,
                   help='query cell(s): a single .cz, a cat multi-cell .cz, a directory of .cz, '
                        'or a 2-column [cell_id, cz_path] table')
    p.add_argument('-s', '--pseudobulks', default=None,
                   help='per-type pseudobulk .cz: a directory (file stem = cell type). '
                        'Optional if -o/--outdir already holds a trained model')
    p.add_argument('-e', '--reference', default=None,
                   help='build_ref reference .cz supplying per-row context (CpG/CpH split)')
    p.add_argument('-o', '--outdir', default=None,
                   help='write/reuse the fitted model under <outdir>/model and predictions '
                        '(predictions.csv + predict_proba.csv) to <outdir>')
    p.add_argument('--cell_counts', default=None,
                   help='optional 2-column [cell_type, count] TSV for the abundance prior')
    p.add_argument('--prior_alpha', type=float, default=0.0,
                   help='abundance-prior strength (0=uniform; needs --cell_counts)')
    p.add_argument('--lambda_cg', type=float, default=1.0, help='CpG channel log-weight')
    p.add_argument('--lambda_ch', type=float, default=1.0, help='CpH channel log-weight')
    p.add_argument('--max_query_cg', type=_int_or_none, default=None,
                   help='randomly keep at most this many covered CpG sites per query cell')
    p.add_argument('--max_query_ch', type=_int_or_none, default=None,
                   help='randomly keep at most this many covered CpH sites per query cell')
    p.add_argument('--alpha0_cg', type=_float_or_none, default=None, help='CpG Beta-prior alpha0 (auto if None)')
    p.add_argument('--beta0_cg', type=_float_or_none, default=None, help='CpG Beta-prior beta0 (auto if None)')
    p.add_argument('--alpha0_ch', type=_float_or_none, default=None, help='CpH Beta-prior alpha0 (auto if None)')
    p.add_argument('--beta0_ch', type=_float_or_none, default=None, help='CpH Beta-prior beta0 (auto if None)')
    p.add_argument('--prior_min_cov', type=int, default=2,
                   help='min coverage for a site to enter empirical prior estimation')
    p.add_argument('--top_cg', type=_int_or_float_or_none, default=None,
                   help='keep top-N (int) or top-fraction (float in (0,1]) discriminative CpG sites')
    p.add_argument('--top_ch', type=_int_or_float_or_none, default=None,
                   help='keep top-N (int) or top-fraction (float in (0,1]) discriminative CpH sites')
    p.add_argument('--min_range_cg', type=float, default=0.0,
                   help='keep a CpG site only if its across-type frequency range exceeds this')
    p.add_argument('--min_range_ch', type=float, default=0.0,
                   help='keep a CpH site only if its across-type frequency range exceeds this')
    p.add_argument('--mc_col', default='mc', help='methylated-count column name')
    p.add_argument('--cov_col', default='cov', help='coverage column name')
    p.add_argument('--context_col', default='context', help='context column name (in the reference)')
    p.add_argument('--abstain_threshold', type=_float_or_none, default=None,
                   help="label a cell 'unassigned' when its top probability is below this")
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel .cz readers / per-cell scoring threads')

    # ---- model.py: deconvolve_bulk -----------------------------------------
    p = sub.add_parser('deconvolve_bulk',
                       help='Deconvolve bulk sample(s) into cell-type fractions',
                       formatter_class=_fmt)
    p.add_argument('-q', '--query', required=True,
                   help='bulk sample(s): a single .cz, a cat multi-sample .cz, a directory of .cz, '
                        'or a 2-column [sample_id, cz_path] table')
    p.add_argument('-s', '--pseudobulks', default=None,
                   help='per-type pseudobulk .cz: a directory (file stem = cell type). '
                        'Optional if -o/--outdir already holds a trained model')
    p.add_argument('-e', '--reference', default=None,
                   help='build_ref reference .cz supplying per-row context (CpG/CpH split)')
    p.add_argument('-o', '--outdir', default=None,
                   help='write/reuse the fitted model under <outdir>/model and write '
                        'fractions.csv to <outdir>')
    p.add_argument('--contexts', default='cg',
                   help="cytosine contexts to use: 'cg', 'ch', or 'cg+ch'")
    p.add_argument('--weight_by_cov', type=_str2bool, default=True,
                   help='weight each site by its bulk coverage (weighted least squares)')
    p.add_argument('--sum_to_one', type=_str2bool, default=True,
                   help='constrain the fractions to sum to 1')
    p.add_argument('--allow_unknown', type=_str2bool, default=False,
                   help="relax to sum<=1 and report the remainder as an 'unknown' fraction")
    p.add_argument('--min_cov', type=int, default=1, help='only use bulk sites with coverage >= this')
    p.add_argument('--alpha0_cg', type=_float_or_none, default=None, help='CpG Beta-prior alpha0 (auto if None)')
    p.add_argument('--beta0_cg', type=_float_or_none, default=None, help='CpG Beta-prior beta0 (auto if None)')
    p.add_argument('--alpha0_ch', type=_float_or_none, default=None, help='CpH Beta-prior alpha0 (auto if None)')
    p.add_argument('--beta0_ch', type=_float_or_none, default=None, help='CpH Beta-prior beta0 (auto if None)')
    p.add_argument('--prior_min_cov', type=int, default=2,
                   help='min coverage for a site to enter empirical prior estimation')
    p.add_argument('--top_cg', type=_int_or_float_or_none, default=None,
                   help='keep top-N (int) or top-fraction (float in (0,1]) discriminative CpG sites')
    p.add_argument('--top_ch', type=_int_or_float_or_none, default=None,
                   help='keep top-N (int) or top-fraction (float in (0,1]) discriminative CpH sites')
    p.add_argument('--min_range_cg', type=float, default=0.0,
                   help='keep a CpG site only if its across-type frequency range exceeds this')
    p.add_argument('--min_range_ch', type=float, default=0.0,
                   help='keep a CpH site only if its across-type frequency range exceeds this')
    p.add_argument('--mc_col', default='mc', help='methylated-count column name')
    p.add_argument('--cov_col', default='cov', help='coverage column name')
    p.add_argument('--context_col', default='context', help='context column name (in the reference)')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='parallel .cz readers / per-sample threads')

    return parser


def main():
    """CLI entry point using argparse for fast startup."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cmd = args.command

    # ---- cz.py commands (Writer/Reader/extract) ----------------------------
    if cmd == 'tocz':
        from .cz import Writer
        w = Writer(output=args.output, formats=args.formats,
                   columns=args.columns, chunk_dims=args.chunk_dims,
                   message=args.message, level=args.level,
                   delta_cols=args.delta_cols)
        w.tocz(input=args.input, usecols=args.usecols,
               key_cols=args.key_cols, sep=args.sep,
               batch_size=args.batch_size, header=args.header,
               skiprows=args.skiprows)

    elif cmd == 'catcz':
        from .cz import Writer
        w = Writer(output=args.output, formats=args.formats,
                   columns=args.columns, chunk_dims=args.chunk_dims,
                   message=args.message)
        # CLI passes either a glob with '*' or a comma-separated path list.
        inp = args.input
        if isinstance(inp, str) and '*' not in inp:
            inp = [p for p in inp.split(',') if p]
        w.catcz(input=inp, chunk_order=args.chunk_order,
                key_added=(args.key_added or None))

    elif cmd == 'view':
        from .cz import Reader
        r = Reader(args.input)
        r.view(show_dims=args.show_dims, header=not args.no_header,
               chunk_order=args.chunk_order, reference=args.reference)

    elif cmd == 'header':
        from .cz import Reader
        r = Reader(args.input)
        r.print_header()
        r.close()

    elif cmd == 'check':
        import sys as _sys
        from .cz import check_cz
        all_ok = True
        for path in args.input:
            ok, reason = check_cz(path)
            all_ok = all_ok and ok
            if not args.quiet:
                status = 'OK' if ok else 'INCOMPLETE'
                print(f"{status}\t{path}" + ('' if ok else f"\t{reason}"))
        _sys.exit(0 if all_ok else 1)

    elif cmd == 'query':
        from .cz import Reader
        r = Reader(args.input)
        r.query(chunk_key=args.chunk_key, start=args.start,
                end=args.end, regions=args.regions,
                query_col=args.query_col, reference=args.reference,
                printout=True)

    elif cmd == 'to_bgzip':
        from .cz import Reader
        r = Reader(args.input)
        r.to_bgzip(output=args.output, reference=args.reference,
                   chunk_order=args.chunk_order, tabix=not args.no_tabix,
                   cov_col=args.cov_col, allc_format=args.allc_format)
        r.close()

    elif cmd == 'summary':
        from .cz import Reader
        r = Reader(args.input)
        if args.blocks:
            r.summary_blocks(printout=True)
        else:
            r.summary_chunks(printout=True)

    elif cmd == 'extract':
        from .cz import extract
        extract(input=args.input, output=args.output,
                index=args.index, batch_size=args.batch_size, jobs=args.jobs)

    # ---- allc.py commands --------------------------------------------------
    elif cmd == 'allc2cz':
        from .allc import allc2cz
        allc2cz(input=args.input, output=args.output,
               reference=args.reference, missing_value=args.missing_value,
               formats=args.formats, columns=args.columns,
               chunk_dims=args.chunk_dims, usecols=args.usecols,
               ref_pos_col=args.ref_pos_col, allc_pos_col=args.allc_pos_col, sep=args.sep,
               chroms=args.chroms, batch_size=args.batch_size,
               sort_col=args.sort_col, delta_cols=args.delta_cols,
               jobs=args.jobs, pattern=args.pattern,
               skip_existing=not args.no_skip_existing)

    elif cmd == 'build_ref':
        from .allc import AllC
        a = AllC(genome=args.genome, output=args.output,
                 pattern=args.pattern, jobs=args.jobs,
                 keep_temp=args.keep_temp, delta=not args.no_delta,
                 chroms=args.chroms)
        a.run()

    elif cmd == 'index':
        kind = getattr(args, 'index_kind', None)
        if kind is None:
            parser.parse_args(['index', '--help'])
            return
        if kind == 'context':
            from .index import index_context
            index_context(input=args.input, output=args.output,
                          pattern=args.pattern, jobs=args.jobs,
                          chunk_keys=args.chunk_keys)
        elif kind == 'regions':
            from .index import index_regions
            index_regions(input=args.input, output=args.output,
                          bed=args.bed, jobs=args.jobs,
                          chunk_keys=args.chunk_keys)
        elif kind == 'probes':
            raise NotImplementedError(
                "`czip index probes` is not implemented yet. "
                "Planned: build a probe_id \u2192 (chrom, primary_id, pos) index "
                "from an illumina EPIC / 450K manifest, reusing the same "
                "vectorised worker / catcz pattern as `index_context` and `index_regions`."
            )
        else:
            raise ValueError(f"unknown index kind: {kind!r}")

    elif cmd == 'merge_cz':
        from .merge import merge_cz
        # --agg: 'sum' / 'mean' / comma-list (parsed into a per-col list).
        agg_arg = args.agg
        if isinstance(agg_arg, str) and ',' in agg_arg:
            agg_arg = [s.strip() for s in agg_arg.split(',') if s.strip()]
        merge_cz(input=args.input,
                 class_table=args.class_table, output=args.output,
                 prefix=args.prefix, jobs=args.jobs,
                 formats=args.formats, chroms=args.chroms,
                 reference=args.reference, keep_cat=args.keep_cat,
                 blocks_per_batch=args.blocks_per_batch, temp=args.temp,
                 bgzip=not args.no_bgzip, batch_size=args.batch_size,
                 ext=args.ext, level=args.level, agg=agg_arg)

    elif cmd == 'merge_cell_type':
        from .merge import merge_cell_type
        merge_cell_type(indir=args.indir, cell_table=args.cell_table,
                        outdir=args.outdir, jobs=args.jobs,
                        chroms=args.chroms, ext=args.ext)

    elif cmd == 'pivot_fraction':
        from .pivot import pivot_fraction
        pivot_fraction(
            indir=args.indir, cz_paths=args.cz_paths,
            output=args.output, prefix=args.prefix, jobs=args.jobs,
            chroms=args.chroms, reference=args.reference,
            keep_cat=args.keep_cat,
            blocks_per_batch=args.blocks_per_batch, temp=args.temp,
            bgzip=not args.no_bgzip, batch_size=args.batch_size,
            ext=args.ext)

    elif cmd == 'pivot_fisher':
        from .pivot import pivot_fisher
        pivot_fisher(
            indir=args.indir, cz_paths=args.cz_paths,
            output=args.output, prefix=args.prefix, jobs=args.jobs,
            chroms=args.chroms, reference=args.reference,
            keep_cat=args.keep_cat,
            blocks_per_batch=args.blocks_per_batch, temp=args.temp,
            bgzip=not args.no_bgzip, batch_size=args.batch_size,
            ext=args.ext)

    elif cmd == 'extractCG':
        from .allc import extractCG
        extractCG(input=args.input, output=args.output,
                  index=args.index, batch_size=args.batch_size,
                  merge_cg=args.merge_cg, jobs=args.jobs)

    elif cmd == 'compare_allc':
        from .allc import compare_allc
        compare_allc(allc1=args.allc1, allc2=args.allc2,
                     drop_zero_cov=not args.keep_zero_cov,
                     dtype=args.dtype, output=args.output,
                     sep=args.sep)

    elif cmd == 'aggregate':
        from .cz import aggregate
        aggregate(input=args.input, output=args.output,
                  index=args.index, intersect=args.intersect,
                  exclude=args.exclude, batch_size=args.batch_size,
                  formats=args.formats, jobs=args.jobs)

    elif cmd == 'call_dmr_array':
        from .dmr import call_dmr_array
        gn = (tuple(args.group_names) if args.group_names else ('A', 'B'))
        call_dmr_array(
            group_a=args.group_a, group_b=args.group_b,
            reference=args.reference, output=args.output,
            beta_col=args.beta_col, test=args.test,
            group_names=gn,
            sidak_p_cutoff=args.sidak_p_cutoff,
            delta_beta_cutoff=args.delta_beta_cutoff,
            min_samples_per_group=args.min_samples_per_group,
            max_dist=args.max_dist, acf_dist=args.acf_dist,
            keep_temp=args.keep_temp, chroms=args.chroms,
            probe_pvalues_output=args.probe_pvalues_output,
            jobs=args.jobs,
        )

    elif cmd == 'annot_dmr':
        from .dmr import annot_dmr
        annot_dmr(input=args.input, matrix=args.matrix,
                  output=args.output, delta_cutoff=args.delta_cutoff)

    elif cmd == 'call_dmr':
        from .dmr import call_dmr
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and isinstance(mc_col, str) and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and isinstance(cov_col, str) and cov_col.isdigit():
            cov_col = int(cov_col)
        gn = args.group_names.split(',') if args.group_names else ('A', 'B')
        call_dmr(group_a=args.group_a, group_b=args.group_b,
                 reference=args.reference, output=args.output,
                 group_names=tuple(gn[:2]),
                 p_value_cutoff=args.p_value_cutoff,
                 frac_delta_cutoff=args.frac_delta_cutoff,
                 min_cov=args.min_cov,
                 min_samples_per_group=args.min_samples_per_group,
                 max_dist=args.max_dist, min_dms=args.min_dms,
                 n_permute=args.n_permute, min_pvalue=args.min_pvalue,
                 max_row_count=args.max_row_count,
                 max_total_count=args.max_total_count,
                 mc_col=mc_col, cov_col=cov_col,
                 index=args.index, dms_output=args.dms_output,
                 chroms=args.chroms, jobs=args.jobs,
                 delta_prefilter=args.delta_prefilter,
                 use_fisher_1v1=args.use_fisher_1v1)

    elif cmd == 'call_dmr_ch':
        from .dmr import call_dmr_ch
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and isinstance(mc_col, str) and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and isinstance(cov_col, str) and cov_col.isdigit():
            cov_col = int(cov_col)
        gn = args.group_names.split(',') if args.group_names else ('A', 'B')
        call_dmr_ch(group_a=args.group_a, group_b=args.group_b,
                    reference=args.reference, output=args.output,
                    bin_size=args.bin_size, context=args.context,
                    group_names=tuple(gn[:2]),
                    p_value_cutoff=args.p_value_cutoff,
                    log2fc_cutoff=args.log2fc_cutoff,
                    abs_delta_cutoff=args.abs_delta_cutoff,
                    normalize=not args.no_normalize,
                    min_cov=args.min_cov,
                    min_samples_per_group=args.min_samples_per_group,
                    max_dist=args.max_dist, min_dms=args.min_dms,
                    n_permute=args.n_permute, min_pvalue=args.min_pvalue,
                    max_row_count=args.max_row_count,
                    max_total_count=args.max_total_count,
                    mc_col=mc_col, cov_col=cov_col,
                    index=args.index, dms_output=args.dms_output,
                    chroms=args.chroms, jobs=args.jobs,
                    delta_prefilter=args.delta_prefilter)

    elif cmd == 'call_dmr_one_vs_rest':
        from .dmr import call_dmr_one_vs_rest
        # Forward only the knobs the user actually set (so the wrapper
        # falls back to call_dmr / call_dmr_ch defaults otherwise).
        forward = {}
        for k in ('p_value_cutoff', 'frac_delta_cutoff', 'log2fc_cutoff',
                  'abs_delta_cutoff', 'bin_size', 'min_cov',
                  'min_samples_per_group', 'max_dist', 'min_dms',
                  'n_permute', 'chroms'):
            v = getattr(args, k, None)
            if v is not None:
                forward[k] = v
        if args.method == 'ch' and args.no_normalize:
            forward['normalize'] = False
        # delta_prefilter / use_fisher_1v1 are honored by the underlying
        # caller; forward them only when the user opted out (defaults are True).
        if not args.delta_prefilter:
            forward['delta_prefilter'] = False
        if args.method == 'cg' and not args.use_fisher_1v1:
            forward['use_fisher_1v1'] = False
        call_dmr_one_vs_rest(
            indir=args.indir, reference=args.reference, outdir=args.outdir,
            ext=args.ext, method=args.method,
            class_table=args.class_table,
            min_class_size=args.min_class_size,
            overwrite=args.overwrite,
            jobs=args.jobs, index=args.index,
            dms_output_dir=args.dms_output_dir,
            auto_merge=args.auto_merge,
            merge_kwargs={'add_fdr': args.add_fdr,
                          'output_format': args.output_format},
            **forward)

    elif cmd == 'call_peaks':
        from .peaks import call_peaks
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and cov_col.isdigit():
            cov_col = int(cov_col)
        call_peaks(input=args.input, reference=args.reference,
                   output=args.output, name=args.name,
                   signal=args.signal, control=args.control,
                   index=args.index,
                   genome_size=args.genome_size,
                   fragment_size=args.fragment_size,
                   qvalue=args.qvalue, broad=args.broad,
                   min_cov=args.min_cov, keep_bed=args.keep_bed,
                   macs3_args=args.macs3_args,
                   mc_col=mc_col, cov_col=cov_col)

    elif cmd == 'call_peaks_bdg':
        from .peaks import call_peaks_bdg
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and cov_col.isdigit():
            cov_col = int(cov_col)
        call_peaks_bdg(input=args.input, reference=args.reference,
                       output=args.output, name=args.name,
                       signal=args.signal, control=args.control,
                       index=args.index, ext=args.ext, method=args.method,
                       cutoff=args.cutoff, min_len=args.min_len,
                       max_gap=args.max_gap, min_cov=args.min_cov,
                       keep_bdg=args.keep_bdg,
                       mc_col=mc_col, cov_col=cov_col)

    elif cmd == 'to_bedgraph':
        from .peaks import to_bedgraph
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and cov_col.isdigit():
            cov_col = int(cov_col)
        to_bedgraph(input=args.input, reference=args.reference,
                    output=args.output, signal=args.signal,
                    index=args.index, min_cov=args.min_cov,
                    mc_col=mc_col, cov_col=cov_col)

    elif cmd == 'bam_to_cz':
        from .bam import bam_to_cz
        bam_to_cz(bam_path=args.input, genome=args.genome,
                  output=args.output,
                  mode=args.mode,
                  count_fmt=args.count_fmt,
                  reference=args.reference,
                  num_upstr_bases=args.num_upstr_bases,
                  num_downstr_bases=args.num_downstr_bases,
                  min_mapq=args.min_mapq,
                  min_base_quality=args.min_base_quality,
                  batch_size=args.batch_size,
                  convert_bam_strandness=args.convert_bam_strandness,
                  save_count_df=args.save_count_df,
                  name_sorted=args.name_sorted,
                  env=args.env,
                  chroms=args.chroms)

    elif cmd == 'name_sort_bam_to_deduped':
        from .bam import name_sort_bam_to_deduped
        name_sort_bam_to_deduped(bam_path=args.input, output=args.output,
                                 stats=args.stats,
                                 remove_duplicates=not args.no_remove_duplicates,
                                 tmp_dir=args.tmp_dir,
                                 sort_threads=args.sort_threads,
                                 sort_mem_mb=args.sort_mem_mb,
                                 index=not args.no_index,
                                 keep_pos_sort=args.keep_pos_sort,
                                 env=args.env)

    elif cmd == 'cz_to_anndata':
        from .features import cz_to_anndata
        inputs = args.input if len(args.input) > 1 else args.input[0]
        obs_df = None
        if args.obs:
            import pandas as pd
            obs_df = pd.read_csv(args.obs, sep='\t', index_col=0)
        # Allow --features <int> for genome-wide bin tiling.
        feats = args.features
        try:
            feats = int(feats)
        except (TypeError, ValueError):
            pass
        cz_to_anndata(cz_inputs=inputs, features=feats,
                      output=args.output, use_samples=args.use_samples,
                      ext=args.ext,
                      pos_col=args.pos_col, mc_col=args.mc_col,
                      cov_col=args.cov_col, obs=obs_df,
                      reference=args.reference,
                      chrom_size=args.chrom_size,
                      exclude_chroms=args.exclude_chroms,
                      blacklist=args.blacklist,
                      flank_bp=args.flank_bp,
                      gtf_id_col=args.gtf_id_col,
                      score=args.score,
                      score_cutoff=args.score_cutoff,
                      jobs=args.jobs)

    elif cmd == 'predict_cell_type':
        from .model import predict_cell_type
        cell_counts = None
        if args.cell_counts:
            import pandas as pd
            _cc = pd.read_csv(args.cell_counts, sep='\t', header=None, comment='#')
            cell_counts = {str(k): int(v)
                           for k, v in zip(_cc.iloc[:, 0], _cc.iloc[:, 1])}
        predict_cell_type(
            query=args.query, pseudobulks=args.pseudobulks,
            reference=args.reference, cell_counts=cell_counts,
            prior_alpha=args.prior_alpha,
            lambda_cg=args.lambda_cg, lambda_ch=args.lambda_ch,
            max_query_cg=args.max_query_cg, max_query_ch=args.max_query_ch,
            alpha0_cg=args.alpha0_cg, beta0_cg=args.beta0_cg,
            alpha0_ch=args.alpha0_ch, beta0_ch=args.beta0_ch,
            prior_min_cov=args.prior_min_cov,
            top_cg=args.top_cg, top_ch=args.top_ch,
            min_range_cg=args.min_range_cg, min_range_ch=args.min_range_ch,
            mc_col=args.mc_col, cov_col=args.cov_col,
            context_col=args.context_col,
            abstain_threshold=args.abstain_threshold,
            n_jobs=args.jobs, outdir=args.outdir)

    elif cmd == 'deconvolve_bulk':
        from .model import deconvolve_bulk
        deconvolve_bulk(
            query=args.query, pseudobulks=args.pseudobulks,
            reference=args.reference, contexts=args.contexts,
            weight_by_cov=args.weight_by_cov, sum_to_one=args.sum_to_one,
            allow_unknown=args.allow_unknown, min_cov=args.min_cov,
            alpha0_cg=args.alpha0_cg, beta0_cg=args.beta0_cg,
            alpha0_ch=args.alpha0_ch, beta0_ch=args.beta0_ch,
            prior_min_cov=args.prior_min_cov,
            top_cg=args.top_cg, top_ch=args.top_ch,
            min_range_cg=args.min_range_cg, min_range_ch=args.min_range_ch,
            mc_col=args.mc_col, cov_col=args.cov_col,
            context_col=args.context_col,
            n_jobs=args.jobs, outdir=args.outdir)


if __name__ == "__main__":
    main()