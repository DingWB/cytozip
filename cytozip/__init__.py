"""cytozip (ChunkZIP) — Chunk-based columnar binary format for fast random access.

Public API:
  - Reader:  Read and query .cz files (local or remote via HTTP).
  - Writer:  Create .cz files from tabular data.
  - AllC:    Extract cytosine positions from a reference genome.
  - allc2cz:  Convert tabix-indexed allc.tsv.gz to .cz format.
  - merge_cz: Merge multiple per-cell .cz files.
  - extract / extractCG / aggregate:  Subset and aggregate .cz data.
  - combp / annot_dmr:  Differential methylation analysis.

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
    'index_regions': 'cz', 'aggregate': 'cz',
    # allc.py — methylation allc-file I/O
    'AllC': 'allc', 'allc2cz': 'allc', 'index_context': 'allc',
    'extractCG': 'allc',
    # bam.py — BAM → .cz
    'bam_to_cz': 'bam',
    # features.py — feature aggregation / anndata
    'cz_to_anndata': 'features', 'parse_features': 'features',
    'parse_gtf': 'features', 'make_genome_bins': 'features',
    # merge.py — per-cell merging pipeline
    'merge_cz': 'merge', 'merge_cell_type': 'merge',
    # pivot.py — per-cell pivot matrices (fraction / fisher)
    'pivot_fraction': 'pivot', 'pivot_fisher': 'pivot',
    # dmr.py — peak calling / DMR analysis
    'call_peaks': 'dmr', 'to_bedgraph': 'dmr',
    'combp': 'dmr', 'annot_dmr': 'dmr',
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
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats, comma-separated')
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
                   help='comma-separated integer column names/indices to store '
                        'as in-block deltas (shrinks strictly-monotonic '
                        'columns like pos; trades some query speed for size)')

    # ---- catcz --------------------------------------------------------------
    p = sub.add_parser('catcz', help='Concatenate multiple .cz files into one', formatter_class=_fmt)
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-I', '--input', required=True, help='input pattern or comma-separated .cz paths')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names')
    p.add_argument('-D', '--chunk_dims', type=_csv_str, default=['chrom'], help='chunk-key (dimension) names')
    p.add_argument('--chunk_order', default=None, help='chunk-key order file or comma-separated')
    p.add_argument('--add_key', action='store_true', help='add filename as extra chunk key')
    p.add_argument('--title', default='filename', help='title for added chunk key')
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

    # ---- query ---------------------------------------------------------------
    p = sub.add_parser('query', help='Query .cz file by chunk-key and position range', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-K', '--chunk_key', default=None, help='chunk-key value to query (e.g. chr1)')
    p.add_argument('-s', '--start', type=int, default=None, help='start position')
    p.add_argument('-e', '--end', type=int, default=None, help='end position')
    p.add_argument('--regions', default=None, help='regions file (tab-separated, no header)')
    p.add_argument('-q', '--query_col', type=_csv_int, default=[0], help='column indices to query on')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')

    # ---- to_allc -------------------------------------------------------------
    p = sub.add_parser('to_allc', help='Convert .cz to allc.tsv.gz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .allc.tsv.gz file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')
    p.add_argument('-K', '--chunk_order', default=None, help='filter/order by chunk-key value (e.g. chr1)')
    p.add_argument('--cov_col', default=None, help='coverage column name; rows with 0 are dropped (default: last data column)')
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

    # ---- allc2cz --------------------------------------------------------------
    p = sub.add_parser('allc2cz', help='Convert tabix-indexed allc.tsv.gz to .cz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True,
                   help='input allc.tsv.gz, OR a directory containing many allc.tsv.gz '
                        '(batch mode: --output must be a directory)')
    p.add_argument('-O', '--output', required=True,
                   help='output .cz file (single-file), or output directory (batch mode)')
    p.add_argument('-r', '--reference', default=None, help='reference .cz file')
    p.add_argument('--missing_value', type=_csv_int, default=[0, 0], help='missing value fill')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names')
    p.add_argument('-D', '--chunk_dims', type=_csv_str, default=['chrom'], help='chunk-key names')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack')
    p.add_argument('--ref_pos_col', type=int, default=0, help='position column index in reference')
    p.add_argument('--allc_pos_col', type=int, default=1, help='position column index in input')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('--chrom_order', default=None, help='chrom order file')
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
    p.add_argument('--no_delta', action='store_true',
                   help='disable DELTA encoding on the pos column (default: on, '
                        'gives ~3x smaller reference files with mild query overhead)')

    # ---- index ---------------------------------------------------------------
    # Nested subcommand: `czip index <kind> ...` — produces a subset
    # coordinate index (.cz file) over a reference allc .cz. Replaces the
    # old `index_context` / `index_regions` names.
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
                    help='context pattern: CGN / CHN / +CGN (strand-specific)')

    # --- index regions (BED-based region subset) ------------------------------
    sp = idx_sub.add_parser('regions',
                            help='Index sites by genomic regions from a BED file',
                            formatter_class=_fmt)
    sp.add_argument('-I', '--input', required=True, help='input reference .cz file')
    sp.add_argument('-O', '--output', default=None, help='output index .cz file')
    sp.add_argument('-b', '--bed', required=True, help='BED file with regions')
    sp.add_argument('-j', '--jobs', type=int, default=4, help='number of parallel processes (CPUs)')

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
                   help='per-column struct formats for the output .cz '
                        '(default H,H = uint16 mc,cov)')
    p.add_argument('--chrom_order', default=None,
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

    # ---- merge_cell_type -----------------------------------------------------
    p = sub.add_parser('merge_cell_type', help='Merge by cell type', formatter_class=_fmt)
    p.add_argument('-i', '--indir', default=None, help='input directory')
    p.add_argument('--cell_table', default=None, help='cell-type table')
    p.add_argument('-O', '--outdir', default=None, help='output directory')
    p.add_argument('-j', '--jobs', type=int, default=64, help='number of parallel processes (CPUs)')
    p.add_argument('--chrom_order', default=None, help='chrom order file')
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
        p.add_argument('--chrom_order', default=None, help='chrom order file')
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

    # ---- aggregate -----------------------------------------------------------
    p = sub.add_parser('aggregate', help='Aggregate records within regions', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('--index', required=True, help='region subset index file')
    p.add_argument('--intersect', default=None, help='intersect filter')
    p.add_argument('--exclude', default=None, help='exclude filter')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per chunk')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats')

    # ---- combp ---------------------------------------------------------------
    p = sub.add_parser('combp', help='Run comb-p on Fisher results', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input Fisher result')
    p.add_argument('-O', '--outdir', default='cpv', help='output directory')
    p.add_argument('-j', '--jobs', type=int, default=24, help='number of parallel processes (CPUs)')
    p.add_argument('--dist', type=int, default=300, help='max distance between sites')
    p.add_argument('--temp', action='store_true', help='keep temp directory')
    p.add_argument('--bed', action='store_true', help='keep bed directory')

    # ---- annot_dmr -----------------------------------------------------------
    p = sub.add_parser('annot_dmr', help='Annotate DMRs', formatter_class=_fmt)
    p.add_argument('-I', '--input', default='merged_dmr.txt', help='merged DMR file')
    p.add_argument('--matrix', default='merged_dmr.cell_class.beta.txt', help='beta matrix file')
    p.add_argument('-O', '--output', default='dmr.annotated.txt', help='output file')
    p.add_argument('--delta_cutoff', type=float, default=None, help='min delta-beta cutoff')

    # ---- call_peaks ----------------------------------------------------------
    p = sub.add_parser('call_peaks', help='Call peaks from methylation .cz using MACS3', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file (pos/strand/context)')
    p.add_argument('-O', '--output', default=None, help='output directory for MACS3 results')
    p.add_argument('-n', '--name', default='peaks', help='name prefix for output files')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth'], help='signal type: unmeth=(cov-mc), meth=mc')
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
    p.add_argument('--min_mapq', type=int, default=10, help='min MAPQ passed to samtools mpileup')
    p.add_argument('--min_base_quality', type=int, default=20, help='min base quality passed to samtools mpileup')
    p.add_argument('-c', '--batch_size', type=int, default=5000, help='rows per batch (one on-disk chunk)')
    p.add_argument('--convert_bam_strandness', action='store_true', help='rewrite BAM so is_forward matches XG/YZ (hisat-3n PE)')
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

    # ---- cz_to_anndata -------------------------------------------------------
    p = sub.add_parser('cz_to_anndata', help='Aggregate many single-cell .cz files over a feature BED into AnnData h5ad', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, nargs='+',
                   help='input .cz file(s) or a directory; may also be one catcz-merged .cz with cell dim')
    p.add_argument('-f', '--features', required=True,
                   help='BED / BED.gz / BED.bgz path, GTF / GTF.gz path, or an int bin size in bp (e.g. 5000) for genome-wide tiling (then --chrom_size is required)')
    p.add_argument('-O', '--output', default=None, help='output .h5ad path')
    p.add_argument('--cell_ids', type=_csv_str, default=None, help='optional override list of cell ids')
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
                add_key=args.add_key, title=args.title)

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

    elif cmd == 'query':
        from .cz import Reader
        r = Reader(args.input)
        r.query(chunk_key=args.chunk_key, start=args.start,
                end=args.end, regions=args.regions,
                query_col=args.query_col, reference=args.reference,
                printout=True)

    elif cmd == 'to_allc':
        from .cz import Reader
        r = Reader(args.input)
        r.to_allc(output=args.output, reference=args.reference,
                  chunk_order=args.chunk_order, tabix=not args.no_tabix,
                  cov_col=args.cov_col)
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
                index=args.index, batch_size=args.batch_size)

    # ---- allc.py commands --------------------------------------------------
    elif cmd == 'allc2cz':
        from .allc import allc2cz
        allc2cz(input=args.input, output=args.output,
               reference=args.reference, missing_value=args.missing_value,
               formats=args.formats, columns=args.columns,
               chunk_dims=args.chunk_dims, usecols=args.usecols,
               ref_pos_col=args.ref_pos_col, allc_pos_col=args.allc_pos_col, sep=args.sep,
               chrom_order=args.chrom_order, batch_size=args.batch_size,
               sort_col=args.sort_col, delta_cols=args.delta_cols,
               jobs=args.jobs, pattern=args.pattern,
               skip_existing=not args.no_skip_existing)

    elif cmd == 'build_ref':
        from .allc import AllC
        a = AllC(genome=args.genome, output=args.output,
                 pattern=args.pattern, jobs=args.jobs,
                 keep_temp=args.keep_temp, delta=not args.no_delta)
        a.run()

    elif cmd == 'index':
        kind = getattr(args, 'index_kind', None)
        if kind is None:
            parser.parse_args(['index', '--help'])
            return
        if kind == 'context':
            from .allc import index_context
            index_context(input=args.input, output=args.output,
                          pattern=args.pattern)
        elif kind == 'regions':
            from .cz import index_regions
            index_regions(input=args.input, output=args.output,
                          bed=args.bed, jobs=args.jobs)
        elif kind == 'probes':
            raise NotImplementedError(
                "`czip index probes` is not implemented yet. "
                "Planned: build a probe_id \u2192 (chrom, primary_id, pos) index "
                "from an illumina EPIC / 450K manifest, using the current "
                "`build_region_index` / `build_context_index` infrastructure as the base."
            )
        else:
            raise ValueError(f"unknown index kind: {kind!r}")

    elif cmd == 'merge_cz':
        from .merge import merge_cz
        merge_cz(input=args.input,
                 class_table=args.class_table, output=args.output,
                 prefix=args.prefix, jobs=args.jobs,
                 formats=args.formats, chrom_order=args.chrom_order,
                 reference=args.reference, keep_cat=args.keep_cat,
                 blocks_per_batch=args.blocks_per_batch, temp=args.temp,
                 bgzip=not args.no_bgzip, batch_size=args.batch_size,
                 ext=args.ext, level=args.level)

    elif cmd == 'merge_cell_type':
        from .merge import merge_cell_type
        merge_cell_type(indir=args.indir, cell_table=args.cell_table,
                        outdir=args.outdir, jobs=args.jobs,
                        chrom_order=args.chrom_order, ext=args.ext)

    elif cmd == 'pivot_fraction':
        from .pivot import pivot_fraction
        pivot_fraction(
            indir=args.indir, cz_paths=args.cz_paths,
            output=args.output, prefix=args.prefix, jobs=args.jobs,
            chrom_order=args.chrom_order, reference=args.reference,
            keep_cat=args.keep_cat,
            blocks_per_batch=args.blocks_per_batch, temp=args.temp,
            bgzip=not args.no_bgzip, batch_size=args.batch_size,
            ext=args.ext)

    elif cmd == 'pivot_fisher':
        from .pivot import pivot_fisher
        pivot_fisher(
            indir=args.indir, cz_paths=args.cz_paths,
            output=args.output, prefix=args.prefix, jobs=args.jobs,
            chrom_order=args.chrom_order, reference=args.reference,
            keep_cat=args.keep_cat,
            blocks_per_batch=args.blocks_per_batch, temp=args.temp,
            bgzip=not args.no_bgzip, batch_size=args.batch_size,
            ext=args.ext)

    elif cmd == 'extractCG':
        from .allc import extractCG
        extractCG(input=args.input, output=args.output,
                  index=args.index, batch_size=args.batch_size,
                  merge_cg=args.merge_cg)

    elif cmd == 'aggregate':
        from .cz import aggregate
        aggregate(input=args.input, output=args.output,
                  index=args.index, intersect=args.intersect,
                  exclude=args.exclude, batch_size=args.batch_size,
                  formats=args.formats)

    elif cmd == 'combp':
        from .dmr import combp
        combp(input=args.input, outdir=args.outdir,
              jobs=args.jobs, dist=args.dist,
              temp=args.temp, bed=args.bed)

    elif cmd == 'annot_dmr':
        from .dmr import annot_dmr
        annot_dmr(input=args.input, matrix=args.matrix,
                  output=args.output, delta_cutoff=args.delta_cutoff)

    elif cmd == 'call_peaks':
        from .dmr import call_peaks
        mc_col = args.mc_col
        cov_col = args.cov_col
        if mc_col is not None and mc_col.isdigit():
            mc_col = int(mc_col)
        if cov_col is not None and cov_col.isdigit():
            cov_col = int(cov_col)
        call_peaks(input=args.input, reference=args.reference,
                   output=args.output, name=args.name,
                   signal=args.signal, index=args.index,
                   genome_size=args.genome_size,
                   fragment_size=args.fragment_size,
                   qvalue=args.qvalue, broad=args.broad,
                   min_cov=args.min_cov, keep_bed=args.keep_bed,
                   macs3_args=args.macs3_args,
                   mc_col=mc_col, cov_col=cov_col)

    elif cmd == 'to_bedgraph':
        from .dmr import to_bedgraph
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
                  save_count_df=args.save_count_df)

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
                      output=args.output, cell_ids=args.cell_ids,
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


if __name__ == "__main__":
    main()
