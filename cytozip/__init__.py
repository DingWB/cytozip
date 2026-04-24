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
    # cz.py
    'Reader': 'cz', 'Writer': 'cz', 'RemoteFile': 'cz', 'extract': 'cz',
    # allc.py
    'AllC': 'allc', 'allc2cz': 'allc', 'index_context': 'allc',
    'index_regions': 'allc', 'merge_cz': 'allc', 'extractCG': 'allc',
    'aggregate': 'allc', 'merge_cell_type': 'allc', 'combp': 'allc',
    'annot_dmr': 'allc', 'call_peaks': 'allc', 'to_bedgraph': 'allc',
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
    p.add_argument('-D', '--dimensions', type=_csv_str, default=['chrom'], help='dimension names, comma-separated')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack, comma-separated')
    p.add_argument('-d', '--dim-cols', type=_csv_int, default=[0], help='dimension column indices')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('--header', default=None, help='header row')
    p.add_argument('--skiprows', type=int, default=0, help='rows to skip')
    p.add_argument('-m', '--message', default='', help='message stored in header')
    p.add_argument('-l', '--level', type=int, default=6, help='compression level')

    # ---- catcz --------------------------------------------------------------
    p = sub.add_parser('catcz', help='Concatenate multiple .cz files into one', formatter_class=_fmt)
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-I', '--input', required=True, help='input pattern or comma-separated .cz paths')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names')
    p.add_argument('-D', '--dimensions', type=_csv_str, default=['chrom'], help='dimension names')
    p.add_argument('--dim-order', default=None, help='dimension order file or comma-separated')
    p.add_argument('--add-dim', action='store_true', help='add filename as extra dimension')
    p.add_argument('--title', default='filename', help='title for added dimension')
    p.add_argument('-m', '--message', default='', help='message stored in header')

    # ---- view ----------------------------------------------------------------
    p = sub.add_parser('view', help='View .cz file contents', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('--show-dim', type=_csv_int, default=None, help='dimension indices to show')
    p.add_argument('--no-header', action='store_true', help='suppress header line')
    p.add_argument('--dimension', default=None, help='filter by dimension')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')

    # ---- header --------------------------------------------------------------
    p = sub.add_parser('header', help='Print header of a .cz file', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')

    # ---- query ---------------------------------------------------------------
    p = sub.add_parser('query', help='Query .cz file by dimension and position range', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-D', '--dimension', default=None, help='dimension value to query (e.g. chr1)')
    p.add_argument('-s', '--start', type=int, default=None, help='start position')
    p.add_argument('-e', '--end', type=int, default=None, help='end position')
    p.add_argument('--regions', default=None, help='regions file (tab-separated, no header)')
    p.add_argument('-q', '--query-col', type=_csv_int, default=[0], help='column indices to query on')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')

    # ---- to_allc -------------------------------------------------------------
    p = sub.add_parser('to_allc', help='Convert .cz to allc.tsv.gz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .allc.tsv.gz file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz for coordinate lookup')
    p.add_argument('-D', '--dimension', default=None, help='filter by dimension')
    p.add_argument('--cov-col', default=None, help='coverage column name; rows with 0 are dropped (default: last data column)')
    p.add_argument('--no-tabix', action='store_true', help='skip tabix indexing')

    # ---- summary_chunks / summary_blocks ------------------------------------
    p = sub.add_parser('summary', help='Print chunk summary of a .cz file', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('--blocks', action='store_true', help='show block-level detail instead of chunk-level')

    # ---- extract -------------------------------------------------------------
    p = sub.add_parser('extract', help='Extract subset of .cz using index', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-s', '--index', required=True, help='subset index file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')

    # ---- allc2cz --------------------------------------------------------------
    p = sub.add_parser('allc2cz', help='Convert tabix-indexed allc.tsv.gz to .cz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input allc.tsv.gz')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz file')
    p.add_argument('--missing-value', type=_csv_int, default=[0, 0], help='missing value fill')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names')
    p.add_argument('-D', '--dimensions', type=_csv_str, default=['chrom'], help='dimension names')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack')
    p.add_argument('--ref-pos-col', type=int, default=0, help='position column index in reference')
    p.add_argument('--allc-pos-col', type=int, default=1, help='position column index in input')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('--chrom-order', default=None, help='chrom order file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('--sort-col', default=None,
                   help='column name or index to index via per-block '
                        'first_coords (enables in-memory bisect for region '
                        'queries). Auto-enabled on "pos" when no reference is used.')

    # ---- build_ref (AllC) ----------------------------------------------------
    p = sub.add_parser('build_ref', help='Extract C positions from reference genome', formatter_class=_fmt)
    p.add_argument('-g', '--genome', required=True, help='reference genome FASTA')
    p.add_argument('-O', '--output', default='hg38_allc.cz', help='output .cz file')
    p.add_argument('-p', '--pattern', default='C', help='nucleotide pattern')
    p.add_argument('-t', '--threads', type=int, default=12, help='parallel jobs')
    p.add_argument('--keep-temp', action='store_true', help='keep temp directory')

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
    sp.add_argument('-t', '--threads', type=int, default=4, help='parallel jobs')

    # --- index probes (methylation array probe manifest) ----------------------
    # Planned: maps illumina EPIC / 450K probe IDs to ref primary_id + pos.
    # Placeholder CLI is in place; implementation will follow.
    sp = idx_sub.add_parser('probes',
                            help='Index methylation array probes (EPIC / 450K) — NOT YET IMPLEMENTED',
                            formatter_class=_fmt)
    sp.add_argument('-I', '--input', required=True, help='input reference .cz file (full-C allc)')
    sp.add_argument('-O', '--output', default=None, help='output probe index .cz file')
    sp.add_argument('-m', '--manifest', required=True, help='illumina manifest CSV')

    # ---- merge_cz ------------------------------------------------------------
    p = sub.add_parser('merge_cz', help='Merge per-cell .cz files', formatter_class=_fmt)
    p.add_argument('-i', '--indir', default=None, help='input directory')
    p.add_argument('--cz-paths', default=None, help='file listing .cz paths')
    p.add_argument('--class-table', default=None, help='cell class table')
    p.add_argument('-O', '--output', default=None, help='output file')
    p.add_argument('--prefix', default=None, help='output prefix')
    p.add_argument('-t', '--threads', type=int, default=12, help='parallel jobs')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats')
    p.add_argument('--chrom-order', default=None, help='chrom order file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz file')
    p.add_argument('--keep-cat', action='store_true', help='keep intermediate cat file')
    p.add_argument('--batchsize', type=int, default=10, help='blocks per batch')
    p.add_argument('--temp', action='store_true', help='keep temp directory')
    p.add_argument('--no-bgzip', action='store_true', help='skip bgzip compression')
    p.add_argument('-c', '--chunksize', type=int, default=50000, help='rows per chunk')
    p.add_argument('--ext', default='.cz', help='input file extension')

    # ---- merge_cell_type -----------------------------------------------------
    p = sub.add_parser('merge_cell_type', help='Merge by cell type', formatter_class=_fmt)
    p.add_argument('-i', '--indir', default=None, help='input directory')
    p.add_argument('--cell-table', default=None, help='cell-type table')
    p.add_argument('-O', '--outdir', default=None, help='output directory')
    p.add_argument('-t', '--threads', type=int, default=64, help='parallel jobs')
    p.add_argument('--chrom-order', default=None, help='chrom order file')
    p.add_argument('--ext', default='.CGN.merged.cz', help='input file extension')

    # ---- extractCG -----------------------------------------------------------
    p = sub.add_parser('extractCG', help='Extract CG-context records', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-s', '--index', required=True, help='CGN subset index file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('--merge-cg', action='store_true', help='merge forward/reverse CG')

    # ---- aggregate -----------------------------------------------------------
    p = sub.add_parser('aggregate', help='Aggregate records within regions', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', required=True, help='output .cz file')
    p.add_argument('-s', '--index', required=True, help='region subset index file')
    p.add_argument('--intersect', default=None, help='intersect filter')
    p.add_argument('--exclude', default=None, help='exclude filter')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats')

    # ---- combp ---------------------------------------------------------------
    p = sub.add_parser('combp', help='Run comb-p on Fisher results', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input Fisher result')
    p.add_argument('-O', '--outdir', default='cpv', help='output directory')
    p.add_argument('-t', '--threads', type=int, default=24, help='parallel jobs')
    p.add_argument('--dist', type=int, default=300, help='max distance between sites')
    p.add_argument('--temp', action='store_true', help='keep temp directory')
    p.add_argument('--bed', action='store_true', help='keep bed directory')

    # ---- annot_dmr -----------------------------------------------------------
    p = sub.add_parser('annot_dmr', help='Annotate DMRs', formatter_class=_fmt)
    p.add_argument('-I', '--input', default='merged_dmr.txt', help='merged DMR file')
    p.add_argument('--matrix', default='merged_dmr.cell_class.beta.txt', help='beta matrix file')
    p.add_argument('-O', '--output', default='dmr.annotated.txt', help='output file')
    p.add_argument('--delta-cutoff', type=float, default=None, help='min delta-beta cutoff')

    # ---- call_peaks ----------------------------------------------------------
    p = sub.add_parser('call_peaks', help='Call peaks from methylation .cz using MACS3', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file (pos/strand/context)')
    p.add_argument('-O', '--output', default=None, help='output directory for MACS3 results')
    p.add_argument('-n', '--name', default='peaks', help='name prefix for output files')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth'], help='signal type: unmeth=(cov-mc), meth=mc')
    p.add_argument('-s', '--index', default=None, help='index file for context filtering (e.g., CpG-only)')
    p.add_argument('-g', '--genome-size', default='mm', help='genome size for MACS3 (hs/mm/integer)')
    p.add_argument('--fragment-size', type=int, default=300, help='pseudo-read fragment size (bp)')
    p.add_argument('-q', '--qvalue', type=float, default=0.05, help='MACS3 q-value cutoff')
    p.add_argument('--broad', action='store_true', help='call broad peaks')
    p.add_argument('--min-cov', type=int, default=1, help='minimum coverage to include a site')
    p.add_argument('--keep-bed', action='store_true', help='keep intermediate pseudo-reads BED')
    p.add_argument('--macs3-args', default='', help='additional MACS3 arguments (quoted string)')
    p.add_argument('--mc-col', default=None, help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov-col', default=None, help='cov column name or 0-based index (default: last column)')

    # ---- to_bedgraph ---------------------------------------------------------
    p = sub.add_parser('to_bedgraph', help='Export methylation signal as bedGraph', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file (mc/cov)')
    p.add_argument('-r', '--reference', required=True, help='reference .cz file')
    p.add_argument('-O', '--output', default=None, help='output bedGraph file')
    p.add_argument('--signal', default='unmeth', choices=['unmeth', 'meth', 'frac_unmeth'], help='signal type')
    p.add_argument('-s', '--index', default=None, help='index file for context filtering')
    p.add_argument('--min-cov', type=int, default=1, help='minimum coverage to include a site')
    p.add_argument('--mc-col', default=None, help='mc column name or 0-based index (default: first column)')
    p.add_argument('--cov-col', default=None, help='cov column name or 0-based index (default: last column)')

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
                   columns=args.columns, dimensions=args.dimensions,
                   message=args.message, level=args.level)
        w.tocz(input=args.input, usecols=args.usecols,
               dim_cols=args.dim_cols, sep=args.sep,
               chunksize=args.chunksize, header=args.header,
               skiprows=args.skiprows)

    elif cmd == 'catcz':
        from .cz import Writer
        w = Writer(output=args.output, formats=args.formats,
                   columns=args.columns, dimensions=args.dimensions,
                   message=args.message)
        w.catcz(input=args.input, dim_order=args.dim_order,
                add_dim=args.add_dim, title=args.title)

    elif cmd == 'view':
        from .cz import Reader
        r = Reader(args.input)
        r.view(show_dim=args.show_dim, header=not args.no_header,
               dimension=args.dimension, reference=args.reference)

    elif cmd == 'header':
        from .cz import Reader
        r = Reader(args.input)
        r.print_header()
        r.close()

    elif cmd == 'query':
        from .cz import Reader
        r = Reader(args.input)
        r.query(dimension=args.dimension, start=args.start,
                end=args.end, regions=args.regions,
                query_col=args.query_col, reference=args.reference,
                printout=True)

    elif cmd == 'to_allc':
        from .cz import Reader
        r = Reader(args.input)
        r.to_allc(output=args.output, reference=args.reference,
                  dimension=args.dimension, tabix=not args.no_tabix,
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
                index=args.index, chunksize=args.chunksize)

    # ---- allc.py commands --------------------------------------------------
    elif cmd == 'allc2cz':
        from .allc import allc2cz
        allc2cz(input=args.input, output=args.output,
               reference=args.reference, missing_value=args.missing_value,
               formats=args.formats, columns=args.columns,
               dimensions=args.dimensions, usecols=args.usecols,
               ref_pos_col=args.ref_pos_col, allc_pos_col=args.allc_pos_col, sep=args.sep,
               chrom_order=args.chrom_order, chunksize=args.chunksize,
               sort_col=args.sort_col)

    elif cmd == 'build_ref':
        from .allc import AllC
        a = AllC(genome=args.genome, output=args.output,
                 pattern=args.pattern, threads=args.threads,
                 keep_temp=args.keep_temp)
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
            from .allc import index_regions
            index_regions(input=args.input, output=args.output,
                          bed=args.bed, threads=args.threads)
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
        from .allc import merge_cz
        merge_cz(indir=args.indir, cz_paths=args.cz_paths,
                 class_table=args.class_table, output=args.output,
                 prefix=args.prefix, threads=args.threads,
                 formats=args.formats, chrom_order=args.chrom_order,
                 reference=args.reference, keep_cat=args.keep_cat,
                 batchsize=args.batchsize, temp=args.temp,
                 bgzip=not args.no_bgzip, chunksize=args.chunksize,
                 ext=args.ext)

    elif cmd == 'merge_cell_type':
        from .allc import merge_cell_type
        merge_cell_type(indir=args.indir, cell_table=args.cell_table,
                        outdir=args.outdir, threads=args.threads,
                        chrom_order=args.chrom_order, ext=args.ext)

    elif cmd == 'extractCG':
        from .allc import extractCG
        extractCG(input=args.input, output=args.output,
                  index=args.index, chunksize=args.chunksize,
                  merge_cg=args.merge_cg)

    elif cmd == 'aggregate':
        from .allc import aggregate
        aggregate(input=args.input, output=args.output,
                  index=args.index, intersect=args.intersect,
                  exclude=args.exclude, chunksize=args.chunksize,
                  formats=args.formats)

    elif cmd == 'combp':
        from .allc import combp
        combp(input=args.input, outdir=args.outdir,
              threads=args.threads, dist=args.dist,
              temp=args.temp, bed=args.bed)

    elif cmd == 'annot_dmr':
        from .allc import annot_dmr
        annot_dmr(input=args.input, matrix=args.matrix,
                  output=args.output, delta_cutoff=args.delta_cutoff)

    elif cmd == 'call_peaks':
        from .allc import call_peaks
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
        from .allc import to_bedgraph
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


if __name__ == "__main__":
    main()
