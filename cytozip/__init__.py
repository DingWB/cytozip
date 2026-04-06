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
    'Reader': 'cz', 'Writer': 'cz', 'extract': 'cz',
    # allc.py
    'AllC': 'allc', 'allc2cz': 'allc', 'generate_ssi1': 'allc',
    'generate_ssi2': 'allc', 'merge_cz': 'allc', 'extractCG': 'allc',
    'aggregate': 'allc', 'merge_cell_type': 'allc', 'combp': 'allc',
    'annot_dmr': 'allc',
}


def __getattr__(name):
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(f'.{mod_name}', __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'cytozip' has no attribute {name!r}")


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

    # ---- summary_chunks / summary_blocks ------------------------------------
    p = sub.add_parser('summary', help='Print chunk summary of a .cz file', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('--blocks', action='store_true', help='show block-level detail instead of chunk-level')

    # ---- extract -------------------------------------------------------------
    p = sub.add_parser('extract', help='Extract subset of .cz using SSI', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--outfile', required=True, help='output .cz file')
    p.add_argument('-s', '--ssi', required=True, help='subset index file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')

    # ---- allc2cz --------------------------------------------------------------
    p = sub.add_parser('allc2cz', help='Convert tabix-indexed allc.tsv.gz to .cz', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input allc.tsv.gz')
    p.add_argument('-O', '--outfile', required=True, help='output .cz file')
    p.add_argument('-r', '--reference', default=None, help='reference .cz file')
    p.add_argument('--missing-value', type=_csv_int, default=[0, 0], help='missing value fill')
    p.add_argument('-F', '--formats', type=_csv_str, default=['B', 'B'], help='column formats')
    p.add_argument('-C', '--columns', type=_csv_str, default=['mc', 'cov'], help='column names')
    p.add_argument('-D', '--dimensions', type=_csv_str, default=['chrom'], help='dimension names')
    p.add_argument('-u', '--usecols', type=_csv_int, default=[4, 5], help='column indices to pack')
    p.add_argument('--pr', type=int, default=0, help='position column index in reference')
    p.add_argument('--pa', type=int, default=1, help='position column index in input')
    p.add_argument('-s', '--sep', default='\t', help='separator')
    p.add_argument('--path-to-chrom', default=None, help='chrom order file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')

    # ---- build_ref (AllC) ----------------------------------------------------
    p = sub.add_parser('build_ref', help='Extract C positions from reference genome', formatter_class=_fmt)
    p.add_argument('-g', '--genome', required=True, help='reference genome FASTA')
    p.add_argument('-O', '--output', default='hg38_allc.cz', help='output .cz file')
    p.add_argument('-p', '--pattern', default='C', help='nucleotide pattern')
    p.add_argument('-n', '--n-jobs', type=int, default=12, help='parallel jobs')
    p.add_argument('--keep-temp', action='store_true', help='keep temp directory')

    # ---- generate_ssi1 -------------------------------------------------------
    p = sub.add_parser('generate_ssi1', help='Generate pattern-based SSI (CGN/CHN)', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', default=None, help='output .ssi file')
    p.add_argument('-p', '--pattern', default='CGN', help='context pattern (CGN/CHN/+CGN)')

    # ---- generate_ssi2 -------------------------------------------------------
    p = sub.add_parser('generate_ssi2', help='Generate region-based SSI from BED', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--output', default=None, help='output .ssi file')
    p.add_argument('-b', '--bed', required=True, help='BED file with regions')
    p.add_argument('-n', '--n-jobs', type=int, default=4, help='parallel jobs')

    # ---- merge_cz ------------------------------------------------------------
    p = sub.add_parser('merge_cz', help='Merge per-cell .cz files', formatter_class=_fmt)
    p.add_argument('-i', '--indir', default=None, help='input directory')
    p.add_argument('--cz-paths', default=None, help='file listing .cz paths')
    p.add_argument('--class-table', default=None, help='cell class table')
    p.add_argument('-O', '--outfile', default=None, help='output file')
    p.add_argument('--prefix', default=None, help='output prefix')
    p.add_argument('-n', '--n-jobs', type=int, default=12, help='parallel jobs')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats')
    p.add_argument('--path-to-chrom', default=None, help='chrom order file')
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
    p.add_argument('-n', '--n-jobs', type=int, default=64, help='parallel jobs')
    p.add_argument('--path-to-chrom', default=None, help='chrom order file')
    p.add_argument('--ext', default='.CGN.merged.cz', help='input file extension')

    # ---- extractCG -----------------------------------------------------------
    p = sub.add_parser('extractCG', help='Extract CG-context records', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--outfile', required=True, help='output .cz file')
    p.add_argument('-s', '--ssi', required=True, help='CGN subset index file')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('--merge-cg', action='store_true', help='merge forward/reverse CG')

    # ---- aggregate -----------------------------------------------------------
    p = sub.add_parser('aggregate', help='Aggregate records within regions', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input .cz file')
    p.add_argument('-O', '--outfile', required=True, help='output .cz file')
    p.add_argument('-s', '--ssi', required=True, help='region subset index file')
    p.add_argument('--intersect', default=None, help='intersect filter')
    p.add_argument('--exclude', default=None, help='exclude filter')
    p.add_argument('-c', '--chunksize', type=int, default=5000, help='rows per chunk')
    p.add_argument('-F', '--formats', type=_csv_str, default=['H', 'H'], help='output formats')

    # ---- combp ---------------------------------------------------------------
    p = sub.add_parser('combp', help='Run comb-p on Fisher results', formatter_class=_fmt)
    p.add_argument('-I', '--input', required=True, help='input Fisher result')
    p.add_argument('-O', '--outdir', default='cpv', help='output directory')
    p.add_argument('-n', '--n-jobs', type=int, default=24, help='parallel jobs')
    p.add_argument('--dist', type=int, default=300, help='max distance between sites')
    p.add_argument('--temp', action='store_true', help='keep temp directory')
    p.add_argument('--bed', action='store_true', help='keep bed directory')

    # ---- annot_dmr -----------------------------------------------------------
    p = sub.add_parser('annot_dmr', help='Annotate DMRs', formatter_class=_fmt)
    p.add_argument('-I', '--input', default='merged_dmr.txt', help='merged DMR file')
    p.add_argument('--matrix', default='merged_dmr.cell_class.beta.txt', help='beta matrix file')
    p.add_argument('-O', '--outfile', default='dmr.annotated.txt', help='output file')
    p.add_argument('--delta-cutoff', type=float, default=None, help='min delta-beta cutoff')

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

    elif cmd == 'summary':
        from .cz import Reader
        r = Reader(args.input)
        if args.blocks:
            r.summary_blocks(printout=True)
        else:
            r.summary_chunks(printout=True)

    elif cmd == 'extract':
        from .cz import extract
        extract(input=args.input, outfile=args.outfile,
                ssi=args.ssi, chunksize=args.chunksize)

    # ---- allc.py commands --------------------------------------------------
    elif cmd == 'allc2cz':
        from .allc import allc2cz
        allc2cz(input=args.input, outfile=args.outfile,
               reference=args.reference, missing_value=args.missing_value,
               formats=args.formats, columns=args.columns,
               dimensions=args.dimensions, usecols=args.usecols,
               pr=args.pr, pa=args.pa, sep=args.sep,
               path_to_chrom=args.path_to_chrom, chunksize=args.chunksize)

    elif cmd == 'build_ref':
        from .allc import AllC
        a = AllC(genome=args.genome, output=args.output,
                 pattern=args.pattern, n_jobs=args.n_jobs,
                 keep_temp=args.keep_temp)
        a.run()

    elif cmd == 'generate_ssi1':
        from .allc import generate_ssi1
        generate_ssi1(input=args.input, output=args.output,
                      pattern=args.pattern)

    elif cmd == 'generate_ssi2':
        from .allc import generate_ssi2
        generate_ssi2(input=args.input, output=args.output,
                      bed=args.bed, n_jobs=args.n_jobs)

    elif cmd == 'merge_cz':
        from .allc import merge_cz
        merge_cz(indir=args.indir, cz_paths=args.cz_paths,
                 class_table=args.class_table, outfile=args.outfile,
                 prefix=args.prefix, n_jobs=args.n_jobs,
                 formats=args.formats, path_to_chrom=args.path_to_chrom,
                 reference=args.reference, keep_cat=args.keep_cat,
                 batchsize=args.batchsize, temp=args.temp,
                 bgzip=not args.no_bgzip, chunksize=args.chunksize,
                 ext=args.ext)

    elif cmd == 'merge_cell_type':
        from .allc import merge_cell_type
        merge_cell_type(indir=args.indir, cell_table=args.cell_table,
                        outdir=args.outdir, n_jobs=args.n_jobs,
                        path_to_chrom=args.path_to_chrom, ext=args.ext)

    elif cmd == 'extractCG':
        from .allc import extractCG
        extractCG(input=args.input, outfile=args.outfile,
                  ssi=args.ssi, chunksize=args.chunksize,
                  merge_cg=args.merge_cg)

    elif cmd == 'aggregate':
        from .allc import aggregate
        aggregate(input=args.input, outfile=args.outfile,
                  ssi=args.ssi, intersect=args.intersect,
                  exclude=args.exclude, chunksize=args.chunksize,
                  formats=args.formats)

    elif cmd == 'combp':
        from .allc import combp
        combp(input=args.input, outdir=args.outdir,
              n_jobs=args.n_jobs, dist=args.dist,
              temp=args.temp, bed=args.bed)

    elif cmd == 'annot_dmr':
        from .allc import annot_dmr
        annot_dmr(input=args.input, matrix=args.matrix,
                  outfile=args.outfile, delta_cutoff=args.delta_cutoff)


if __name__ == "__main__":
    main()
