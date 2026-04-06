"""cytozip (ChunkZIP) — Chunk-based columnar binary format for fast random access.

Public API:
  - Reader:  Read and query .cz files (local or remote via HTTP).
  - Writer:  Create .cz files from tabular data.
  - AllC:    Extract cytosine positions from a reference genome.
  - bed2cz:  Convert tabix-indexed allc.tsv.gz to .cz format.
  - merge_cz: Merge multiple per-cell .cz files.
  - extract / extractCG / aggregate:  Subset and aggregate .cz data.
  - combp / annot_dmr:  Differential methylation analysis.

CLI entry point: ``cytozip <command> [options]``
"""
from .cz import (
    Reader,
    Writer,
    extract
)
from .allc import (AllC, generate_ssi1, bed2cz, generate_ssi2,
                   merge_cz, extractCG, aggregate,
                   merge_cell_type,
                   combp, annot_dmr
                   )

from ._version import version as __version__


def main():
    """CLI entry point powered by Python Fire."""
    import fire
    # Suppress Fire's interactive pager for cleaner piped output.
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(
        {
            "Writer": Writer,
            'Reader': Reader,
            'AllC': AllC,
            'bed2cz': bed2cz,
            'generate_ssi1': generate_ssi1,
            'generate_ssi2': generate_ssi2,
            'merge_cz': merge_cz,
            'merge_cell_type': merge_cell_type,
            'extract': extract,
            'extractCG': extractCG,
            'aggregate': aggregate,
            "combp": combp,
            'annot_dmr': annot_dmr,
        },serialize=lambda x:_safe_print(x)
	)

def _safe_print(x):
    try:
        if x is not None:
            print(x)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
