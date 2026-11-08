"""Shared helpers for htslib-backed dependencies (pysam, ``_bam_pileup``).

htslib prints warnings such as::

    [W::hts_idx_load3] The index file is older than the data file: x.allc.tsv.gz.tbi

whenever a ``.tbi`` / ``.bai`` has an older mtime than its data file (typical
after ``cp`` / ``rsync`` of a dataset). The index is almost always still valid,
so cytozip lowers the htslib log level to *errors only* before touching such
files. Override with the ``CYTOZIP_HTS_LOG_LEVEL`` environment variable
(htslib scale: 0 = off, 1 = error [default], 3 = warning, 4 = info, 5 = debug).
"""
import os

HTS_LOG_LEVEL_ENV = 'CYTOZIP_HTS_LOG_LEVEL'
_DEFAULT_LEVEL = 1  # HTS_LOG_ERROR


def hts_log_level():
    """Return the htslib log level requested for cytozip (int)."""
    try:
        return int(os.environ.get(HTS_LOG_LEVEL_ENV, _DEFAULT_LEVEL))
    except ValueError:
        return _DEFAULT_LEVEL


def import_pysam():
    """``import pysam`` with htslib logging set to :func:`hts_log_level`.

    Call this instead of ``import pysam`` everywhere in cytozip (including
    multiprocessing workers, which re-import in a fresh interpreter under
    the ``spawn`` start method).
    """
    import pysam
    try:
        pysam.set_verbosity(hts_log_level())
    except Exception:  # very old pysam without set_verbosity
        pass
    return pysam
