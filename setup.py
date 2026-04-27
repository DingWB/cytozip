# -*- coding: utf-8 -*-
"""
@author: DingWB
"""
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
from pathlib import Path
from Cython.Build import cythonize
from setuptools import Extension
import os
import numpy as np

# Prefer plain gcc/g++ for building the extension to avoid linking against
# MPI compilers (mpicc) that may be present in some environments. This
# forces a more portable .so that does not depend on libmpi.
os.environ['CC'] = os.environ.get('CC', 'gcc')
os.environ['CXX'] = os.environ.get('CXX', 'g++')
if 'mpicc' in os.path.basename(os.environ.get('CC', '')):
    print('WARNING: CC appears to be mpicc; consider building with CC=gcc CXX=g++ to avoid MPI linkage')
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# ---- Optional htslib detection ----------------------------------------------
# The ``cytozip._bam_pileup`` extension links against libhts. If the htslib
# headers/libs are not available (e.g. user installed via plain ``pip``
# without a prior ``conda install -c bioconda htslib``), we silently skip
# building it. ``cytozip.bam.bam_to_cz`` automatically falls back to the
# ``samtools mpileup`` subprocess backend at runtime in that case.
def _have_htslib():
    """Return True iff <htslib/sam.h> is reachable on the search path used
    by the active build environment (CONDA_PREFIX/include + CPPFLAGS-ish)."""
    candidates = []
    cp = os.environ.get('CONDA_PREFIX', '')
    if cp:
        candidates.append(os.path.join(cp, 'include', 'htslib', 'sam.h'))
    for inc in ('/usr/include', '/usr/local/include',
                '/opt/homebrew/include', '/opt/local/include'):
        candidates.append(os.path.join(inc, 'htslib', 'sam.h'))
    return any(os.path.isfile(p) for p in candidates)


_HAVE_HTSLIB = _have_htslib() or os.environ.get('CYTOZIP_FORCE_HTSLIB') == '1'
if not _HAVE_HTSLIB:
    print(
        'NOTE: htslib headers not found; skipping cytozip._bam_pileup '
        'extension. bam_to_cz() will fall back to the `samtools mpileup` '
        'subprocess backend. To enable the fast in-process pileup '
        'backend, install htslib (e.g. `conda install -c bioconda htslib`) '
        'and reinstall cytozip.'
    )

setup(
    name="cytozip",
	setup_requires=['setuptools_scm'],
	use_scm_version=True,  # version=__version__,
    description="cytozip: Chunk based ZIP for cytosine data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Wubin Ding",
    author_email="ding.wu.bin.gm@gmail.com",
    url="https://github.com/DingWB/cytozip",
    packages=find_packages(exclude=('docs',)),
    install_requires=['pandas', 'numpy', 'loguru', 'pysam',
                      'fast-fisher', 'fsspec', 's3fs', 'gcsfs',
                      'requests', 'anndata', 'scipy'],
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.tsv', '*.csv', '*.fa', '*Snakefile', '*ipynb']
    },
    entry_points={
        'console_scripts':
            [
                'czip=cytozip:main',
            ],
    },

    ext_modules=cythonize(
        [
            Extension(
                "cytozip.cz_accel",
                ["cytozip/cz_accel.pyx"],
                extra_compile_args=[
                    "-Wno-unreachable-code-fallthrough",
                    "-Wno-unused-result",
                    "-Wno-sign-compare",
                    "-O3",
                    "-fopenmp",
                ],
                extra_link_args=["-fopenmp"], # for multiple threads in cython.
                libraries=["deflate"],
                # Pick up libdeflate from the active conda/virtual env first,
                # then fall back to system paths. Build-time: compile error
                # if libdeflate-dev is not installed.
                include_dirs=[
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include')
                ] if os.environ.get('CONDA_PREFIX') else [],
                library_dirs=[
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')
                ] if os.environ.get('CONDA_PREFIX') else [],
                runtime_library_dirs=[
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')
                ] if os.environ.get('CONDA_PREFIX') else [],
            ),
        ] + ([
            # In-process htslib mpileup wrapper. Optional: only built if
            # htslib headers + libs are available (typically via conda's
            # ``htslib`` package). If they're missing, the extension is
            # silently skipped and bam_to_cz falls back to the
            # ``samtools mpileup`` subprocess backend.
            Extension(
                "cytozip._bam_pileup",
                ["cytozip/_bam_pileup.pyx"],
                extra_compile_args=[
                    "-Wno-unreachable-code-fallthrough",
                    "-Wno-unused-result",
                    "-Wno-sign-compare",
                    "-O3",
                ],
                libraries=["hts"],
                include_dirs=([
                    np.get_include(),
                ] + ([
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include')
                ] if os.environ.get('CONDA_PREFIX') else [])),
                library_dirs=[
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')
                ] if os.environ.get('CONDA_PREFIX') else [],
                runtime_library_dirs=[
                    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')
                ] if os.environ.get('CONDA_PREFIX') else [],
            ),
        ] if _HAVE_HTSLIB else []),
        language_level="3",
    ),
)


# python setup.py build_ext --inplace