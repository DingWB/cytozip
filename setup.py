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
#
# Env-var convention:
#   * ``$CONDA_PREFIX`` — set when a user runs ``pip install .`` inside an
#     activated conda env (htslib lives under ``$CONDA_PREFIX/{include,lib}``).
#   * ``$PREFIX``       — set by ``conda-build``; points to the *host* env
#     (which is where htslib headers/libs are actually installed during a
#     recipe build). ``$CONDA_PREFIX`` in that context points to the
#     *build* env and does NOT contain htslib.
# We inspect BOTH so the same setup.py works in ``pip install`` and
# ``conda build`` without patching.
_CONDA_ENV_PREFIXES = [
    p for p in (os.environ.get('PREFIX'), os.environ.get('CONDA_PREFIX'))
    if p
]


def _have_htslib():
    """Return True iff <htslib/sam.h> is reachable on any known search path."""
    candidates = []
    for p in _CONDA_ENV_PREFIXES:
        candidates.append(os.path.join(p, 'include', 'htslib', 'sam.h'))
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
    packages=find_packages(exclude=(
        # Never install these as Python packages, even if a stray
        # ``__init__.py`` sneaks in. Also mirrored in MANIFEST.in so
        # they never end up in the sdist to begin with.
        'docs', 'docs.*',
        'notebooks', 'notebooks.*',
        'tests', 'tests.*',
        'data', 'data.*',
        'cytozip_example_data', 'cytozip_example_data.*',
        'conda-recipe', 'conda-recipe.*',
        'build', 'build.*',
    )),
    install_requires=['pandas', 'numpy', 'loguru', 'pysam',
                      'fsspec',
                      'requests', 'anndata', 'scipy'],
    # fast-fisher is optional (PyPI-only; used only by pivot_fisher).
    # Install with: pip install cytozip[pivot]
    extras_require={
        'pivot': ['fast-fisher'],
    },
    include_package_data=True,
    # Only ship data files that live INSIDE the cytozip/ package. The
    # empty-key ``''`` used to pull ``*.ipynb`` etc. from every package,
    # which was error-prone. Currently ``cytozip/`` needs the browser
    # reader ``cz_reader.mjs`` at runtime.
    package_data={
        'cytozip': ['*.mjs'],
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
                # Pick up libdeflate from any conda prefix on the search
                # path (``$PREFIX`` in conda-build, ``$CONDA_PREFIX`` in
                # a plain activated env). Build-time: compile error if
                # libdeflate-dev is not installed anywhere.
                include_dirs=[os.path.join(p, 'include')
                              for p in _CONDA_ENV_PREFIXES],
                library_dirs=[os.path.join(p, 'lib')
                              for p in _CONDA_ENV_PREFIXES],
                runtime_library_dirs=[os.path.join(p, 'lib')
                                      for p in _CONDA_ENV_PREFIXES],
            ),
            Extension(
                "cytozip.dmr_accel",
                ["cytozip/dmr_accel.pyx"],
                extra_compile_args=[
                    "-Wno-unreachable-code-fallthrough",
                    "-Wno-unused-result",
                    "-Wno-sign-compare",
                    "-O3",
                    "-fopenmp",
                ],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include()],
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
                include_dirs=([np.get_include()]
                              + [os.path.join(p, 'include')
                                 for p in _CONDA_ENV_PREFIXES]),
                library_dirs=[os.path.join(p, 'lib')
                              for p in _CONDA_ENV_PREFIXES],
                runtime_library_dirs=[os.path.join(p, 'lib')
                                      for p in _CONDA_ENV_PREFIXES],
            ),
        ] if _HAVE_HTSLIB else []),
        language_level="3",
    ),
)


# python setup.py build_ext --inplace