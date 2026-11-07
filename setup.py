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
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import numpy as np

_IS_DARWIN = sys.platform == 'darwin'
_IS_WINDOWS = sys.platform.startswith('win')

# On Linux prefer plain gcc/g++ for building the extension to avoid linking
# against MPI compilers (mpicc) that may be present in some environments
# (forces a more portable .so that does not depend on libmpi). On macOS the
# default is clang (Apple or conda-forge); leave it alone.
if not _IS_DARWIN and not _IS_WINDOWS:
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


def _brew_prefix(formula=None):
    """``brew --prefix [formula]`` or None (macOS Homebrew)."""
    brew = shutil.which('brew')
    if not brew:
        return None
    try:
        out = subprocess.run([brew, '--prefix'] + ([formula] if formula else []),
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    p = out.stdout.strip()
    return p if out.returncode == 0 and p and os.path.isdir(p) else None


def _search_prefixes():
    """Prefixes whose ``include/`` / ``lib/`` are searched for libdeflate,
    htslib and OpenMP: conda env(s) first, then (macOS) Homebrew — including
    keg-only formulas such as ``libomp`` — and MacPorts, then /usr/local."""
    prefixes = list(_CONDA_ENV_PREFIXES)
    if _IS_DARWIN:
        for f in (None, 'libdeflate', 'htslib', 'libomp'):
            p = _brew_prefix(f)
            if p and p not in prefixes:
                prefixes.append(p)
        for p in ('/opt/homebrew', '/opt/local', '/usr/local'):
            if os.path.isdir(p) and p not in prefixes:
                prefixes.append(p)
    elif not _IS_WINDOWS:
        if os.path.isdir('/usr/local') and '/usr/local' not in prefixes:
            prefixes.append('/usr/local')
    return prefixes


_PREFIXES = _search_prefixes()
_INCLUDE_DIRS = [os.path.join(p, 'include') for p in _PREFIXES]
_LIBRARY_DIRS = [os.path.join(p, 'lib') for p in _PREFIXES]
# rpath so the extension finds libdeflate / libhts / libomp at import time
# without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH (conda and Homebrew both put
# their dylibs outside the system search path).
_RPATH_ARGS = ([] if _IS_WINDOWS
               else [f'-Wl,-rpath,{d}' for d in _LIBRARY_DIRS if os.path.isdir(d)])


def _have_htslib():
    """Return True iff <htslib/sam.h> is reachable on any known search path."""
    candidates = [os.path.join(d, 'htslib', 'sam.h') for d in _INCLUDE_DIRS]
    for inc in ('/usr/include', '/usr/local/include',
                '/opt/homebrew/include', '/opt/local/include'):
        candidates.append(os.path.join(inc, 'htslib', 'sam.h'))
    return any(os.path.isfile(p) for p in candidates)


# ---- OpenMP detection --------------------------------------------------------
# gcc and conda-forge clang accept ``-fopenmp`` directly. Apple's clang needs
# ``-Xpreprocessor -fopenmp`` plus Homebrew's ``libomp`` (keg-only: its
# include/lib dirs must be passed explicitly). Stock Apple clang without
# libomp has no OpenMP at all: then the extensions are built single-threaded
# through ``cytozip/_omp_shim.h`` (``prange`` becomes a serial loop) instead
# of failing the build. Override with ``CYTOZIP_OPENMP=0`` (force off) or
# ``CYTOZIP_OPENMP=1`` (force the first flag set without probing).
_OMP_TEST_SRC = """
#include <omp.h>
#include <stdio.h>
int main(void) {
    int n = 0;
#pragma omp parallel
    { n = omp_get_num_threads(); }
    printf("%d", n);
    return 0;
}
"""


def _openmp_flag_candidates():
    if _IS_WINDOWS:
        return [(['/openmp'], [])]
    cands = [(['-fopenmp'], ['-fopenmp'])]
    if _IS_DARWIN:
        omp_inc = [d for d in _INCLUDE_DIRS if os.path.isfile(os.path.join(d, 'omp.h'))]
        omp_lib = [d for d in _LIBRARY_DIRS if any(
            os.path.isfile(os.path.join(d, f)) for f in ('libomp.dylib', 'libomp.a', 'libgomp.dylib'))]
        cflags = ['-Xpreprocessor', '-fopenmp'] + [f'-I{d}' for d in omp_inc]
        lflags = ['-lomp'] + [f'-L{d}' for d in omp_lib]
        cands.append((cflags, lflags))
    return cands


def _probe_openmp():
    """Return ``(compile_args, link_args)`` that build a tiny OpenMP program,
    or ``([], [])`` when none do (-> serial build via the shim)."""
    setting = os.environ.get('CYTOZIP_OPENMP')
    cands = _openmp_flag_candidates()
    if setting == '0':
        return [], []
    if setting == '1':
        return cands[0]
    cc = os.environ.get('CC') or sysconfig.get_config_var('CC') or ('cl' if _IS_WINDOWS else 'cc')
    cc = cc.split()[0]
    if not shutil.which(cc):
        return cands[0]  # cannot probe; assume the conventional flag
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'omp.c')
        exe = os.path.join(td, 'omp.exe' if _IS_WINDOWS else 'omp')
        with open(src, 'w') as fh:
            fh.write(_OMP_TEST_SRC)
        for cflags, lflags in cands:
            cmd = ([cc, src] + cflags + lflags + ['-o', exe]) if not _IS_WINDOWS \
                else [cc, src] + cflags + ['/Fe:' + exe]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                return cflags, lflags
    return [], []


_OMP_CFLAGS, _OMP_LFLAGS = _probe_openmp()
if _OMP_CFLAGS:
    print(f'OpenMP: enabled ({" ".join(_OMP_CFLAGS)})')
else:
    print('NOTE: no working OpenMP toolchain found; building cytozip '
          'extensions single-threaded (install `libomp` via Homebrew or use '
          'conda-forge compilers + llvm-openmp for multi-threaded inflate/DMR).')

if _IS_WINDOWS:
    _WARN_FLAGS = []
    _OPT_FLAGS = ['/O2']
else:
    _WARN_FLAGS = ['-Wno-unreachable-code-fallthrough', '-Wno-unused-result',
                   '-Wno-sign-compare']
    _OPT_FLAGS = ['-O3']
_COMMON_CFLAGS = _WARN_FLAGS + _OPT_FLAGS


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
                      'requests', 'anndata', 'scipy', 'biopython'],
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
                extra_compile_args=_COMMON_CFLAGS + _OMP_CFLAGS,
                extra_link_args=_OMP_LFLAGS + _RPATH_ARGS,
                libraries=["deflate"],
                # Pick up libdeflate from any conda prefix (``$PREFIX`` in
                # conda-build, ``$CONDA_PREFIX`` in an activated env) or, on
                # macOS, Homebrew / MacPorts. Build-time: compile error if
                # libdeflate-dev is not installed anywhere.
                include_dirs=['cytozip'] + _INCLUDE_DIRS,
                library_dirs=_LIBRARY_DIRS,
            ),
            Extension(
                "cytozip.dmr_accel",
                ["cytozip/dmr_accel.pyx"],
                extra_compile_args=_COMMON_CFLAGS + _OMP_CFLAGS,
                extra_link_args=_OMP_LFLAGS + _RPATH_ARGS,
                include_dirs=['cytozip', np.get_include()] + _INCLUDE_DIRS,
                library_dirs=_LIBRARY_DIRS,
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
                extra_compile_args=_COMMON_CFLAGS,
                extra_link_args=_RPATH_ARGS,
                libraries=["hts"],
                include_dirs=[np.get_include()] + _INCLUDE_DIRS,
                library_dirs=_LIBRARY_DIRS,
            ),
        ] if _HAVE_HTSLIB else []),
        language_level="3",
    ),
)


# python setup.py build_ext --inplace