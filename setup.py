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
from setuptools.command.build_ext import build_ext as _build_ext
import os

class build_ext(_build_ext):
    """Override build_ext to force plain gcc, ignoring CC=mpicc or similar
    wrappers set by HPC/conda environments.  cz_accel has no MPI dependency
    and linking against libmpi makes the .so unportable."""
    def build_extensions(self):
        cc = self.compiler.compiler[0] if self.compiler.compiler else 'gcc'
        if 'mpi' in os.path.basename(cc).lower():
            self.compiler.set_executable('compiler', ['gcc'])
            self.compiler.set_executable('compiler_so', ['gcc', '-fPIC'])
            self.compiler.set_executable('compiler_cxx', ['g++'])
            self.compiler.set_executable('linker_so', ['gcc', '-shared'])
        super().build_extensions()
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

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
    install_requires=['pandas', 'numpy', 'cython', 'fast-fisher'],
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
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(
        [
            Extension(
                "cytozip.cz_accel",
                ["cytozip/cz_accel.pyx"],
                extra_compile_args=[
                    "-Wno-unreachable-code-fallthrough",
                    "-Wno-unused-result",
                    "-Wno-sign-compare",
                ],
            )
        ],
        language_level="3",
    ),
)


# python setup.py build_ext --inplace