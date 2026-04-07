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

# Prefer plain gcc/g++ for building the extension to avoid linking against
# MPI compilers (mpicc) that may be present in some environments. This
# forces a more portable .so that does not depend on libmpi.
os.environ['CC'] = os.environ.get('CC', 'gcc')
os.environ['CXX'] = os.environ.get('CXX', 'g++')
if 'mpicc' in os.path.basename(os.environ.get('CC', '')):
    print('WARNING: CC appears to be mpicc; consider building with CC=gcc CXX=g++ to avoid MPI linkage')
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
    install_requires=['pandas', 'numpy', 'cython', 'pysam','fast-fisher'],
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
                ],
                libraries=["z"],
            )
        ],
        language_level="3",
    ),
)


# python setup.py build_ext --inplace