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
    install_requires=['pandas', 'fire', 'numpy', 'cython', 'fast-fisher'],
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

    # Build the acceleration extension(s). If Cython is available this will
    # compile the .pyx module we added (`cytozip/cz_accel.pyx`). Silence a few
    # clang warnings that are benign on macOS by passing extra compile args.
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