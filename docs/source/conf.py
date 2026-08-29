# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
print(sys.path)

from cytozip import __version__

# -- Project information -----------------------------------------------------

project = "cytozip"
copyright = "2026, Wubin Ding"
author = "Wubin Ding"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "recommonmark",
    "sphinx.ext.napoleon",
    "sphinxcontrib.jquery",
    "sphinx_search.extension",
    "sphinx_copybutton",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Parse .md files as Markdown (via recommonmark) and .rst as
# reStructuredText. Do NOT map .md to "restructuredtext": that forces the
# RST parser onto Markdown files, which produces spurious "Unexpected
# indentation" / "Inline literal start-string without end-string" errors on
# every fenced code block and `inline code` span.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = "sphinx"
todo_include_todos = False

autosectionlabel_prefix_document = True
# autosectionlabel would otherwise emit "duplicate label" warnings when two
# notebooks/docs share a heading text (e.g. "Advanced features"). Restrict
# it to top-level headings so cross-document collisions are rare.
autosectionlabel_maxdepth = 1
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"  # Read the Docs; pip install --upgrade sphinx-rtd-theme
# documentation: https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html

html_theme_options = {
    "analytics_id": "G-6CXLY8R88P",
    "collapse_navigation": False,
    "globaltoc_collapse": False,
    "globaltoc_maxdepth": 3,
    "sidebarwidth": 200,  # sidebarwidth
    "navigation_depth": 6,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

html_context = {
    "display_github": True,
    "github_user": "DingWB",
    "github_repo": "PyComplexHeatmap",
    "github_version": "main/docs/source/",
}

htmlhelp_basename = "PyComplexHeatmapDoc"

latex_documents = [
    (
        master_doc,
        "cytozip.tex",
        "cytozip Documentation",
        "Wubin Ding",
        "manual",
    ),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "cytozip", "CZIP Documentation", [author], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "cytozip",
        "CZIP Documentation",
        author,
        "cytozip",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Change the width of content, add the following to the css
html_css_files = [
    "css/custom.css",
]
# .wy-nav-content {
#     max-width: 75% !important;
# }
