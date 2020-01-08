
# Generate rst files with
# sphinx-apidoc -f -e -o source/ ../predictit
# Only other important file is index.rst

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

import sys
import pathlib
import datetime

script_dir = pathlib.Path(__file__).resolve()
lib_path = script_dir.parents[1].as_posix()
sys.path.insert(0, lib_path)


# -- Project information -----------------------------------------------------

project = 'predictit'
copyright = '2020, Daniel Malachov'
author = 'Daniel Malachov'

# The full version, including alpha/beta/rc tags
release = datetime.datetime.now().strftime('%d-%m-%Y')

source_suffix = ['.rst', '.md']

# -- General configuration ---------------------------------------------------
html_theme_options = {
    'github_user': 'Malachov',
    'github_repo': 'predictit',
    'github_banner': True
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx.ext.intersphinx',
                'sphinx.ext.viewcode',
                'sphinx.ext.githubpages',
                'sphinx.ext.imgmath',
                'm2r'
]


html_sidebars = { '**': ['about.html', 'navi.html', 'searchbox.html']}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
