# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import alabaster
import os
import sys
from pathlib import Path
file_path = os.path.dirname( __file__ )
lib_path = Path(file_path).parents[1]

sys.path.insert(0, os.path.abspath(lib_path))

import predictit
version = predictit.__version__
# -- Project information -----------------------------------------------------

project = 'predictit'
copyright = '2019, Daniel Malachov'
author = 'Daniel Malachov'

# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------
html_theme_path = [alabaster.get_path()]
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  'alabaster',
                'sphinx.ext.autodoc',
                'sphinx.ext.intersphinx',
                'sphinx.ext.ifconfig',
                'sphinx.ext.viewcode',
                'sphinx.ext.githubpages',
                'sphinx.ext.imgmath',
                'm2r'

]

html_theme = 'alabaster'

source_suffix = ['.rst', '.md']

master_doc = 'index'

autosectionlabel_prefix_document = True

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}
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
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']