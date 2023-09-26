# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path('../../').absolute()))

from firepy import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'boblica'
copyright = '2023, Benedek Kiss'
author = 'Benedek Kiss'

# The full version, including alpha/beta/rc tags
version = __version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/boblica-logo.png'
html_favicon = '_static/boblica-icon.png'

# -- Read the docs theme options -------------------------------------------------
html_theme_options = {
    'logo_only': False,
    'style_nav_header_background': '#333131',
}

# -- Autodoc options ----------------------------------------------------------
autodoc_typehints = 'description'
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'
