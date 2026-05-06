# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GLM-HMM tutorials'
copyright = '2026, Camila Maura'
author = 'Camila Maura'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# The Root document
root_doc = "index"


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',  # Links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
    'myst_nb',
    'sphinx_contributors',
    'sphinx_code_tabs',
    'sphinx.ext.mathjax',
    'sphinx_math_dollar', # necessary to use $$ in math from https://www.sympy.org/sphinx-math-dollar/
    'sphinx_autodoc_typehints',
    'sphinx_togglebutton',
    'matplotlib.sphinxext.plot_directive',
    "matplotlib.sphinxext.mathmpl",
    "sphinx.ext.intersphinx",
]


myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]

# configs for mathjax
mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
