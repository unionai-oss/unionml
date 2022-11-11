# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import logging as pylogging
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import sys

from sphinx.util import logging

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "unionml"
copyright = "2022, unionai-oss"
author = "unionai-oss"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx-prompt",
    "sphinx_tabs.tabs",
    "sphinxcontrib.youtube",
    "sphinxcontrib.mermaid",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "UnionML"
html_logo = "_static/images/union-logo.svg"
html_favicon = "_static/images/union-logo.svg"

announcement = """
UnionML is in Beta 🏗 &nbsp;! Join us on
<a href='https://flyte-org.slack.com/archives/C03JL38L65V' target='_blank'>Slack</a>
and <a href='https://github.com/unionai-oss/unionml/discussions' target='_blank'>Github Discussions</a>,
and if you like this project, <a href='https://github.com/unionai-oss/unionml' target='_blank'>give us a star ⭐️! </a>
"""

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#e59f12",
        "color-brand-content": "#e59f12",
        "color-announcement-background": "#FEE7B8",
        "color-announcement-text": "#535353",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FDB51D",
        "color-brand-content": "#FDB51D",
        "color-announcement-background": "#493100",
    },
    "announcement": announcement,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_js_files = [
    "js/custom.js",
]

html_css_files = [
    "css/custom.css",
]

nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "cache")
nb_execution_timeout = 600
nb_execution_raise_on_error = True

if "READTHEDOCS" in os.environ:
    # don't run the docs
    nb_execution_excludepatterns = ["tutorials/*.md"]

autoclass_content = "both"
autodoc_default_flags = ["members", "undoc-members"]
autosummary_generate = True

# strip prompts in example code snippets
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# mermaid
mermaid_params = [
    "--height",
    "350",
    "--backgroundColor",
    "transparent",
]

# intersphinx configuration
intersphinx_mapping = {
    "flytekit": ("https://flyte.readthedocs.io/projects/flytekit/en/latest/", None),
    "flytekitplugins": ("https://docs.flyte.org/projects/flytekit/en/latest/", None),
    "flyte": ("https://flyte.readthedocs.io/en/latest/", None),
    "flyteidl": ("https://docs.flyte.org/projects/flyteidl/en/latest", None),
    "flytectl": ("https://docs.flyte.org/projects/flytectl/en/latest/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "torch": ("https://pytorch.org/docs/master/", None),
}


# this is a workaround to filter out forward reference issue in sphinx_autodoc_typehints
class FilterFlytekitPluginsTypeAnnotationWarning(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            "Cannot resolve forward reference in type annotations of " '"unionml.dataset.Dataset.from_sqlalchemy_task"'
        )


IGNORE_APPLICATION_WARNINGS = (
    "the sphinxcontrib.youtube extension does not declare",
    "doing serial read",
    "Inline emphasis start-string without end-string.",
)


class SphinxApplicationWarningsFilter(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        return not record.getMessage().startswith(IGNORE_APPLICATION_WARNINGS)


class SphinxDocUtilsWarningsFilter(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        return not (
            # ignore specific error that occurs due to the `serve_command` function, which uses
            # the uvicorn serve command docstring.
            record.getMessage().startswith("Inline emphasis start-string without end-string")
            and re.match(r".+cli_reference.rst:\d+", record.location)
        )


logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(FilterFlytekitPluginsTypeAnnotationWarning())
logging.getLogger("sphinx.application").logger.addFilter(SphinxApplicationWarningsFilter())
logging.getLogger("sphinx.util.docutils").logger.addFilter(SphinxDocUtilsWarningsFilter())
