# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from enum import Enum

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "e3j"
copyright = "2024, InstaDeep Ltd"
author = "Olivier Peltre"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# single backticks `code`
default_role = "code"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
]

autosummary_generate = True
html_favicon = "icon.ico"

# Render the `Values:` section of enum docstrings like an `Args:` list.
napoleon_custom_sections = [("Values", "params_style")]

# Merge the __init__ docstring (with its napoleon `Args:` section) into the
# class-level documentation, right under the class signature. This avoids the
# redundancy of listing constructor parameters both in the class signature and
# again in a separately-rendered __init__ method.
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "e3j"
html_static_path = ["_static"]

html_css_files = ["custom.css"]

# -- $ and $$ for mathjax with sphinx_math_dollar ------------------------------

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}


# -- Prevent sphinx from skipping __call__ (the __init__ docstring is merged
#    into the class body via autoclass_content = "both")
def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


# -- Strip the noisy Enum constructor signature from documented enum classes.
def strip_enum_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and isinstance(obj, type) and issubclass(obj, Enum):
        return ("", return_annotation)
    return None


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("autodoc-process-signature", strip_enum_signature)
