import os
import sys

import configs
from pathlib import Path

sys.path.insert(0, os.path.abspath(Path(configs.__file__).parent.parent))
current_directory = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

project = " LIS² (Large Image Split Segmentation)"
copyright = "2026, Julien Rabault et al."
author = "Julien Rabault et all."
release = "v1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
]
html_logo = '_static/lis2-normal.png'
html_favicon = '_static/lis2-small.png'

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]
