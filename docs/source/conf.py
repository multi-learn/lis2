import os
import sys

import configs
from pathlib import Path


sys.path.insert(0, os.path.abspath(Path(configs.__file__).parent.parent))
current_directory = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

project = "BigSF - ToolBox"
copyright = "2025, Julien et seulement Julien"
author = "Julien et seulement Julien"
release = "v0.1"

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
html_theme = "sphinx_rtd_theme"
