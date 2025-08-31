from __future__ import annotations

import os
import sys
from datetime import datetime

project = "scorecardpl"
author = "scorecardpl contributors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Mock heavy external deps to speed up RTD builds
autodoc_mock_imports = [
    "polars",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = os.environ.get("SPHINX_HTML_THEME", "sphinx_rtd_theme")
html_static_path = ["_static"]

source_suffix = {
    ".md": "markdown",
}

# MyST config
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "substitution",
]

# Ensure local package import for autodoc
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
