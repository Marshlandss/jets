"""
Read machine-specific directories from environment variables.
"""
# Imports: standard library
import os
from pathlib import Path

DIR_REPO   = Path(__file__).resolve().parents[2]
DIR_INPUT  = Path(os.environ.get("JETS_DIR_INPUT",  DIR_REPO / "data" / "input")).expanduser()
DIR_OUTPUT = Path(os.environ.get("JETS_DIR_OUTPUT", DIR_REPO / "data" / "output")).expanduser()
