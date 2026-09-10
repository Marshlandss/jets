"""
Read machine-specific directories from environment variables.
"""
import os
from pathlib import Path

try:
    DIR_LOAD = Path(os.environ["JETS_DIR_LOAD"]).expanduser()
    DIR_SAVE = Path(os.environ["JETS_DIR_SAVE"]).expanduser()
except KeyError as error:
    raise RuntimeError(f"Environment variable {error} is not set; see README.md.") from None