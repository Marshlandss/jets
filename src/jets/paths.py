"""
Provide the directories that the scripts read from (both 'DIR_INPUT' and 'DIR_OUTPUT') and write to (only 'DIR_OUTPUT').

'DIR_INPUT'  holds data this package did not produce (jet system catalogues and FITS cutouts, BORG SDSS cubes);
'DIR_OUTPUT' holds everything it did, including files that other scripts read in turn.
Both default to subdirectories of the repository's 'data/', which assumes an editable installation.
Override defaults with the environment variables 'JETS_DIR_INPUT' and 'JETS_DIR_OUTPUT' to store data elsewhere.
"""

# Imports: standard library
import os
from pathlib import Path

DIR_REPO   = Path(__file__).resolve().parents[2]
DIR_INPUT  = Path(os.environ.get("JETS_DIR_INPUT",  DIR_REPO / "data" / "input")).expanduser()
DIR_OUTPUT = Path(os.environ.get("JETS_DIR_OUTPUT", DIR_REPO / "data" / "output")).expanduser()
