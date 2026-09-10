# Black hole jets
This is a Python package that enables measurements of black hole jet orientations from radio astronomical images, and cosmic web filament orientations from 3D large-scale structure maps. It also enables analyses of the relationship between these orientations. Finally, the package produces publication-ready plots.

# Installation
```bash
# Option 1: Clone over HTTPS (no SSH keys needed; recommended).
git clone https://github.com/Marshlandss/jets.git
# Option 2: Clone over SSH (SSH keys needed).
git clone git@github.com:Marshlandss/jets.git
```
```bash
cd jets

# Option 1: Install dependencies with conda (recommended).
conda env create -f environment.yaml
conda activate jets
# Option 2: Install dependencies with pip.
pip install -e .
```
The package is now installed in editable mode: a `git pull` updates your installation immediately.
(Re-run `pip install -e .` if dependencies change.)

# Configuration
Before running any code, set two environment variables:
- `JETS_DIR_LOAD`: the directory containing the input catalogue, BORG SDSS cubes, and a subdirectory `fits/` with the radio cutouts;
- `JETS_DIR_SAVE`: the directory to which output (tables and plots) is written; the code creates any subdirectories it needs.

Choose either of these options:

```bash
# Option 1 (all OSs): Store the variables in the conda environment.
conda activate jets
conda env config vars set JETS_DIR_LOAD="/path/to/load" JETS_DIR_SAVE="/path/to/save"
conda deactivate
conda activate jets            # Re-activate so that the variables take effect.
conda env config vars list     # Check.

# Option 2 (macOS and Linux): Add these lines to your shell profile (e.g. ~/.zshrc on macOS).
# Then open a new terminal.
export JETS_DIR_LOAD="/path/to/load"
export JETS_DIR_SAVE="/path/to/save"
```
Surround paths that contain spaces with quotes: e.g. `"G:/My Drive/..."`.
If you run code from an IDE, check that its run configuration sees these variables.

# Usage
<div align="center">
  <img src="figures/158.51625_18.680278_notsub.png" alt="A Mpc-scale jet system: radio view, optical host galaxy view, and jet and filament orientation in 2D" width="80%">
</div>