# CLAUDE.md

Conventions for the `jets` repository (github.com/Marshlandss/jets). Read this before writing or reviewing code here, and follow it over your own defaults. When the repository and this file disagree, say so. Feel free to challenge conventions when they become cumbersome or illogical.

## What this repository is

A Python package (`src/jets/`) and scripts (`scripts/`) for measuring the orientations of Mpc-scale AGN jets and of the Cosmic Web filaments around their hosts, and the alignment between the two. The package holds computation; the scripts hold I/O and orchestration. A script loads, calls into the package, saves.
A package function does not print (except for progress reporting in long-running loops), read files it was not handed, or know about directories.

## Language and spelling

- English throughout, with **Oxford spelling**: British spelling, but `-ize`/`-ization` where the suffix is the Greek `-izein`: realize, organize, voxelize, initialize, normalize, maximize, visualize, localize. Keep `-ise` where it is part of the stem (advise, comprise, excise, exercise, revise) and `-yse` (analyse). This applies to identifiers, file names and data file names, not only to comments.
- Years in citations use the Holocene calendar: Jasche et al. (12015), Tuominen et al. (12021).

## Naming

- **File names**: `snake_case`, lowercase. `find_` for scripts that search for orientations, `plot_` for scripts that make figures.
- **Inside the code**: `camelCase` for functions, methods, parameters, variables and attributes. Constants in
  `config.py` are `UPPER_SNAKE_CASE`.
- **Abbreviations in capitals**, without implying a constant: `RNG` (random number generator), `FOF`
  (a `FilamentOrientationFinder` instance).
- Functions acting on one kind of object may carry it as a prefix: `cubeAverageDown`, `cubeGenerateFilamentProfileBeta`.
- Descriptive over short: `numberOfVoxelsFine`, not `N`; `indexAltitude`, not `iAlt`.
- Names describe quantities, never units: `radiusCore  # in Mpc`, not `radiusCoreMpc`. Exception: two variables holding the same quantity in different units.

## Units and comments

- Every physical quantity carries a unit comment at its definition: e.g. `# in deg` and `# in Mpc`. Dimensionless quantities are not exempted: e.g. `# in 1` or `# in %`. Write compound units with slashes: e.g. `# in g/m^3` and `# in km/s/Mpc`.
- Within a block of assignments, align the `=` signs and trailing comments. Alignment is per block, not per file.
- Comments explain intent, not mechanics: e.g. no `# increment i`.
- No commented-out code. Superseded code lives in git history. Delete it.

## Code style

- Spaces around `=` in keyword arguments and defaults: `RNG.normal(size = 3)`, `def f(axis = None)`.
- No parentheses around `if` conditions: `if axis is None:`.
- Imports grouped and labelled, in this order, importing only what is used:
  ```python
  # Imports: standard library
  # Imports: third-party
  # Imports: first-party
  ```
- Docstrings for every public function and class: a summary paragraph, then `Parameters` and `Returns` blocks giving type or shape, unit, and indexing convention where it matters (`indexed [iz, iy, ix]`). Code names in single quotes: `'cutout'`.
- Every text file ends with a newline.

## Data, paths and configuration

- Directories come only from `jets.paths`: `DIR_INPUT` (defaults to `data/input/`) and `DIR_OUTPUT` (defaults to `data/output/`), overridable through `JETS_DIR_INPUT` and `JETS_DIR_OUTPUT`. No absolute paths in code.
- `input/` holds what this repository's code does not produce: (jet system) catalogues, (radio and optical) images, and (BORG SDSS large-scale structure) reconstructions. `output/` holds everything it does produce, including files that other scripts read.
- Both directories are gitignored; only small derived files are explicitly whitelisted. Large data (`*.npz`, `*.h5`, FITS) never enters the repository.
- Physical constants, search parameters, and simulation parameters live in `config.py`, each with a unit comment and a source. Scripts import them and never repeat the literals. `config.py` contains no computation beyond arithmetic on its own constants.
- Random numbers come from `np.random.default_rng(SEED)` with `SEED` from `config.py`, passed explicitly.

## Array conventions

- BORG SDSS density cubes, and any cutout handed to `FilamentOrientationFinder`, are indexed `[iz, iy, ix]`. Voxel
  indices from the Excel catalogues are `(x, y, z)`. Synthetic cubes from `filament_simulation` are indexed
  `[ix, iy, iz]` and must be transposed with `np.transpose(cube, (2, 1, 0))` before reaching the finder.
- Densities are in units of the present-day cosmic mean (`DENSITY_MEAN_TODAY`), as BORG provides them.
- Orientations are (azimuth, altitude) in degrees, azimuth in [0, 360), altitude in [0, 90]; filament axes are undirected.

## Refactoring rule

Refactoring must not change results. Before and after restructuring code on the real-data pipeline, reproduce a stored output (e.g. one jet system's `(91, 360)` column-density map) to numerical tolerance. If a change is expected to shift results (a constant, a unit conversion), say so and quantify it.

## For Claude specifically

- Check the repository before making claims about it. Never assert that a file, function, path or dependency exists or is missing without looking.
- Do not invent directory layouts, file names or data keys; ask, or say you are guessing.
- Do not add dependencies without adding them to `pyproject.toml`.
- When asked for a design, give instructions and small verifiable code fragments rather than large generated modules.
- When remarking that a docstring or similar is insufficient, do not merely complain, but suggest a concrete way to fix it.
