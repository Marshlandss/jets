"""
Estimate (a part of) the angular error of filament orientations measured from BORG SDSS density cubes,
by voxelizing synthetic straight filaments with known orientations and running 'FilamentOrientationFinder' on them.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import SEED, FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP, FILAMENT_NUMBER_OF_SIMULATIONS
from jets.filament_orientation import FilamentOrientationFinder
from jets.filament_simulation import cubeGenerateFilamentProfileBeta, cubeAverageDown
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import axialSeparation, convertCartesianToSpherical

# Initialise settings.
numberOfVoxelsFine   = 205 # in 1; per side

# Initialise filament orientation finding.
FOF                  = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)
numberOfVoxelsCoarse = 2 * FOF.voxelRadius + 1 # in 1; per side
RNG                  = np.random.default_rng(SEED)
azimuthsTrue         = np.full(FILAMENT_NUMBER_OF_SIMULATIONS, np.nan) # in deg
altitudesTrue        = np.full(FILAMENT_NUMBER_OF_SIMULATIONS, np.nan) # in deg
azimuthsBest         = np.full(FILAMENT_NUMBER_OF_SIMULATIONS, np.nan) # in deg
altitudesBest        = np.full(FILAMENT_NUMBER_OF_SIMULATIONS, np.nan) # in deg
errorsAngular        = np.full(FILAMENT_NUMBER_OF_SIMULATIONS, np.nan) # in deg

# Loop over simulations.
for i in range(FILAMENT_NUMBER_OF_SIMULATIONS):
    # Generate a fine-grained filament with a random orientation and offset, and degrade it to BORG resolution.
    cube, axis, _ = cubeGenerateFilamentProfileBeta(numberOfVoxelsFine, numberOfVoxelsCoarse, RNG)
    cubeCoarse    = cubeAverageDown(cube, numberOfVoxelsCoarse)

    # Calculate column densities for a swathe of line segment orientations.
    # The synthetic cube is indexed [ix, iy, iz]; the FOF expects [iz, iy, ix].
    columnDensities = FOF.columnDensities(np.transpose(cubeCoarse, (2, 1, 0)))

    # Store the ground-truth and recovered filament orientations, as well as the error between them.
    azimuthsTrue[i], altitudesTrue[i] = convertCartesianToSpherical(*axis)
    axisBest, _                       = FOF.findBestPlateau(columnDensities)
    azimuthsBest[i], altitudesBest[i] = convertCartesianToSpherical(*axisBest)
    errorsAngular[i]                  = axialSeparation(azimuthsTrue[i], altitudesTrue[i], azimuthsBest[i], altitudesBest[i])[0, 0]

    if (i + 1) % 10 == 0:
        print(f"Simulation {i + 1} of {FILAMENT_NUMBER_OF_SIMULATIONS}: error {errorsAngular[i]:.1f} deg")

# Store angular errors.
np.save(DIR_OUTPUT / f"filament_voxelization_errors_{FILAMENT_NUMBER_OF_SIMULATIONS}.npy", errorsAngular)
