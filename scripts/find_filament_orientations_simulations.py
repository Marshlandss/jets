"""
Estimate (a part of) the angular error of filament orientations measured from BORG SDSS density cubes,
by voxelising synthetic straight filaments with known orientations and running 'FilamentOrientationFinder' on them.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import SEED, FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP
from jets.filament_orientation import FilamentOrientationFinder
from jets.filament_simulation import cubeGenerateFilamentProfileBeta, cubeAverageDown
from jets.paths import DIR_SAVE
from jets.sphere_utils import axialSeparation, convertCartesianToSpherical

# Initialise settings.
numberOfRealisations = int(1e4) # in 1
numberOfVoxelsFine   = 205      # in 1; per side

# Initialise filament orientation finding.
FOF                  = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)
numberOfVoxelsCoarse = 2 * FOF.voxelRadius + 1 # in 1; per side
RNG                  = np.random.default_rng(SEED)
azimuthsTrue         = np.full(numberOfRealisations, np.nan) # in deg
altitudesTrue        = np.full(numberOfRealisations, np.nan) # in deg
azimuthsBest         = np.full(numberOfRealisations, np.nan) # in deg
altitudesBest        = np.full(numberOfRealisations, np.nan) # in deg
errorsAngular        = np.full(numberOfRealisations, np.nan) # in deg

# Loop over realisations.
for i in range(numberOfRealisations):
    # Generate a fine-grained filament with a random orientation and offset, and degrade it to BORG resolution.
    cube, axis, offset = cubeGenerateFilamentProfileBeta(numberOfVoxelsFine, numberOfVoxelsCoarse, RNG)
    cubeCoarse         = cubeAverageDown(cube, numberOfVoxelsCoarse)

    # Calculate column densities for a swathe of line segment orientations.
    # The synthetic cube is indexed [ix, iy, iz]; the FOF expects [iz, iy, ix].
    columnDensities = FOF.columnDensities(np.transpose(cubeCoarse, (2, 1, 0)))

    # Store the ground-truth and recovered filament orientations, as well as the error between them.
    iAlt, iAz                         = FOF.findBest(columnDensities)
    azimuthsTrue[i], altitudesTrue[i] = convertCartesianToSpherical(*axis)
    azimuthsBest[i], altitudesBest[i] = FOF.azimuths[iAz], FOF.altitudes[iAlt]
    errorsAngular[i]                  = axialSeparation(azimuthsTrue[i], altitudesTrue[i], azimuthsBest[i], altitudesBest[i])[0, 0]

    if (i + 1) % 100 == 0:
        print(f"Realisation {i + 1} of {numberOfRealisations}: error {errorsAngular[i]:.1f} deg")

# Store angular errors.
np.save(DIR_SAVE / f"filament_voxelisation_errors_{numberOfRealisations}.npy", errorsAngular)