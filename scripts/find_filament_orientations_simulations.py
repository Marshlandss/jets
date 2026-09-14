"""
Estimate the angular error of filament orientations recovered from BORG SDSS density cubes,
by voxelising synthetic straight filaments with known orientations and running 'FilamentOrientationFinder' on them.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import SEED
from jets.filament_orientation import FilamentOrientationFinder
from jets.filament_simulation import make_beta_cylinder_density_cube, average_down
from jets.paths import DIR_SAVE
from jets.sphere_utils import axialSeparation, convertCartesianToSpherical

# Initialise settings.
numberOfRealisations = int(1e4) # in 1
numberOfVoxelsFine   = 205      # in 1; per side
lambdaMax            = 2.5      # in BORG voxel side lengths
stepAngle            = 1.       # in deg

FOF                  = FilamentOrientationFinder(lambdaMax = lambdaMax, stepAngle = stepAngle)
numberOfVoxelsCoarse = 2 * FOF.voxelRadius + 1 # in 1; per side
RNG                  = np.random.default_rng(SEED)


azimuthsTrue   = np.full(numberOfRealisations, np.nan) # in deg
altitudesTrue  = np.full(numberOfRealisations, np.nan) # in deg
azimuthsBest   = np.full(numberOfRealisations, np.nan) # in deg
altitudesBest  = np.full(numberOfRealisations, np.nan) # in deg
errorsAngular  = np.full(numberOfRealisations, np.nan) # in deg

for i in range(numberOfRealisations):
    # Generate a fine-grained filament with a random orientation and offset, and degrade it to BORG resolution.
    cube, axis, offset = make_beta_cylinder_density_cube(numberOfVoxelsFine, RNG, radiusCore = 1.2, beta = 2.0, rho0 = 1.6e-23)
    cubeCoarse         = average_down(cube, numberOfVoxelsCoarse)

    # The synthetic cube is indexed [ix, iy, iz]; the finder expects [iz, iy, ix].
    columnDensities = FOF.columnDensities(np.transpose(cubeCoarse, (2, 1, 0)))
    iAlt, iAz       = FOF.findBest(columnDensities)

    axes.append(axis)
    offsets.append(offset)
    densitiesSmall = average_down(cube, 5)

axes =[]
offsets=[]
FOF.findBest([np.array([128,128,128])] * 10000, densitiesMean)
azimuths  = np.full(len(axes), np.nan)
altitudes = np.full(len(axes), np.nan)
distancesAng = np.full(len(axes), np.nan)

for axis,i in zip(axes,range(len(axes))):
    x, y, z = axis
    alt_rad = np.arcsin(z)
    az_rad = np.arctan2(y, x)
    alt_deg = np.degrees(alt_rad)
    az_deg = np.degrees(az_rad) % 360.0
    azimuths[i] = az_deg
    altitudes[i] = alt_deg
    distance1 = distanceOnSphere(az_deg, alt_deg, FOF.azimuthsBest[i], FOF.altitudesBest[i])[0,0]
    distance2 = distanceOnSphere(az_deg, alt_deg, FOF.azimuthsBest[i] + 180, -1 * FOF.altitudesBest[i])[0,0]
    distance  = min(distance1, distance2)
    distancesAng[i] = distance
    print(az_deg, alt_deg, FOF.azimuthsBest[i], FOF.altitudesBest[i], distance)

np.save(DIR_SAVE / f"filament_voxelisation_errors_{numberOfRealisations}.npy", errorsAngular)