"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using the BORG SDSS posterior mean density cube.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndicesList
from jets.paths import DIR_INPUT, DIR_OUTPUT

# Initialize settings.
labelSample  = "Mpc"
labelsMethod = ("d", "a") # "d": direct, "a": adjusted

# Initialize paths.
pathDensities = DIR_INPUT  / "reconstructions" / "borg_sdss_density.npz"
pathExcel     = DIR_OUTPUT / f"catalogue_filament_{labelSample}_mean.xlsx"

# Initialize filament orientation finding.
FOF           = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)

# Load the BORG SDSS posterior mean.
densitiesMean = np.load(pathDensities)["mean"] + 1 # in today's mean matter density
print(f"Loaded density cube of shape {densitiesMean.shape}; mean {np.mean(densitiesMean):.3f}, range [{np.amin(densitiesMean):.3f}, {np.amax(densitiesMean):.3f}].")

for labelMethod in labelsMethod:
    print(f"Finding filament orientations for method '{labelMethod}'...")
    # Load all host galaxy voxel indices.
    voxelIndicesList = loadVoxelIndicesList(pathExcel, labelMethod)
    # Find all column densities and best orientations.
    columnDensitiesAll, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesMean, voxelIndicesList)
    # Save all column densities.
    np.save(DIR_OUTPUT / f"column_densities_{labelSample}_mean_{labelMethod}.npy", columnDensitiesAll.astype(np.float32))
    # Save best orientations.
    writeFilamentOrientations(pathExcel, azimuthsBest, altitudesBest, columnDensitiesBest, labelMethod)
