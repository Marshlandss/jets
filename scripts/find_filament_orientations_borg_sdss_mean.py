"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using the BORG SDSS posterior mean density cube.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndicesList
from jets.paths import DIR_INPUT, DIR_OUTPUT

# Initialise settings.
methods       = ["d", "a"] # "d": direct, "a": adjusted

# Initialise paths.
pathDensities = DIR_INPUT  / "reconstructions" / "borg_sdss_density.npz"
pathExcel     = DIR_OUTPUT / "Mpc_filament_pa_exact_1.xlsx"

# Initialise filament orientation finding.
FOF           = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)

# Load the BORG SDSS posterior mean.
densitiesMean = np.load(pathDensities)["mean"] + 1 # in today's mean matter density
print(f"Loaded density cube of shape {densitiesMean.shape}; mean {np.mean(densitiesMean):.3f}, range [{np.amin(densitiesMean):.3f}, {np.amax(densitiesMean):.3f}].")

for method in methods:
    print(f"Finding filament orientations for method '{method}'...")
    # Load all host galaxy voxel indices.
    voxelIndicesList = loadVoxelIndicesList(pathExcel, method)
    # Find all column densities and best orientations.
    columnDensitiesAll, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesMean, voxelIndicesList)

    np.save(DIR_OUTPUT / f"Mpc_column_densities_all_{method}.npy", columnDensitiesAll)
    writeFilamentOrientations(pathExcel, azimuthsBest, altitudesBest, columnDensitiesBest, method)
