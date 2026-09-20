"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using 'BORG_NUMBER_OF_REALIZATIONS' BORG SDSS posterior realizations.
"""
# Imports: third-party
import h5py
# Imports: first-party
from jets.config import FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP, BORG_NUMBER_OF_REALIZATIONS, BORG_INDEX_REALIZATION_START, BORG_INDEX_REALIZATION_STEP
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndicesList
from jets.paths import DIR_INPUT, DIR_OUTPUT

# Initialise settings.
labelsSample = ("Mpc", "kpc")
labelsMethod = ("d", "a") # "d": direct, "a": adjusted

# Initialize filament orientation finding.
FOF          = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)

for labelSample in labelsSample:
    for i in range(BORG_NUMBER_OF_REALIZATIONS):
        indexRealization = BORG_INDEX_REALIZATION_START + i * BORG_INDEX_REALIZATION_STEP
        pathDensities    = DIR_INPUT  / "reconstructions" / f"final_density_{indexRealization}.h5"
        pathExcel        = DIR_OUTPUT / f"catalogue_filament_{labelSample}_{indexRealization}.xlsx"
        print(f"Working on realization {indexRealization} ({i + 1} of {BORG_NUMBER_OF_REALIZATIONS})...")

        # Load the BORG SDSS realization.
        with h5py.File(pathDensities, "r") as file:
            densitiesRealization = file["scalars"]["field"][()] + 1 # in today's mean matter density

        for labelMethod in labelsMethod:
            # For the adjusted method, host galaxy localization is density realization–dependent,
            # and thus voxel indices must be loaded in the loop over realizations.
            voxelIndicesList = loadVoxelIndicesList(pathExcel, labelMethod)
            _, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesRealization, voxelIndicesList)
            writeFilamentOrientations(pathExcel, azimuthsBest, altitudesBest, columnDensitiesBest, labelMethod)
