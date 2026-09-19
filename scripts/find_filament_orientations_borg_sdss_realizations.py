"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using 41 BORG SDSS posterior realizations.
"""
# Imports: third-party
import h5py
# Imports: first-party
from jets.config import FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndicesList
from jets.paths import DIR_INPUT, DIR_OUTPUT

# Initialise settings.
labelsSample         = ("Mpc", "kpc")
labelsMethod         = ("d", "a") # "d": direct, "a": adjusted
numberOfRealizations = 41   # in 1
indexFirst           = 2000 # in 1; index of the first realization
indexStep            = 250  # in 1; index spacing between consecutive realizations

# Initialize filament orientation finding.
FOF                  = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)

for labelSample in labelsSample:
    for i in range(numberOfRealizations):
        indexRealization = indexFirst + i * indexStep
        pathDensities    = DIR_INPUT  / "reconstructions" / f"final_density_{indexRealization}.h5"
        pathExcel        = DIR_OUTPUT / f"catalogue_filament_{labelSample}_{indexRealization}.xlsx"
        print(f"Working on realization {indexRealization} ({i + 1} of {numberOfRealizations})...")

        # Load the BORG SDSS realization.
        with h5py.File(pathDensities, "r") as file:
            densitiesRealization = file["scalars"]["field"][()] + 1 # in today's mean matter density

        for labelMethod in labelsMethod:
            # For the adjusted method, host galaxy localization is density realization–dependent,
            # and thus voxel indices must be loaded in the loop over realizations.
            voxelIndicesList = loadVoxelIndicesList(pathExcel, labelMethod)
            _, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesRealization, voxelIndicesList)
            writeFilamentOrientations(pathExcel, azimuthsBest, altitudesBest, columnDensitiesBest, labelMethod)
