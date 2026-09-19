"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using 41 BORG SDSS posterior realisations.
"""
# Imports: standard library
import shutil
# Imports: third-party
import h5py
# Imports: first-party
from jets.config import FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndicesList
from jets.paths import DIR_INPUT, DIR_OUTPUT

# Initialise settings.
methods              = ["d", "a"] # "d": direct, "a": adjusted
numberOfRealisations = 41   # in 1
indexFirst           = 2000 # in 1; index of the first realisation
indexStep            = 250  # in 1; index spacing between consecutive realisations

# Initialise filament orientation finding.
FOF                  = FilamentOrientationFinder(FILAMENT_LAMBDA_MAX, FILAMENT_ANGLE_STEP)

for i in range(numberOfRealisations):
    indexRealisation = indexFirst + i * indexStep
    pathDensities    = DIR_INPUT  / "reconstructions" / f"final_density_{indexRealisation}.h5"
    pathExcelLoad    = DIR_OUTPUT / f"fpa_{indexRealisation}.xlsx"
    pathExcelWrite   = DIR_OUTPUT / f"fpa_{indexRealisation}_exact_1.xlsx"
    print(f"Working on realisation {indexRealisation} ({i + 1} of {numberOfRealisations})...")

    # Load the BORG SDSS realisation.
    with h5py.File(pathDensities, "r") as file:
        densitiesRealisation = file["scalars"]["field"][()] + 1 # in today's mean matter density

    # Copy the catalogue, so that the filament orientations are added to a new file.
    if not pathExcelWrite.exists():
        shutil.copy(pathExcelLoad, pathExcelWrite)

    for method in methods:
        # For the adjusted method, host galaxy localisation is density realisation–dependent, and thus must be loaded in the loop over realisations.
        voxelIndicesList = loadVoxelIndicesList(pathExcelLoad, method)
        _, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesRealisation, voxelIndicesList)
        writeFilamentOrientations(pathExcelWrite, azimuthsBest, altitudesBest, columnDensitiesBest, method)
