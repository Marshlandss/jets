"""
Find the Cosmic Web filament orientations of Mpc-scale jet systems, using 41 BORG SDSS posterior realisations.
"""
# Imports: standard library
import shutil
# Imports: third-party
import h5py
# Imports: first-party
from jets.filament_orientation import FilamentOrientationFinder, findFilamentOrientations, writeFilamentOrientations, loadVoxelIndices
from jets.paths import DIR_LOAD, DIR_SAVE

# Initialise settings.
lambdaMax            = 2.5  # in BORG voxel side lengths
stepAngle            = 1.   # in deg
methods              = ["d", "a"] # "d": direct, "a": adjusted
numberOfRealisations = 41   # in 1
indexFirst           = 2000 # in 1; index of the first realisation
indexStep            = 250  # in 1; index spacing between consecutive realisations

# Initialise filament orientation finding.
FOF                  = FilamentOrientationFinder(lambdaMax, stepAngle)

for i in range(numberOfRealisations):
    indexRealisation = indexFirst + i * indexStep
    pathDensities    = DIR_LOAD / "borg_sdss" / "final_density" / f"final_density_{indexRealisation}.h5"
    pathExcelLoad    = DIR_SAVE / f"fpa_{indexRealisation}.xlsx"
    pathExcelWrite   = DIR_SAVE / f"fpa_{indexRealisation}_exact_1.xlsx"
    print(f"Working on realisation {indexRealisation} ({i + 1} of {numberOfRealisations})...")

    # Load the BORG SDSS realisation.
    with h5py.File(pathDensities, "r") as file:
        densitiesRealisation = file["scalars"]["field"][()] + 1 # in today's mean matter density

    # Copy the catalogue, so that the filament orientations are added to a new file.
    if not pathExcelWrite.exists():
        shutil.copy(pathExcelLoad, pathExcelWrite)

    for method in methods:
        voxelIndicesList = loadVoxelIndices(pathExcelLoad, method)
        _, azimuthsBest, altitudesBest, columnDensitiesBest = findFilamentOrientations(FOF, densitiesRealisation, voxelIndicesList)
        writeFilamentOrientations(pathExcelWrite, azimuthsBest, altitudesBest, columnDensitiesBest, method)

'''
# Find and write to file the best filament orientations for Mpc-scale jet systems in BORG SDSS realisations (direct method).
numberOfExcelSheets = 41 # in 1
for i in range(0, numberOfExcelSheets):
    realisationIndexString  = str(2000 + i * 250)
    pathBORGSDSSRealisation = directoryDataBORGSDSS + "final_density/" + "final_density_" + realisationIndexString + ".h5"
    pathExcelLoad           = directoryExcel + "fpa_" + realisationIndexString + ".xlsx"
    pathExcelWrite          = directoryExcel + "fpa_" + realisationIndexString + "_exact_1.xlsx"

    print("Working on realisation '" + realisationIndexString + "'...")
    print(pathBORGSDSSRealisation)
    print(pathExcelLoad)
    print(pathExcelWrite)

    # Load BORG SDSS realisation.
    with h5py.File(pathBORGSDSSRealisation, "r") as hf:
        densitiesSample = hf["scalars"]["field"][()] + 1 # in today's mean matter density
    print(densitiesSample.shape)
    print(np.amin(densitiesSample), np.amax(densitiesSample))
    print(np.mean(densitiesSample))

    # Load host galaxy voxel indices.
    df                     = pd.read_excel(pathExcelLoad)
    voxelIndicesListDirect = [np.array(ast.literal_eval(s)) for s in df["voxel_index_r (x,y,z)"]] # list of NumPy arrays, each containing 3 integers
    voxelIndicesListAdjust = [np.array(ast.literal_eval(s)) for s in df["voxel_index_j (x,y,z)"]] # list of NumPy arrays, each containing 3 integers

    # Create 'pathExcelWrite' if it doesn't exist yet.
    if (not os.path.exists(pathExcelWrite)):
        command = f"cp '{pathExcelLoad}' '{pathExcelWrite}'"
        print(command)
        os.system(command)

    # Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS realisation (direct method).
    FOF.findBest(voxelIndicesListDirect, densitiesSample)
    FOF.write(pathExcelWrite, methodDirect = True)
    # Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS realisation (adjusted method).
    FOF.findBest(voxelIndicesListAdjust, densitiesSample)
    FOF.write(pathExcelWrite, methodDirect = False)
'''