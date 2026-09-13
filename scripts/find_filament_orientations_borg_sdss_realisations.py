
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
