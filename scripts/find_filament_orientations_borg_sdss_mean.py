# Imports: third-party
import numpy as np
import pandas as pd
# Imports: first-party
from jets.filament_orientation import FilamentOrientationFinder

directoryDataBORGSDSS  = "/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/PhD/Ardor Telae/data/BORG SDSS/"

# Initialise FOF.
lambdaMax              = 2.5 # in 2.93 Mpc / h
stepAngle              = 1.  # in deg
FOF                    = FilamentOrientationFinder(lambdaMax = lambdaMax, stepAngle = stepAngle)

# Initialise directory paths.
directoryExcel         = "/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_Mpc/"

# Load host galaxy voxel indices.
pathExcel              = directoryExcel + "Mpc_filament_pa_exact_1.xlsx"#"kpc_filament_pa_exact_1.xlsx"
df                     = pd.read_excel(pathExcel)
voxelIndicesListDirect = [np.array(ast.literal_eval(s)) for s in df["voxel_index_r (x,y,z)"]] # list of NumPy arrays, each containing 3 integers (voxel indices)
voxelIndicesListAdjust = [np.array(ast.literal_eval(s)) for s in df["voxel_index_j (x,y,z)"]] # list of NumPy arrays, each containing 3 integers (voxel indices)

# Load BORG SDSS mean.
dataBORGSDSSDensity    = np.load(directoryDataBORGSDSS + "borg_sdss_density.npz")
densitiesMean          = dataBORGSDSSDensity["mean"] + 1 # in today's mean matter density
print(densitiesMean.shape)
print(np.amin(densitiesMean), np.amax(densitiesMean))
print(np.mean(densitiesMean))

# Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS mean (direct method).
FOF.findBest(voxelIndicesListDirect, densitiesMean)
from matplotlib import pyplot as plt
for i in range(10):
    plt.imshow(FOF.columnDensities[i+135], origin = "lower", aspect = "auto")
    plt.title(i+135)
    plt.show()
FOF.writeNumPy(directoryExcel + "Mpc_column_densities_all_r.npy")


#FOF.write(pathExcel, methodDirect = True)
# Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS mean (adjusted method).
FOF.findBest(voxelIndicesListAdjust, densitiesMean)
from matplotlib import pyplot as plt
for i in range(10):
    plt.imshow(FOF.columnDensities[i+135], origin = "lower", aspect = "auto")
    plt.title(i+135)
    plt.show()
FOF.writeNumPy(directoryExcel + "Mpc_column_densities_all_j.npy")
#FOF.write(pathExcel, methodDirect = False)