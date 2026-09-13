
directoryDataBORGSDSS  = "/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/PhD/Ardor Telae/data/BORG SDSS/"

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
#'''
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
#'''