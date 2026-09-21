"""
Measure the distribution of the filament orientation error: BORG SDSS statistical error and host galaxy localization error taken together.

For each jet system, the 'measured filament axis' is the one found in the BORG SDSS posterior mean density cube with one of the host galaxy localization methods.
The plausible true axes are those found in the posterior realizations, with both localization methods.
This script calculates and saves the angles between the latter and the former.
Those angles form the error distribution that the inference of the jet Watson concentration parameter κ marginalizes over.
The reference axis is the axis that the inference actually uses, rather than the realizations' own principal axis,
because the quantity of interest is how far the true axis may be from the axis used in the APAD data.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import BORG_INDEX_REALIZATION_START, BORG_INDEX_REALIZATION_STEP, BORG_NUMBER_OF_REALIZATIONS
from jets.filament_orientation import loadFilamentAxes, loadVoxelIndicesList
from jets.paths import DIR_OUTPUT

# Initialize settings.
labelSample       = "Mpc"
labelsMethod      = ("d", "a") # "d": direct method, "a": adjusted method
pathCatalogueMean = DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_mean.xlsx"
pathErrors        = DIR_OUTPUT / f"filament_orientation_errors_localization_posterior_{labelSample}.npy"

# Load the axes found in the posterior mean density cube; shape (numberOfJetSystems, numberOfMethods, 3).
axesMean           = np.stack([loadFilamentAxes(pathCatalogueMean, labelMethod) for labelMethod in labelsMethod], axis = 1)
numberOfJetSystems = axesMean.shape[0] # in 1

# Load the axes found in the posterior realizations; shape (numberOfJetSystems, numberOfRealizations, numberOfMethods, 3).
axesRealizations = np.full((numberOfJetSystems, BORG_NUMBER_OF_REALIZATIONS, len(labelsMethod), 3), np.nan)
for i in range(BORG_NUMBER_OF_REALIZATIONS):
    indexRealization         = BORG_INDEX_REALIZATION_START + i * BORG_INDEX_REALIZATION_STEP
    pathCatalogueRealization = DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_{indexRealization}.xlsx"
    print(f"Loading '{pathCatalogueRealization.name}' ({i + 1} of {BORG_NUMBER_OF_REALIZATIONS})...")
    for j, labelMethod in enumerate(labelsMethod):
        axesRealizations[ : , i, j] = loadFilamentAxes(pathCatalogueRealization, labelMethod)

print(axesMean.shape)
import sys
sys.exit()

directoryExcelSheets = "/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_kpc/"
numberOfJetSystems   = 777#242
numberOfExcelSheets  = 41
# These arrays have 3 dimensions; the shape could be (242, 10, 2). (For each jet system, we have a measurement for each of the 'numberOfExcelSheets' BORG SDSS realisations, and for 2 methods: direct and adjusted.)
XsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
YsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
ZsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
for i in range(0, numberOfExcelSheets):
    fileName = "fpa_" + str(2000 + i * 250) + "_exact_1.xlsx"
    print("Loading '" + fileName + "'...")
    XsDM, YsDM, ZsDM = loadFilamentVectors(directoryExcelSheets + fileName, method = "r")
    XsAM, YsAM, ZsAM = loadFilamentVectors(directoryExcelSheets + fileName, method = "j")
    XsAll[ : , i, 0] = XsDM
    YsAll[ : , i, 0] = YsDM
    ZsAll[ : , i, 0] = ZsDM
    XsAll[ : , i, 1] = XsAM
    YsAll[ : , i, 1] = YsAM
    ZsAll[ : , i, 1] = ZsAM

# Load Xs, Ys, and Zs of the filament orientations deduced from the BORG SDSS mean.
XsMean = np.full((numberOfJetSystems, 2), np.nan)
YsMean = np.full((numberOfJetSystems, 2), np.nan)
ZsMean = np.full((numberOfJetSystems, 2), np.nan)
fileName = "kpc_filament_pa_exact_1.xlsx"
print("Loading '" + fileName + "'...")
XsDM, YsDM, ZsDM = loadFilamentVectors(directoryExcelSheets + fileName, method = "r")
XsAM, YsAM, ZsAM = loadFilamentVectors(directoryExcelSheets + fileName, method = "j")
XsMean[ : , 0] = XsDM
YsMean[ : , 0] = YsDM
ZsMean[ : , 0] = ZsDM
XsMean[ : , 1] = XsAM
YsMean[ : , 1] = YsAM
ZsMean[ : , 1] = ZsAM

pyplot.scatter(XsDM, YsDM)
pyplot.show()

# Load azimuths and altitudes for both the direct method (DM) and the adjusted method (AM).
XsDM, YsDM, ZsDM, voxelIndicesDM = loadFilamentVectors(pathExcel, method = "r", returnVoxelIndices = True)
XsAM, YsAM, ZsAM, voxelIndicesAM = loadFilamentVectors(pathExcel, method = "j", returnVoxelIndices = True)

# df                 = pd.read_excel(pathExcel)
# anglesDM           = df["best_angle_r (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
# anglesAM           = df["best_angle_j (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
# voxelIndicesDM     = df["voxel_index_r (x,y,z)"].apply(ast.literal_eval).to_numpy()
# voxelIndicesAM     = df["voxel_index_j (x,y,z)"].apply(ast.literal_eval).to_numpy()
# numberOfJetSystems = anglesDM.shape[0] # in 1
# anglesDMAzimuth    = np.full(numberOfJetSystems, np.nan) # in deg
# anglesDMAltitude   = np.full(numberOfJetSystems, np.nan) # in deg
# anglesAMAzimuth    = np.full(numberOfJetSystems, np.nan) # in deg
# anglesAMAltitude   = np.full(numberOfJetSystems, np.nan) # in deg
# for i in range(numberOfJetSystems):
#     anglesDMAzimuth[i]  = anglesDM[i][0]
#     anglesDMAltitude[i] = anglesDM[i][1]
#     anglesAMAzimuth[i]  = anglesAM[i][0]
#     anglesAMAltitude[i] = anglesAM[i][1]
#
# # Calculate Cartesian unit vectors.
# XsDM = np.cos(np.radians(anglesDMAltitude)) * np.cos(np.radians(anglesDMAzimuth)) # in 1
# YsDM = np.cos(np.radians(anglesDMAltitude)) * np.sin(np.radians(anglesDMAzimuth)) # in 1
# ZsDM = np.sin(np.radians(anglesDMAltitude))                                       # in 1
# XsAM = np.cos(np.radians(anglesAMAltitude)) * np.cos(np.radians(anglesAMAzimuth)) # in 1
# YsAM = np.cos(np.radians(anglesAMAltitude)) * np.sin(np.radians(anglesAMAzimuth)) # in 1
# ZsAM = np.sin(np.radians(anglesAMAltitude))                                       # in 1

# Calculate angle differences.
dotProducts = np.clip(XsDM * XsAM + YsDM * YsAM + ZsDM * ZsAM, -1, 1) # Add 'np.clip' because of numerical error.
anglesDelta = np.degrees(np.arccos(dotProducts)) # in deg

# Determine for which jet systems the DM and AM identified the same host galaxy voxel.
areSameVoxel = np.equal(voxelIndicesDM, voxelIndicesAM)
print(np.sum(areSameVoxel))

# For the moment, include all jet systems for which 'anglesDelta' exceeds 'angleDeltaThreshold'.
angleDeltaThreshold = 1. # in deg
for a in anglesDelta[anglesDelta > angleDeltaThreshold]:
    print(a)

# Calculate MLE filament measurement error kappa.
MLEExpressionData = np.mean(np.square(dotProducts[anglesDelta > angleDeltaThreshold]))
MLEKappa          = watson.MLEKappa(MLEExpressionData)

#plt.hist(anglesDelta[~areSameVoxel], bins = np.linspace(0, 90, num = 6 + 1, endpoint = True))
pyplot.hist(anglesDelta[anglesDelta > angleDeltaThreshold], bins = np.linspace(0, 90, num = 18 + 1, endpoint = True))
pyplot.xticks(np.linspace(0, 90, num = 6 + 1, endpoint = True))
pyplot.title(r"$\kappa_\mathrm{f} = " + "{:.2f}".format(MLEKappa) + "$")
pyplot.show()
