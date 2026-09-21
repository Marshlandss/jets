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
from jets.jet_utils import loadJetSystemNames
from jets.paths import DIR_OUTPUT

# Initialize settings.
labelSample       = "Mpc"
labelsMethod      = ("d", "a") # "d": direct method, "a": adjusted method
pathCatalogueMean = DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_mean.xlsx"
pathErrors        = DIR_OUTPUT / f"filament_orientation_errors_localization_posterior_{labelSample}.npy"

# Load the axes found in the posterior mean density cube; shape (numberOfJetSystems, numberOfMethods, 3).
axesMean           = np.stack([loadFilamentAxes(pathCatalogueMean, labelMethod) for labelMethod in labelsMethod], axis = 1)
numberOfJetSystems = axesMean.shape[0] # in 1
namesMean          = loadJetSystemNames(pathCatalogueMean)

# Load the axes found in the posterior realizations; shape (numberOfJetSystems, numberOfRealizations, numberOfMethods, 3).
axesRealizations = np.full((numberOfJetSystems, BORG_NUMBER_OF_REALIZATIONS, len(labelsMethod), 3), np.nan)
for i in range(BORG_NUMBER_OF_REALIZATIONS):
    indexRealization         = BORG_INDEX_REALIZATION_START + i * BORG_INDEX_REALIZATION_STEP
    pathCatalogueRealization = DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_{indexRealization}.xlsx"

    namesRealization = loadJetSystemNames(pathCatalogueRealization)
    if not np.array_equal(namesMean, namesRealization):
        raise ValueError(f"'{pathCatalogueRealization.name}' does not list the same jet systems in the same order as '{pathCatalogueMean.name}'.")

    print(f"Loading '{pathCatalogueRealization.name}' ({i + 1} of {BORG_NUMBER_OF_REALIZATIONS})...")
    for j, labelMethod in enumerate(labelsMethod):
        axesRealizations[ : , i, j] = loadFilamentAxes(pathCatalogueRealization, labelMethod)


# Calculate the angles between the realization axes and the mean cube axes, for both choices of reference method.
# Shape (numberOfJetSystems, numberOfRealizations, numberOfMethods, numberOfMethods), indexed
# [jet system, realization, localization method of the realization axis, localization method of the reference axis].
# Calculate dot products, which, for unit vectors, yields the cosine of the angle between them.
cosines       = np.abs(np.einsum("srmi,sni->srmn", axesRealizations, axesMean)) # in 1; absolute, as axes are undirected
# Calculate angles between axes.
errorsAngular = np.degrees(np.arccos(np.clip(cosines, 0, 1)))                   # in deg
np.save(pathErrors, errorsAngular)
print(f"Saved filament orientation errors to '{pathErrors}'.")


# Collect statistics on how methodological variation in host galaxy localization affects BORG SDSS mean cube filament axes.
# How often do the two methods select the same host galaxy voxel?
# What is the distribution of the angle between the axes that they lead to?
voxelIndices  = np.stack([np.array(loadVoxelIndicesList(pathCatalogueMean, labelMethod)) for labelMethod in labelsMethod], axis = 1)
areSameVoxel  = np.all(voxelIndices[ : , 0] == voxelIndices[ : , 1], axis = 1)
anglesMethods = np.degrees(np.arccos(np.clip(np.abs(np.einsum("si,si->s", axesMean[ : , 0], axesMean[ : , 1])), 0, 1))) # in deg
print(f"\nHost galaxy localization, mean cube: the two methods share a voxel for {np.sum(areSameVoxel)} of "
      f"{numberOfJetSystems} jet systems; their axes differ by {np.median(anglesMethods):.2f} deg (median), "
      f"{np.percentile(anglesMethods, 10):.2f} deg (10th percentile), "
      f"{np.percentile(anglesMethods, 90):.2f} deg (90th percentile).")
