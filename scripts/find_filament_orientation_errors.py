# Imports: standard library
import ast
# Imports: third-party
import numpy as np
import pandas as pd


def loadFilamentVectors(pathExcel, method = "r", returnVoxelIndices = False):
    """
    """
    dataFrame          = pd.read_excel(pathExcel)
    angles             = dataFrame["best_angle_" + method + " (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
    if returnVoxelIndices:
        voxelIndices = dataFrame["voxel_index_" + method + " (x,y,z)"].apply(ast.literal_eval).to_numpy()

    numberOfJetSystems = angles.shape[0] # in 1
    anglesAzimuth      = np.full(numberOfJetSystems, np.nan) # in deg
    anglesAltitude     = np.full(numberOfJetSystems, np.nan) # in deg
    for i in range(numberOfJetSystems):
        anglesAzimuth[i]  = angles[i][0]
        anglesAltitude[i] = angles[i][1]

    # Calculate Cartesian unit vectors.
    Xs = np.cos(np.radians(anglesAltitude)) * np.cos(np.radians(anglesAzimuth)) # in 1
    Ys = np.cos(np.radians(anglesAltitude)) * np.sin(np.radians(anglesAzimuth)) # in 1
    Zs = np.sin(np.radians(anglesAltitude))                                           # in 1

    if returnVoxelIndices:
        return Xs, Ys, Zs, voxelIndices
    else:
        return Xs, Ys, Zs
