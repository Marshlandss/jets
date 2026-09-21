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


def mean_axis_from_components(Xs, Ys, Zs, w=None, normalize=True):
    """
    Compute the mean axis (±m) from axis data on S^2 given as components.

    Parameters
    ----------
    Xs, Ys, Zs : array_like, shape (N,)
        Components of the vectors.
    w : array_like, shape (N,), optional
        Non-negative weights. If None, all weights are 1.
    normalize : bool, default True
        If True, normalize each (X,Y,Z) to unit length before processing.

    Returns
    -------
    m : ndarray, shape (3,)
        Unit vector representing the mean axis (sign arbitrary).
    evals : ndarray, shape (3,)
        Eigenvalues of the scatter matrix (ascending order).
    """

    Xs = np.asarray(Xs, float)
    Ys = np.asarray(Ys, float)
    Zs = np.asarray(Zs, float)

    # Stack into (N, 3)
    X = np.column_stack((Xs, Ys, Zs))
    print(X.shape)

    if normalize:
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-length vector encountered.")
        X = X / norms[:, None]

    N = X.shape[0]

    if w is None:
        w = np.ones(N)
    else:
        w = np.asarray(w, float)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")

    # Scatter / second-moment matrix
    M = (X.T * w) @ X / w.sum()

    # Eigen-decomposition (symmetric -> eigh)
    evals, evecs = np.linalg.eigh(M)

    # Top eigenvector = mean axis
    m = evecs[:, np.argmax(evals)]
    m /= np.linalg.norm(m)

    return m, evals
