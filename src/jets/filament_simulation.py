# Imports: third-party
from astropy import units as u
import numpy as np
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC, FILAMENT_BETA, FILAMENT_DENSITY_CENTRAL, FILAMENT_RADIUS_CORE

def cubeGenerateFilamentProfileBeta(
    numberOfVoxelsFine,   # in 1
    numberOfVoxelsCoarse, # in 1
    RNG,
    beta           = FILAMENT_BETA,            # in 1
    densityCentral = FILAMENT_DENSITY_CENTRAL, # in g/m^3; central density
    radiusCore     = FILAMENT_RADIUS_CORE,     # in Mpc
    axis           = None):                    # in 1
    """
    Generate a fine-grained mass density cube containing a single straight filament with a beta-profile cross-section,
    rho(d) = densityCentral (1 + (d / radiusCore)^2)^(-3 beta / 2), where d is the perpendicular distance to the filament axis.

    The cube spans 'numberOfVoxelsCoarse' BORG voxels per side at 'numberOfVoxelsFine' fine cells per side.
    The filament axis has a random (isotropic) orientation and passes through a random point within half a BORG voxel of the cube centre,
    so that 'cubeAverageDown' samples the voxelisation error at a random phase.

    Parameters
    ----------
    numberOfVoxelsFine   : int; fine cells per side, must be divisible by 'numberOfVoxelsCoarse'
    numberOfVoxelsCoarse : int; coarse cells per side
    RNG                  : numpy.random.Generator
    beta                 : float; beta-profile exponent (in 1)
    densityCentral       : float; central mass density (in g m^-3)
    radiusCore           : float; core radius of the beta profile (in Mpc, comoving)
    axis                 : array of shape (3,) or None; direction of the filament axis, normalised internally.
                           If None (default), an isotropically random direction is drawn from 'RNG'.
                           Supply a fixed direction to test the pipeline.

    Returns
    -------
    cube   : array of shape (numberOfVoxelsFine, numberOfVoxelsFine, numberOfVoxelsFine), indexed [ix, iy, iz]; mass density (in g m^-3)
    axis   : array of shape (3,); unit vector along the filament
    offset : array of shape (3,); point on the filament axis (in Mpc, comoving)
    """
    # Fine grid coordinates
    dx_mpc        = BORG_VOXEL_SIZE_MPC * numberOfVoxelsCoarse / numberOfVoxelsFine
    coords        = (np.arange(numberOfVoxelsFine) - (numberOfVoxelsFine - 1) / 2) * dx_mpc
    x, y, z       = np.meshgrid(coords, coords, coords, indexing = "ij")

    # If 'axis' is given, use it; otherwise, generate a random axis.
    if axis is None:
        axis = RNG.normal(size = 3)
    else:
        axis = np.asarray(axis, dtype = float)
    axis /= np.linalg.norm(axis)

    # Offset within half a low-res voxel
    half_voxel = 0.5 * BORG_VOXEL_SIZE_MPC
    offset     = RNG.uniform(-half_voxel, half_voxel, size = 3) # Rather than 'np.array([0.,0.,0.])'.

    # Shifted coordinates
    xs = x - offset[0]
    ys = y - offset[1]
    zs = z - offset[2]

    # Perpendicular distance
    r_dot_a = axis[0] * xs + axis[1] * ys + axis[2] * zs
    r2      = xs ** 2 + ys ** 2 + zs ** 2
    d_perp2 = np.maximum(r2 - r_dot_a**2, 0.0)
    d_perp  = np.sqrt(d_perp2)

    # Beta-profile density
    cube = densityCentral * (1 + (d_perp / radiusCore) ** 2) ** (-1.5 * beta)

    return cube, axis, offset


def cubeAverageDown(cube, numberOfVoxelsCoarse):
    """
    Degrade a cube to 'numberOfVoxelsCoarse' voxels per side by averaging over blocks of (numberOfVoxelsFine / numberOfVoxelsCoarse)^3 fine cells, mimicking BORG's resolution.
    """
    if len(set(cube.shape)) != 1:
        raise ValueError("This function only works on arrays equally sized along all dimensions.")
    numberOfVoxelsFine = cube.shape[0]
    if numberOfVoxelsFine % numberOfVoxelsCoarse != 0:
        raise ValueError(f"'cube' has 'numberOfVoxelsFine' = {numberOfVoxelsFine} cells per side, which is not divisible by 'numberOfVoxelsCoarse' = {numberOfVoxelsCoarse}.")

    m = numberOfVoxelsFine // numberOfVoxelsCoarse
    return cube.reshape(numberOfVoxelsCoarse, m, numberOfVoxelsCoarse, m, numberOfVoxelsCoarse, m).mean(axis = (1, 3, 5))


def cubeColumnDensityAlongAxis(cube, cubeLength, axisIndex):
    """
    Integrate a mass density cube (in g m^-3) along the axis with index 'axisIndex', giving the column density (in g m^-2) as a 2D map.
    The cell size follows from 'cubeLength' (in Mpc; comoving) and the cube's shape.
    """
    mpc_to_m = u.Mpc.to(u.m) # in m; 3.0857e22
    dx_m     = (cubeLength / cube.shape[axisIndex]) * mpc_to_m # in m (per pixel)
    return np.sum(cube, axis = axisIndex) * dx_m