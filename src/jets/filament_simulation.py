# Imports: third-party
from astropy import units as u
import numpy as np
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC

def cubeGenerateFilamentProfileBeta(
    numberOfVoxelsFine,
    RNG,
    radiusCore = 1.2,      # in Mpc
    beta        = 2.0,     # in 1
    rho0        = 1.6e-23, # in g/m^3; central density
    axis        = None,
    numberOfVoxelsCoarse = 5):
    """
    Generate a fine-grained mass density cube containing a single straight filament with a beta-profile cross-section,
    rho(d) = rho0 (1 + (d / radiusCore)^2)^(-3 beta / 2), where d is the perpendicular distance to the filament axis.

    The cube spans 'numberOfVoxels' BORG voxels per side (≈ 20.9 Mpc comoving) at numberOfVoxelsFine fine cells per side.
    The filament axis has a random (isotropic) orientation and passes through a random point within half a BORG voxel of the cube centre,
    so that 'average_down' samples the voxelisation error at a random phase.

    Parameters
    ----------
    numberOfVoxelsFine : int; fine cells per side, must be divisible by 'numberOfVoxelsCoarse'
    RNG                : numpy.random.Generator
    radiusCore         : float; core radius of the beta profile (in Mpc, comoving)
    beta               : float; beta-profile exponent (in 1)
    rho0               : float; central mass density (in g m^-3)

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
        axis  = RNG.normal(size = 3)
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
    cube = rho0 * (1 + (d_perp / radiusCore) ** 2) ** (-1.5 * beta)

    return cube, axis, offset


def cubeAverageDown(cube, numberOfVoxelsCoarse):
    """
    Degrade a cube to 'numberOfVoxelsCoarse' voxels per side by averaging over blocks of (numberOfVoxelsFine / numberOfVoxelsCoarse)^3 fine cells, mimicking BORG's resolution.
    """
    if len(set(cube.shape)) != 1:
        raise ValueError("This function only works on arrays equally sized along all dimensions.")
    numberOfVoxelsFine = cube.shape[0]
    if numberOfVoxelsFine % numberOfVoxelsCoarse != 0:
        raise ValueError(f"'numberOfVoxelsFine' must be divisible by {numberOfVoxelsCoarse}.")

    m = numberOfVoxelsFine // numberOfVoxelsCoarse
    return cube.reshape(numberOfVoxelsCoarse, m, numberOfVoxelsCoarse, m, numberOfVoxelsCoarse, m).mean(axis = (1, 3, 5))


def cubeColumnDensityAlongAxis(cube, cube_size_mpc, axis_index):
    """
    Integrate a mass density cube (in g m^-3) along the axis with index 'axis_index', giving the column density (in g m^-2) as a 2D map.
    The cell size follows from 'cube_size_mpc' (comoving) and the cube's shape.
    """
    mpc_to_m = u.Mpc.to(u.m) # in m; 3.0857e22
    dx_m     = (cube_size_mpc / cube.shape[axis_index]) * mpc_to_m
    return np.sum(cube, axis = axis_index) * dx_m