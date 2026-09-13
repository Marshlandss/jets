# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC

def make_beta_cylinder_density_cube(
    N,
    rng,
    radius_mpc = 1.2,     # in Mpc
    beta       = 2.0,     # in 1
    rho0       = 1.6e-23, # in g/m^3; central density
):
    """
    Generate a fine-grained mass density cube containing a single straight filament with a beta-profile cross section,
    rho(d) = rho0 (1 + (d / radius_mpc)^2)^(-3 beta / 2), where d is the perpendicular distance to the filament axis.

    The cube spans 5 BORG voxels per side (≈ 20.9 Mpc comoving) at N fine cells per side. The filament axis has a random
    (isotropic) orientation and passes through a random point within half a BORG voxel of the cube centre, so that
    'average_down_to_5' samples the voxelisation error at a random phase.

    Parameters
    ----------
    N          : int; fine cells per side, must be divisible by 5
    rng        : numpy.random.Generator
    radius_mpc : float; core radius of the beta profile (in Mpc, comoving)
    beta       : float; beta-profile exponent (in 1)
    rho0       : float; central mass density (in g m^-3)

    Returns
    -------
    cube   : array of shape (N, N, N); mass density (in g m^-3)
    axis   : array of shape (3,); unit vector along the filament
    offset : array of shape (3,); point on the filament axis (in Mpc, comoving)
    """
    # Fine grid coordinates
    cube_size_mpc = 5 * BORG_VOXEL_SIZE_MPC # Total cube size = 5 voxels → ≈ 20.87 Mpc
    dx_mpc        = cube_size_mpc / N
    coords        = (np.arange(N) - (N - 1) / 2) * dx_mpc
    x, y, z       = np.meshgrid(coords, coords, coords, indexing="ij")

    # Random cylinder axis
    axis  = rng.normal(size = 3)
    axis /= np.linalg.norm(axis)

    # Offset within half a low-res voxel
    half_voxel = 0.5 * BORG_VOXEL_SIZE_MPC
    offset     = rng.uniform(-half_voxel, half_voxel, size=3) # Rather than 'np.array([0.,0.,0.])'.

    # Shifted coordinates
    xs = x - offset[0]
    ys = y - offset[1]
    zs = z - offset[2]

    # Perpendicular distance
    r_dot_a = axis[0] * xs + axis[1] * ys + axis[2] * zs
    r2      = xs**2 + ys**2 + zs**2
    d_perp2 = np.maximum(r2 - r_dot_a**2, 0.0)
    d_perp  = np.sqrt(d_perp2)

    # Beta-profile density
    cube = rho0 * (1 + (d_perp / radius_mpc)**2)**(-1.5 * beta)

    return cube, axis, offset


def average_down(cube, numberOfVoxels):
    """
    Degrade a cube to 'numberOfVoxels' voxels per side by averaging over blocks of (N / numberOfVoxels)^3 fine cells, mimicking BORG's resolution.
    """
    N = cube.shape[0]
    if N % numberOfVoxels != 0:
        raise ValueError(f"N must be divisible by {numberOfVoxels}.")
    m = N // numberOfVoxels
    return cube.reshape(numberOfVoxels, m, numberOfVoxels, m, numberOfVoxels, m).mean(axis=(1, 3, 5))


def column_density_along_axis(cube, cube_size_mpc, axis_index):
    """
    Integrate a mass density cube (in g m^-3) along its first axis, giving the column density (in g m^-2) as a 2D map.
    The cell size follows from 'cube_size_mpc' (comoving) and the cube's shape.
    """
    mpc_to_m = u.Mpc.to(u.m) # in m; 3.0857e22
    dx_m     = (cube_size_mpc / cube.shape[axis_index]) * mpc_to_m
    return np.sum(cube, axis = axis_index) * dx_m