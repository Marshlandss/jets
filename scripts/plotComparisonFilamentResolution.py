# Imports: third-party
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC, SEED
from jets.paths import DIR_SAVE
from jets.plot_utils import plotGalaxySpiral

plt.rcParams.update({"text.usetex": True})

def make_beta_cylinder_density_cube(
    N,
    radius_mpc=1.2,
    beta=2.0,
    rho0=1.6e-23,   # central density in g/m^3
    rng = None,
):
    rng = np.random.default_rng() if rng is None else rng

    # Total cube size = 5 voxels → ≈ 20.87 Mpc
    cube_size_mpc = BORG_VOXEL_SIZE_MPC * 5

    # Fine grid coordinates
    dx_mpc = cube_size_mpc / N
    coords = (np.arange(N) - (N - 1) / 2) * dx_mpc
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Random cylinder axis
    axis = rng.normal(size = 3)
    axis /= np.linalg.norm(axis)

    # Offset within half a low-res voxel
    half_voxel = 0.5 * BORG_VOXEL_SIZE_MPC
    offset = rng.uniform(-half_voxel, half_voxel, size=3) # Rather than 'np.array([0.,0.,0.])'.

    # Shifted coordinates
    xs = x - offset[0]
    ys = y - offset[1]
    zs = z - offset[2]

    # Perpendicular distance
    r_dot_a = axis[0]*xs + axis[1]*ys + axis[2]*zs
    r2 = xs**2 + ys**2 + zs**2
    d_perp2 = np.maximum(r2 - r_dot_a**2, 0.0)
    d_perp = np.sqrt(d_perp2)

    # Beta-profile density
    cube = rho0 * (1 + (d_perp / radius_mpc)**2)**(-1.5 * beta)

    return cube, axis, offset


def average_down_to_5(cube):
    N = cube.shape[0]
    if N % 5 != 0:
        raise ValueError("N must be divisible by 5.")
    m = N // 5
    return cube.reshape(5, m, 5, m, 5, m).mean(axis=(1, 3, 5))


def column_density_along_x(cube, cube_size_mpc):
    mpc_to_m = 3.085677581491367e22
    dx_m = (cube_size_mpc / cube.shape[0]) * mpc_to_m
    return np.sum(cube, axis=0) * dx_m


# ============================================================
# Main
# ============================================================
N    = 205
vmin = 0.0
vmax = 1.0

rng                = np.random.default_rng(SEED)
cube, axis, offset = make_beta_cylinder_density_cube(N = N, radius_mpc = 1.2, beta = 2.0, rho0 = 1.6e-23, rng = rng) # Parameters from Tuominen et al. (2021).
cube5              = average_down_to_5(cube)

cube_size_mpc = BORG_VOXEL_SIZE_MPC * 5
col_fine      = column_density_along_x(cube,  cube_size_mpc)
col_coarse    = column_density_along_x(cube5, cube_size_mpc)

half        = cube_size_mpc / 2.
extent      = [-half, half, -half, half]
ticks       = [-10.4, -5.2, 0.0, 5.2, 10.4]
ticklabels0 = ["$-10.4$", "$-5.2$", "$0$", "$+5.2$", "$+10.4$"]
ticklabels1 = ["", "$-5.2$", "$0$", "$+5.2$", "$+10.4$"]

fig = plt.figure(figsize = (6., 3.))
gs  = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.01)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
cax = fig.add_subplot(gs[0, 2])

im0 = ax0.imshow(col_fine, interpolation = "bilinear", origin = "lower", cmap = cm.lipari, vmin = vmin, vmax = vmax, extent = extent)
ax0.set_title(r"\textbf{filament:} ground truth",fontsize=10)
ax0.set_xlabel(r"comoving $x\ (\mathrm{Mpc})$")
ax0.set_ylabel(r"comoving $y\ (\mathrm{Mpc})$")
ax0.set_xticks(ticks)
ax0.set_xticklabels(ticklabels0,fontsize = 10)
ax0.set_yticks(ticks)
ax0.set_yticklabels(ticklabels0,fontsize = 10)
ax0.set_xlim(extent[0], extent[1])
ax0.set_ylim(extent[2], extent[3])

im1 = ax1.imshow(col_coarse, interpolation = "nearest", origin = "lower", cmap = cm.lipari, vmin = vmin, vmax = vmax, extent = extent)
ax1.set_title(r"\textbf{filament:} reconstruction resolution",fontsize=10)
ax1.set_xlabel(r"comoving $x\ (\mathrm{Mpc})$",fontsize=10)
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticklabels1,fontsize=10)
ax1.set_xlim(extent[0], extent[1])
ax1.set_ylim(extent[2], extent[3])
ax1.tick_params(labelleft = False)

plotGalaxySpiral(.25, centreX = 0., centreY = 0., radiusBulgeRelative=.15, ax=ax1)

# Draw grid.
edges = -half + np.arange(6) * BORG_VOXEL_SIZE_MPC  # 6 edges → 5 cells
for ax in (ax0, ax1):
    for e in edges:
        ax.axvline(e, color = ".4", lw = 0.2, alpha = 0.3, zorder = 3)
        ax.axhline(e, color = ".4", lw = 0.2, alpha = 0.3, zorder = 3)

# Draw colour bar attached to the right panel.
cbar = fig.colorbar(im1, cax = cax)
cbar.set_label(r"Cosmic Web column density $\sigma_\mathrm{CW}\ (\mathrm{g\ m^{-2}})$",fontsize=10)
cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('${x:.1f}$'))
plt.subplots_adjust(left = 0.11, right = 0.91, top = 0.9, bottom = 0.15, wspace = 0.01)

# Save figure.
pathFigure = DIR_SAVE / "comparisonFilamentResolution.pdf"
plt.savefig(pathFigure, dpi = 1000)
print(f"Saved figure to '{pathFigure}'.")
plt.close()