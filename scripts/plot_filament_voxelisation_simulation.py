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
from jets.filament_simulation import cubeGenerateFilamentProfileBeta, cubeAverageDown, column_density_along_axis

plt.rcParams.update({
    "text.usetex":     True,
    "font.size":       10,
    "axes.titlesize":  10,
    "axes.labelsize":  10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10})

# ============================================================
# Main
# ============================================================
numberOfVoxelsFine   = 205 # in 1
numberOfVoxelsCoarse = 5   # in 1
vmin                 = 0.0
vmax                 = 1.0

RNG                = np.random.default_rng(SEED)
cube, axis, offset = cubeGenerateFilamentProfileBeta(numberOfVoxelsFine = numberOfVoxelsFine, RNG = RNG, radiusCore = 1.2, beta = 2.0, rho0 = 1.6e-23) # Parameters from Tuominen et al. (2021).
cube_coarse        = cubeAverageDown(cube, numberOfVoxelsCoarse)
cube_size_mpc      = BORG_VOXEL_SIZE_MPC * numberOfVoxelsCoarse
col_fine           = column_density_along_axis(cube,        cube_size_mpc, axis_index = 0)
col_coarse         = column_density_along_axis(cube_coarse, cube_size_mpc, axis_index = 0)


half        = cube_size_mpc / 2.
extent      = [-half, half, -half, half]
ticks       = [-10.4, -5.2, 0.0, 5.2, 10.4]
ticklabels0 = ["$-10.4$", "$-5.2$", "$0$", "$+5.2$", "$+10.4$"]
ticklabels1 = ["", "$-5.2$", "$0$", "$+5.2$", "$+10.4$"]

fig = plt.figure(figsize = (6., 3.))
gs  = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.01)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1], sharey = ax0)
cax = fig.add_subplot(gs[0, 2])

im0 = ax0.imshow(col_fine, interpolation = "bilinear", origin = "lower", cmap = cm.lipari, vmin = vmin, vmax = vmax, extent = extent)
ax0.set_title(r"\textbf{filament:} ground truth")
ax0.set_xlabel(r"comoving $x\ (\mathrm{Mpc})$")
ax0.set_ylabel(r"comoving $y\ (\mathrm{Mpc})$")
ax0.set_xticks(ticks)
ax0.set_xticklabels(ticklabels0)
ax0.set_yticks(ticks)
ax0.set_yticklabels(ticklabels0)
ax0.set_xlim(extent[0], extent[1])
ax0.set_ylim(extent[2], extent[3])

im1 = ax1.imshow(col_coarse, interpolation = "nearest", origin = "lower", cmap = cm.lipari, vmin = vmin, vmax = vmax, extent = extent)
ax1.set_title(r"\textbf{filament:} reconstruction resolution")
ax1.set_xlabel(r"comoving $x\ (\mathrm{Mpc})$")
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticklabels1)
ax1.set_xlim(extent[0], extent[1])
ax1.set_ylim(extent[2], extent[3])
ax1.tick_params(labelleft = False)

# Draw galaxy symbol.
plotGalaxySpiral(.25, centreX = 0., centreY = 0., radiusBulgeRelative=.15, ax=ax1)

# Draw grid.
edges = -half + np.arange(6) * BORG_VOXEL_SIZE_MPC  # 6 edges → 5 cells
for ax in (ax0, ax1):
    for e in edges:
        ax.axvline(e, color = ".4", lw = 0.2, alpha = 0.3, zorder = 3)
        ax.axhline(e, color = ".4", lw = 0.2, alpha = 0.3, zorder = 3)

# Draw colour bar attached to the right panel.
cbar = fig.colorbar(im1, cax = cax)
cbar.set_label(r"Cosmic Web column density $\sigma_\mathrm{CW}\ (\mathrm{g\ m^{-2}})$")
cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('${x:.1f}$'))
plt.subplots_adjust(left = 0.11, right = 0.91, top = 0.9, bottom = 0.15, wspace = 0.01)

# Save figure.
pathFigure = DIR_SAVE / "filament_voxelisation_simulation.pdf"
plt.savefig(pathFigure, dpi = 1000)
print(f"Saved figure to '{pathFigure}'.")
plt.close()