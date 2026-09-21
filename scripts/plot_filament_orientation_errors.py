"""
Visualize all filament orientation error distributions.
"""
# Imports: third-party
import matplotlib.pyplot as plt
import numpy as np
# Imports: first-party
from jets.config import SEED, FILAMENT_NUMBER_OF_SIMULATIONS
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import composeAngularErrors


# Initialize settings.
numberOfSamples        = int(1e6)
numberOfSystematics    = 3  # 1: voxelization errors only; each extra systematic error source is drawn from the same distribution
errorAngularDegreesMin = 0  # in deg
errorAngularDegreesMax = 60 # in deg
numberOfBins           = 30
colour                 = "mediumseagreen"

# Initialize random number generator and voxelization-induced error data.
RNG                  = np.random.default_rng(SEED)
errorsAngularDegrees = np.load(DIR_OUTPUT / f"filament_orientation_errors_voxelization_{FILAMENT_NUMBER_OF_SIMULATIONS}.npy") # in deg
errorsAngularRadians = np.radians(errorsAngularDegrees) # in rad


# Bootstrap the voxelization errors; every additional systematic is assumed to be an independent draw from the same distribution.
# For 'numberOfSystematics = 2', non-voxelization systematics are assumed comparable to voxelization systematics.
gamma                   = RNG.choice(errorsAngularRadians, size = numberOfSamples, replace = True)
for _ in range(numberOfSystematics - 1):
    beta  = RNG.choice(errorsAngularRadians, size = numberOfSamples, replace = True)
    gamma = composeAngularErrors(gamma, beta, RNG)
errorsAngularDegreesAll = np.degrees(gamma)

print("voxelization only: median = %5.2f deg, mean = %5.2f deg, N = %d"
      % (np.median(errorsAngularDegrees),    np.mean(errorsAngularDegrees),    len(errorsAngularDegrees)))
print("all systematics:   median = %5.2f deg, mean = %5.2f deg, N = %d"
      % (np.median(errorsAngularDegreesAll), np.mean(errorsAngularDegreesAll), len(errorsAngularDegreesAll)))


# Plot the error distributions.
plt.rcParams.update({
    "text.usetex":     True,
    "font.size":       11,
    "axes.labelsize":  11,
    "legend.fontsize": 9})

bins = np.linspace(errorAngularDegreesMin, errorAngularDegreesMax, num = numberOfBins + 1)

fig, (axTop, axBottom) = plt.subplots(2, 1, figsize = (6, 4), sharex = True, constrained_layout = True)

for ax, data, label in [
    (axTop,    errorsAngularDegrees,    r"voxelization only"),
    (axBottom, errorsAngularDegreesAll, r"all systematics (%d comparable components)" % numberOfSystematics),
]:
    ax.hist(data, bins = bins, density = True, color = colour, edgecolor = "white", linewidth = 0.3)
    median = np.median(data)
    ax.axvline(median, color = "black", linestyle = "--", linewidth = 1, label = r"median: $%.1f^\circ$" % median)
    ax.set_ylabel(r"probab. density ($\mathrm{deg}^{-1}$)")
    ax.set_xlim(errorAngularDegreesMin, errorAngularDegreesMax)
    ax.legend(title = label, loc = "upper right", frameon = False, alignment = "right")

axBottom.set_xlabel(r"filament orientation error $\angle(\hat{f}, \hat{f}_\mathrm{m,0})$ ($^\circ$)")

pathFigure = DIR_OUTPUT / "plot_filament_orientation_errors.pdf"
fig.savefig(pathFigure)
print(f"Saved plot to '{pathFigure}'.")
