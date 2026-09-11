"""
Combine voxelisation-induced filament orientation errors with other, unmodelled  systematic errors of comparable magnitude via the spherical law of cosines.

Geometry (unit sphere S^2):
    T  = true filament axis
    P1 = axis after voxelisation error  (arc T-P1  = alpha)
    P2 = axis after an additional error (arc P1-P2 = beta)
The total error gamma is the arc T-P2. In the spherical triangle T-P1-P2, the
angle at P1 between arcs P1-T and P1-P2 is phi, so

    cos(gamma) = cos(alpha) cos(beta) + sin(alpha) sin(beta) cos(phi).

Assuming the direction of the second error is isotropic, phi ~ Uniform(0, 2 pi).
"""
# Imports: third-party
import matplotlib.pyplot as plt
import numpy as np
# Imports: first-party
from jets.config import SEED
from jets.paths import DIR_LOAD, DIR_SAVE

def composeAngularErrors(alpha, beta, rng):
    """
    Spherical law of cosines: total angular displacement gamma (rad) after two
    successive displacements alpha and beta (rad) with an isotropic relative
    azimuth phi.
    """
    phi      = rng.uniform(0, 2 * np.pi, size = len(alpha))
    cosGamma = np.cos(alpha) * np.cos(beta) + np.sin(alpha) * np.sin(beta) * np.cos(phi)
    return np.arccos(np.clip(cosGamma, -1, 1))


# Initialise settings.
numberOfSamples        = int(1e6)
numberOfSystematics    = 2  # 1: voxelisation errors only; each extra systematic error source is drawn from the same distribution
errorAngularDegreesMin = 0  # in deg
errorAngularDegreesMax = 60 # in deg
numberOfBins           = 30
colour                 = "mediumseagreen"

# Initialise random number generator and voxelisation-induced error data.
rng                  = np.random.default_rng(SEED)
errorsAngularDegrees = np.load(DIR_LOAD / "filament_voxelisation_errors_10000.npy") # in deg
errorsAngularRadians = np.radians(errorsAngularDegrees)                             # in rad


# Bootstrap the voxelisation errors; every additional systematic is assumed to be an independent draw from the same distribution.
# For 'numberOfSystematics = 2', non-voxelisation systematics are assumed comparable to voxelisation systematics.
gamma                   = rng.choice(errorsAngularRadians, size = numberOfSamples, replace = True)
for _ in range(numberOfSystematics - 1):
    beta  = rng.choice(errorsAngularRadians, size = numberOfSamples, replace = True)
    gamma = composeAngularErrors(gamma, beta, rng)
errorsAngularDegreesAll = np.degrees(gamma)

print("voxelisation only: median = %5.2f deg, mean = %5.2f deg, N = %d"
      % (np.median(errorsAngularDegrees),    np.mean(errorsAngularDegrees),    len(errorsAngularDegrees)))
print("all systematics:   median = %5.2f deg, mean = %5.2f deg, N = %d"
      % (np.median(errorsAngularDegreesAll), np.mean(errorsAngularDegreesAll), len(errorsAngularDegreesAll)))


# Plot the error distributions.
plt.rcParams.update({
    "text.usetex":     True,
    "font.size":       11,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
})

bins = np.linspace(errorAngularDegreesMin, errorAngularDegreesMax, num = numberOfBins + 1)

fig, (axTop, axBottom) = plt.subplots(2, 1, figsize = (6, 4), sharex = True, constrained_layout = True)

for ax, data, label in [
    (axTop,    errorsAngularDegrees,    r"voxelisation only"),
    (axBottom, errorsAngularDegreesAll, r"all systematics (%d comparable components)" % numberOfSystematics),
]:
    ax.hist(data, bins = bins, density = True, color = colour, edgecolor = "white", linewidth = 0.3)
    median = np.median(data)
    ax.axvline(median, color = "black", linestyle = "--", linewidth = 1, label = r"median: $%.1f^\circ$" % median)
    ax.set_ylabel(r"probab. density ($\mathrm{deg}^{-1}$)")
    ax.set_xlim(errorAngularDegreesMin, errorAngularDegreesMax)
    ax.legend(title = label, loc = "upper right", frameon = False, alignment = "right")

axBottom.set_xlabel(r"filament orientation error $\angle(\hat{f}, \hat{f}_\mathrm{m,0})$ ($^\circ$)")

pathFigure = DIR_SAVE / "filament_voxelisation_errors.pdf"
fig.savefig(pathFigure)
print(f"Saved figure to '{pathFigure}'.")