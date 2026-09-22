"""
Plot the filament orientation error distributions (FOEDs), for each reference method, in two figures.

Figure 1 (components), two panels:
    top:    voxelization (simulated);
    bottom: host galaxy localization and BORG SDSS posterior (measured; pooled over jet systems, realizations, and
            localization methods of the realization axes).
    Both panels show the isotropic distribution, for which the measured axis carries no information about the true one.
Figure 2 (budgets), three panels, composed from the components via the spherical law of cosines:
    B1 (minimal), B2 (fiducial), and B3 (illustrative). The B2 panel also shows its Watson + isotropic description,
    and that description's isotropic part.
Both figures share the horizontal axis [0, 90] deg; legends quote <P_2>, the factor by which each distribution attenuates
the alignment quadrupole.

This script only reads. The components come from 'find_filament_orientation_errors_voxelization.py' and
'find_filament_orientation_errors_localization_posterior.py', the budgets from 'find_filament_orientation_errors_total.py',
and the Watson + isotropic parameters from 'fit_filament_orientation_errors_total.py'. As <P_2> factors multiply under
composition, the script checks the budget files against the component files.
"""
# Imports: standard library
import json
# Imports: third-party
import matplotlib.pyplot as plt
import numpy as np
# Imports: first-party
from jets import watson
from jets.config import FILAMENT_NUMBER_OF_SIMULATIONS
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import polynomialLegendreMeanSample


def plotHistogram(ax, errors, label):
    """
    Draw the probability density of 'errors' as a filled histogram, in units of 10^-2 deg^-1, labelled with the distribution's <P_2>.

    Parameters
    ----------
    ax     : matplotlib Axes; the panel to draw in
    errors : array of shape (n,); filament orientation errors, in deg, in [0, 90]
    label  : str; legend label, to which the distribution's <P_2> is appended
    """
    PDs, _ = np.histogram(errors, bins = bins, density = True) # in deg^-1
    ax.hist(bins[ : -1], bins = bins, weights = scalePD * PDs, color = colourHistograms, edgecolor = "white", linewidth = .3,
            label = r"%s ($\langle P_2 \rangle = %.2f$)" % (label, polynomialLegendreMeanSample(np.radians(errors), 2)))


plt.rcParams.update({
    "text.usetex":     True,
    "font.size":       11,
    "axes.labelsize":  11,
    "legend.fontsize": 9})

# Initialize settings.
labelSample                     = "Mpc"
labelsMethod                    = ("d", "a")                                                 # of the reference axis; "d": direct method, "a": adjusted method
labelsBudget                    = {"B1" : "minimal", "B2" : "fiducial", "B3" : "illustrative"}
labelBudgetDescribed            = "B2"                                                       # the budget shown with its Watson + isotropic description
numbersOfComponentsVoxelization = {"B1" : 1, "B2" : 3, "B3" : 4}                             # in 1; voxelization, plus the assumed components drawn from its distribution in 'find_filament_orientation_errors_total.py'
symbolVoxelization              = r"A_\mathrm{vox}"
symbolLocalizationPosterior     = r"A_\mathrm{lp}"
symbolTotal                     = r"A_\mathrm{f}"
numberOfBins                    = 30                                                         # in 1; over [0, 90] deg
scalePD                         = 100                                                        # in 1; probability densities are shown in units of 10^-2 deg^-1
headroom                        = 1.3                                                        # in 1; upper vertical limit relative to the automatic one, leaving room for legends
colourHistograms                = "mediumseagreen"
colourModel                     = "black"
colourIsotropic                 = "grey"
labelAxisX                      = r"filament orientation error $a$ ($^\circ$)"

# Load the components.
errorsVoxelization          = np.load(DIR_OUTPUT / f"filament_orientation_errors_voxelization_{FILAMENT_NUMBER_OF_SIMULATIONS}.npy") # in deg
errorsLocalizationPosterior = np.load(DIR_OUTPUT / f"filament_orientation_errors_localization_posterior_{labelSample}.npy")          # in deg; indexed [jet system, realization, method of realization axis, method of reference axis]

bins           = np.linspace(0, 90, num = numberOfBins + 1)       # in deg
angles         = np.linspace(0, 90, num = 900 + 1)                # in deg
PDsIsotropic   = scalePD * watson.PDsPolarAngle(angles, 0)        # in 10^-2 deg^-1; sin(a), expressed per deg
P2Voxelization = polynomialLegendreMeanSample(np.radians(errorsVoxelization), 2)  # in 1


for n, labelMethodReference in enumerate(labelsMethod):
    errorsPooled  = errorsLocalizationPosterior[ : , : , : , n].ravel() # in deg
    errorsBudgets = {labelBudget : np.load(DIR_OUTPUT / f"filament_orientation_errors_total_{labelBudget}_{labelSample}_{labelMethodReference}.npy")
                     for labelBudget in labelsBudget}                    # in deg
    P2Pooled      = polynomialLegendreMeanSample(np.radians(errorsPooled), 2)           # in 1
    with open(DIR_OUTPUT / f"filament_orientation_errors_total_parametric_{labelSample}_{labelMethodReference}.json") as file:
        parameters = json.load(file)[labelBudgetDescribed]

    # Report summary statistics, and check each budget's <P_2> against the product of its components' <P_2>.
    print(f"Reference method '{labelMethodReference}':")
    for label, errors in [("voxelization", errorsVoxelization), ("localization + posterior", errorsPooled),
                          *errorsBudgets.items()]:
        print(f"    {label:24s}: {np.median(errors):5.2f} deg (median), {np.percentile(errors, 10):5.2f} deg (10th percentile), "
              f"{np.percentile(errors, 90):5.2f} deg (90th percentile); <P_2> = {polynomialLegendreMeanSample(np.radians(errors), 2):.3f}; N = {errors.size}")
    for labelBudget, errors in errorsBudgets.items():
        P2File       = polynomialLegendreMeanSample(np.radians(errors), 2)                                       # in 1
        P2Components = P2Pooled * P2Voxelization ** numbersOfComponentsVoxelization[labelBudget] # in 1
        print(f"    {labelBudget}: <P_2> = {P2File:.4f} (file), {P2Components:.4f} (components); difference {P2File - P2Components:+.4f}")


    # Figure 1: the components.
    fig, axes = plt.subplots(2, 1, figsize = (6, 4), sharex = True, constrained_layout = True)
    plotHistogram(axes[0], errorsVoxelization, "voxelization")
    plotHistogram(axes[1], errorsPooled,       "localization + posterior")
    for ax, symbol in zip(axes, (symbolVoxelization, symbolLocalizationPosterior)):
        ax.plot(angles, PDsIsotropic, color = colourIsotropic, linestyle = "--", linewidth = 1, label = r"isotropic ($\langle P_2 \rangle = 0$)")
        ax.set_ylim(bottom = 0)
        ax.set_ylabel(r"$f_{%s}$ ($10^{-2}\ \mathrm{deg}^{-1}$)" % symbol)
        ax.legend(loc = "upper right", frameon = False)
    axes[1].set_xlim(0, 90)
    axes[1].set_xticks(np.arange(0, 90 + 1, 10))
    axes[1].set_xlabel(labelAxisX)

    pathFigure = DIR_OUTPUT / f"plot_filament_orientation_errors_components_{labelSample}_{labelMethodReference}.pdf"
    fig.savefig(pathFigure)
    plt.close(fig)
    print(f"Saved plot to '{pathFigure}'.")


    # Figure 2: the budgets.
    fig, axes = plt.subplots(3, 1, figsize = (6, 6), sharex = True, sharey = True, constrained_layout = True)
    for ax, (labelBudget, errors) in zip(axes, errorsBudgets.items()):
        plotHistogram(ax, errors, f"{labelBudget}, {labelsBudget[labelBudget]}")
        if labelBudget == labelBudgetDescribed:
            kappa, weightWatson = parameters["kappa"], parameters["weightWatson"] # in 1, in 1
            PDsModel            = scalePD * (weightWatson * watson.PDsPolarAngle(angles, kappa) + (1 - weightWatson) * watson.PDsPolarAngle(angles, 0)) # in 10^-2 deg^-1
            ax.plot(angles, PDsModel, color = colourModel, linewidth = 1.2,
                    label = r"Watson + isotropic ($\kappa = %.2f$, $w = %.2f$)" % (kappa, weightWatson))
            ax.plot(angles, (1 - weightWatson) * PDsIsotropic, color = colourModel, linestyle = "--", linewidth = 1, label = r"isotropic part")
        ax.set_ylabel(r"$f_{%s}$ ($10^{-2}\ \mathrm{deg}^{-1}$)" % symbolTotal)
    axes[0].set_ylim(0, headroom * axes[0].get_ylim()[1])
    for ax in axes:
        ax.legend(loc = "upper right", frameon = False)
    axes[2].set_xlim(0, 90)
    axes[2].set_xticks(np.arange(0, 90 + 1, 10))
    axes[2].set_xlabel(labelAxisX)

    pathFigure = DIR_OUTPUT / f"plot_filament_orientation_errors_total_{labelSample}_{labelMethodReference}.pdf"
    fig.savefig(pathFigure)
    plt.close(fig)
    print(f"Saved plot to '{pathFigure}'.")


'''
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
'''
