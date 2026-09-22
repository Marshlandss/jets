"""
Plot the filament orientation error distributions (FOEDs), for each reference method, in two figures.

Figure 1 (components), two panels:
    top:    voxelization (simulated);
    bottom: host galaxy localization and BORG SDSS posterior (measured; pooled over jet systems, realizations, and
            localization methods of the realization axes).
    Both panels show the uniform distribution on the sphere, for which the measured axis carries no information about the true one.
Figure 2 (budgets), three panels, composed from the components via the spherical law of cosines:
    B1 (minimal), B2 (fiducial), and B3 (extended). The B2 and B3 panels also show their Watson--uniform descriptions.
Both figures share the horizontal axis [0, 90] deg; legends quote E[P_2(cos A)], the factor by which each distribution of
filament orientation errors A attenuates the alignment quadrupole.

This script only reads. The components come from 'find_filament_orientation_errors_voxelization.py' and
'find_filament_orientation_errors_localization_posterior.py', the budgets from 'find_filament_orientation_errors_total.py',
and the Watson--uniform parameters from 'fit_filament_orientation_errors_total.py'.
As E[P_2] factors multiply under composition, the script checks the budget files against the component files.
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


def plotHistogram(ax, errors, symbol, bins, scale, colour, label):
    """
    Draw the probability density of 'errors' as a filled histogram, labelled with the distribution's E[P_2(cos A)].

    Parameters
    ----------
    ax     : matplotlib Axes; the panel to draw in
    errors : array of shape (n,); filament orientation errors, in deg, in [0, 90]
    symbol : str; LaTeX code (without dollar signs) of the random variable that 'errors' samples, such as 'symbolVoxelization'
    bins   : array of shape (m + 1,); histogram bin edges, in deg
    scale  : float; factor by which the probability densities (in deg^-1) are multiplied before drawing, in 1
    colour : str; bar colour
    label  : str; legend label, to which the distribution's E[P_2(cos A)] is appended
    """
    PDs, _ = np.histogram(errors, bins = bins, density = True) # in deg^-1
    ax.hist(bins[ : -1], bins = bins, weights = scale * PDs, color = colour, edgecolor = "white", linewidth = .3,
            label = r"%s: $\mathbb{E}[P_2(\cos %s)] = %.2f$" % (label, symbol, polynomialLegendreMeanSample(np.radians(errors), 2)))


plt.rcParams.update({
    "text.usetex":         True,
    "text.latex.preamble": r"\usepackage{amssymb}", # for \mathbb{E}
    "font.size":           11,
    "axes.labelsize":      11,
    "legend.fontsize":     9})

# Initialize settings.
labelSample                     = "Mpc"
labelsMethod                    = ("d", "a")                                                 # of the reference axis; "d": direct method, "a": adjusted method
labelsBudget                    = {"B1" : "minimal", "B2" : "fiducial", "B3" : "extended"}
labelsBudgetDescribed           = ("B2", "B3")                                               # the budget shown with its Watson--uniform description
numbersOfComponentsVoxelization = {"B1" : 1, "B2" : 3, "B3" : 4}                             # in 1; voxelization, plus the assumed components drawn from its distribution in 'find_filament_orientation_errors_total.py'
symbolVoxelization              = r"A_\mathrm{v}"
symbolLocalizationPosterior     = r"A_\mathrm{l,p}"
symbolTotal                     = r"A_\mathrm{f}"
numberOfBins                    = 36                                                         # in 1; over [0, 90] deg
scalePD                         = 100                                                        # in 1; probability densities are shown in units of 10^-2 1/deg
PDMaxComponents                 = 8.                                                         # in 1e-2 1/deg; upper vertical limit of the component panels
PDMaxBudgets                    = 2.1                                                        # in 1e-2 1/deg; upper vertical limit of the budget panels
colourHistograms                = "indianred"
colourModel                     = "grey"
colourUniform                   = "grey"
labelAxisX                      = r"filament orientation error $a$ ($^\circ$)"

# Load the components.
errorsVoxelization          = np.load(DIR_OUTPUT / f"filament_orientation_errors_voxelization_{FILAMENT_NUMBER_OF_SIMULATIONS}.npy") # in deg
errorsLocalizationPosterior = np.load(DIR_OUTPUT / f"filament_orientation_errors_localization_posterior_{labelSample}.npy")          # in deg; indexed [jet system, realization, method of realization axis, method of reference axis]

# For axes distributed uniformly on the sphere, the angle between them has probability density sin(a) (in rad^-1) on [0, 90] deg.
bins           = np.linspace(0, 90, num = numberOfBins + 1)                      # in deg
angles         = np.linspace(0, 90, num = 900 + 1)                               # in deg
PDsUniform     = scalePD * watson.PDsPolarAngle(angles, 0)                       # in 10^-2 deg^-1
P2Voxelization = polynomialLegendreMeanSample(np.radians(errorsVoxelization), 2) # in 1


for n, labelMethodReference in enumerate(labelsMethod):
    errorsPooled  = errorsLocalizationPosterior[ : , : , : , n].ravel()                  # in deg
    errorsBudgets = {labelBudget : np.load(DIR_OUTPUT / f"filament_orientation_errors_total_{labelBudget}_{labelSample}_{labelMethodReference}.npy")
                     for labelBudget in labelsBudget}                                   # in deg
    P2Pooled      = polynomialLegendreMeanSample(np.radians(errorsPooled), 2)           # in 1
    with open(DIR_OUTPUT / f"filament_orientation_errors_total_parametric_{labelSample}_{labelMethodReference}.json") as file:
        parametersBudgets = json.load(file)  # per budget: Watson concentration 'kappa' and uniform weight 'weightUniform'; in 1

    # Report summary statistics, and check each budget's E[P_2] against the product of its components' E[P_2].
    print(f"Reference method '{labelMethodReference}':")
    for label, errors in [("voxelization", errorsVoxelization), ("localization--posterior", errorsPooled), *errorsBudgets.items()]:
        print(f"    {label:24s}: {np.median(errors):5.2f} deg (median), {np.percentile(errors, 10):5.2f} deg (10th percentile), "
              f"{np.percentile(errors, 90):5.2f} deg (90th percentile); E[P_2] = {polynomialLegendreMeanSample(np.radians(errors), 2):.3f}; "
              f"N = {errors.size}")
    for labelBudget, errors in errorsBudgets.items():
        P2File       = polynomialLegendreMeanSample(np.radians(errors), 2)                      # in 1
        P2Components = P2Pooled * P2Voxelization ** numbersOfComponentsVoxelization[labelBudget] # in 1
        print(f"    {labelBudget}: E[P_2] = {P2File:.4f} (file), {P2Components:.4f} (components); difference {P2File - P2Components:+.4f}")


    # Figure 1: the components.
    fig, axes = plt.subplots(2, 1, figsize = (6, 4), sharex = True, constrained_layout = True)
    plotHistogram(axes[0], errorsVoxelization, symbolVoxelization,          bins, scalePD, colourHistograms, r"\textbf{voxelization} error")
    plotHistogram(axes[1], errorsPooled,       symbolLocalizationPosterior, bins, scalePD, colourHistograms, r"\textbf{localization--posterior} error")
    for ax, symbol in zip(axes, (symbolVoxelization, symbolLocalizationPosterior)):
        ax.plot(angles, PDsUniform, color = colourUniform, linestyle = "--", linewidth = 1, label = r"uniform on $\mathbb{S}^2$: $\mathbb{E}[P_2(Z)] = 0$")
        ax.set_ylim(bottom = 0, top = PDMaxComponents)
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
        plotHistogram(ax, errors, symbolTotal, bins, scalePD, colourHistograms, r"\textbf{" + f"{labelsBudget[labelBudget]}" + "} error budget")#f"{labelBudget}, {labelsBudget[labelBudget]}")
        if labelBudget in labelsBudgetDescribed:
            kappa, weightUniform = parametersBudgets[labelBudget]["kappa"], parametersBudgets[labelBudget]["weightUniform"] # in 1, in 1
            PDsModel             = scalePD * ((1 - weightUniform) * watson.PDsPolarAngle(angles, kappa) + weightUniform * watson.PDsPolarAngle(angles, 0)) # in 10^-2 1/deg
            ax.plot(angles, PDsModel, color = colourModel, linestyle = "--", linewidth = 1,
                    label = r"Watson--uniform ($\kappa = %.2f$, $w = %.2f$)" % (kappa, weightUniform))
        ax.set_ylabel(r"$f_{%s}$ ($10^{-2}\ \mathrm{deg}^{-1}$)" % symbolTotal)
    axes[0].set_ylim(0, PDMaxBudgets)
    for ax in axes:
        ax.legend(loc = "upper right", frameon = False)
    axes[2].set_xlim(0, 90)
    axes[2].set_xticks(np.arange(0, 90 + 1, 10))
    axes[2].set_xlabel(labelAxisX)

    pathFigure = DIR_OUTPUT / f"plot_filament_orientation_errors_total_{labelSample}_{labelMethodReference}.pdf"
    fig.savefig(pathFigure)
    plt.close(fig)
    print(f"Saved plot to '{pathFigure}'.")
