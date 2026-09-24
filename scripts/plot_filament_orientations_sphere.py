"""
Plot filament axes found with the BORG SDSS posterior on the upper unit hemisphere, seen from above.

Each panel shows one jet system and, for both host galaxy localization methods (direct and adjusted):
    the axes found in the individual posterior realizations (dots);
    the axis found in the posterior mean cube (stars), which is the axis that enters the APADs;
    the principal axis of the realization axes (diamonds), for comparison with the mean cube axis.
Faint great-circle arcs connect each realization axis to the reference axis (the mean cube axis of the reference method);
their angular lengths are the filament orientation errors of 'find_filament_orientation_errors_localization_posterior.py'.
Arc parts on the lower hemisphere are drawn dashed, as if seen through the sphere; there, the arc heads for the reference
axis's antipode (hollow star), which represents the same undirected axis.

The script makes the paper figure, a row of example jet systems with the direct method as reference, and optionally one
figure per jet system and reference method. It also reports, over all jet systems, the angle between each method's
principal axis and its mean cube axis.
"""
# Imports: third-party
from cmcrameri import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
# Imports: first-party
from jets.config import BORG_INDEX_REALIZATION_START, BORG_INDEX_REALIZATION_STEP, BORG_NUMBER_OF_REALIZATIONS
from jets.filament_orientation import loadFilamentAxes, loadFilamentAxesRealizations
from jets.jet_utils import loadJetSystemNames
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import axialSeparationCartesian, axisPrincipal, geodesicArc


def plotArcAxial(ax, axisStart, axisEnd, **kwargs):
    """
    Draw the great-circle arc between two undirected axes, projected onto the plane z = 0 as seen from z > 0.
    The arc starts at 'axisStart' and ends at whichever of 'axisEnd' and its antipode is nearer; parts on the lower
    hemisphere (z < 0) are drawn dashed.

    Parameters
    ----------
    ax        : matplotlib Axes
    axisStart : array of shape (3,); Cartesian unit vector with a non-negative z-component
    axisEnd   : array of shape (3,); Cartesian unit vector
    kwargs    : passed to 'ax.plot', e.g. 'color', 'lw' and 'alpha'
    """
    if axisStart @ axisEnd < 0:
        axisEnd = -axisEnd
    points      = geodesicArc(axisStart, axisEnd)
    areVisible  = points[ : , 2] >= 0
    # Replace the points of the other part by NaNs, so that each part is drawn as one line.
    pointsFront = np.where(areVisible[ : , None], points, np.nan)
    pointsBack  = np.where(areVisible[ : , None], np.nan, points)
    ax.plot(pointsFront[ : , 0], pointsFront[ : , 1], ls = "-",  **kwargs)
    ax.plot(pointsBack [ : , 0], pointsBack [ : , 1], ls = "--", **kwargs)


def plotHemisphere(ax, axesRealizations, axesMean, axesPrincipal, axisReference, coloursMethod, colourMap):
    """
    Draw one jet system's filament axes on the upper unit hemisphere, projected onto the plane z = 0 as seen from z > 0,
    over a background shaded by z.

    Parameters
    ----------
    ax               : matplotlib Axes
    axesRealizations : array of shape (numberOfRealizations, numberOfMethods, 3); Cartesian unit vectors with z >= 0
    axesMean         : array of shape (numberOfMethods, 3); Cartesian unit vectors with z >= 0
    axesPrincipal    : array of shape (numberOfMethods, 3); Cartesian unit vectors with z >= 0
    axisReference    : array of shape (3,); Cartesian unit vector with z >= 0; the arcs' end point
    coloursMethod    : sequence of length 'numberOfMethods'; marker colour per method
    colourMap        : matplotlib Colormap; background shading

    Returns
    -------
    matplotlib AxesImage; the background, for a colour bar
    """
    # Shade the upper hemisphere by z, the sine of the altitude; mask (i.e. leave transparent) pixels outside the unit disk.
    numberOfPixels = 1001                                                                               # in 1; per side
    gridXs         = np.linspace(-1, 1, num = numberOfPixels)                                           # in 1
    gridYs         = np.linspace(-1, 1, num = numberOfPixels)                                           # in 1
    radiiSquared   = np.square(gridXs)[None, : ] + np.square(gridYs)[ : , None]                         # in 1
    gridZs         = np.ma.masked_where(radiiSquared > 1, np.sqrt(np.clip(1 - radiiSquared, 0, None)))  # in 1
    image          = ax.imshow(gridZs, cmap = colourMap, origin = "lower", extent = (-1, 1, -1, 1), vmin = 0, vmax = 1)
    ax.add_patch(plt.Circle((0, 0), 1, color = colourMap(0.), fill = False, lw = 1))

    # Draw the reference axis as a diameter, from the axis to its antipode.
    ax.plot([axisReference[0], -axisReference[0]], [axisReference[1], -axisReference[1]], color = ".25", ls = ":", lw = 1, zorder = 2)
    ax.scatter(0, 0, color = ".25", s = 8, zorder = 2)

    # Draw the arcs from the realization axes to the reference axis.
    for axisRealization in axesRealizations.reshape(-1, 3):
        plotArcAxial(ax, axisRealization, axisReference, color = ".35", lw = .5, alpha = .5, zorder = 3)

    # Draw the realization axes, the principal axes, and the mean cube axes.
    for indexMethod, colour in enumerate(coloursMethod):
        ax.scatter(axesRealizations[ : , indexMethod, 0], axesRealizations[ : , indexMethod, 1], color = colour, lw = 0, s = 12, zorder = 5)
        ax.scatter(axesPrincipal[indexMethod, 0], axesPrincipal[indexMethod, 1], color = colour, marker = "D", s = 18, lw = .6, edgecolor = ".3", zorder = 6)
        ax.scatter(axesMean     [indexMethod, 0], axesMean     [indexMethod, 1], color = colour, marker = "*", s = 70, lw = .6, edgecolor = ".3", zorder = 7)

    # Draw the reference axis's antipode, towards which arcs on the lower hemisphere head.
    ax.scatter(-axisReference[0], -axisReference[1], color = "white", marker = "*", s = 35, lw = .6, edgecolor = ".25", zorder = 7)

    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    return image


def makeHandlesLegend(labelsMethod, namesMethod, coloursMethod):
    """
    Return legend handles for the realization axes, mean cube axes and principal axes of each method in 'labelsMethod'.
    Handles are ordered by marker, then by method, so that a legend with three columns has one column per marker.
    """
    handles = []
    for marker, size, labelMarker in (("o", 3.5, "realizations"), ("*", 8, "mean cube"), ("D", 3.5, "principal axis")):
        for labelMethod in labelsMethod:
            edgeWidth = 0 if marker == "o" else .6
            handles.append(Line2D([], [], ls = "", marker = marker, ms = size, mew = edgeWidth, mec = ".3",
                                  color = coloursMethod[labelMethod], label = f"{namesMethod[labelMethod]}: {labelMarker}"))
    return handles


plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8})

# Initialize settings. 'colourMap' uses only the lighter part of 'bilbao', whose darkest reds would compete with the adjusted-method markers.
labelSample               = "Mpc"
labelsMethod              = ("d", "a")         # "d": direct method, "a": adjusted method
namesMethod               = {"d" : "direct", "a" : "adjusted"}
coloursMethod             = {"d" : "cornflowerblue", "a" : "tomato"}
colourMap                 = ListedColormap(cm.bilbao(np.linspace(.35, 1, 256)))
namesJetSystemPaper       = ("J153932+380949", "J085514+491136", "J153127+052137") # examples for the paper figure
labelMethodReferencePaper = "d"
widthFigurePaper          = 7.                 # in inch
plottingPerJetSystem      = False              # if True, also plot one figure per jet system and reference method
labelsMethodReference     = ("d", "a")         # reference methods for the per-jet-system figures
pathCatalogueMean         = DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_mean.xlsx"
pathsCatalogueRealization = [DIR_OUTPUT / "catalogues" / f"catalogue_filament_{labelSample}_{BORG_INDEX_REALIZATION_START + i * BORG_INDEX_REALIZATION_STEP}.xlsx"
                             for i in range(BORG_NUMBER_OF_REALIZATIONS)]
pathFigurePaper           = DIR_OUTPUT / f"plot_filament_orientations_sphere_{labelSample}_{labelMethodReferencePaper}.pdf"
directoryFigures          = DIR_OUTPUT / f"plots_filament_orientations_sphere_{labelSample}"

# Load the axes found in the posterior mean cube; shape (numberOfJetSystems, numberOfMethods, 3).
axesMean           = np.stack([loadFilamentAxes(pathCatalogueMean, labelMethod) for labelMethod in labelsMethod], axis = 1)
namesJetSystem     = loadJetSystemNames(pathCatalogueMean)
numberOfJetSystems = len(namesJetSystem) # in 1

# Load the axes found in the posterior realizations; shape (numberOfJetSystems, numberOfRealizations, numberOfMethods, 3).
axesRealizations = loadFilamentAxesRealizations(pathsCatalogueRealization, labelsMethod, namesJetSystem)

# Calculate the principal axis of the realization axes, per jet system and method; shape (numberOfJetSystems, numberOfMethods, 3).
axesPrincipal = np.full_like(axesMean, np.nan)
for indexJetSystem in range(numberOfJetSystems):
    for indexMethod in range(len(labelsMethod)):
        axesPrincipal[indexJetSystem, indexMethod], _ = axisPrincipal(axesRealizations[indexJetSystem, : , indexMethod])

# Report how far the principal axes lie from the mean cube axes.
print()
for indexMethod, labelMethod in enumerate(labelsMethod):
    anglesPrincipalMean = axialSeparationCartesian(axesPrincipal[ : , indexMethod], axesMean[ : , indexMethod]) # in deg
    print(f"{namesMethod[labelMethod].capitalize()} method: angle between principal axis of realization axes and mean cube axis: "
          f"{np.median(anglesPrincipalMean):.1f} deg (median), {np.percentile(anglesPrincipalMean, 90):.1f} deg (90th percentile), "
          f"{np.max(anglesPrincipalMean):.1f} deg (maximum).")
print()

handlesLegend = makeHandlesLegend(labelsMethod, namesMethod, coloursMethod)
coloursPanel  = [coloursMethod[labelMethod] for labelMethod in labelsMethod]

# Plot the paper figure: a row of example jet systems.
fig, axes = plt.subplots(1, len(namesJetSystemPaper), figsize = (widthFigurePaper, .42 * widthFigurePaper), sharey = True)
for ax, nameJetSystem in zip(axes, namesJetSystemPaper):
    indexJetSystem = list(namesJetSystem).index(nameJetSystem)
    image          = plotHemisphere(ax, axesRealizations[indexJetSystem], axesMean[indexJetSystem], axesPrincipal[indexJetSystem],
                                    axesMean[indexJetSystem, labelsMethod.index(labelMethodReferencePaper)], coloursPanel, colourMap)
    ax.set_title(nameJetSystem, fontsize = 9)
    ax.set_xlabel(r"$x\ (1)$")
axes[0].set_ylabel(r"$y\ (1)$")
fig.subplots_adjust(left = .08, right = .91, bottom = .27, top = .93, wspace = .06)
colourBar = fig.colorbar(image, cax = fig.add_axes((.925, .27, .012, .66)))
colourBar.set_label(r"$z\ (1)$")
fig.legend(handles = handlesLegend, loc = "lower center", ncol = 3, frameon = False,
           bbox_to_anchor = (.49, 0), handletextpad = .2, columnspacing = 1.2, labelspacing = .3)
print(f"Saving plot to '{pathFigurePaper}'...")
fig.savefig(pathFigurePaper)
plt.close(fig)

# Plot one figure per jet system and reference method, if requested.
if plottingPerJetSystem:
    directoryFigures.mkdir(parents = True, exist_ok = True)
    for indexJetSystem in range(numberOfJetSystems):
        for labelMethodReference in labelsMethodReference:
            fig, ax = plt.subplots(figsize = (6, 4))
            image   = plotHemisphere(ax, axesRealizations[indexJetSystem], axesMean[indexJetSystem], axesPrincipal[indexJetSystem],
                                     axesMean[indexJetSystem, labelsMethod.index(labelMethodReference)], coloursPanel, colourMap)
            ax.set_xlabel(r"$x\ (1)$")
            ax.set_ylabel(r"$y\ (1)$")
            ax.set_title(rf"{namesJetSystem[indexJetSystem]} $\vert$ reference: {namesMethod[labelMethodReference]} method", fontsize = 10)
            ax.legend(handles = handlesLegend, loc = "center left", bbox_to_anchor = (1.25, .5), frameon = False)
            colourBar = fig.colorbar(image, ax = ax, fraction = .046, pad = .03)
            colourBar.set_label(r"$z\ (1)$")

            pathFigure = directoryFigures / f"plot_filament_orientations_sphere_{indexJetSystem:03d}_{labelMethodReference}.pdf"
            print(f"Saving plot to '{pathFigure}'...")
            fig.savefig(pathFigure, bbox_inches = "tight")
            plt.close(fig)