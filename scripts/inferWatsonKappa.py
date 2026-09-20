"""
Martijn Oei & Ruby Yao, 2026
"""
# Imports: standard library
import ast
# Imports: third-party
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
# Imports: first-party
from jets.simulatePADifferences import PADifferencesInferrer
from jets import watson

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{gensymb}"


# Initialize settings.
kappaJ                 = +2.       # in 1
kappaF                 = np.inf # in 1
numberOfSamples        = int(1e6)  # in 1
pathExcel              = "/Users/martijnoei/Downloads/Mpc_filament_pa.xlsx"
pathExcelPADifferences = "/Users/martijnoei/Downloads/Mpc_pa_diff-2.xlsx"
plotDirectory          = "/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/PhD/Ardor Telae/figures/GRGs/"
inferKappaF            = False
plotOrientationsSphere = False
plotPDsPolarAngleRel   = False

inferrer               = PADifferencesInferrer(numberOfSamples)


def loadFilamentVectors(pathExcel, method = "r", returnVoxelIndices = False):
    """
    """
    dataFrame          = pd.read_excel(pathExcel)
    angles             = dataFrame["best_angle_" + method + " (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
    if returnVoxelIndices:
        voxelIndices = dataFrame["voxel_index_" + method + " (x,y,z)"].apply(ast.literal_eval).to_numpy()

    numberOfJetSystems = angles.shape[0] # in 1
    anglesAzimuth      = np.full(numberOfJetSystems, np.nan) # in deg
    anglesAltitude     = np.full(numberOfJetSystems, np.nan) # in deg
    for i in range(numberOfJetSystems):
        anglesAzimuth[i]  = angles[i][0]
        anglesAltitude[i] = angles[i][1]

    # Calculate Cartesian unit vectors.
    Xs = np.cos(np.radians(anglesAltitude)) * np.cos(np.radians(anglesAzimuth)) # in 1
    Ys = np.cos(np.radians(anglesAltitude)) * np.sin(np.radians(anglesAzimuth)) # in 1
    Zs = np.sin(np.radians(anglesAltitude))                                           # in 1

    if returnVoxelIndices:
        return Xs, Ys, Zs, voxelIndices
    else:
        return Xs, Ys, Zs


def mean_axis_from_components(Xs, Ys, Zs, w=None, normalize=True):
    """
    Compute the mean axis (±m) from axis data on S^2 given as components.

    Parameters
    ----------
    Xs, Ys, Zs : array_like, shape (N,)
        Components of the vectors.
    w : array_like, shape (N,), optional
        Non-negative weights. If None, all weights are 1.
    normalize : bool, default True
        If True, normalize each (X,Y,Z) to unit length before processing.

    Returns
    -------
    m : ndarray, shape (3,)
        Unit vector representing the mean axis (sign arbitrary).
    evals : ndarray, shape (3,)
        Eigenvalues of the scatter matrix (ascending order).
    """

    Xs = np.asarray(Xs, float)
    Ys = np.asarray(Ys, float)
    Zs = np.asarray(Zs, float)

    # Stack into (N, 3)
    X = np.column_stack((Xs, Ys, Zs))
    print(X.shape)

    if normalize:
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-length vector encountered.")
        X = X / norms[:, None]

    N = X.shape[0]

    if w is None:
        w = np.ones(N)
    else:
        w = np.asarray(w, float)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")

    # Scatter / second-moment matrix
    M = (X.T * w) @ X / w.sum()

    # Eigen-decomposition (symmetric -> eigh)
    evals, evecs = np.linalg.eigh(M)

    # Top eigenvector = mean axis
    m = evecs[:, np.argmax(evals)]
    m /= np.linalg.norm(m)

    return m, evals


if (inferKappaF):
    # Load azimuths and altitudes for both the direct method (DM) and the adjusted method (AM).
    XsDM, YsDM, ZsDM, voxelIndicesDM = loadFilamentVectors(pathExcel, method = "r", returnVoxelIndices = True)
    XsAM, YsAM, ZsAM, voxelIndicesAM = loadFilamentVectors(pathExcel, method = "j", returnVoxelIndices = True)

    # df                 = pd.read_excel(pathExcel)
    # anglesDM           = df["best_angle_r (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
    # anglesAM           = df["best_angle_j (az,alt)(deg)"].apply(ast.literal_eval).to_numpy()
    # voxelIndicesDM     = df["voxel_index_r (x,y,z)"].apply(ast.literal_eval).to_numpy()
    # voxelIndicesAM     = df["voxel_index_j (x,y,z)"].apply(ast.literal_eval).to_numpy()
    # numberOfJetSystems = anglesDM.shape[0] # in 1
    # anglesDMAzimuth    = np.full(numberOfJetSystems, np.nan) # in deg
    # anglesDMAltitude   = np.full(numberOfJetSystems, np.nan) # in deg
    # anglesAMAzimuth    = np.full(numberOfJetSystems, np.nan) # in deg
    # anglesAMAltitude   = np.full(numberOfJetSystems, np.nan) # in deg
    # for i in range(numberOfJetSystems):
    #     anglesDMAzimuth[i]  = anglesDM[i][0]
    #     anglesDMAltitude[i] = anglesDM[i][1]
    #     anglesAMAzimuth[i]  = anglesAM[i][0]
    #     anglesAMAltitude[i] = anglesAM[i][1]
    #
    # # Calculate Cartesian unit vectors.
    # XsDM = np.cos(np.radians(anglesDMAltitude)) * np.cos(np.radians(anglesDMAzimuth)) # in 1
    # YsDM = np.cos(np.radians(anglesDMAltitude)) * np.sin(np.radians(anglesDMAzimuth)) # in 1
    # ZsDM = np.sin(np.radians(anglesDMAltitude))                                       # in 1
    # XsAM = np.cos(np.radians(anglesAMAltitude)) * np.cos(np.radians(anglesAMAzimuth)) # in 1
    # YsAM = np.cos(np.radians(anglesAMAltitude)) * np.sin(np.radians(anglesAMAzimuth)) # in 1
    # ZsAM = np.sin(np.radians(anglesAMAltitude))                                       # in 1

    # Calculate angle differences.
    dotProducts = np.clip(XsDM * XsAM + YsDM * YsAM + ZsDM * ZsAM, -1, 1) # Add 'np.clip' because of numerical error.
    anglesDelta = np.degrees(np.arccos(dotProducts)) # in deg

    # Determine for which jet systems the DM and AM identified the same host galaxy voxel.
    areSameVoxel = np.equal(voxelIndicesDM, voxelIndicesAM)
    print(np.sum(areSameVoxel))

    # For the moment, include all jet systems for which 'anglesDelta' exceeds 'angleDeltaThreshold'.
    angleDeltaThreshold = 1. # in deg
    for a in anglesDelta[anglesDelta > angleDeltaThreshold]:
        print(a)

    # Calculate MLE filament measurement error kappa.
    MLEExpressionData = np.mean(np.square(dotProducts[anglesDelta > angleDeltaThreshold]))
    MLEKappa          = watson.MLEKappa(MLEExpressionData)

    #plt.hist(anglesDelta[~areSameVoxel], bins = np.linspace(0, 90, num = 6 + 1, endpoint = True))
    pyplot.hist(anglesDelta[anglesDelta > angleDeltaThreshold], bins = np.linspace(0, 90, num = 18 + 1, endpoint = True))
    pyplot.xticks(np.linspace(0, 90, num = 6 + 1, endpoint = True))
    pyplot.title(r"$\kappa_\mathrm{f} = " + "{:.2f}".format(MLEKappa) + "$")
    pyplot.show()



directoryExcelSheets = "/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_kpc/"
numberOfJetSystems   = 777#242
numberOfExcelSheets  = 41
# These arrays have 3 dimensions; the shape could be (242, 10, 2). (For each jet system, we have a measurement for each of the 'numberOfExcelSheets' BORG SDSS realisations, and for 2 methods: direct and adjusted.)
XsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
YsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
ZsAll                = np.full((numberOfJetSystems, numberOfExcelSheets, 2), np.nan)
for i in range(0, numberOfExcelSheets):
    fileName = "fpa_" + str(2000 + i * 250) + "_exact_1.xlsx"
    print("Loading '" + fileName + "'...")
    XsDM, YsDM, ZsDM = loadFilamentVectors(directoryExcelSheets + fileName, method = "r")
    XsAM, YsAM, ZsAM = loadFilamentVectors(directoryExcelSheets + fileName, method = "j")
    XsAll[ : , i, 0] = XsDM
    YsAll[ : , i, 0] = YsDM
    ZsAll[ : , i, 0] = ZsDM
    XsAll[ : , i, 1] = XsAM
    YsAll[ : , i, 1] = YsAM
    ZsAll[ : , i, 1] = ZsAM

# Load Xs, Ys, and Zs of the filament orientations deduced from the BORG SDSS mean.
XsMean = np.full((numberOfJetSystems, 2), np.nan)
YsMean = np.full((numberOfJetSystems, 2), np.nan)
ZsMean = np.full((numberOfJetSystems, 2), np.nan)
fileName = "kpc_filament_pa_exact_1.xlsx"
print("Loading '" + fileName + "'...")
XsDM, YsDM, ZsDM = loadFilamentVectors(directoryExcelSheets + fileName, method = "r")
XsAM, YsAM, ZsAM = loadFilamentVectors(directoryExcelSheets + fileName, method = "j")
XsMean[ : , 0] = XsDM
YsMean[ : , 0] = YsDM
ZsMean[ : , 0] = ZsDM
XsMean[ : , 1] = XsAM
YsMean[ : , 1] = YsAM
ZsMean[ : , 1] = ZsAM

pyplot.scatter(XsDM, YsDM)
pyplot.show()
#sys.exit()


# Determine the principal axes (PA).
XsPA = np.full(numberOfJetSystems, np.nan)
YsPA = np.full(numberOfJetSystems, np.nan)
ZsPA = np.full(numberOfJetSystems, np.nan)

for indexJetSystem in range(numberOfJetSystems):
    m, evals = mean_axis_from_components(XsAll[indexJetSystem].flatten(), YsAll[indexJetSystem].flatten(), ZsAll[indexJetSystem].flatten())
    # For consistency, store the principal axis such that the z-component is positive. This will later be assumed in the plotting.
    if (m[2] < 0):
        m *= -1
    XsPA[indexJetSystem] = m[0]
    YsPA[indexJetSystem] = m[1]
    ZsPA[indexJetSystem] = m[2]

    print(indexJetSystem, "Principal axis (±):", m, "Eigenvalues:", evals, evals[2]/evals[1])

altitudesPA = np.degrees(np.arcsin(ZsPA))
azimuthsPA  = (np.degrees(np.atan2(YsPA, XsPA)) + 360) % 360
print(np.amin(altitudesPA), np.amax(altitudesPA))
print(np.amin(azimuthsPA), np.amax(azimuthsPA))
for alt, az in zip(altitudesPA, azimuthsPA):
    print(alt,az)


import pandas as pd

# --- Your arrays (must match number of Excel rows) ---
# altitudesPA = ...
# azimuthsPA  = ...

# Path to the Excel file
excel_path = "/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_kpc/kpc_filament_pa_principal_axis.xlsx"   # change if needed

# 1. Load Excel
df = pd.read_excel(excel_path)

# Safety check (optional but smart)
if len(df) != len(altitudesPA):
    raise ValueError("Array length does not match number of Excel rows.")

# 2. Create the (az, alt) string column
df["best_angle_r (az,alt)(deg)"] = [f"[{az:.1f}, {alt:.1f}]" for az, alt in zip(azimuthsPA, altitudesPA)]
df["best_angle_j (az,alt)(deg)"] = df["best_angle_r (az,alt)(deg)"]

# 3. Write back to Excel (overwrite file)
df.to_excel(excel_path, index = False)

print("Excel updated successfully.")



# Calculate, for each jet system, the dot products of its individual vectors with its principal axis vector.
dotProductsPA    = np.clip(XsAll * XsPA[ : , None, None] + YsAll * YsPA[ : , None, None] + ZsAll * ZsPA[ : , None, None], -1, +1)
anglesPA         = np.degrees(np.arccos(np.abs(dotProductsPA)))
print(np.amin(dotProductsPA), np.amax(dotProductsPA))
print(np.mean(anglesPA), np.median(anglesPA))
np.savetxt(directoryExcelSheets + "anglesPrincipalAxis.txt", anglesPA.flatten())

dotProductsMeanPA = np.clip(XsMean * XsPA[ : , None] + YsMean * YsPA[ : , None] + ZsMean * ZsPA[ : , None], -1, +1)
anglesMeanPA      = np.degrees(np.arccos(np.abs(dotProductsMeanPA)))
print(anglesMeanPA.shape)
pyplot.hist(anglesMeanPA[ : , 0], bins = np.linspace(0, 90, num = 30 + 1, endpoint = True))
pyplot.title("angular distances between fil. orientation in BORG SDSS mean (DM) and principal axis")
pyplot.show()
pyplot.hist(anglesMeanPA[ : , 1], bins = np.linspace(0, 90, num = 30 + 1, endpoint = True))
pyplot.title("angular distances between fil. orientation in BORG SDSS mean (AM) and principal axis")
pyplot.show()


print("komen ze per jet system:")
for i in range(numberOfJetSystems):
    print(i, "{:.2f}".format(np.mean(anglesPA[i, : , : ])), "deg")


if plotOrientationsSphere:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def toPix(coordinatePhysical):
        return .5 * (numberOfPixels - 1) * (coordinatePhysical + 1)

    with matplotlib.rc_context():
        import ultraplot
        colourMapObject = ultraplot.colormaps["browns6_r"]#"yellows3_r"]

    # Generate grid of positive-z coordinates.
    numberOfPixels = 1000 + 1 # number of pixels per side
    gridXs         = np.linspace(-1, +1, num = numberOfPixels, endpoint = True)
    gridYs         = np.linspace(-1, +1, num = numberOfPixels, endpoint = True)
    gridZs         = np.sqrt(1 - np.square(gridXs)[None, : ] - np.square(gridYs)[: , None])

    # Loop over jet systems, plot a sphere for each.
    for indexJetSystem in range(numberOfJetSystems):
        pyplot.figure(figsize = (5, 4))
        im = pyplot.imshow(gridZs, cmap = colourMapObject, origin = "lower")
        pyplot.plot(toPix(np.array([XsPA[indexJetSystem], -1 * XsPA[indexJetSystem]])), toPix(np.array([YsPA[indexJetSystem], -1 * YsPA[indexJetSystem]])), c = "lavender", ls = "--")
        for indexRealisation in range(numberOfExcelSheets):
            for indexMethod in range(2):
                XStart              = XsAll[indexJetSystem, indexRealisation, indexMethod]
                YStart              = YsAll[indexJetSystem, indexRealisation, indexMethod]
                ZStart              = np.sqrt(1 - (XStart ** 2 + YStart ** 2)) # Note that this implicitly relies on us only exploring filament orientations in the upper (z > 0) hemisphere. #ZsAll[indexJetSystem, indexRealisation, 0]
                XEnd                = XsPA[indexJetSystem]
                YEnd                = YsPA[indexJetSystem]
                ZEnd                = np.sqrt(1 - (XEnd ** 2 + YEnd ** 2)) # Note that we assume that the z-component of the mean axis is positive. #ZsPA[indexJetSystem]

                dotProduct          = XStart * XEnd + YStart * YEnd + ZStart * ZEnd
                greatCircleDistance = np.arccos(dotProduct) # in rad
                if (greatCircleDistance > np.pi / 2):
                    XEnd *= -1
                    YEnd *= -1
                    ZEnd *= -1
                    dotProduct *= -1
                    greatCircleDistance = np.pi - greatCircleDistance
                    #continue
                ts = np.linspace(0, 1, num = 100 + 1, endpoint=True)
                angles              = ts * greatCircleDistance
                XStartPerp          = XEnd - dotProduct * XStart
                YStartPerp          = YEnd - dotProduct * YStart
                ZStartPerp          = ZEnd - dotProduct * ZStart
                length              = np.sqrt(np.square(XStartPerp) + np.square(YStartPerp) + np.square(ZStartPerp))
                XStartPerp         /= length
                YStartPerp         /= length
                ZStartPerp         /= length
                XsGeodesic          = np.cos(angles) * XStart + np.sin(angles) * XStartPerp
                YsGeodesic          = np.cos(angles) * YStart + np.sin(angles) * YStartPerp
                ZsGeodesic          = np.cos(angles) * ZStart + np.sin(angles) * ZStartPerp

                areVisible = (ZsGeodesic > 0)
                pyplot.plot(toPix(np.array([0, XStart])), toPix(np.array([0, YStart])), color = "gray", ls = "-", alpha = .08)
                pyplot.plot(toPix(XsGeodesic[areVisible]), toPix(YsGeodesic[areVisible]), color = "gray", ls = "-")#, alpha = .3)
                pyplot.plot(toPix(XsGeodesic[~areVisible]), toPix(YsGeodesic[~areVisible]), color="gray", ls="--")#, alpha = .3)


        pyplot.scatter(toPix(XsAll[indexJetSystem, : , 0]), toPix(YsAll[indexJetSystem, : , 0]), color = "cornflowerblue", lw = 0, s = 40, zorder = 5) # alpha = .5,midnightblue
        pyplot.scatter(toPix(XsMean[indexJetSystem, 0]), toPix(YsMean[indexJetSystem, 0]), color = "cornflowerblue", marker = "*", s = 50, zorder = 6, lw = 1., edgecolor = ".3")
        pyplot.scatter(toPix(XsAll[indexJetSystem, : , 1]), toPix(YsAll[indexJetSystem, : , 1]), color = "tomato", lw = 0, s = 40, zorder = 5) # alpha = .5,crimson
        pyplot.scatter(toPix(XsMean[indexJetSystem, 1]), toPix(YsMean[indexJetSystem, 1]), color="tomato", marker="*", s = 50, zorder = 6, lw = 1., edgecolor = ".3")
        pyplot.scatter([toPix(XsPA[indexJetSystem])], [toPix(YsPA[indexJetSystem])], color = "lavender", marker = "*", zorder = 7, s = 50)
        pyplot.scatter([toPix(-XsPA[indexJetSystem])], [toPix(-YsPA[indexJetSystem])], color = "lavender", marker = "*", zorder = 7, s = 20)#,alpha = .5)#facecolors = "none", edgecolors = "lavender",ls = "--",

        pyplot.scatter([toPix(0)], [toPix(0)], c = "lavender", s = 10)
        #pyplot.scatter([XsPA[indexJetSystem]],[YsPA[indexJetSystem]],c = "green")
        #pyplot.scatter(XsDiff.flatten(), YsDiff.flatten(),c = "green", alpha = .2)
        #pyplot.scatter(XsAll.flatten(), YsAll.flatten(),c = "green", alpha = .2)
        pyplot.xticks(np.linspace(0, numberOfPixels - 1, num = 5, endpoint = True), ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"])
        pyplot.yticks(np.linspace(0, numberOfPixels - 1, num = 5, endpoint = True), ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"])
        pyplot.xlabel(r"$x\ (1)$")
        pyplot.ylabel(r"$y\ (1)$")
        #pyplot.scatter(XsAll - XsPA[ : , None, None], YsPA,c = "green")
        #for r in np.sin(np.radians(np.linspace(0, 90, num = 3 + 1, endpoint = True))):
        #    circle = pyplot.Circle((toPix(0), toPix(0)), toPix(r), color="gray", fill=False)

        #from matplotlib import cm
        #colourMapObject = cm.get_cmap(colourMap)
        circle = pyplot.Circle((toPix(0), toPix(0)), toPix(0), color=colourMapObject(0.0), fill=False, lw = 1)
        # Add the patch to the axes
        pyplot.gca().add_patch(circle)
        pyplot.gca().set_aspect("equal")

        divider = make_axes_locatable(pyplot.gca())
        cax     = divider.append_axes("right", size = "2%", pad = 0.05)
        cb      = pyplot.gcf().colorbar(im, cax = cax)
        cb.set_label(r"$z\ (1)$", fontsize = 12)
        #pyplot.title(r"direct method: blue $\vert$ adjusted method: red")
        pyplot.subplots_adjust(top = .98)
        pyplot.savefig(directoryExcelSheets + f"filamentOrientationsSphereExact_{indexJetSystem}.pdf")
        pyplot.close()



def PDsVMFModifiedPolarAngle(angles, kappa, assumeDegrees = True):
    """
    Calculate von Mises--Fisher distribution over the [0, 90 deg] interval (rather than the [0, 180 deg] interval).
    """
    if (assumeDegrees):
        angles = np.radians(angles)
    PDs = kappa / (np.exp(kappa) - 1) * np.exp(kappa * np.abs(np.cos(angles))) * np.sin(angles)
    if (assumeDegrees):
        PDs *= np.pi / 180
    return PDs


plotAngles = np.linspace(0, 90, num = 90 + 1, endpoint = True)


# Initialise bins.
# The number of bins matters! If we do many samples, we can permit ourselves smaller bins.
numberOfBins = 30 # in 1
binEdges     = np.linspace(0, 90, num = numberOfBins + 1, endpoint = True) # in deg;
binWidth     = binEdges[1] - binEdges[0] # in deg
binCentres   = (binEdges[ : -1] + binEdges[1 : ]) * .5


# Read LoTSS--BORG SDSS APADs from Excel.
df = pd.read_excel(pathExcelPADifferences)
PAsDeltaAbsData = df["angle_difference_r (deg)"].dropna()

numberOfData     = len(PAsDeltaAbsData)
counts, binEdges = np.histogram(PAsDeltaAbsData, bins = binEdges, density = False) # counts: in 1

pyplot.bar(binCentres, counts, width = binWidth)
pyplot.show()

kappaJs        = np.linspace(-8., 0., num = 40 + 1, endpoint = True) # in 1
logLikelihoods = np.full_like(kappaJs, np.nan)

xgrid = np.linspace(0, 90, 900 + 1, endpoint = True)
for i, kappaJ in enumerate(kappaJs):
    # Monte Carlo simulate absolute PA difference RVs.
    inferrer.samplePAsDeltaAbs(kappaJ = kappaJ, alpha = 1.7, beta = 5.7, weightUniform = 3.6e-1, distributionF = "beta rectangular")#kappaF = kappaF,

    # Calculate Bernstein-based log-likelihood.
    #w, pdf, cdf = fit_bernstein_density_samples(inferrer.PAsDeltaAbs, n=4)
    #PDs = pdf(xgrid)
    #pdf(PAsDeltaAbsData)
    #logLikelihoods[i] = np.sum(np.log(np.interp(PAsDeltaAbsData, xgrid, PDs)))

    # Calculate histogram-based log-likelihood.
    PDs, binEdges = np.histogram(inferrer.PAsDeltaAbs, bins = binEdges, density = True) # PDs: in 1 / deg
    logLikelihoods[i] = np.sum(counts * np.log(PDs * 90)) / numberOfData # in 1

    # if (i < 4):
    #     p = pdf(xgrid)  # always >= 0
    #     F = cdf(xgrid)  # monotone from 0 to 1
    #     pyplot.bar(binCentres, PDs, width=binWidth)
    #     pyplot.plot(xgrid, p, c = "red")
    #     pyplot.show()

    print(i, kappaJ, logLikelihoods[i])


#kappaJs        = np.linspace(-10., 0., num = 50 + 1, endpoint = True) # in 1
#kappaF         = 2.7 # in 1
#inferrer.samplePAsDeltaAbs(kappaJ = kappaJ, kappaF = kappaF)


if plotPDsPolarAngleRel:
    # Probability density relative to probability density of uniform distribution on the sphere.
    angles = np.linspace(0, 90, num = 900 + 1, endpoint = True)
    pyplot.figure(figsize = (6, 3))
    for kappaJ in [-4.5, -4.3, -4.1]:#[-4.8, -4.6, -4.4, -4.2, -4.0]:
        pyplot.plot(angles, np.sqrt(-1 * kappaJ / np.pi) * 2 / erf(np.sqrt(-1 * kappaJ)) * np.exp(kappaJ * np.square(np.cos(np.radians(angles)))), c = "mediumseagreen")
    pyplot.xlabel(r"polar angle $a\ (\degree)$")
    pyplot.gca().set_yscale("log")
    pyplot.tight_layout()
    pyplot.show()


x = kappaJs
y = logLikelihoods


# POLYNOMIAL
n = 5  # <-- set polynomial degree; it seems that 4 is too low (you still see 'ripples')
# Fit polynomial (highest power first)
p = np.poly1d(np.polyfit(x, y, deg=n))
# Max of polynomial on the x-range spanned by your data
dp = p.deriv()                 # derivative polynomial
crit = dp.r                    # stationary points (may be complex)
# keep real critical points within [min(x), max(x)]
xmin, xmax = np.min(x), np.max(x)
crit_real = crit[np.isreal(crit)].real
crit_in = crit_real[(crit_real >= xmin) & (crit_real <= xmax)]
# also consider endpoints (global max on interval can be at boundaries)
candidates = np.concatenate([crit_in, [xmin, xmax]])
vals = p(candidates)
x_max = candidates[np.argmax(vals)]
y_max = vals.max()
print(crit, crit_real)
print(x_max, y_max)
print("Best kappaJ (polynomial):", "{:.3f}".format(x_max))

from scipy.optimize import curve_fit

def logL_linear_exp(x, a, b, A, xt, w):
    return a + b*x - A*np.exp((x - xt)/w)
p0 = [
    0.0,#-1085.7,#np.mean(y),              # a
    0.07,  # b (estimate slope on left), (y[1]-y[0])/(x[1]-x[0])
    0.2,                        # A
    -3.3,#x[np.argmax(y)],         # xt near the knee / MLE
    1.2                         # w (controls sharpness)
]

popt, pcov = curve_fit(
    logL_linear_exp,
    x, y,
    p0=p0,
    maxfev=int(1e7)
)
print(popt)

from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

# choose smoothing: s=0 interpolates; increase s to smooth noise
spl = UnivariateSpline(x, y, k=4, s=10)  # tweak s (try 0, 0.1, 1, 5...)

res = minimize_scalar(lambda t: -spl(t), bounds=(x.min(), x.max()), method="bounded")
x_mle = res.x
logL_max = spl(x_mle)

# curvature -> approx sigma:  logL ~ logLmax - (x-x0)^2/(2 sigma^2)
#d2 = spl.derivative(2)(x_mle)
#sigma = np.sqrt(-1.0/d2) if d2 < 0 else np.nan

a,b,A,xt,w = popt
kappaMLELinExp = np.log(b * w / A) * w + xt
print("Best kappaJ (spline):", "{:.3f}".format(x_mle))
print("Best kappaJ (parametric)", "{:.3f}".format(kappaMLELinExp))
print("MLE x* =", x_mle)
print("logL(x*) =", logL_max)



# Evaluate polynomial on the grid
plotKappas = np.linspace(-10, 0, num = 1000 +1, endpoint = True)
plotLLs    = p(plotKappas)
plotLLsSpline = spl(plotKappas)
plotLLsLinExp = logL_linear_exp(plotKappas, popt[0], popt[1], popt[2], popt[3], popt[4])
#pyplot.plot(kappaJs, logLikelihoods)
#pyplot.plot(plotKappas, plotLLs)
#pyplot.plot(plotKappas, plotLLsSpline, c = "red")

pyplot.figure(figsize = (6, 3))
pyplot.scatter(kappaJs, logLikelihoods * 1e2, lw = 0, c = "mediumseagreen", alpha = .3)
pyplot.plot(plotKappas, plotLLsLinExp * 1e2, c = "mediumseagreen")
pyplot.plot([kappaMLELinExp, kappaMLELinExp], [0, (a + b * (kappaMLELinExp - w)) * 1e2], c = "black", alpha = .2, ls = "-.")
pyplot.scatter([kappaMLELinExp],[(a + b * (kappaMLELinExp - w)) * 1e2], c = "mediumseagreen", marker = "*", zorder = 5)
pyplot.text(-9.8, .05, r"$\mathcal{L}_\mathrm{fit}(\kappa_\mathrm{j}) = a_1 + a_2\kappa_\mathrm{j} -a_3\exp{\frac{\kappa_\mathrm{j}-a_4}{a_5}}$", ha = "left", c = "gray")
pyplot.xlim(-10, 0)
pyplot.ylim(0, 1.2)
pyplot.xlabel(r"Watson distribution concentration $\kappa_\mathrm{j}\ (1)$")
pyplot.ylabel(r"average log-likelihood $\tilde{\mathcal{L}}\ (10^{-2})$")
pyplot.subplots_adjust(left = .1, bottom = 0.15, right = .98, top = .98)
pyplot.savefig(plotDirectory + "logLikelihood.pdf")
pyplot.close()

pyplot.plot(kappaJs, np.exp(242 * logLikelihoods))
pyplot.show()



likelihoods = np.exp(logLikelihoods - np.amax(logLikelihoods))
pyplot.plot(kappaJs, likelihoods)
pyplot.scatter(kappaJs, likelihoods)
pyplot.show()

kappaJBest = kappaJs[np.argmax(logLikelihoods)]
print("Best kappaJ (highest LL of array):", "{:.3f}".format(kappaJBest))
#print("Best fit kappa_j:", kappaJBest)

inferrer.samplePAsDeltaAbs(kappaJ = x_max, kappaF = kappaF)
pyplot.figure(figsize=(6, 4))
pyplot.hist(inferrer.PAsDeltaAbs, bins=6)
pyplot.xlabel("simulated PA differences (deg)")
pyplot.ylabel("Count")
pyplot.tight_layout()
pyplot.show()

print(PAsDeltaAbsData.shape)
print("Integral over PDF:", np.sum(PDs) * binWidth)

# Plot histogram with 6 equal-width bins
pyplot.figure(figsize=(6, 4))
pyplot.hist(PAsDeltaAbsData, bins=6)
pyplot.xlabel("angle_difference_j (deg)")
pyplot.ylabel("Count")
pyplot.title("Histogram of angle_difference_j")
pyplot.tight_layout()
pyplot.show()

angles       = np.linspace(0, 90, num = 900 + 1, endpoint = True)
PDsJ         = inferrer.PDsWatsonPolarAngle(angles, x_max)
PDsReference = inferrer.PDsWatsonPolarAngle(angles, 0)
pyplot.plot(angles, PDsJ)
pyplot.plot(angles, PDsReference)
pyplot.xticks(np.linspace(0, 90, num = 9 + 1, endpoint = True))
pyplot.xlabel("true angle between jet axis and filament axis (deg)")
pyplot.ylabel("probability density")
pyplot.title(r"$\kappa_\mathrm{j} = " + str(x_max) + "$")
pyplot.show()
#np.sqrt(-1 * kappaJ / np.pi) * 2 / erf(np.sqrt(-1 * kappaJ)) * np.exp(kappaJ * np.square(np.cos(np.radians(angles)))) * np.sin(np.radians(angles))
#np.sin(np.radians(angles))



#print(np.amin(dotProducts), np.amax(dotProducts))
#print(anglesDirect)
# print(type(anglesDirect))
# print(anglesDirect.shape)
#areSameVoxel = np.full(numberOfJetSystems, False)
#areSameVoxel2 = np.equal(voxelIndicesDM, voxelIndicesAM)
# for i in range(numberOfJetSystems):
#     if (voxelIndicesDM[i] == voxelIndicesAM[i]):
#         areSameVoxel[i] = True
#     print(voxelIndicesDM[i], voxelIndicesAM[i], areSameVoxel[i])
# print(np.array_equal(areSameVoxel, areSameVoxel2), "?")
# print(anglesDM[i], anglesDMAzimuth[i], anglesDMAltitude[i])
# print(anglesAM[i], anglesAMAzimuth[i], anglesAMAltitude[i])
# print("")

#if (np.round(angle, 3) == 0.)
# inBin = (anglesDelta > 5.) * (anglesDelta <= 10.)
# for a,b in zip(voxelIndicesDM[inBin], anglesDM[inBin]):
#     print(a,b)
# print("This one's weird:", anglesDelta[127])
# print(np.mean(anglesDelta), np.std(anglesDelta), np.median(anglesDelta))
# print(len(anglesDelta[~areSameVoxel]))

RHS             = np.mean(np.square(inferrer.ZsJ))
print(inferrer.MLEKappa(RHS))
kappaz = np.linspace(-20, 20, num = 4000 + 1, endpoint = True)
pyplot.plot(kappaz, watson.MLEExpressionKappa(kappaz))
pyplot.axhline(RHS)
pyplot.show()

# pyplot.plot(kappas, LHSs)
# pyplot.axhline(RHS)
# pyplot.axhline(0.6577638454896202, c = "red")
# pyplot.axhline(0.6342351098670317, c = "green")
# pyplot.grid(True)
# pyplot.show()

# kappas = np.linspace(10, 0, num = 1000, endpoint = False)[::-1]
# print(kappas)
# LHSs = 1 / np.sqrt(kappas * np.pi) * np.exp(kappas) / erfi(np.sqrt(kappas)) - 1 / (2 * kappas)
# kappasNeg = np.linspace(-9, 0, num = 9000, endpoint = False)
# LHSsNeg   = -1 * np.exp(kappasNeg) / (np.sqrt(-1 * kappasNeg * np.pi) * erf(np.sqrt(-1 * kappasNeg))) - 1 / (2 * kappasNeg)
# pyplot.plot(kappas, LHSs, c = "mediumseagreen")
# pyplot.plot(kappasNeg, LHSsNeg, c = "mediumseagreen")
# pyplot.plot(kappasNeg, 1 / 3. + 4 / 45. * kappasNeg, c = "mediumseagreen", alpha = .3)

#print(np.amin(PAs), np.amax(PAs))
# pyplot.hist(ZsM, bins = np.linspace(-1, 1, num = 40 + 1, endpoint = True))
# pyplot.show()

# pyplot.hist(np.degrees(np.arccos(np.abs(ZsM))), bins = np.linspace(0, 90, num = 18 + 1, endpoint = True))
# pyplot.xticks(np.linspace(0, 90, num = 9 + 1, endpoint = True))
# #pyplot.hist(np.degrees(np.arccos(np.abs(ZsM))), bins = np.linspace(0, 90, num = 6 + 1, endpoint = True))
# #pyplot.xticks(np.linspace(0, 90, num = 6 + 1, endpoint = True))
# pyplot.xlabel("filament angle error (deg)")
# pyplot.title(r"$\kappa_\mathrm{m} = " + str(kappaM) + "$")
# pyplot.show()

# pyplot.hist(PAsDeltaAbs, bins = np.linspace(0, 90, num = 18 + 1, endpoint = True))
# pyplot.xticks(np.linspace(0, 90, num = 9 + 1, endpoint = True))
# pyplot.xlabel(r"jet–filament PA difference $|\Delta\phi_\mathrm{jf}|$")
# pyplot.title(r"$\kappa_\mathrm{j} = " + str(kappaJ) + r", \kappa_\mathrm{m} = " + str(kappaM) + "$")
# pyplot.show()


# pyplot.figure(figsize = (6, 3))
# from matplotlib import cm
# kappas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]#[-6, -4, -2, 0, 2, 4, 6]
# cmap   = cm.get_cmap("managua_r", len(kappas))  # discrete, maximally spaced
# pyplot.figure(figsize=(6, 3))
# for i, kappa in enumerate(kappas):
#     pyplot.plot(
#         angles,
#         PDsWatsonPolarAngle(angles, kappa, assumeDegrees=True),
#         color=cmap(i),
#         label=rf"$\kappa={kappa}$")
#     if (kappa > 0):
#         pyplot.scatter([np.degrees(np.arcsin(1 / np.sqrt(2 * kappa)))],[np.sqrt(2 / np.pi) / erfi(np.sqrt(kappa)) * np.exp(kappa - .5)])
# #for kappa in [-6, -4, -2, 0, 2, 4, 6]:
# #    pyplot.plot(angles, PDsWatsonPolarAngle(angles, kappa, assumeDegrees = True))
# pyplot.show()

print("MLE kappa:", MLEKappa(MLEExpressionData))

print("getalleke:", MLEExpressionData)
print("getalleke:", np.mean(np.square(dotProducts[anglesDelta > 0])))  # 0.6577638454896202
print("getalleke:", np.mean(np.square(dotProducts[~areSameVoxel])))  # 0.6342351098670317

# print("quantity die wordt gemaximaliseerd:")
    # print("met de mean:")
    # print(np.sum(np.square(XsAll[indexJetSystem].flatten() * XsMean[indexJetSystem] + YsAll[indexJetSystem].flatten() * YsMean[indexJetSystem] + ZsAll[indexJetSystem].flatten() * ZsMean[indexJetSystem])))
    # print("met z-as:")
    # print(np.sum(np.square(XsAll[indexJetSystem].flatten() * 0 + YsAll[indexJetSystem].flatten() * 0 + ZsAll[indexJetSystem].flatten() * 1)))
    # print("afstandjes")
    # print("met de mean:")
    # distancesM = np.degrees(np.arccos(np.abs(XsAll[indexJetSystem].flatten() * XsMean[indexJetSystem] + YsAll[indexJetSystem].flatten() * YsMean[indexJetSystem] + ZsAll[indexJetSystem].flatten() * ZsMean[indexJetSystem])))
    # print(distancesM)
    # print(np.mean(distancesM))
    # print("met z-as:")
    # distancesZ = np.degrees(np.arccos(np.abs(XsAll[indexJetSystem].flatten() * 0 + YsAll[indexJetSystem].flatten() * 0 + ZsAll[indexJetSystem].flatten() * 1)))
    # print(distancesZ)
    # print(np.mean(distancesZ))
    #XsMean[indexJetSystem] = 0
    #YsMean[indexJetSystem] = 0
    #ZsMean[indexJetSystem] = 1


#import sys
#sys.exit()
'''
# Calculate, for each jet system, the mean vector.
XsMean             = np.mean(XsAll, axis = (1, 2))
YsMean             = np.mean(YsAll, axis = (1, 2))
ZsMean             = np.mean(ZsAll, axis = (1, 2))
lengthsMean        = np.sqrt(np.square(XsMean) + np.square(YsMean) + np.square(ZsMean))
XsMean            /= lengthsMean
YsMean            /= lengthsMean
ZsMean            /= lengthsMean
'''
pyplot.hist(anglesMean.flatten(), bins = np.linspace(0, 90, num = 30 + 1, endpoint = True), density = True)
pyplot.title("HAAR PLOT")
# pyplot.plot(plotAngles, inferrer.PDsWatsonPolarAngle(plotAngles, kappaMLE, assumeDegrees = True), c = "red")
# for k in np.linspace(1, 10, num = 10, endpoint = True):
#     plotPDsVMFM = PDsVMFModifiedPolarAngle(plotAngles, k)
#     pyplot.plot(plotAngles, plotPDsVMFM)
#     print(np.trapezoid(plotPDsVMFM, plotAngles), "WOEFPOEP")
pyplot.show()
'''
XsDiff  = XsAll - XsMean[ : , None, None]
YsDiff  = YsAll - YsMean[ : , None, None]
ZsDiff  = ZsAll - ZsMean[ : , None, None]
lengths = np.sqrt(np.square(XsDiff) + np.square(YsDiff) + np.square(ZsDiff))
XsDiff /= lengths
YsDiff /= lengths
ZsDiff /= lengths

angless = np.degrees(np.arccos(np.abs(ZsDiff)))
pyplot.hist(angless.flatten(), bins = np.linspace(0, 90, num = 9 + 1, endpoint = True), density = True)
pyplot.title("whazzup")
pyplot.show()
'''
# pyplot.imshow(anglesMean[:,:,0], aspect = "auto")
# pyplot.show()
#lengthsDMMean        = np.sqrt(np.square(XsDMMean) + np.square(YsDMMean) + np.square(ZsDMMean))
#print(lengthsDMMean)
#print(np.sum(np.isnan(XsDMAll)))

print(np.mean(anglesMean, axis = (1,2)))

#kappaMLE = inferrer.MLEKappa(np.mean(np.square(np.cos(np.radians(anglesMean.flatten()))))) # in 1
#print("MLE", kappaMLE)
# from scipy.special import erfi, erfinv
# def erfiinv(x):
#     # Coerce to a NumPy array with a numeric dtype SciPy supports
#     x = np.asarray(x)
#
#     # If it's object (common with lists containing None/Decimal/SymPy), force complex128
#     if x.dtype == object:
#         x = x.astype(np.complex128)
#
#     # If it's real, keep it real; if it's complex, ensure complex128
#     if np.iscomplexobj(x):
#         x = x.astype(np.complex128)
#     else:
#         x = x.astype(np.float64)
#     return -1j * erfinv(1j * x)
#
# kappas = np.linspace(0.1, 5, num = 50, endpoint = True)
# medians = np.arccos(1 / np.sqrt(kappas) * erfiinv(.5 * erfi(np.sqrt(kappas))))
# pyplot.plot(kappas, medians)
# pyplot.show()


# pyplot.hist(np.mean(anglesMeanDM, axis = 1), bins = np.linspace(0, 90, num = 9 + 1, endpoint = True))
# pyplot.xticks(np.linspace(0, 90, num = 9 + 1, endpoint = True))
# pyplot.xlabel("mean angular distance from indiv. filament vectors to mean filament vectors (deg)")
# pyplot.ylabel("number of Mpc-scale jet systems")
# pyplot.show()

# from scipy.optimize import minimize
# from scipy.special import betaln, logsumexp
#
# def fit_bernstein_density_samples(samples_deg, n=10):
#     """
#     Fit a Bernstein polynomial density on [0,90] directly to samples via MLE.
#
#     Model on u in [0,1]:
#         p(u) = sum_{k=0}^n w_k * Beta(u; k+1, n-k+1)
#     with w_k >= 0 and sum w_k = 1.
#
#     Returns:
#         w (n+1,)
#         pdf_deg(xdeg): callable on degrees
#         cdf_deg(xdeg): callable on degrees (exact mixture CDF)
#     """
#     x = np.asarray(samples_deg, dtype=float)
#     u = np.clip(x / 90.0, 1e-12, 1 - 1e-12)
#
#     k = np.arange(n + 1)
#     a = k + 1.0
#     b = (n - k) + 1.0
#
#     # log Beta pdf for each sample i and component k:
#     # log f_{k}(u_i) = (a-1)log u + (b-1)log(1-u) - log B(a,b)
#     log_pdf = ((a - 1)[None, :] * np.log(u)[:, None] +
#                (b - 1)[None, :] * np.log1p(-u)[:, None] -
#                betaln(a, b)[None, :])
#
#     def nll(z):
#         # softmax to enforce weights on simplex
#         z = z - np.max(z)
#         w = np.exp(z)
#         w = w / w.sum()
#
#         # log p(u_i) = logsumexp_k( log w_k + log_pdf[i,k] )
#         return -np.sum(logsumexp(np.log(w)[None, :] + log_pdf, axis=1))
#
#     z0 = np.zeros(n + 1)  # uniform weights initialisation
#     res = minimize(nll, z0, method="L-BFGS-B")
#     z = res.x - np.max(res.x)
#     w = np.exp(z)
#     w = w / w.sum()
#
#     # Precompute for evaluation on grids
#     from scipy.stats import beta as beta_dist
#
#     def pdf_deg(xdeg):
#         xdeg = np.asarray(xdeg, dtype=float)
#         uu = np.clip(xdeg / 90.0, 0.0, 1.0)
#         # mixture pdf on [0,1]
#         pdf_u = np.zeros_like(uu)
#         for kk in range(n + 1):
#             pdf_u += w[kk] * beta_dist.pdf(uu, a[kk], b[kk])
#         return pdf_u / 90.0  # scale back to degrees
#
#     def cdf_deg(xdeg):
#         xdeg = np.asarray(xdeg, dtype=float)
#         uu = np.clip(xdeg / 90.0, 0.0, 1.0)
#         cdf_u = np.zeros_like(uu)
#         for kk in range(n + 1):
#             cdf_u += w[kk] * beta_dist.cdf(uu, a[kk], b[kk])
#         return cdf_u
#
#     return w, pdf_deg, cdf_deg
