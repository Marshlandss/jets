# Imports: third-party
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure(figsize = (5, 4))
    im = plt.imshow(gridZs, cmap = colourMapObject, origin = "lower")
    plt.plot(toPix(np.array([XsPA[indexJetSystem], -1 * XsPA[indexJetSystem]])), toPix(np.array([YsPA[indexJetSystem], -1 * YsPA[indexJetSystem]])), c = "lavender", ls = "--")
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
            plt.plot(toPix(np.array([0, XStart])), toPix(np.array([0, YStart])), color = "gray", ls = "-", alpha = .08)
            plt.plot(toPix(XsGeodesic[areVisible]), toPix(YsGeodesic[areVisible]), color = "gray", ls = "-")#, alpha = .3)
            plt.plot(toPix(XsGeodesic[~areVisible]), toPix(YsGeodesic[~areVisible]), color="gray", ls="--")#, alpha = .3)


    plt.scatter(toPix(XsAll[indexJetSystem, : , 0]), toPix(YsAll[indexJetSystem, : , 0]), color = "cornflowerblue", lw = 0, s = 40, zorder = 5) # alpha = .5,midnightblue
    plt.scatter(toPix(XsMean[indexJetSystem, 0]), toPix(YsMean[indexJetSystem, 0]), color = "cornflowerblue", marker = "*", s = 50, zorder = 6, lw = 1., edgecolor = ".3")
    plt.scatter(toPix(XsAll[indexJetSystem, : , 1]), toPix(YsAll[indexJetSystem, : , 1]), color = "tomato", lw = 0, s = 40, zorder = 5) # alpha = .5,crimson
    plt.scatter(toPix(XsMean[indexJetSystem, 1]), toPix(YsMean[indexJetSystem, 1]), color="tomato", marker="*", s = 50, zorder = 6, lw = 1., edgecolor = ".3")
    plt.scatter([toPix(XsPA[indexJetSystem])], [toPix(YsPA[indexJetSystem])], color = "lavender", marker = "*", zorder = 7, s = 50)
    plt.scatter([toPix(-XsPA[indexJetSystem])], [toPix(-YsPA[indexJetSystem])], color = "lavender", marker = "*", zorder = 7, s = 20)#,alpha = .5)#facecolors = "none", edgecolors = "lavender",ls = "--",

    plt.scatter([toPix(0)], [toPix(0)], c = "lavender", s = 10)
    #plt.scatter([XsPA[indexJetSystem]],[YsPA[indexJetSystem]],c = "green")
    #plt.scatter(XsDiff.flatten(), YsDiff.flatten(),c = "green", alpha = .2)
    #plt.scatter(XsAll.flatten(), YsAll.flatten(),c = "green", alpha = .2)
    plt.xticks(np.linspace(0, numberOfPixels - 1, num = 5, endpoint = True), ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"])
    plt.yticks(np.linspace(0, numberOfPixels - 1, num = 5, endpoint = True), ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"])
    plt.xlabel(r"$x\ (1)$")
    plt.ylabel(r"$y\ (1)$")
    #plt.scatter(XsAll - XsPA[ : , None, None], YsPA,c = "green")
    #for r in np.sin(np.radians(np.linspace(0, 90, num = 3 + 1, endpoint = True))):
    #    circle = plt.Circle((toPix(0), toPix(0)), toPix(r), color="gray", fill=False)

    #from matplotlib import cm
    #colourMapObject = cm.get_cmap(colourMap)
    circle = plt.Circle((toPix(0), toPix(0)), toPix(0), color=colourMapObject(0.0), fill=False, lw = 1)
    # Add the patch to the axes
    plt.gca().add_patch(circle)
    plt.gca().set_aspect("equal")

    divider = make_axes_locatable(plt.gca())
    cax     = divider.append_axes("right", size = "2%", pad = 0.05)
    cb      = plt.gcf().colorbar(im, cax = cax)
    cb.set_label(r"$z\ (1)$", fontsize = 12)
    #plt.title(r"direct method: blue $\vert$ adjusted method: red")
    plt.subplots_adjust(top = .98)
    plt.savefig(directoryExcelSheets + f"filamentOrientationsSphereExact_{indexJetSystem}.pdf")
    plt.close()
