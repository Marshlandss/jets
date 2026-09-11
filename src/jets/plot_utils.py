"""
Plotting helpers shared by the figure scripts.
"""
# Imports: third-party
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import numpy as np


def spiralArmOutline(xs, ys, widths, numberOfPointsCap = 8):
    """
    Return the outline (as an array of shape (M, 2)) of a ribbon of locally varying width around a curve.

    The ribbon has full width 'widths[i]' at vertex '(xs[i], ys[i])', measured perpendicular to the curve. Its far end is closed
    with a semicircular cap; its near end is left flat (in 'plotGalaxySpiral' it is hidden under the bulge). All quantities are
    in data units, so the ribbon scales with the axes exactly as a patch does.
    """
    # Unit tangents and normals along the curve.
    tangentsX = np.gradient(xs)                                                   # in 1
    tangentsY = np.gradient(ys)                                                   # in 1
    norms     = np.hypot(tangentsX, tangentsY)                                    # in 1
    normalsX  = -tangentsY / norms                                                # in 1
    normalsY  =  tangentsX / norms                                                # in 1

    # Offset the curve to either side.
    halfWidths = .5 * widths                                                      # in 1
    leftX      = xs + halfWidths * normalsX                                       # in 1
    leftY      = ys + halfWidths * normalsY                                       # in 1
    rightX     = xs - halfWidths * normalsX                                       # in 1
    rightY     = ys - halfWidths * normalsY                                       # in 1

    # Semicircular cap at the far end, traversed from the left side to the right side.
    angleNormal = np.arctan2(normalsY[-1], normalsX[-1])                          # in rad
    anglesCap   = angleNormal - np.linspace(0, np.pi, num = numberOfPointsCap + 2)[1 : -1] # in rad
    capX        = xs[-1] + halfWidths[-1] * np.cos(anglesCap)                     # in 1
    capY        = ys[-1] + halfWidths[-1] * np.sin(anglesCap)                     # in 1

    outlineX = np.concatenate([leftX, capX, rightX[ : : -1]])                     # in 1
    outlineY = np.concatenate([leftY, capY, rightY[ : : -1]])                     # in 1
    return np.column_stack([outlineX, outlineY])


def plotGalaxySpiral(A,                                     # in 1; galactic radius (data units)
                     B                    = .25,            # in 1; Ringermacher and Mead's arm-winding parameter
                     N                    = 4.,             # in 1; Ringermacher and Mead's arm-tightness parameter
                     numberOfPointsSpiral = 200,            # in 1; vertices along each arm's centreline
                     radiusBulgeRelative  = .08,            # in 1; relative to galactic radius
                     centreX              = 0.,             # in 1
                     centreY              = 0.,             # in 1
                     colourSpiral         = "white",
                     colourBulge          = "white",
                     colourDisk           = "white",
                     alphaSpiral          = .4,
                     alphaBulge           = .8,
                     alphaDisk            = .2,
                     lineWidthMinRelative = .2,             # in 1; arm width at the tip relative to that at the centre
                     radiiDiskRelative    = (.4, .6, .8, 1.), # in 1; relative to galactic radius
                     zorder               = 3,
                     ax                   = None
                     ):
    """
    Draw a two-armed spiral galaxy (Ringermacher and Mead, 12009) with a bulge and concentric disk rings.

    Everything is drawn in data units: the arms are filled polygons whose width tapers from twice the bulge radius at the
    centre to 'lineWidthMinRelative' times that at the tip, so the drawing scales with the axes irrespective of figure size,
    layout engine, matplotlib version, or the order in which the figure is assembled.
    """
    if ax is None:
        ax = plt.gca()

    # Arm centreline (Ringermacher and Mead, 12009, eq. 1).
    phis     = np.linspace(0, 2 * N * np.arctan(1 / (np.e * B)), num = numberOfPointsSpiral, endpoint = True) # in rad
    rs       = np.zeros_like(phis)                                                                            # in 1
    rs[1 : ] = A / np.log(B * np.tan(phis[1 : ] / (2 * N)))                                                   # in 1
    xs       = rs * np.cos(phis)                                                                              # in 1
    ys       = rs * np.sin(phis)                                                                              # in 1

    # Bulge and disk rings.
    radiusBulge = radiusBulgeRelative * A                                                                     # in 1
    ax.add_patch(Circle((centreX, centreY), radiusBulge, color = colourBulge, alpha = alphaBulge, lw = 0, zorder = zorder))
    for radiusDiskRelative in radiiDiskRelative:
        ax.add_patch(Circle((centreX, centreY), radiusDiskRelative * A, color = colourDisk, alpha = alphaDisk, lw = 0, zorder = zorder))

    # Arms: one polygon each, so that the alpha is uniform (no overlapping segments).
    widthMax = 2 * radiusBulge                                                                                # in 1
    widths   = np.linspace(widthMax, lineWidthMinRelative * widthMax, num = numberOfPointsSpiral)             # in 1
    for sign in (1, -1):
        outline = spiralArmOutline(centreX + sign * xs, centreY + sign * ys, widths)
        ax.add_patch(Polygon(outline, closed = True, color = colourSpiral, alpha = alphaSpiral, lw = 0, zorder = zorder))