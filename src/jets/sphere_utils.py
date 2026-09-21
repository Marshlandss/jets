# Imports: third-party
import numpy as np

def distanceOnSphere(longitudes1, latitudes1, longitudes2, latitudes2, unitsDegree = True):
    """
    Calculate the great-circle distance between every point of '(longitudes1, latitudes1)' and every point of '(longitudes2, latitudes2)'.
    The spherical law of cosines formula leads to numerical errors when points are very near (e.g. close to eachother).
    The haversine                formula leads to numerical errors when points are very far  (e.g. close to antipodal).
    The Vincenty formula is accurate in all cases, but is computationally more demanding.
    """

    # Convert to NumPy arrays.
    longitudes1        = np.atleast_1d(longitudes1) # in deg or rad
    latitudes1         = np.atleast_1d(latitudes1)  # in deg or rad
    longitudes2        = np.atleast_1d(longitudes2) # in deg or rad
    latitudes2         = np.atleast_1d(latitudes2)  # in deg or rad

    # Convert to radians.
    if unitsDegree:
        longitudes1 = np.radians(longitudes1) # in rad
        latitudes1  = np.radians(latitudes1)  # in rad
        longitudes2 = np.radians(longitudes2) # in rad
        latitudes2  = np.radians(latitudes2)  # in rad

    # To avoid duplicating calculations, we pre-calculate all factors of the Vincenty formula.
    deltaLongitudes    = longitudes1[ : , None] - longitudes2[None, : ] # in rad
    cosDeltaLongitudes = np.cos(deltaLongitudes)                     # in 1
    sinDeltaLongitudes = np.sin(deltaLongitudes)                     # in 1
    cosLatitudes1      = np.cos(latitudes1)                          # in 1
    cosLatitudes2      = np.cos(latitudes2)                          # in 1
    sinLatitudes1      = np.sin(latitudes1)                          # in 1
    sinLatitudes2      = np.sin(latitudes2)                          # in 1

    # Apply the Vincenty formula.
    # This is more optimal than using the Haversine formula (https://en.wikipedia.org/wiki/Haversine_formula),
    # which is ill-conditioned when solving for c when c is small.
    # The angular distances resulting from the Vincenty formula fall between 0 and pi rad.
    # This appears in tension with https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html, which claims a range of -pi to pi rad.
    distances          = np.arctan2(np.sqrt(np.square(cosLatitudes2[None, : ] * sinDeltaLongitudes) + np.square(cosLatitudes1[ : , None] * sinLatitudes2[None, : ] - sinLatitudes1[ : , None] * cosLatitudes2[None, : ] * cosDeltaLongitudes)), sinLatitudes1[ : , None] * sinLatitudes2[None, : ] + cosLatitudes1[ : , None] * cosLatitudes2[None, : ] * cosDeltaLongitudes) # in rad

    if unitsDegree:
        distances = np.degrees(distances) # in deg

    return distances


def convertSphericalToCartesian(azimuths, altitudes):
    """
    Convert 'azimuths' and 'altitudes' in degrees to Cartesian unit vectors, given by 'xs', 'ys', and 'zs'.
    """
    xs = np.cos(np.radians(altitudes)) * np.cos(np.radians(azimuths))
    ys = np.cos(np.radians(altitudes)) * np.sin(np.radians(azimuths))
    zs = np.sin(np.radians(altitudes))
    return xs, ys, zs


def convertCartesianToSpherical(xs, ys, zs):
    """
    Convert Cartesian vectors, given by 'xs', 'ys', and 'zs', to 'azimuths' and 'altitudes' in degrees.
    Works for both unit vectors and non-unit vectors.
    """
    azimuths  = np.degrees(np.arctan2(ys, xs)) % 360.
    altitudes = np.degrees(np.arctan2(zs, np.hypot(xs, ys)))
    return azimuths, altitudes


def axialSeparation(azimuths1, altitudes1, azimuths2, altitudes2):
    """
    Compute angle between two axes (undirected lines), in degrees:
    the smaller of the separations between direction 1 and direction 2, or direction 1 and direction 2's antipode.
    """
    distances1 = distanceOnSphere(azimuths1, altitudes1, azimuths2,         altitudes2)
    distances2 = distanceOnSphere(azimuths1, altitudes1, azimuths2 + 180., -altitudes2)
    return np.minimum(distances1, distances2)


def axisPrincipal(vectors, weights = None):
    """
    Calculate the principal axis of a set of undirected axes: the eigenvector of the (weighted) scatter matrix with the
    largest eigenvalue. Axes are undirected, so this is the right notion of a mean; the vector mean would cancel
    antipodal pairs. The returned axis points into the upper hemisphere, matching the (azimuth, altitude) convention.

    Parameters
    ----------
    vectors : array of shape (n, 3); Cartesian unit vectors. Lengths are not normalized here, so a vector that is not
              of unit length acts as if it carried an extra weight.
    weights : array of shape (n,) or None; non-negative weights, e.g. the solid angle each orientation represents.
              None gives every axis equal weight.

    Returns
    -------
    axis        : array of shape (3,); unit vector with a non-negative z-component
    eigenvalues : array of shape (3,); scatter matrix eigenvalues in ascending order. Their ratios measure how
                  elongated the set of axes is; the eigenvalues themselves scale with 'weights'.
    """
    if weights is None:
        scatter = vectors.T @ vectors
    else:
        scatter = (vectors * weights[ : , None]).T @ vectors
    eigenvalues, eigenvectors = np.linalg.eigh(scatter) # ascending; eigenvectors are normalized: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
    axis                      = eigenvectors[ : , -1]

    # Make sure 'axis' is a vector pointing in the upper hemisphere.
    if axis[2] < 0:
        axis = -axis

    return axis, eigenvalues


def composeAngularErrors(alpha, beta, RNG):
    """
    Compose two successive angular errors on the unit sphere via the spherical law of cosines, assuming the errors have an isotropic relative azimuth phi.
    We call the two successive displacements 'alpha' and 'beta', and the total angular displacement 'gamma'. All are in radians.

    Geometry (unit sphere S^2):
    I  = initial axis
    P1 = axis after the first error  (alpha = arc I-P1)
    P2 = axis after the second error (beta  = arc P1-P2)
    The total error gamma = arc I-P2. In the spherical triangle I-P1-P2, the angle at P1 between arcs I-P1 and P1-P2 is phi, so

    cos(gamma) = cos(alpha) cos(beta) + sin(alpha) sin(beta) cos(phi).
    """
    # Assuming the direction of the second error is isotropic, phi ~ Uniform(0, 2 pi).
    phi      = RNG.uniform(0, 2 * np.pi, size = len(alpha))
    # Apply spherical law of cosines.
    cosGamma = np.cos(alpha) * np.cos(beta) + np.sin(alpha) * np.sin(beta) * np.cos(phi)
    # The result is the angle between directed vectors, in [0, pi]; for axes, fold it with min(gamma, pi - gamma) after the last composition.
    return np.arccos(np.clip(cosGamma, -1, 1))
