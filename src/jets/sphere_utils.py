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
