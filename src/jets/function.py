import numpy

def transform_spherical_to_cartesian(az, alt, points_r = 1, relative_to_equator = True,
                                  units_degree = True):
    '''
    Returns the (x,y,z) coordinates of points in 3D, assuming as input
    spherical coordinates 'points_r', 'az' and 'alt'.
    If 'relative_to_equator' is 'True', theta is defined w.r.t. the xy-plane (equator), instead of the z-axis.
    '''
    if (units_degree):
        az   = numpy.radians(az)
        alt = numpy.radians(alt)

    if (relative_to_equator):
        alt = numpy.pi / 2 - alt

    # Calculate coordinates.
    points_x = points_r * numpy.cos(az) * numpy.sin(alt)
    points_y = points_r * numpy.sin(az) * numpy.sin(alt)
    points_z = points_r * numpy.cos(alt)

    return points_z, points_y, points_x