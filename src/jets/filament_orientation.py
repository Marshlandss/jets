"""
Martijn Simon Soen Liong Oei, September 12026 H.E.
"""
# Imports: standard library
import ast
# Imports: third-party
from astropy import units as u
import numpy as np
import pandas as pd
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC, DENSITY_MEAN_TODAY
from jets.sphere_utils import convertSphericalToCartesian


class FilamentOrientationFinder:
    """
    Find the 3D orientation of the Cosmic Web filament that runs through a given voxel of a density cube.
    Do so by maximising the column density along a line segment through that voxel over a grid of orientations.
    Because a filament axis is undirected, only the upper hemisphere is sampled: azimuth in [0, 360) deg, altitude in [0, 90] deg.
    """
    def __init__(self, lambdaMax, # in 1
                       stepAngle, # in deg
                ):
        self.lambdaMax           = lambdaMax
        self.stepAngle           = stepAngle
        self.voxelRadius         = int(np.ceil(lambdaMax - .5))
        self.numberOfAzimuths    = int(360. / stepAngle)     # in 1
        self.numberOfAltitudes   = int(90.  / stepAngle) + 1 # in 1
        self.azimuths            = np.linspace(0., 360., num = self.numberOfAzimuths,  endpoint = False)
        self.altitudes           = np.linspace(0., 90.,  num = self.numberOfAltitudes, endpoint = True)

        azimuthsRadians          = np.radians(self.azimuths)
        altitudesRadians         = np.radians(self.altitudes)
        azimuthsCos              = np.cos(azimuthsRadians)
        azimuthsSin              = np.sin(azimuthsRadians)
        altitudesCos             = np.cos(altitudesRadians)
        altitudesSin             = np.sin(altitudesRadians)

        # Calculate, for each (altitude, azimuth) combination, the x-component of the corresponding unit vector.
        xsFilament               = altitudesCos[:, None] * azimuthsCos[None, :]
        # Calculate, for each (altitude, azimuth) combination, the y-component of the corresponding unit vector.
        ysFilament               = altitudesCos[:, None] * azimuthsSin[None, :]
        # Calculate, for each (altitude, azimuth) combination, the z-component of the corresponding unit vector.
        zsFilament               = altitudesSin[:, None] * np.ones_like(self.azimuths)[None, :]

        self.xsFilamentSign      = np.sign(xsFilament)
        self.ysFilamentSign      = np.sign(ysFilament)
        self.zsFilamentSign      = np.sign(zsFilament)

        jMax                     = int(np.floor(lambdaMax + .5))
        js                       = np.arange(1, jMax + 1)
        self.lambdasCrossingX    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(xsFilament[ : , : , None]))).astype(np.float32)
        self.lambdasCrossingY    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(ysFilament[ : , : , None]))).astype(np.float32)
        self.lambdasCrossingZ    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(zsFilament[ : , : , None]))).astype(np.float32)

        # Initialise indices of the central voxel in a smaller 'cutout cube'.
        self.voxelIndicesCentre  = np.array([self.voxelRadius, self.voxelRadius, self.voxelRadius])

        # Initialise constants for column density calculation.
        self.metresPerMegaparsec = u.Mpc.to(u.m) # in 1


    def cutout(self, densities, voxelIndices):
        """
        Excise the (2 voxelRadius + 1)^3 cube centred on 'voxelIndices' from 'densities'.
        'densities' is indexed [iz, iy, ix] (BORG SDSS layout); 'voxelIndices' is (x, y, z), as in the Excel catalogue.
        """
        ix, iy, iz = voxelIndices
        r          = self.voxelRadius
        return densities[iz - r : iz + r + 1, iy - r : iy + r + 1, ix - r : ix + r + 1]


    def columnDensities(self, cutout):
        """
        Calculate the column density through the centre of 'cutout' for every orientation on the (altitude, azimuth) grid.

        For each orientation, a line segment passes through the centre of the central voxel and extends 'self.lambdaMax' voxel side
        lengths in both directions. The column density is the integral of the mass density along this segment, evaluated
        exactly for a piecewise-constant density field: each voxel the segment crosses contributes its density times the length
        of the segment inside it. The crossing points are precomputed in '__init__' and do not depend on the data.

        Parameters
        ----------
        cutout : array of shape (2 voxelRadius + 1,) * 3, indexed [iz, iy, ix]; mass density in units of the present-day
                 cosmic mean ('DENSITY_MEAN_TODAY'), as in the BORG SDSS cubes. Use 'cutout' to excise it from a full cube.

        Returns
        -------
        columnDensities : array of shape (numberOfAltitudes, numberOfAzimuths); column density (in g m^-2), with
                          columnDensities[i, j] corresponding to orientation ('altitudes[i]', 'azimuths[j]')
        """
        shapeExpected = (2 * self.voxelRadius + 1,) * 3
        if cutout.shape != shapeExpected:
            raise ValueError(f"The cutout's shape is {cutout.shape}, while {shapeExpected} was expected.")

        columnDensities = np.full((self.numberOfAltitudes, self.numberOfAzimuths), np.nan)

        for indexAltitude in range(self.numberOfAltitudes):
            for indexAzimuth in range(self.numberOfAzimuths):
                lambdasCrossingXCurrent = list(self.lambdasCrossingX[indexAltitude, indexAzimuth])
                lambdasCrossingYCurrent = list(self.lambdasCrossingY[indexAltitude, indexAzimuth])
                lambdasCrossingZCurrent = list(self.lambdasCrossingZ[indexAltitude, indexAzimuth])
                lambdaCrossingPrevious  = 0.
                voxelIndicesDeviation   = np.array([0, 0, 0])
                columnDensity           = 0.
                while lambdaCrossingPrevious < self.lambdaMax:
                    # Find the lambda value of the next border crossing.
                    lambdaCrossingNext = min(lambdasCrossingXCurrent[0], lambdasCrossingYCurrent[0], lambdasCrossingZCurrent[0])
                    # Calculate the lambda interval of the line segment in the current voxel(s).
                    lambdaInterval     = min(lambdaCrossingNext, self.lambdaMax) - lambdaCrossingPrevious
                    # Add the column density contributions of the current voxels.
                    columnDensity     += lambdaInterval * (cutout[tuple(self.voxelIndicesCentre + voxelIndicesDeviation)] + cutout[tuple(self.voxelIndicesCentre - voxelIndicesDeviation)])

                    if lambdaCrossingNext == lambdasCrossingXCurrent[0]:
                        lambdasCrossingXCurrent.pop(0)
                        indexChange = 2
                        signChange  = self.xsFilamentSign[indexAltitude, indexAzimuth]
                    elif lambdaCrossingNext == lambdasCrossingYCurrent[0]:
                        lambdasCrossingYCurrent.pop(0)
                        indexChange = 1
                        signChange  = self.ysFilamentSign[indexAltitude, indexAzimuth]
                    else:
                        lambdasCrossingZCurrent.pop(0)
                        indexChange = 0
                        signChange  = self.zsFilamentSign[indexAltitude, indexAzimuth]
                    voxelIndicesDeviation[indexChange] += 1 * signChange
                    lambdaCrossingPrevious = lambdaCrossingNext

                columnDensity *= BORG_VOXEL_SIZE_MPC * self.metresPerMegaparsec * DENSITY_MEAN_TODAY  # in g/m^2
                columnDensities[indexAltitude, indexAzimuth] = columnDensity
        return columnDensities


    def findBest(self, columnDensities):
        """
        Return the (altitude, azimuth) index pair of the maximum of 'columnDensities'.
        On ties, the first maximum in row-major order is returned (the same rule as the original strict '<' comparison).
        """
        return np.unravel_index(np.argmax(columnDensities), columnDensities.shape)



def findFilamentOrientations(FOF, densities, voxelIndicesList):
    """
    Find the filament orientation of every jet system whose voxel coordinates are listed in 'voxelIndicesList'.
    Do so by maximising the Cosmic Web column density through its host over all orientations on the (altitude, azimuth) grid of 'FOF'.

    Parameters
    ----------
    FOF              : FilamentOrientationFinder; determines the orientation grid and the integration length 'lambdaMax'
    densities        : array of shape (N, N, N), indexed [iz, iy, ix]; mass density in units of the present-day cosmic
                       mean, e.g. the BORG SDSS posterior mean cube or a single realisation
    voxelIndicesList : sequence of length 'numberOfJetSystems'; the (x, y, z) voxel indices of each jet system's host

    Returns
    -------
    columnDensitiesAll  : array of shape (numberOfJetSystems, numberOfAltitudes, numberOfAzimuths); column density (in g m^-2)
    azimuthsBest        : array of shape (numberOfJetSystems,); azimuth        (in deg) of the best-fitting orientation
    altitudesBest       : array of shape (numberOfJetSystems,); altitude       (in deg) of the best-fitting orientation
    columnDensitiesBest : array of shape (numberOfJetSystems,); column density (in g m^-2) at that orientation
    """
    numberOfJetSystems  = len(voxelIndicesList) # in 1

    columnDensitiesAll  = np.full((numberOfJetSystems, FOF.numberOfAltitudes, FOF.numberOfAzimuths), np.nan) # in g/m^2
    azimuthsBest        = np.full(numberOfJetSystems, np.nan) # in deg
    altitudesBest       = np.full(numberOfJetSystems, np.nan) # in deg
    columnDensitiesBest = np.full(numberOfJetSystems, np.nan) # in g/m^2
    for indexJetSystem, voxelIndices in enumerate(voxelIndicesList):
        columnDensitiesAll [indexJetSystem] = FOF.columnDensities(FOF.cutout(densities, voxelIndices))
        indexAltitudeBest, indexAzimuthBest = FOF.findBest(columnDensitiesAll[indexJetSystem])
        azimuthsBest       [indexJetSystem] = FOF.azimuths [indexAzimuthBest]
        altitudesBest      [indexJetSystem] = FOF.altitudes[indexAltitudeBest]
        columnDensitiesBest[indexJetSystem] = columnDensitiesAll[indexJetSystem, indexAltitudeBest, indexAzimuthBest]
    return columnDensitiesAll, azimuthsBest, altitudesBest, columnDensitiesBest


def writeFilamentOrientations(pathExcel, azimuthsBest, altitudesBest, columnDensitiesBest, method):
    """
    Add best filament orientations to the jet system catalogue at 'pathExcel'. 'method' is "d" (direct) or "a" (adjusted).
    """
    xs, ys, zs = convertSphericalToCartesian(azimuthsBest, altitudesBest)
    dataFrame  = pd.read_excel(pathExcel)
    dataFrame[f"best_angle_{method} (az,alt)(deg)"]     = [f"[{az:.1f}, {alt:.1f}]" for az, alt in zip(azimuthsBest, altitudesBest)]
    dataFrame[f"cartesian_best_angle_{method} (x,y,z)"] = [f"[{x:.5f}, {y:.5f}, {z:.5f}]" for x, y, z in zip(xs, ys, zs)]
    dataFrame[f"column_density_{method} (g/m^2)"]       = np.round(columnDensitiesBest, 3)
    dataFrame.to_excel(pathExcel, index = False)


def loadVoxelIndicesList(pathExcel, method):
    """
    Load the (x, y, z) voxel indices of the jet system hosts from the catalogue at 'pathExcel'.
    'method' is "d" (direct) or "a" (adjusted).

    Return a list of 1D NumPy arrays. Each array contains three indices.
    """
    columnName = {"d" : "voxel_index_r (x,y,z)", "a" : "voxel_index_j (x,y,z)"}[method]
    dataFrame  = pd.read_excel(pathExcel)
    return [np.array(ast.literal_eval(string)) for string in dataFrame[columnName]]