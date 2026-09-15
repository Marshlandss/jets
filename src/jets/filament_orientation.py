"""
Martijn Simon Soen Liong Oei, September 12026 H.E.
"""
# Imports: third-party
from astropy import units as u
import numpy as np
import pandas as pd
import time
# Imports: first-party
from jets.config import BORG_VOXEL_SIZE_MPC, DENSITY_MEAN_TODAY

class FilamentOrientationFinder:
    """
    """
    def __init__(self, lambdaMax = 2.5, # in 1
                       stepAngle = 10., # in deg
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
        """



    def findBest(self, voxelIndicesList, densities):
        """
        Return the (altitude, azimuth) index pair of the maximum of 'columnDensities'.
        On ties, the first maximum in row-major order is returned — the same rule as the original strict '<' comparison.
        """
        numberOfJetSystems  = len(voxelIndicesList)  # in 1
        print(numberOfJetSystems, type(voxelIndicesList))
        print(voxelIndicesList[0], type(voxelIndicesList[0]), voxelIndicesList[0][0], type(voxelIndicesList[0][0]))

        self.azimuthsBest        = np.full(numberOfJetSystems, np.nan)
        self.altitudesBest       = np.full(numberOfJetSystems, np.nan)
        self.columnDensitiesBest = np.full(numberOfJetSystems, np.nan)
        self.columnDensities     = np.full((numberOfJetSystems, self.numberOfAltitudes, self.numberOfAzimuths), np.nan)
        #print(self.columnDensities.shape)

        timeStart = time.time()
        for indexJetSystem in range(numberOfJetSystems):
            print(f"Finding filament orientation for jet system {indexJetSystem + 1}...")
            # Grab the voxel indices of the current jet system.
            voxelIndices       = voxelIndicesList[indexJetSystem]

            # Excise a smaller cube from the big cube.
            #densitiesSmall     = densities[voxelIndices[2] - self.voxelRadius : voxelIndices[2] + self.voxelRadius + 1, voxelIndices[1] - self.voxelRadius : voxelIndices[1] + self.voxelRadius + 1, voxelIndices[0] - self.voxelRadius : voxelIndices[0] + self.voxelRadius + 1]

            # Initialise the loop over filament orientations.
            indexAltitudeBest  = None
            indexAzimuthBest   = None
            columnDensityBest  = None

            for indexAltitude in range(self.numberOfAltitudes):
                for indexAzimuth in range(self.numberOfAzimuths):
                    lambdasCrossingXCurrent = list(self.lambdasCrossingX[indexAltitude, indexAzimuth])
                    lambdasCrossingYCurrent = list(self.lambdasCrossingY[indexAltitude, indexAzimuth])
                    lambdasCrossingZCurrent = list(self.lambdasCrossingZ[indexAltitude, indexAzimuth])
                    #print(lambdasCrossingXCurrent, lambdasCrossingYCurrent, lambdasCrossingZCurrent)
                    #print(type(lambdasCrossingXCurrent), type(lambdasCrossingYCurrent), type(lambdasCrossingZCurrent))
                    lambdaCrossingPrevious = 0.
                    voxelIndicesDeviation  = np.array([0, 0, 0])
                    columnDensity          = 0.
                    while lambdaCrossingPrevious < lambdaMax:
                        # Find the lambda value of the next border crossing.
                        lambdaCrossingNext = min(lambdasCrossingXCurrent[0], lambdasCrossingYCurrent[0], lambdasCrossingZCurrent[0])
                        # Calculate the lambda interval of the line segment in the current voxel(s).
                        lambdaInterval     = min(lambdaCrossingNext, lambdaMax) - lambdaCrossingPrevious
                        #print(lambdaCrossingPrevious, lambdaCrossingNext, lambdaInterval, voxelIndicesDeviation, densitiesMeanSmall[tuple(voxelIndicesCentre + voxelIndicesDeviation)])
                        # Add the column density contributions of the current voxels.
                        columnDensity     += lambdaInterval * (densitiesSmall[tuple(self.voxelIndicesCentre + voxelIndicesDeviation)] + densitiesSmall[tuple(self.voxelIndicesCentre - voxelIndicesDeviation)])

                        if   lambdaCrossingNext == lambdasCrossingXCurrent[0]:
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

                    columnDensity *= BORG_VOXEL_SIZE_MPC * self.metresPerMegaparsec * DENSITY_MEAN_TODAY # in g/m^2
                    if (columnDensityBest == None or columnDensityBest < columnDensity):
                        columnDensityBest = columnDensity
                        indexAltitudeBest = indexAltitude
                        indexAzimuthBest  = indexAzimuth

                    self.columnDensities[indexJetSystem, indexAltitude, indexAzimuth] = columnDensity
                    #print(columnDensityBest, columnDensity)

            self.azimuthsBest       [indexJetSystem] = self.azimuths [indexAzimuthBest]
            self.altitudesBest      [indexJetSystem] = self.altitudes[indexAltitudeBest]
            self.columnDensitiesBest[indexJetSystem] = columnDensityBest

            #print(azimuths[indexAzimuthBest], altitudes[indexAltitudeBest], columnDensityBest)
            #columnDensities = np.full((numberOfAltitudes, numberOfAzimuths, numberOfJetSystems))

        #self.columnDensitiesBest *= BORG_VOXEL_SIZE_MPC * self.metresPerMegaparsec * DENSITY_MEAN_TODAY # in g/m^2
        #self.columnDensities     *= BORG_VOXEL_SIZE_MPC * self.metresPerMegaparsec * DENSITY_MEAN_TODAY # in g/m^2
        timeEnd = time.time()
        print(timeEnd - timeStart)

        # Calculate Cartesian coordinates for the best unit vectors.
        self.xsBest = np.cos(np.radians(self.altitudesBest)) * np.cos(np.radians(self.azimuthsBest))
        self.ysBest = np.cos(np.radians(self.altitudesBest)) * np.sin(np.radians(self.azimuthsBest))
        self.zsBest = np.sin(np.radians(self.altitudesBest))


        #for az,alt,cd in zip(self.azimuthsBest, self.altitudesBest, self.columnDensitiesBest):
        #    print(az,alt,cd)
        '''
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8,7))
        plt.scatter(self.xsBest, self.ysBest, c = np.arange(numberOfJetSystems))
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.show()
        '''


    def write(self, pathExcel, methodDirect = True):
        """
        """
        if methodDirect:
            methodLetter = "r"
        else:
            methodLetter = "j"
        DataFrame = pd.read_excel(pathExcel)
        DataFrame["best_angle_"           + methodLetter + " (az,alt)(deg)"] = [f"[{az:.1f}, {alt:.1f}]" for az, alt in zip(self.azimuthsBest, self.altitudesBest)]
        DataFrame["cartesian_best_angle_" + methodLetter + " (x,y,z)"]       = [f"[{x:.5f}, {y:.5f}, {z:.5f}]" for x, y, z in zip(self.xsBest, self.ysBest, self.zsBest)]
        DataFrame["column_density_"       + methodLetter + " (g/m^2)"]       = np.round(self.columnDensitiesBest, 3)
        DataFrame.to_excel(pathExcel, index = False)


    def writeNumPy(self, pathNumPy):
        """
        """
        np.save(pathNumPy, self.columnDensities)