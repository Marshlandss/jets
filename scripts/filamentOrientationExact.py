"""
Martijn Simon Soen Liong Oei, February 12026 H.E.
"""
import ast, copy, os
import numpy as np
import pandas as pd
import h5py, sys, time

def make_beta_cylinder_density_cube(
    N,
    numberOfVoxels=5,
    radius_mpc=1.2,
    beta=2.0,
    rho0=1.6e-23,   # central density in g/m^3
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng

    # Low-res voxel size (BORG / SDSS scale)
    h = 0.702
    voxel_lowres_mpc = 750 / (256 * h)      # ≈ 4.17 Mpc

    # Total cube size = 5 voxels → ≈ 20.87 Mpc
    cube_size_mpc = voxel_lowres_mpc * numberOfVoxels

    # Fine grid coordinates
    dx_mpc = cube_size_mpc / N
    coords = (np.arange(N) - (N - 1) / 2) * dx_mpc
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Random cylinder axis
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)

    # Offset within half a low-res voxel
    half_voxel = 0.5 * voxel_lowres_mpc
    offset = rng.uniform(-half_voxel, half_voxel, size=3)

    # Shifted coordinates
    xs = x - offset[0]
    ys = y - offset[1]
    zs = z - offset[2]

    # Perpendicular distance
    r_dot_a = axis[2]*xs + axis[1]*ys + axis[0]*zs
    r2 = xs**2 + ys**2 + zs**2
    d_perp2 = np.maximum(r2 - r_dot_a**2, 0.0)
    d_perp = np.sqrt(d_perp2)

    # Beta-profile density
    cube = rho0 * (1. + (d_perp / radius_mpc)**2)**(-1.5 * beta)

    return cube, axis, offset


def average_down_to_5(cube):
    N = cube.shape[0]
    if N % 5 != 0:
        raise ValueError("N must be divisible by 5.")
    m = N // 5
    return cube.reshape(5, m, 5, m, 5, m).mean(axis=(1, 3, 5))


class FilamentOrientationFinder:
    """
    """
    def __init__(self, lambdaMax = 2.5, # in 1
                       stepAngle = 10., # in deg
                       littleH   = .7   # in 1
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
        print(xsFilament.shape, ysFilament.shape, zsFilament.shape)

        self.xsFilamentSign      = np.sign(xsFilament)
        self.ysFilamentSign      = np.sign(ysFilament)
        self.zsFilamentSign      = np.sign(zsFilament)

        jMax                     = int(np.floor(lambdaMax + .5))
        js                       = np.arange(1, jMax + 1)
        self.lambdasCrossingX    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(xsFilament[ : , : , None]))).astype(np.float32)
        self.lambdasCrossingY    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(ysFilament[ : , : , None]))).astype(np.float32)
        self.lambdasCrossingZ    = ((2 * js[None, None, : ] - 1) / (2 * np.abs(zsFilament[ : , : , None]))).astype(np.float32)
        print(self.lambdasCrossingX.shape, self.lambdasCrossingY.shape, self.lambdasCrossingZ.shape)

        # Initialise indices of the central voxel in a smaller 'cutout cube'.
        self.voxelIndicesCentre  = np.array([self.voxelRadius, self.voxelRadius, self.voxelRadius])

        # Initialise constants for column density calculation.
        self.metresPerMegaparsec = 3.0857e22              # in 1
        self.densityMeanToday    = 2.6e-24                # in g/m^3
        self.lengthVoxel         = (750. / littleH) / 256 # in Mpc


    def findBest(self, voxelIndicesList, densities):
        """
        """
        numberOfJetSystems  = len(voxelIndicesList)  # in 1
        print(numberOfJetSystems, type(voxelIndicesList))
        print(voxelIndicesList[0], type(voxelIndicesList[0]), voxelIndicesList[0][0], type(voxelIndicesList[0][0]))

        self.azimuthsBest        = np.full(numberOfJetSystems, np.nan)
        self.altitudesBest       = np.full(numberOfJetSystems, np.nan)
        self.columnDensitiesBest = np.full(numberOfJetSystems, np.nan)
        self.columnDensities     = np.full((numberOfJetSystems, self.numberOfAltitudes, self.numberOfAzimuths), np.nan)
        #print(self.columnDensities.shape)
        #import sys
        #sys.exit()

        timeStart = time.time()
        for indexJetSystem in range(numberOfJetSystems):
            print(f"Finding filament orientation for jet system {indexJetSystem + 1}...")
            # Grab the voxel indices of the current jet system.
            voxelIndices       = voxelIndicesList[indexJetSystem]

            # Excise a smaller cube from the big cube.
            #densitiesSmall     = densities[voxelIndices[2] - self.voxelRadius : voxelIndices[2] + self.voxelRadius + 1, voxelIndices[1] - self.voxelRadius : voxelIndices[1] + self.voxelRadius + 1, voxelIndices[0] - self.voxelRadius : voxelIndices[0] + self.voxelRadius + 1]
            cube, axis, offset = make_beta_cylinder_density_cube(N=205,radius_mpc=1.2,beta=2.0,rho0=1.6e-23)
            axes.append(axis)
            offsets.append(offset)
            densitiesSmall = average_down_to_5(cube)

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

                    columnDensity *= self.lengthVoxel * self.metresPerMegaparsec * self.densityMeanToday # in g/m^2
                    if (columnDensityBest == None or columnDensityBest < columnDensity):
                        columnDensityBest = columnDensity
                        indexAltitudeBest = indexAltitude
                        indexAzimuthBest = indexAzimuth

                    self.columnDensities[indexJetSystem, indexAltitude, indexAzimuth] = columnDensity
                    #print(columnDensityBest, columnDensity)

            self.azimuthsBest       [indexJetSystem] = self.azimuths [indexAzimuthBest]
            self.altitudesBest      [indexJetSystem] = self.altitudes[indexAltitudeBest]
            self.columnDensitiesBest[indexJetSystem] = columnDensityBest

            #print(azimuths[indexAzimuthBest], altitudes[indexAltitudeBest], columnDensityBest)
            #columnDensities = np.full((numberOfAltitudes, numberOfAzimuths, numberOfJetSystems))

        #self.columnDensitiesBest *= self.lengthVoxel * self.metresPerMegaparsec * self.densityMeanToday # in g/m^2
        #self.columnDensities     *= self.lengthVoxel * self.metresPerMegaparsec * self.densityMeanToday # in g/m^2
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


# Initialise FOF.
lambdaMax              = 2.5 # in 2.93 Mpc / h
stepAngle              = 1.  # in deg
FOF                    = FilamentOrientationFinder(lambdaMax = lambdaMax, stepAngle = stepAngle)

# Initialise directory paths.
directoryDataBORGSDSS  = "/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/PhD/Ardor Telae/data/BORG SDSS/"
directoryExcel         = "/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_Mpc/"

#'''
# Load BORG SDSS mean.
dataBORGSDSSDensity    = np.load(directoryDataBORGSDSS + "borg_sdss_density.npz")
densitiesMean          = dataBORGSDSSDensity["mean"] + 1 # in today's mean matter density
print(densitiesMean.shape)
print(np.amin(densitiesMean), np.amax(densitiesMean))
print(np.mean(densitiesMean))

# Load host galaxy voxel indices.
pathExcel              = directoryExcel + "Mpc_filament_pa_exact_1.xlsx"#"kpc_filament_pa_exact_1.xlsx"
df                     = pd.read_excel(pathExcel)
voxelIndicesListDirect = [np.array(ast.literal_eval(s)) for s in df["voxel_index_r (x,y,z)"]] # list of NumPy arrays, each containing 3 integers (voxel indices)
voxelIndicesListAdjust = [np.array(ast.literal_eval(s)) for s in df["voxel_index_j (x,y,z)"]] # list of NumPy arrays, each containing 3 integers (voxel indices)


axes =[]
offsets=[]
FOF.findBest([np.array([128,128,128])] * 10000, densitiesMean)
azimuths  = np.full(len(axes), np.nan)
altitudes = np.full(len(axes), np.nan)
distancesAng = np.full(len(axes), np.nan)
from functionsArdorTelae import distanceOnSphere
for axis,i in zip(axes,range(len(axes))):
    x, y, z = axis
    alt_rad = np.arcsin(z)
    az_rad = np.arctan2(y, x)
    alt_deg = np.degrees(alt_rad)
    az_deg = np.degrees(az_rad) % 360.0
    azimuths[i] = az_deg
    altitudes[i] = alt_deg
    distance1 = distanceOnSphere(az_deg, alt_deg, FOF.azimuthsBest[i], FOF.altitudesBest[i])[0,0]
    distance2 = distanceOnSphere(az_deg, alt_deg, FOF.azimuthsBest[i] + 180, -1 * FOF.altitudesBest[i])[0,0]
    distance  = min(distance1, distance2)
    distancesAng[i] = distance
    print(az_deg, alt_deg, FOF.azimuthsBest[i], FOF.altitudesBest[i], distance)
np.save("/Users/martijnoei/Library/CloudStorage/Dropbox-Personal/Martijn/Caltech/Caltech Connection/comparisonFilamentResolution_errors.npy", distancesAng)
from matplotlib import pyplot as plt
plt.hist(distancesAng,bins=np.linspace(0,40,num=20+1))
plt.show()
#print(axes)
import sys
sys.exit()

# Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS mean (direct method).
FOF.findBest(voxelIndicesListDirect, densitiesMean)
from matplotlib import pyplot as plt
for i in range(10):
    plt.imshow(FOF.columnDensities[i+135], origin = "lower", aspect = "auto")
    plt.title(i+135)
    plt.show()
FOF.writeNumPy(directoryExcel + "Mpc_column_densities_all_r.npy")
#'''
#FOF.write(pathExcel, methodDirect = True)
# Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS mean (adjusted method).
FOF.findBest(voxelIndicesListAdjust, densitiesMean)
from matplotlib import pyplot as plt
for i in range(10):
    plt.imshow(FOF.columnDensities[i+135], origin = "lower", aspect = "auto")
    plt.title(i+135)
    plt.show()
FOF.writeNumPy(directoryExcel + "Mpc_column_densities_all_j.npy")
#FOF.write(pathExcel, methodDirect = False)
#'''
import sys
sys.exit()

# Find and write to file the best filament orientations for Mpc-scale jet systems in BORG SDSS realisations (direct method).
numberOfExcelSheets = 41 # in 1
for i in range(0, numberOfExcelSheets):
    realisationIndexString  = str(2000 + i * 250)
    pathBORGSDSSRealisation = directoryDataBORGSDSS + "final_density/" + "final_density_" + realisationIndexString + ".h5"
    pathExcelLoad           = directoryExcel + "fpa_" + realisationIndexString + ".xlsx"
    pathExcelWrite          = directoryExcel + "fpa_" + realisationIndexString + "_exact_1.xlsx"

    print("Working on realisation '" + realisationIndexString + "'...")
    print(pathBORGSDSSRealisation)
    print(pathExcelLoad)
    print(pathExcelWrite)

    # Load BORG SDSS realisation.
    with h5py.File(pathBORGSDSSRealisation, "r") as hf:
        densitiesSample = hf["scalars"]["field"][()] + 1 # in today's mean matter density
    print(densitiesSample.shape)
    print(np.amin(densitiesSample), np.amax(densitiesSample))
    print(np.mean(densitiesSample))

    # Load host galaxy voxel indices.
    df                     = pd.read_excel(pathExcelLoad)
    voxelIndicesListDirect = [np.array(ast.literal_eval(s)) for s in df["voxel_index_r (x,y,z)"]] # list of NumPy arrays, each containing 3 integers
    voxelIndicesListAdjust = [np.array(ast.literal_eval(s)) for s in df["voxel_index_j (x,y,z)"]] # list of NumPy arrays, each containing 3 integers

    # Create 'pathExcelWrite' if it doesn't exist yet.
    if (not os.path.exists(pathExcelWrite)):
        command = f"cp '{pathExcelLoad}' '{pathExcelWrite}'"
        print(command)
        os.system(command)

    # Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS realisation (direct method).
    FOF.findBest(voxelIndicesListDirect, densitiesSample)
    FOF.write(pathExcelWrite, methodDirect = True)
    # Find and write to file the best filament orientations for Mpc-scale jet systems in the BORG SDSS realisation (adjusted method).
    FOF.findBest(voxelIndicesListAdjust, densitiesSample)
    FOF.write(pathExcelWrite, methodDirect = False)



#voxelRadius         = int(np.ceil(lambdaMax - .5))
#numberOfAzimuths  = int(360 / stepAngle)    # in 1
#numberOfAltitudes = int(90 / stepAngle) + 1 # in 1
#azimuths          = np.linspace(0., 360., num = numberOfAzimuths, endpoint = False)
#altitudes         = np.linspace(0., 90., num = numberOfAltitudes, endpoint = True)
# azimuthsRadians   = np.radians(azimuths)
# altitudesRadians  = np.radians(altitudes)
# azimuthsCos       = np.cos(azimuthsRadians)
# azimuthsSin       = np.sin(azimuthsRadians)
# altitudesCos      = np.cos(altitudesRadians)
# altitudesSin      = np.sin(altitudesRadians)
# xsFilament        = altitudesCos[ : , None] * azimuthsCos[None, : ]
# ysFilament        = altitudesCos[ : , None] * azimuthsSin[None, : ]
# zsFilament        = altitudesSin[ : , None] * np.ones_like(azimuths)[None, : ]
# xsFilamentSign    = np.sign(xsFilament)
# ysFilamentSign    = np.sign(ysFilament)
# zsFilamentSign    = np.sign(zsFilament)

#jMax             = int(np.floor(lambdaMax + .5))
#js               = np.arange(1, jMax + 1)
#print(js)
# lambdasCrossingX = (2 * js[None, None, : ] - 1) / (2 * np.abs(xsFilament[ : , : , None]))
# lambdasCrossingY = (2 * js[None, None, : ] - 1) / (2 * np.abs(ysFilament[ : , : , None]))
# lambdasCrossingZ = (2 * js[None, None, : ] - 1) / (2 * np.abs(zsFilament[ : , : , None]))

# voxelIndicesCentre  = np.array([voxelRadius, voxelRadius, voxelRadius])
# metresPerMegaparsec = 3.0857e22         # in 1
# densityMeanToday    = 2.6e-24           # in g/m^3
# lengthVoxel         = (750. / .7) / 256 # in Mpc

#voxelIndices        = np.array([194, 224, 119])#np.array([212, 187, 75])#np.array([195, 238, 112])#np.array([195, 239, 112])#

# print(voxelIndicesS[0], type(voxelIndicesS[0]))
# print(np.array(voxelIndicesS[0]), type(np.array(voxelIndicesS[0])))
# print(np.array(voxelIndicesS[0])[0])
# print(voxelIndicesS.shape)

#densitiesMeanSmall = densitiesMean[voxelIndices[2] - voxelRadius : voxelIndices[2] + voxelRadius + 1, voxelIndices[1] - voxelRadius : voxelIndices[1] + voxelRadius + 1, voxelIndices[0] - voxelRadius : voxelIndices[0] + voxelRadius + 1]
#xs = np.cos(np.radians(self.altitudesBest)) * np.cos(np.radians(self.azimuthsBest))
#ys = np.cos(np.radians(self.altitudesBest)) * np.sin(np.radians(self.azimuthsBest))
# print(type(df["voxel_index_r (x,y,z)"][0]))
# print(voxelIndicesListDirect[:5])
#final_density_2000.h5"
#print(np.array_equal(voxelIndicesListDirect, voxelIndicesListDirectS)) # Should be True and is True.
#print(np.array_equal(voxelIndicesListAdjust, voxelIndicesListAdjustS))

# def random_unit_vector(rng=None):
#     rng = np.random.default_rng() if rng is None else rng
#     v = rng.normal(size=3)
#     return v / np.linalg.norm(v)
# def make_beta_cylinder_density_cube(
#     N,
#     cube_size_mpc=20.87,
#     radius_mpc=1.2,
#     beta=2.0,
#     rho0=1.6e-23,   # central density in g/m^3
#     rng=None,
# ):
#     rng = np.random.default_rng() if rng is None else rng
#
#     dx_mpc = cube_size_mpc / N
#     coords = (np.arange(N) - (N - 1) / 2) * dx_mpc
#     x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
#
#     axis = random_unit_vector(rng)
#     #axis = [0.,0.,1.] # x,y,z
#
#     r_dot_a = axis[2] * x + axis[1] * y + axis[0] * z
#     r2 = x**2 + y**2 + z**2
#     d_perp2 = np.maximum(r2 - r_dot_a**2, 0.0)
#     d_perp = np.sqrt(d_perp2)
#
#     cube = rho0 * (1.0 + (d_perp / radius_mpc) ** 2) ** (-1.5 * beta)
#     return cube, axis