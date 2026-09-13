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