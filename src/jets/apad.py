"""
Martijn Oei & Ruby Yao, September 12026
"""
import numpy as np


class PADifferencesInferrer:
    """
    """
    def __init__(self, numberOfSamples):
        """
        """
        self.numberOfSamples = numberOfSamples


    def samplePAsDeltaAbs(self, kappaJ = 0, kappaF = np.inf, distributionJ = "Watson", distributionF = "Watson"):
        """
        """
        print("kappaJ = ", kappaJ, "kappaF = ", kappaF)
        if (distributionJ == "Watson"):
            self.ZsJ = self.sampleZsWatson(kappaJ)
        else:
            print("Only the Watson distribution has been implemented.")
        if (distributionF == "Watson"):
            self.ZsF = self.sampleZsWatson(kappaF)
        else:
            print("Only the Watson distribution has been implemented.")

        # This function only works correctly if 'Xfs' is shared among the jets and filaments.
        self.Xfs         = np.random.uniform(-1, 1, self.numberOfSamples)
        Zfs              = np.sqrt(1 - np.square(self.Xfs))

        self.PhisJ       = np.random.uniform(0, 2 * np.pi, self.numberOfSamples)
        self.PhisF       = np.random.uniform(0, 2 * np.pi, self.numberOfSamples)
        XsJ              = np.sqrt(1 - np.square(self.ZsJ))
        XsF              = np.sqrt(1 - np.square(self.ZsF))
        tanPAsJ          = XsJ * np.sin(self.PhisJ) / (self.ZsJ * Zfs - XsJ * self.Xfs * np.cos(self.PhisJ))
        tanPAsF          = XsF * np.sin(self.PhisF) / (self.ZsF * Zfs - XsF * self.Xfs * np.cos(self.PhisF))
        self.PAsJ        = np.degrees(np.arctan(tanPAsJ))
        self.PAsF        = np.degrees(np.arctan(tanPAsF))
        self.PAsDeltaAbs = np.abs(self.PAsJ - self.PAsF)
        self.PAsDeltaAbs = np.minimum(self.PAsDeltaAbs, 180 - self.PAsDeltaAbs)
        return self.PAsDeltaAbs
