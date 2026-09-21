"""
Martijn Oei & Ruby Yao, September 12026
"""
# Imports: third-party
from scipy.special import erf, erfinv, erfi
import numpy as np


class PADifferencesInferrer:
    """
    """
    def __init__(self, numberOfSamples):
        """
        """
        self.numberOfSamples = numberOfSamples


    def sampleZsWatson(self,
                       kappa,
                       xStepGrid = 1e-3): # Achieve a resolution of at least 'xStepGrid'.
        """
        """
        FZs = np.random.uniform(0, 1, self.numberOfSamples)
        if   (kappa < 0):
            Zs = erfinv((2 * FZs - 1) * erf(np.sqrt(-kappa))) / np.sqrt(-kappa)
        elif (kappa == 0):
            Zs = 2 * FZs - 1
        elif (kappa > 0):
            if (kappa == np.inf):
                # This case represents full polar alignment. 'Zs' is populated with floats -1. and +1.; they occur with equal probability.
                Zs = np.round(np.random.rand(self.numberOfSamples)) * 2 - 1 # Create numbers 0. and 1. through rounding, then 0. and 2., then -1. and +1.
            else:
                # Calculate inverse imaginary error function arguments (IIEFAs).
                IIEFAs    = (2 * FZs - 1) * erfi(np.sqrt(kappa))
                # Initialise a suitable imaginary error function (restricted) domain and, equivalently, a suitable inverse imaginary error function (restricted) codomain.
                xMin      = -1 * np.sqrt(kappa) # Since 2 * FZs - 1 is at least -1, note that erfiinv(-1 * erfi(np.sqrt(kappa))) = erfiinv(erfi(-1 * np.sqrt(kappa))) = -1 * np.sqrt(kappa).
                xMax      = +1 * np.sqrt(kappa) # Since 2 * FZs - 1 is at most  +1, note that erfiinv(+1 * erfi(np.sqrt(kappa))) =                                         +1 * np.sqrt(kappa).
                xs        = np.linspace(xMin, xMax, num = int(np.ceil((xMax - xMin) / xStepGrid)) + 1, endpoint = True)
                # Calculate imaginary error function values.
                IEFs      = erfi(xs)
                # Calculate inverse imaginary error function values.
                IIEFs     = np.interp(IIEFAs, IEFs, xs)
                # Calculate Zs.
                Zs        = IIEFs / np.sqrt(kappa)
        return Zs


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
