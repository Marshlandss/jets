"""
Martijn Oei & Ruby Yao, 2026
"""

import numpy
from scipy.special import erf, erfinv, erfi, betainc


class PADifferencesInferrer:
    """
    """
    def __init__(self, numberOfSamples):
        """
        """
        self.numberOfSamples = numberOfSamples


    def sampleZsWatson(self,
                       kappa,
                       xStepGrid = 1e-2): # Achieve a resolution of at least 'xStepGrid'.
        """
        """
        FZs = numpy.random.uniform(0, 1, self.numberOfSamples)
        if   (kappa < 0):
            Zs = erfinv((2 * FZs - 1) * erf(numpy.sqrt(-kappa))) / numpy.sqrt(-kappa)
        elif (kappa == 0):
            Zs = 2 * FZs - 1
        elif (kappa > 0):
            if (kappa == numpy.inf):
                # This case represents full polar alignment. 'Zs' is populated with floats -1. and +1.; they occur with equal probability.
                Zs = numpy.round(numpy.random.rand(self.numberOfSamples)) * 2 - 1 # Create numbers 0. and 1. through rounding, then 0. and 2., then -1. and +1.
            else:
                # Calculate inverse imaginary error function arguments (IIEFAs).
                IIEFAs    = (2 * FZs - 1) * erfi(numpy.sqrt(kappa))
                # Initialise a suitable imaginary error function (restricted) domain and, equivalently, a suitable inverse imaginary error function (restricted) codomain.
                xMin      = -1 * numpy.sqrt(kappa) # Since 2 * FZs - 1 is at least -1, note that erfiinv(-1 * erfi(numpy.sqrt(kappa))) = erfiinv(erfi(-1 * numpy.sqrt(kappa))) = -1 * numpy.sqrt(kappa).
                xMax      = +1 * numpy.sqrt(kappa) # Since 2 * FZs - 1 is at most  +1, note that erfiinv(+1 * erfi(numpy.sqrt(kappa))) =                                         +1 * numpy.sqrt(kappa).
                xs        = numpy.linspace(xMin, xMax, num = int(numpy.ceil((xMax - xMin) / xStepGrid)) + 1, endpoint = True)
                # Calculate imaginary error function values.
                IEFs      = erfi(xs)
                # Calculate inverse imaginary error function values.
                IIEFs     = numpy.interp(IIEFAs, IEFs, xs)
                # Calculate Zs.
                Zs        = IIEFs / numpy.sqrt(kappa)
        return Zs


    def sampleZsBetaRectangular(self, alpha, beta, weightUniform):
        """
        """
        CDFXs = numpy.linspace(0, 1, num = 1000 + 1, endpoint = True)
        CDFYs = CDFXs * weightUniform + betainc(alpha, beta, CDFXs) * (1 - weightUniform)

        FABars = numpy.random.uniform(0, 1, self.numberOfSamples)
        ABars  = numpy.interp(FABars, CDFYs, CDFXs)
        As     = ABars * numpy.pi / 2
        Zs     = numpy.cos(As) * (numpy.random.randint(0, high = 2, size = self.numberOfSamples) * 2 - 1)
        return Zs

    def samplePAsDeltaAbs(self, kappaJ = 0, kappaF = numpy.inf, alpha = 0., beta = 0., weightUniform = 1., distributionJ = "Watson", distributionF = "Watson"):
        """
        """
        print("kappaJ = ", kappaJ, "kappaF = ", kappaF, "alpha = ", alpha, "beta = ", beta, "weightUniform = ", weightUniform)
        if (distributionJ == "Watson"):
            self.ZsJ = self.sampleZsWatson(kappaJ)
        else:
            print("Only the Watson distribution has been implemented.")
        if (distributionF == "Watson"):
            self.ZsF = self.sampleZsWatson(kappaF)
        elif (distributionF == "beta rectangular"):
            self.ZsF = self.sampleZsBetaRectangular(alpha, beta, weightUniform)
        else:
            print("Only the Watson distribution has been implemented.")

        # This function only works correctly if 'Xfs' is shared among the jets and filaments.
        self.Xfs         = numpy.random.uniform(-1, 1, self.numberOfSamples)
        Zfs              = numpy.sqrt(1 - numpy.square(self.Xfs))

        self.PhisJ       = numpy.random.uniform(0, 2 * numpy.pi, self.numberOfSamples)
        self.PhisF       = numpy.random.uniform(0, 2 * numpy.pi, self.numberOfSamples)
        XsJ              = numpy.sqrt(1 - numpy.square(self.ZsJ))
        XsF              = numpy.sqrt(1 - numpy.square(self.ZsF))
        tanPAsJ          = XsJ * numpy.sin(self.PhisJ) / (self.ZsJ * Zfs - XsJ * self.Xfs * numpy.cos(self.PhisJ))
        tanPAsF          = XsF * numpy.sin(self.PhisF) / (self.ZsF * Zfs - XsF * self.Xfs * numpy.cos(self.PhisF))
        self.PAsJ        = numpy.degrees(numpy.arctan(tanPAsJ))
        self.PAsF        = numpy.degrees(numpy.arctan(tanPAsF))
        self.PAsDeltaAbs = numpy.abs(self.PAsJ - self.PAsF)
        self.PAsDeltaAbs = numpy.minimum(self.PAsDeltaAbs, 180 - self.PAsDeltaAbs)
        return self.PAsDeltaAbs


    def PDsWatsonPolarAngle(self, angles, kappa, assumeDegrees = True):
        if (assumeDegrees):
            angles = numpy.radians(angles)
        if   (kappa < 0):
            PDs = numpy.sqrt(-1 * kappa / numpy.pi) * 2 / erf(numpy.sqrt(-1 * kappa)) * numpy.exp(kappa * numpy.square(numpy.cos(angles))) * numpy.sin(angles)
        elif (kappa == 0):
            PDs = numpy.sin(angles)
        elif (kappa > 0):
            PDs = numpy.sqrt(kappa / numpy.pi) * 2 / erfi(numpy.sqrt(kappa)) * numpy.exp(kappa * numpy.square(numpy.cos(angles))) * numpy.sin(angles)
        if (assumeDegrees):
            PDs *= numpy.pi / 180
        return PDs


    def MLEExpressionKappa(self, kappas):
        MLEExpressions = numpy.full_like(kappas, numpy.nan)
        kappasPositive = kappas[kappas > 0]
        kappasNegative = kappas[kappas < 0]
        MLEExpressions[kappas >  0] = numpy.exp(kappasPositive) / (numpy.sqrt(numpy.pi * kappasPositive) * erfi(numpy.sqrt(kappasPositive))) - 1. / (2 * kappasPositive)
        MLEExpressions[kappas == 0] = 1 / 3.
        MLEExpressions[kappas <  0] = -1 * numpy.exp(kappasNegative) / (numpy.sqrt(numpy.pi * -1 * kappasNegative) * erf(numpy.sqrt(-1 * kappasNegative))) - 1. / (2 * kappasNegative)
        return MLEExpressions


    def MLEKappa(self, MLEExpressionData, kappaHalfWidth = 10., kappaStepSize = 1e-2):
        kappas              = numpy.linspace(-1 * kappaHalfWidth, kappaHalfWidth, num = int(numpy.ceil(2 * kappaHalfWidth / kappaStepSize)) + 1, endpoint = True)
        MLEExpressionsKappa = self.MLEExpressionKappa(kappas)
        kappaMLE            = numpy.interp(MLEExpressionData, MLEExpressionsKappa, kappas)
        return kappaMLE
