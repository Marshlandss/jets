"""
Watson distribution quantities that depend on the concentration 'kappa' alone.

The (antipodally symmetric) Watson distribution on the sphere with mean axis mu and concentration kappa has a
probability density proportional to exp(kappa (mu . x)^2). Writing Z := mu . x = cos A, with A the polar angle
measured from mu, this module provides the probability density of A, and the concentration's maximum likelihood
estimate given the sample mean of Z^2.

As the distribution is antipodally symmetric, x and -x are the same axis: A is taken in [0, pi / 2], so that it is
the angle between an axis and the mean axis, and its density integrates to 1 over that range (Mardia and Jupp,
12000; Watson, 11965).
Positive kappa concentrates the distribution around the mean axis mu (polar clustering); negative kappa concentrates it
around the equator (girdle clustering); kappa = 0 is the uniform distribution on the sphere.
"""
# Imports: third-party
from scipy.special import erf, erfinv, erfi
import numpy as np


def PDsPolarAngle(angles, kappa, assumeDegrees = True):
    """
    Calculate the probability density of the polar angle A for a Watson distribution of concentration 'kappa'.
    The density is f_A(a) = exp(kappa cos^2 a) sin a / M(1 / 2, 3 / 2, kappa), with M being Kummer's confluent
    hypergeometric function, which equals sqrt(pi) erfi(sqrt(kappa)) / (2 sqrt(kappa)) for kappa > 0 and
    sqrt(pi) erf(sqrt(-kappa)) / (2 sqrt(-kappa)) for kappa < 0.
    It integrates to 1 over a in [0, pi / 2], the angle between an axis and the mean axis; evaluating it beyond pi / 2 double-counts the antipodal half.

    Parameters
    ----------
    angles : np.ndarray
        Polar angles at which to evaluate the density, in [0, 90 deg] if 'assumeDegrees' and in [0, pi / 2 rad] if not.
    kappa : float
        Watson distribution concentration; in 1. Must be finite: kappa = +-np.inf gives NaN.
    assumeDegrees : bool
        If True, 'angles' are in deg and the density is returned in deg^-1; if False, both are in rad.

    Returns
    -------
    np.ndarray
        Probability density at 'angles'; in deg^-1 if 'assumeDegrees', else in rad^-1. Same shape as 'angles'.
    """
    if (assumeDegrees):
        angles = np.radians(angles)
    if   (kappa < 0):
        PDs = np.sqrt(-1 * kappa / np.pi) * 2 / erf(np.sqrt(-1 * kappa)) * np.exp(kappa * np.square(np.cos(angles))) * np.sin(angles)
    elif (kappa == 0):
        PDs = np.sin(angles)
    elif (kappa > 0):
        PDs = np.sqrt(kappa / np.pi) * 2 / erfi(np.sqrt(kappa)) * np.exp(kappa * np.square(np.cos(angles))) * np.sin(angles)
    if (assumeDegrees):
        PDs *= np.pi / 180
    return PDs


def MLEExpressionKappa(kappas):
    """
    Calculate E[Z^2] as a function of the Watson distribution concentration, where Z is the cosine of the polar angle.

    The maximum likelihood estimate of the concentration is the 'kappa' for which this expression equals the sample
    mean of Z^2; see 'MLEKappa'. The expression increases monotonically from 0 (kappa -> -infinity) through 1 / 3
    (kappa = 0, the uniform distribution) to 1 (kappa -> +infinity).

    Parameters
    ----------
    kappas : np.ndarray
        Watson distribution concentrations; in 1. Must be a float array, as the result is built with 'np.full_like'.

    Returns
    -------
    np.ndarray
        E[Z^2] at 'kappas'; in 1. Same shape as 'kappas'.
    """
    MLEExpressions = np.full_like(kappas, np.nan)
    kappasPositive = kappas[kappas > 0]
    kappasNegative = kappas[kappas < 0]
    MLEExpressions[kappas >  0] = np.exp(kappasPositive) / (np.sqrt(np.pi * kappasPositive) * erfi(np.sqrt(kappasPositive))) - 1. / (2 * kappasPositive)
    MLEExpressions[kappas == 0] = 1 / 3.
    MLEExpressions[kappas <  0] = -1 * np.exp(kappasNegative) / (np.sqrt(np.pi * -1 * kappasNegative) * erf(np.sqrt(-1 * kappasNegative))) - 1. / (2 * kappasNegative)
    return MLEExpressions


def MLEKappa(MLEExpressionData, kappaHalfWidth = 10., kappaStepSize = 1e-2):
    """
    Calculate the maximum likelihood estimate of the Watson distribution concentration from the sample mean of Z^2.

    The estimate is found by inverting 'MLEExpressionKappa' on a grid through linear interpolation, and is therefore
    clipped to [-'kappaHalfWidth', +'kappaHalfWidth'].

    Parameters
    ----------
    MLEExpressionData : float or np.ndarray
        Sample mean of Z^2, with Z the cosine of the polar angle; in 1.
    kappaHalfWidth : float
        Half-width of the concentration grid searched; in 1.
    kappaStepSize : float
        Spacing of the concentration grid searched; in 1.

    Returns
    -------
    float or np.ndarray
        Maximum likelihood estimate of the concentration; in 1. Same shape as 'MLEExpressionData'.
    """
    kappas              = np.linspace(-1 * kappaHalfWidth, kappaHalfWidth, num = int(np.ceil(2 * kappaHalfWidth / kappaStepSize)) + 1, endpoint = True)
    MLEExpressionsKappa = MLEExpressionKappa(kappas)
    kappaMLE            = np.interp(MLEExpressionData, MLEExpressionsKappa, kappas)
    return kappaMLE


def sampleZsWatson(kappa,
                   numberOfSamples,   # in 1
                   xStepGrid = 1e-3): # Achieve a resolution of at least 'xStepGrid'.
    """
    """
    FZs = np.random.uniform(0, 1, numberOfSamples)
    if   (kappa < 0):
        Zs = erfinv((2 * FZs - 1) * erf(np.sqrt(-kappa))) / np.sqrt(-kappa)
    elif (kappa == 0):
        Zs = 2 * FZs - 1
    elif (kappa > 0):
        if (kappa == np.inf):
            # This case represents full polar alignment. 'Zs' is populated with floats -1. and +1.; they occur with equal probability.
            Zs = np.round(np.random.rand(numberOfSamples)) * 2 - 1 # Create numbers 0. and 1. through rounding, then 0. and 2., then -1. and +1.
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
