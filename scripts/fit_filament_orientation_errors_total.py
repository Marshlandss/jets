"""
Describe the filament orientation error budgets parametrically, as a mixture of a Watson distribution and the uniform
distribution on the sphere.

The two parameters are fixed by each budget's E[P_2] and E[P_4]: the factors by which the filament orientation errors attenuate
the quadrupole and hexadecapole of the jet--filament alignment. The inference uses the empirical budgets; the parametric
description serves the paper (and the test that swapping it in for the fiducial budget leaves the inference unchanged).
"""
# Imports: standard library
import json
# Imports: third-party
import numpy as np
# Imports: first-party
from jets import watson
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import polynomialLegendreMeanSample

# Initialize settings.
labelSample  = "Mpc"
labelsMethod = ("d", "a")          # of the reference axis; "d": direct method, "a": adjusted method
labelsBudget = ("B1", "B2", "B3")

for labelMethodReference in labelsMethod:
    parameters = {}
    for labelBudget in labelsBudget:
        errors = np.radians(np.load(DIR_OUTPUT / f"filament_orientation_errors_total_{labelBudget}_{labelSample}_{labelMethodReference}.npy")) # in rad
        polynomialLegendreMeanSample2 = polynomialLegendreMeanSample(errors, 2)                                               # in 1
        polynomialLegendreMeanSample4 = polynomialLegendreMeanSample(errors, 4)                                               # in 1
        kappa, weightUniform          = watson.fitWatsonUniform(polynomialLegendreMeanSample2, polynomialLegendreMeanSample4) # in 1, in 1
        parameters[labelBudget]       = {"kappa" : kappa, "weightUniform" : weightUniform}
        print(f"Budget {labelBudget}, reference method '{labelMethodReference}': "
              f"E[P_2] = {polynomialLegendreMeanSample2:.4f}, E[P_4] = {polynomialLegendreMeanSample4:.4f} "
              f"-> kappa = {kappa:.3f}, weightUniform = {weightUniform:.3f}")

    pathParameters = DIR_OUTPUT / f"filament_orientation_errors_total_parametric_{labelSample}_{labelMethodReference}.json"
    with open(pathParameters, "w") as file:
        json.dump(parameters, file, indent = 4)
        file.write("\n")
    print(f"Saved parametric descriptions to '{pathParameters}'.")
