"""
Compose filament orientation error budgets from their components, via the spherical law of cosines.
We consider the following four budgets:
    B0: no errors
    B1: host galaxy localization and BORG SDSS posterior (measured, pooled over jet systems) + voxelization (simulated)
    B2: B1 + galaxy bias + gravity model
    B3: B2 + lump of other systematics

Each assumed component (galaxy bias, gravity model, lump of other systematics) is drawn from the voxelization error distribution:
it is assumed comparable to, and independent of, voxelization.
B0 (no filament orientation error) needs no file.
The budgets are nested: B2 composes on top of B1's samples and B3 on top of B2's,
so that the differences between budgets reflect the added components rather than independent Monte Carlo noise.
Composition assumes that the direction of each component is isotropic about the axis it perturbs.
Under that assumption the distribution of the total error does not depend on the order in which the components are composed.
"""
# Imports: third-party
import numpy as np
# Imports: first-party
from jets.config import SEED, FILAMENT_NUMBER_OF_SIMULATIONS
from jets.paths import DIR_OUTPUT
from jets.sphere_utils import composeAngularErrors

# Initialize settings.
labelSample           = "Mpc"
labelsMethod          = ("d", "a")                                            # "d": direct method, "a": adjusted method; of the reference axis
systematicsAdditional = ("galaxy bias", "gravity model", "other systematics") # composed in this order after B1
labelsBudget          = {0 : "B1", 2 : "B2", 3 : "B3"}                        # number of additional systematics composed -> budget label
numberOfSamples       = int(1e6)                                              # in 1; per budget

# Load the error components.
errorsLocalizationPosterior = np.radians(np.load(DIR_OUTPUT / f"filament_orientation_errors_localization_posterior_{labelSample}.npy"))          # in rad
errorsVoxelization          = np.radians(np.load(DIR_OUTPUT / f"filament_orientation_errors_voxelization_{FILAMENT_NUMBER_OF_SIMULATIONS}.npy")) # in rad

RNG = np.random.default_rng(SEED)
for n, labelMethodReference in enumerate(labelsMethod):
    # Pool the localization and posterior errors over jet systems, realizations, and localization methods of the realization axes.
    errorsPooled = errorsLocalizationPosterior[ : , : , : , n].ravel() # in rad

    # Compose the measured components, then add the assumed components one by one.
    errorsTotal = composeAngularErrors(RNG.choice(errorsPooled,       size = numberOfSamples),
                                       RNG.choice(errorsVoxelization, size = numberOfSamples), RNG) # in rad
    for numberOfSystematicsAdditional in range(len(systematicsAdditional) + 1):
        if (numberOfSystematicsAdditional > 0):
            errorsTotal = composeAngularErrors(errorsTotal, RNG.choice(errorsVoxelization, size = numberOfSamples), RNG)
        if (numberOfSystematicsAdditional in labelsBudget):
            labelBudget        = labelsBudget[numberOfSystematicsAdditional]
            errorsTotalDegrees = np.degrees(np.minimum(errorsTotal, np.pi - errorsTotal)) # in deg; folded into [0, 90], as axes are undirected
            pathErrors         = DIR_OUTPUT / f"filament_orientation_errors_total_{labelBudget}_{labelSample}_{labelMethodReference}.npy"
            np.save(pathErrors, errorsTotalDegrees)
            print(f"Saved filament orientation errors to '{pathErrors}'.")
            print(f"    budget {labelBudget}, reference method '{labelMethodReference}': {np.median(errorsTotalDegrees):5.2f} deg (median), "
                  f"{np.percentile(errorsTotalDegrees, 10):5.2f} deg (10th percentile), {np.percentile(errorsTotalDegrees, 90):5.2f} deg (90th percentile)")
