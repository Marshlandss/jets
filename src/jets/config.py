# Imports: third-party
import numpy as np

# === Constants ===
STEP_SIZE             = 1.
NUMBER_OF_STEPS       = int(180 / STEP_SIZE)
ANGLES                = np.linspace(0, np.pi, num = NUMBER_OF_STEPS, endpoint = False) # in radians

# --- Modifiable parameters (for varying results) ---
SEED                  = 0
BEST_ANGLE_THRESHOLD  = .85
NAN_PERCENTAGE_CUTOFF = 43  # in %
PIXEL_SHIFT           = 1   # shifts plot by indicated number of pixels to the top right; Martijn: 0, Martin: 1

# --- BORG SDSS reconstruction ---
LITTLE_H              = 0.702                                                    # in 1; H_0 / (100 km s^-1 Mpc^-1), matching Jasche et al. (12015)
BORG_BOX_SIZE_MPC_H   = 750.                                                     # in Mpc / h; comoving
BORG_NUMBER_OF_VOXELS = 256                                                      # in 1; per axis
BORG_VOXEL_SIZE_MPC   = BORG_BOX_SIZE_MPC_H / (BORG_NUMBER_OF_VOXELS * LITTLE_H) # in Mpc (≈ 4.17); comoving

# --- Saving ---
SAVING_PLOTS          = True

# OVERWRITE FILES -
# If True:  Replaces all files
# If False: Only adds missing files 
#           OR if file is placed in the wrong folder, file will be removed and recreated in the correct folder
# Set False if you have already generated plots and are sorting them manually (using the manual adjustment lists)
OVERWRITE_FILES       = False
SAVE_FORMAT           = "pdf"