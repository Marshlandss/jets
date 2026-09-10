import numpy as np

# === Constants ===
STEP_SIZE             = 1
NUMBER_OF_STEPS       = int(180/STEP_SIZE)
ANGLES                = np.linspace(0, np.pi, num = NUMBER_OF_STEPS, endpoint = False) # in radians

# --- Modifiable Parameters (For varying results) --- #
BEST_ANGLE_THRESHOLD  = .85
NAN_PERCENTAGE_CUTOFF = 43        # in percent
PIXEL_SHIFT           = 1         # shifts plot by indicated number of pixels to the top right; Martijn: 0, Martin: 1

# --- Save Parameters ---
SAVING_PLOTS          = True

# OVERWRITE FILES -
# If True: Replaces all files
# If False: Only adds missing files 
#           OR if file is placed in the wrong folder, file will be removed and recreated
#              in the correct folder
# Set False if you have already generated plots and are sorting them manually (using the manual adjustment lists)
OVERWRITE_FILES       = False
SAVE_FORMAT           = "pdf"