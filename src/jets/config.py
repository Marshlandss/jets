
import numpy as np

# === Constants ===

STEP_SIZE               = 1
NUMBER_OF_STEPS         = int(180/STEP_SIZE)
ANGLES                  = np.linspace(0, np.pi, num = NUMBER_OF_STEPS, endpoint = False) # In radians

# --- Modifiable Parameters (For varying results) --- #
BEST_ANGLE_THRESHOLD    = .85       
NAN_PERCENTAGE_CUTOFF   = 43        # in percent
PIXEL_SHIFT             = 1         # shifts plot by indicated number of pixels to the top right; Martijin: 0, Hardcastle: 1

# --- Save Parameters ---
SAVING_PLOTS             = False
SAVE_FORMAT              = "pdf"