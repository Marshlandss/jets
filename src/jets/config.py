# Set random seed.
SEED = 0

# Set jet orientation search parameters.
JET_ANGLE_STEP        = 1.  # in deg; angular resolution of the jet orientation search
BEST_ANGLE_THRESHOLD  = .85
PIXEL_SHIFT           = 1   # shifts plot by indicated number of pixels to the top right; Martijn: 0, Martin: 1
NAN_PERCENTAGE_CUTOFF = 43. # in %

# Set cosmology and BORG SDSS parameters.
LITTLE_H                     = 0.702                                                    # in 1; H_0 / (100 km/s/Mpc), matching Jasche et al. (12015)
DENSITY_MEAN_TODAY           = 2.6e-24                                                  # in g/m^3
BORG_BOX_SIZE_MPC_H          = 750.                                                     # in Mpc/h; comoving
BORG_NUMBER_OF_VOXELS        = 256                                                      # in 1; per axis
BORG_VOXEL_SIZE_MPC          = BORG_BOX_SIZE_MPC_H / (BORG_NUMBER_OF_VOXELS * LITTLE_H) # in Mpc (≈ 4.17); comoving
BORG_NUMBER_OF_REALIZATIONS  = 41                                                       # in 1
BORG_INDEX_REALIZATION_START = 2000                                                     # index of the loop's first realization
BORG_INDEX_REALIZATION_STEP  = 250                                                      # index spacing between consecutive realizations

# Set voxelization error simulation parameters.
FILAMENT_NUMBER_OF_SIMULATIONS = int(1e4) # in 1
FILAMENT_TOLERANCE_REL         = 1e-3     # in 1
# Set beta-profile filament parameters from Tuominen et al. (12021).
FILAMENT_BETA                  = 2.0      # in 1
FILAMENT_DENSITY_CENTRAL       = 1.6e-23  # in g/m^3
FILAMENT_RADIUS_CORE           = 1.2      # in Mpc

# Set filament orientation search parameters.
FILAMENT_ANGLE_STEP = 1.  # in deg; angular resolution of the filament orientation search
FILAMENT_LAMBDA_MAX = 2.5 # in BORG voxel side lengths; half-length of the line segment integrated over

# Set saving parameters.
SAVING_PLOTS    = True
# If True:  Replaces all files
# If False: Only adds missing files 
#           OR if file is placed in the wrong folder, file will be removed and recreated in the correct folder
# Set False if you have already generated plots and are sorting them manually (using the manual adjustment lists)
OVERWRITE_FILES = False
SAVE_FORMAT     = "pdf"
