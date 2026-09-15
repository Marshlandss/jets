# Set random seed.
SEED = 0

# Set jet orientation finding parameters.
STEP_SIZE             = 1.
BEST_ANGLE_THRESHOLD  = .85
PIXEL_SHIFT           = 1.   # shifts plot by indicated number of pixels to the top right; Martijn: 0, Martin: 1
NAN_PERCENTAGE_CUTOFF = 43.  # in %

# Set cosmology and BORG SDSS parameters.
LITTLE_H              = 0.702                                                    # in 1; H_0 / (100 km s^-1 Mpc^-1), matching Jasche et al. (12015)
DENSITY_MEAN_TODAY    = 2.6e-24                                                  # in g/m^3
BORG_BOX_SIZE_MPC_H   = 750.                                                     # in Mpc / h; comoving
BORG_NUMBER_OF_VOXELS = 256                                                      # in 1; per axis
BORG_VOXEL_SIZE_MPC   = BORG_BOX_SIZE_MPC_H / (BORG_NUMBER_OF_VOXELS * LITTLE_H) # in Mpc (≈ 4.17); comoving

# Set beta-profile filament parameters from Tuominen et al. (2021), used for voxelisation error simulations.
FILAMENT_RADIUS_CORE     = 1.2     # in Mpc
FILAMENT_BETA            = 2.0     # in 1
FILAMENT_DENSITY_CENTRAL = 1.6e-23 # in g m^-3

# Set saving parameters.
SAVING_PLOTS          = True
# If True:  Replaces all files
# If False: Only adds missing files 
#           OR if file is placed in the wrong folder, file will be removed and recreated in the correct folder
# Set False if you have already generated plots and are sorting them manually (using the manual adjustment lists)
OVERWRITE_FILES       = False
SAVE_FORMAT           = "pdf"