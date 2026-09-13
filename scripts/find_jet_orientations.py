# Imports: Python
import os
# Imports: third-party
from astropy.io import fits
import pandas as pd
# Imports: first-party
from jets.config import ANGLES, BEST_ANGLE_THRESHOLD, NAN_PERCENTAGE_CUTOFF, PIXEL_SHIFT, SAVING_PLOTS, SAVE_FORMAT, OVERWRITE_FILES
from jets.paths import DIR_LOAD, DIR_SAVE
from jets.jet_system import JetSystem
from jets.jet_plotter import JetPlotter
from jets.manual_adjustments import M_LIST, SA_LIST
from jets.jet_utils import format_sheet, check_if_file_exists_in_correct_location, check_adjustment_status
from jets.plot_styles import set_plot_styles

# === Plot Formatting ===
axis_font_size    = set_plot_styles()


# === Load data ===
# --- For Martijn's catalogue ---
path_catalogue_go = DIR_LOAD / "GGO_catalogue_2025_02_with3C236.fits"
directory_fits    = DIR_LOAD / "fits"

# --- For Martin's catalogue ---
#path_catalogue_go = DIR_LOAD / "agn-v1.1.fits"
#directory_fits    = DIR_LOAD / "cleaned_fits_without_dups"

hdu_list          = fits.open(path_catalogue_go)

# --- For Martijn's catalogue ---
go_right_ascensions     = hdu_list[1].data["host_right_ascension_(deg)"]           # in deg
go_declinations         = hdu_list[1].data["host_declination_(deg)"]               # in deg
go_length_angular_means = hdu_list[1].data["outflow_length_angular_mean_(arcmin)"] # in arcmin
go_redshift_means       = hdu_list[1].data["host_redshift_mean_(1)"]
go_redshift_stdev       = hdu_list[1].data["host_redshift_SD_(1)"]
areSpectroscopic        = (go_redshift_stdev < 0.001)

# --- For Martin's catalogue ---
#import numpy
#go_right_ascensions     = hdu_list[1].data["optRA"]     # in deg
#go_declinations         = hdu_list[1].data["optDec"]    # in deg
#go_length_angular_means = hdu_list[1].data['LAS'] / 60  # in arcmin
#go_redshift_means       = hdu_list[1].data['z_best']
#z_hetdex                = hdu_list[1].data["z_hetdex"]
#z_desi                  = hdu_list[1].data["z_desi"]
#zspec_sdss              = hdu_list[1].data["zspec_sdss"]
#areSpectroscopic        = ~numpy.isnan(z_hetdex) | ~numpy.isnan(z_desi) | ~numpy.isnan(zspec_sdss)

hdu_list.close()


# === Create Folders ===
# The following paths are for saving output.
path_excel     = DIR_SAVE / f"{DIR_SAVE.name}.xlsx"
save_plots_loc = DIR_SAVE / "jet_orientation_plots"
save_a_loc     = save_plots_loc / "a"
save_sa_loc    = save_plots_loc / "sa"
save_m_loc     = save_plots_loc / "m"

# --- Creates folders if they do not exist ---
print()
for location in (save_a_loc, save_sa_loc, save_m_loc):
    location.mkdir(parents = True, exist_ok = True)

# --- Create Excel sheet if it does not exist ---
if not os.path.exists(path_excel):
    columns = {
        "right_ascension (deg)"        : [],
        "declination (deg)"            : [],
        "best_angle (deg)"             : [],
        "radius (arcmin)"              : [],
        "number_of_increases"          : [],
        "uncertainty_best_angle (deg)" : [],
        "nan_percentage (%)"           : [],
        "noise_percentage (%)"         : [],
        "adjustment_status"            : [],
        "redshift"                     : []
    }
    # Sets excel sheet columns
    df = pd.DataFrame(columns = columns)
    df.to_excel(path_excel, index = False)


# === Select cutouts ===
suffixes           = ("_sub_masked10.fits", "_sub.fits", ".fits")
cutouts_per_system = {}
# Keep only the most processed cutout of each jet system: masked > subtracted > unaltered.
for filename in sorted(os.listdir(directory_fits)):
    for rank, suffix in enumerate(suffixes):
        if filename.endswith(suffix):
            cutouts_per_system.setdefault(filename.removesuffix(suffix), []).append((rank, filename))
            break
filenames = [min(cutouts)[1] for cutouts in cutouts_per_system.values()]


# === Create plots and update excel ===
if OVERWRITE_FILES:
    print("!!! Overwriting data and files")
else:
    print("!!! Only creating files that are not in folders/are placed in the wrong folder")

counter = 0
for filename in filenames:
    counter += 1

    # === TEST CODE ==========================================================================
    # if not filename.startswith(("6.8218","122.1484","209.9408", "218.23212", "248.3243")):
    # if not filename.startswith(("127.864559","247.016890","348.0052")):
    if not filename.startswith(("134.235")):
        continue
    # ========================================================================================

    print("-" * 60)

    file_path           = os.path.join(directory_fits, filename)

    fits_data           = filename.split("_")
    ra                  = float(fits_data[0])
    dec                 = float(fits_data[1])
    image_width_arcmins = float(fits_data[2][0: fits_data[2].index("A")])

    save_format         = SAVE_FORMAT
    name                = filename[:-5] + "_jet_orientation." + save_format
    
    save_a              = save_a_loc  / name
    save_sa             = save_sa_loc / name
    save_m              = save_m_loc  / name

    adjustment_status   = check_adjustment_status(ra, dec, M_LIST, SA_LIST)

    print("JET", ra, "+", dec, "(", counter, "/", len(filenames), ")")

    if OVERWRITE_FILES or not check_if_file_exists_in_correct_location(adjustment_status, save_a, save_sa, save_m):

        jet_system = JetSystem(
            file_path, ra, dec, image_width_arcmins, adjustment_status, SA_LIST, M_LIST, ANGLES,
            # Data
            go_right_ascensions, go_declinations, go_length_angular_means, go_redshift_means, areSpectroscopic,
            # Thresholds and Parameters
            BEST_ANGLE_THRESHOLD, NAN_PERCENTAGE_CUTOFF,
            # File save locations
            path_excel,
            # Save status
            PIXEL_SHIFT
        )

        if SAVING_PLOTS:
            print("\n> Creating and saving plot... ")
            jet_plotter = JetPlotter(jet_system, save_a, save_sa, save_m, SAVE_FORMAT, axis_font_size)

    print("-" * 60)

format_sheet(path_excel)