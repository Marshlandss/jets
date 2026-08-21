# === Imports ===

import numpy
import os
import pandas as pd
from astropy.io import fits
from pathlib import Path

from src.jets.jet_system import JetSystem
from src.jets.jet_plotter import JetPlotter
from src.jets.manual_adjustments import M_LIST, SA_LIST
from src.jets.config import (
    ANGLES,
    BEST_ANGLE_THRESHOLD,
    NAN_PERCENTAGE_CUTOFF,
    PIXEL_SHIFT,
    SAVE_DIR,
    SAVING_PLOTS,
    SAVE_FORMAT, 
    OVERWRITE_FILES
)
from src.jets.jet_utils import format_sheet, check_if_file_exists_in_correct_location, check_adjustment_status
from src.jets.plot_styles import set_plot_styles

# === Plot Formatting ===
axis_font_size = set_plot_styles()

# === Load data ===

# User should edit these pathes -----------------------------------------------------------
# --- For Martijn's Catalogue ---
GGO_CATALOGUE_DIR           = "G:/My Drive/Internships/caltech/astro_project/analysis/Mpc/jet_orientation_Mpc/data/GGO_catalogue_2025_02_with3C236.fits"
DIR_MAP_FITS                = "G:/My Drive/Internships/caltech/astro_project/analysis/Mpc/jet_orientation_Mpc/data/fits/"

# --- For Hardcastle's Catalogue ---
# GGO_CATALOGUE_DIR           = "G:/My Drive/Internships/caltech/astro_project/analysis/kpc/jet_orientation_kpc/data/agn-v1.1.fits"
# DIR_MAP_FITS                = "G:/My Drive/Internships/caltech/astro_project/analysis/kpc/jet_orientation_kpc/data/cleaned_fits_without_dups/"
# ------------------------------------------------------------------------------------------

# --- Data from `GGO_CATALOGUE_DIR` ---
hdu_list                    = fits.open(GGO_CATALOGUE_DIR)

# --- For Martijn's Catalogue ---
ggo_right_ascensions        = hdu_list[1].data["host_right_ascension_(deg)"]           # in deg
ggo_declinations            = hdu_list[1].data["host_declination_(deg)"]               # in deg
ggo_length_angular_means    = hdu_list[1].data["outflow_length_angular_mean_(arcmin)"] # in arcmin
ggo_redshift_means          = hdu_list[1].data["host_redshift_mean_(1)"]
ggo_redshift_stdev          = hdu_list[1].data["host_redshift_SD_(1)"]
areSpectroscopic            = (ggo_redshift_stdev < 0.001)

# --- For Hardcastle's Catalogue ---
# ggo_right_ascensions        = hdu_list[1].data["optRA"]     # in deg
# ggo_declinations            = hdu_list[1].data["optDec"]    # in deg
# ggo_length_angular_means    = hdu_list[1].data['LAS'] / 60  # in arcmin
# ggo_redshift_means          = hdu_list[1].data['z_best']
# z_hetdex                    = hdu_list[1].data["z_hetdex"]
# z_desi                      = hdu_list[1].data["z_desi"]
# zspec_sdss                  = hdu_list[1].data["zspec_sdss"]
# areSpectroscopic            = ~numpy.isnan(z_hetdex) | ~numpy.isnan(z_desi) | ~numpy.isnan(zspec_sdss)   # For Hardcastle's Catalogue

hdu_list.close()

# === Create Folders ===

# --- Save Directories ---
excel_file                  = SAVE_DIR + SAVE_DIR.split('/')[-2] + ".xlsx"
save_plots_loc              = SAVE_DIR + "jet_orientation_plots/"
save_a_loc                  = save_plots_loc + "a/"
save_sa_loc                 = save_plots_loc + "sa/"
save_m_loc                  = save_plots_loc + "m/"

# --- Creates folders if they do not exist ---
print()
if not os.path.exists(save_plots_loc):

    print("> Creating folders...")

    locations = [save_a_loc, save_sa_loc, save_m_loc]

    for location in locations:

        path = Path(location)
        path.mkdir(parents = True, exist_ok = True)

# --- Creates Excel sheet if it does not exist ---
if not os.path.exists(excel_file):
    columns = {
        "right_ascension (deg)"         : [],
        "declination (deg)"             : [],
        "best_angle (deg)"              : [],
        "radius (arcmin)"               : [],
        "number_of_increases"           : [],
        "uncertainty_best_angle (deg)"  : [],
        "nan_percentage (%)"            : [],
        "noise_percentage (%)"          : [],
        "adjustment_status"             : [],
        "redshift"                      : []
    }
    # Sets excel sheet columns
    df = pd.DataFrame(columns = columns)
    df.to_excel(excel_file, index = False)

# === Selects cutouts ===

# Removes unaltered cutout if a subtracted version is found; only one of cutout of each jet system is left in the folder
filenames = os.listdir(DIR_MAP_FITS)
for filename in filenames.copy():
    if os.path.isfile(DIR_MAP_FITS + filename[slice(0,filename.index(".fits"))] + "_sub.fits"):
        if os.path.isfile(DIR_MAP_FITS + filename[slice(0,filename.index(".fits"))] + "_sub_masked10.fits"):
            filenames.remove(filename[slice(0,filename.index(".fits"))] + "_sub.fits")
        filenames.remove(filename)

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

    file_path = os.path.join(DIR_MAP_FITS, filename)

    fits_data               = filename.split("_")
    ra                      = float(fits_data[0])
    dec                     = float(fits_data[1])
    image_width_arcmins     = float(fits_data[2][0: fits_data[2].index("A")])

    save_format             = SAVE_FORMAT
    name                    = filename[:-5] + "_jet_orientation." + save_format
    
    save_a                  = save_a_loc + name
    save_sa                 = save_sa_loc + name
    save_m                  = save_m_loc + name

    adjustment_status = check_adjustment_status(ra, dec, M_LIST, SA_LIST)

    print("JET", ra, "+", dec, "(", counter, "/", len(filenames), ")")

    if OVERWRITE_FILES or not check_if_file_exists_in_correct_location(adjustment_status, save_a, save_sa, save_m):

        jet_system = JetSystem(
            file_path, ra, dec, image_width_arcmins, adjustment_status, SA_LIST, M_LIST, ANGLES,
            # Data
            ggo_right_ascensions, ggo_declinations, ggo_length_angular_means, ggo_redshift_means, areSpectroscopic,
            # Thresholds and Parameters
            BEST_ANGLE_THRESHOLD, NAN_PERCENTAGE_CUTOFF,
            # File save locations
            excel_file,
            # Save status
            PIXEL_SHIFT
        )

        if SAVING_PLOTS:
            print("\n> Creating and saving plot... ")
            jet_plotter = JetPlotter(jet_system, save_a, save_sa, save_m, SAVE_FORMAT, axis_font_size)

    print("-" * 60)

format_sheet(excel_file)