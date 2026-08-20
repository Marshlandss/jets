# === Imports ===

import numpy
import os
import pandas as pd
from astropy.io import fits
from pathlib import Path

from src.jets.jet_system import JetSystem
from src.jets.manual_adjustments import manual_list, semi_automatic_list
from src.jets.config import (
    ANGLES,
    BEST_ANGLE_THRESHOLD,
    NAN_PERCENTAGE_CUTOFF,
    PIXEL_SHIFT,
    SAVING_PLOTS,
    SAVE_FORMAT
)
from src.jets.jet_utils import format_sheet
from src.jets.plot_styles import set_plot_styles

# === Plot Formatting ===
axis_font_size = set_plot_styles()

# === Load data ===

# User should edit these pathes -----------------------------------------------------------
# --- For Martijn's Catalogue ---
GGO_CATALOGUE_DIR           = "G:/My Drive/Internships/caltech/astro_project/analysis/Mpc/jet_orientation_Mpc/data/GGO_catalogue_2025_02_with3C236.fits"
DIR_MAP_FITS                = "G:/My Drive/Internships/caltech/astro_project/analysis/Mpc/jet_orientation_Mpc/data/fits/"
SAVE_DIR                    = "G:/My Drive/Internships/caltech/astro_project/analysis/Mpc/jet_orientation_Mpc/results/with_length_angular_means/"

# --- For Hardcastle's Catalogue ---
# GGO_CATALOGUE_DIR           = "G:/My Drive/Internships/caltech/astro_project/analysis/kpc/jet_orientation_kpc/data/agn-v1.1.fits"
# DIR_MAP_FITS                = "G:/My Drive/Internships/caltech/astro_project/analysis/kpc/jet_orientation_kpc/data/cleaned_fits_without_dups/"
# SAVE_DIR                    = "G:/My Drive/Internships/caltech/astro_project/analysis/kpc/jet_orientation_kpc/results/with_length_angular_means/"
# ------------------------------------------------------------------------------------------

# --- Data from `GGO_CATALOGUE_DIR` ---
hdu_list                    = fits.open(GGO_CATALOGUE_DIR)

# --- For Martijn's Catalogue ---
ggo_right_ascensions        = hdu_list[1].data["host_right_ascension_(deg)"]           # in deg
ggo_declinations            = hdu_list[1].data["host_declination_(deg)"]               # in deg
ggo_length_angular_means    = hdu_list[1].data["outflow_length_angular_mean_(arcmin)"] # in arcmin
ggo_redshift_means          = hdu_list[1].data["host_redshift_mean_(1)"]
ggo_redshift_stdev        = hdu_list[1].data["host_redshift_SD_(1)"]
areSpectroscopic          = (ggo_redshift_stdev < 0.001)

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
excel_file              = SAVE_DIR + SAVE_DIR.split('/')[-2] + ".xlsx"
save_plots_loc          = SAVE_DIR + "jet_orientation_plots/"
save_auto_loc           = SAVE_DIR + "auto/"
save_manual_loc         = SAVE_DIR + "manual/"
save_semi_automatic_loc    = SAVE_DIR + "some_semi_automatic/"

# --- Creates folders if they do not exist ---
if not os.path.exists(save_plots_loc):

    print("Creating folders...")

    locations = [save_auto_loc, save_manual_loc, save_semi_automatic_loc]

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

counter = 0
for filename in filenames:

    counter += 1

    # === TEST CODE ==========================================================================
    if not filename.startswith(("6.8218","122.1484","209.9408", "218.23212", "248.3243")):
        continue
    # ========================================================================================

    print(counter, "/", len(filenames), "-" * 50)

    file_path = os.path.join(DIR_MAP_FITS, filename)
    print('"' + filename + '"')

    jet_system = JetSystem(
        file_path, ANGLES,
        # Data
        ggo_right_ascensions, ggo_declinations, ggo_length_angular_means, ggo_redshift_means, areSpectroscopic,
        # Thresholds and Parameters
        BEST_ANGLE_THRESHOLD, NAN_PERCENTAGE_CUTOFF,
        # File save locations
        excel_file, save_auto_loc, save_manual_loc, save_semi_automatic_loc,
        # Save status
        SAVING_PLOTS, PIXEL_SHIFT, SAVE_FORMAT,
        # Lists
        manual_list, semi_automatic_list,
        # Plot style
        axis_font_size
    )
    print("-" * 60)

format_sheet(excel_file)