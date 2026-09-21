# Imports: standard library
import os
# Imports: third-party
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd


def format_sheet(excel_location):
    """
    Sort the catalogue at 'excel_location' by right ascension and widen each column to fit its widest cell.
    Rewrites the file in place through the 'xlsxwriter' engine, so any formatting the file already carried is lost.

    Parameters
    ----------
    excel_location : path to an Excel catalogue with a 'right_ascension (deg)' column
    """
    df = pd.read_excel(excel_location)

    # Sort Excel sheet by x index of central voxel and adjusts column size to accommodate data
    df_updated = df.sort_values(by = "right_ascension (deg)", ascending = True)

    # Adjust width of Excel columns based on the widest cell
    with pd.ExcelWriter(excel_location, engine = "xlsxwriter") as writer:
        df_updated.to_excel(writer, sheet_name = "Sheet1", index = False)

        worksheet = writer.sheets["Sheet1"]

        for col_idx, col in enumerate(df_updated.columns):
            max_length = max(df_updated[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(col_idx, col_idx, max_length)


def check_adjustment_status(ra, dec, m_list, sa_list):
    """
    Check what adjustment status is being applied based on the manually inputted lists:
    > manual and semi_automatic lists
    """

    status_mapping = {
        "m"       : m_list, # manual
        "sa"      : sa_list, # semi-automatic
    }

    for status, coord_list in status_mapping.items():
        for entry in coord_list:
            right_ascension, declination = map(float, entry[0].split("_"))

            if ra == right_ascension and dec == declination:
                return status

    return "a" # automatic


def check_if_file_exists_in_correct_location(adjustment_status, a_loc, sa_loc, m_loc):
    """
    Identifies if the current jet system has already been saved in to one of the folders.
    Checks if its adjustment_status aligns with the folder it has been saved to.
    """

    mapping = {"a" : a_loc, "sa" : sa_loc, "m" : m_loc}

    for status, save_location in mapping.items():
        if os.path.exists(save_location) :
            print("> File exists in location:", save_location)
            if adjustment_status == status :
                # File is saved to the correct folder
                print("> File exists in correct location.")
                return True
            else:
                # File exists in a folder, however not the correct folder
                print("> File does not exist in correct location... Removing files...")
                os.remove(save_location)
                return False

    # File does not exist in any folders
    print("> File does not exist yet.")
    return False


def defineJetSystemNames(rightAscensions, declinations):
    """
    Designate jet systems by their host galaxy coordinates, in the format 'JHHMMSS+DDMMSS', rounded to whole seconds.

    Host galaxy coordinates affect the inferred jet orientation, so a change in them should show in the designation:
    A change of an arcsecond or more in declination always does. A second of time in right ascension, however, spans
    15" on the sky at declination 0 deg (and 15" * cos(declination) in general), so smaller right ascension changes may not.

    Parameters
    ----------
    rightAscensions : array_like
        Right ascensions of the host galaxies; in deg.
    declinations : array_like
        Declinations of the host galaxies; in deg.

    Returns
    -------
    np.ndarray of str
        Designations, one per jet system. Raises a ValueError if two jet systems receive the same designation.
    """
    coordinates = SkyCoord(rightAscensions, declinations, unit = "deg")
    names       = np.char.add(np.char.add("J", coordinates.ra.to_string(unit = "hourangle", sep = "", precision = 0, pad = True)),
                              coordinates.dec.to_string(sep = "", precision = 0, pad = True, alwayssign = True))
    if len(set(names)) < len(names):
        raise ValueError("Two jet systems share a designation.")
    return names


def saveJetSystemNames(pathExcel):
    """
    Give the catalogue at 'pathExcel' a first column 'name' with jet system designations, replacing any existing one.
    """
    dataFrame = pd.read_excel(pathExcel)
    names     = defineJetSystemNames(dataFrame["right_ascension (deg)"], dataFrame["declination (deg)"])
    dataFrame = dataFrame.drop(columns = "name", errors = "ignore")
    dataFrame.insert(0, "name", names)
    dataFrame.to_excel(pathExcel, index = False)


def loadJetSystemNames(pathExcel):
    """
    Load the jet system designations from the catalogue at 'pathExcel'.
    """
    return pd.read_excel(pathExcel)["name"].to_numpy(dtype = str)
