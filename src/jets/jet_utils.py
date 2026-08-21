import os
import pandas as pd

def format_sheet(excel_location):
    """

    """
    df = pd.read_excel(excel_location)

    # Sorts Excel sheet by x index of central voxel and adjusts column size to accommodate data
    df_updated = df.sort_values(by="right_ascension (deg)", ascending=True)

    # Adjusts width of Excel columns based on the widest cell
    with pd.ExcelWriter(excel_location, engine="xlsxwriter") as writer:
        df_updated.to_excel(writer, sheet_name="Sheet1", index=False)

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

    mapping = {
        "a"                 : a_loc,
        "sa"                : sa_loc,
        "m"                 : m_loc,
    }

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