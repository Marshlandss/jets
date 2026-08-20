
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