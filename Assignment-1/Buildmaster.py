


# import pandas as pd
# from pathlib import Path

# PRICE_DIR = Path("IBEX35/prices")
# OUT_FILE = Path("Master_Price_Database.csv")

# all_dfs = []

# for f in PRICE_DIR.glob("*.csv"):
#     df = pd.read_csv(f)
#     df["Ticker"] = f.name
#     all_dfs.append(df)

# master_df = pd.concat(all_dfs, ignore_index=True)

# master_df.to_csv(OUT_FILE, index=False)

# print("Master price database created")
# print("Total rows:", len(master_df))
# print("Total stocks:", master_df["Ticker"].nunique())






import pandas as pd
from pathlib import Path

# ===== CONFIG =====
CSV_DIR = Path("IBEX35/results/2010_6_to_2025_12")   # folder with CSVs
OUTPUT_EXCEL = Path("IBEX35/master_restricted_data.xlsx")

# Excel sheet name limit
MAX_SHEET_NAME_LEN = 31

# ===== CREATE MASTER EXCEL =====
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    for csv_file in CSV_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)

        # Sheet name = CSV file name (without extension)
        sheet_name = csv_file.stem[:MAX_SHEET_NAME_LEN]

        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        print(f"✓ Added sheet: {sheet_name}")

print(f"\nMaster Excel created at: {OUTPUT_EXCEL.resolve()}")

