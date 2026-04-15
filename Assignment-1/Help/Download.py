# # ===================Index======================

# # import yfinance as yf
# # from pathlib import Path
# # BASE_DIR = Path("IBEX35")
# # INDEX_DIR = BASE_DIR / "index"
# # INDEX_DIR.mkdir(parents=True, exist_ok=True)

# # ibex = yf.download(
# #     "^IBEX",
# #     start="2010-01-01",
# #     end="2025-01-01",
# #     progress=False
# # )

# # ibex.reset_index(inplace=True)
# # ibex.to_csv(
# #     INDEX_DIR / "IBEX35_index.csv",
# #     index=False,
# #     encoding="utf-8"
# # )

# # print("IBEX 35 index data saved as CSV")



# # ===================Constituents======================

# import pandas as pd
# import requests
# from pathlib import Path
# OUTPUT_DIR = Path("data/constituents/biannual")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# WIKI_API = "https://en.wikipedia.org/w/api.php"
# HEADERS = {
#     "User-Agent": "AcademicResearchBot/1.0 (contact: b23406@students.iitmandi.ac.in)"
# }


# def get_revision_before(date_str):
#     params = {
#         "action": "query",
#         "format": "json",
#         "prop": "revisions",
#         "titles": "IBEX_35",
#         "rvlimit": 1,
#         "rvdir": "older",
#         "rvstart": date_str,
#         "rvprop": "ids"
#     }
#     r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
#     r.raise_for_status()
#     pages = r.json()["query"]["pages"]
#     page = next(iter(pages.values()))
#     return page["revisions"][0]["revid"]


# def extract_companies_from_revision(revid):
#     url = f"https://en.wikipedia.org/w/index.php?oldid={revid}"
#     response = requests.get(url, headers=HEADERS, timeout=20)
#     response.raise_for_status()
#     html = response.text
#     tables = pd.read_html(html)
#     for table in tables:
#         if "Company" in table.columns:
#             return table["Company"].dropna().unique().tolist()
#     return []


# def generate_biannual_periods(start_year=2010, end_year=2026):
#     periods = []
#     for year in range(start_year, end_year + 1):
#         periods.append((year, "Jun", f"{year}-06-30T23:59:59Z"))
#         periods.append((year, "Dec", f"{year}-12-31T23:59:59Z"))
#     return periods


# periods = generate_biannual_periods()
# for year, label, date_str in periods:
#     tag = f"{year}_{label}"
#     print(f"Processing {tag} ...")
#     try:
#         revid = get_revision_before(date_str)
#         companies = extract_companies_from_revision(revid)
#         if len(companies) < 30:
#             print(f"⚠ Incomplete data for {tag} ({len(companies)} companies)")
#             continue
#         df = pd.DataFrame({"Company": companies})
#         df.to_csv(
#             OUTPUT_DIR / f"ibex_{tag}.csv",
#             index=False,
#             encoding="utf-8"
#         )
#         print(f"✅ Saved {tag} ({len(companies)} companies)")
#     except Exception as e:
#         print(f"❌ Failed for {tag}: {e}")




# # ===================Master Companies======================


# import pandas as pd
# from pathlib import Path
# SRC_DIR = Path("IBEX35/constituents")
# OUT_FILE = Path("IBEX35/constituents/master_company_list.csv")
# companies = set()
# for f in SRC_DIR.glob("ibex_*.csv"):
#     df = pd.read_csv(f)
#     companies.update(df["Company"].dropna().str.strip())
# df_master = pd.DataFrame(sorted(companies), columns=["Company"])
# df_master.to_csv(OUT_FILE, index=False, encoding="utf-8")
# print(f"Total unique companies: {len(df_master)}")


# # ===================Download======================


# import pandas as pd
# import yfinance as yf
# from pathlib import Path
# import time
# INPUT_FILE = Path("IBEX35/constituents/company_canonical_map.csv")
# OUTPUT_DIR = Path("IBEX35/prices")
# START_DATE = "2010-01-01"
# END_DATE   = "2026-02-01"
# SLEEP_SEC = 1  # polite rate limiting
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# df = pd.read_csv(INPUT_FILE)
# # Keep only tradable entities
# df = df[df["Tradable"].str.upper() == "YES"]
# # Drop duplicates (one download per economic entity)
# df = df.drop_duplicates(subset="CanonicalName")
# print(f"Total tradable companies to download: {len(df)}\n")
# failed = []

# for _, row in df.iterrows():
#     company = row["CanonicalName"]
#     base_ticker = row["BaseTicker"].strip()
#     ticker = f"{base_ticker}"
#     print(f"Downloading: {company} ({ticker})")
#     try:
#         data = yf.download(
#             ticker,
#             start=START_DATE,
#             end=END_DATE,
#             progress=False,
#             auto_adjust=False
#         )
#         if data.empty:
#             print(f"⚠ No data returned for {ticker}\n")
#             failed.append(ticker)
#             continue
#         data.reset_index(inplace=True)
#         data["Company"] = company
#         data["Ticker"] = ticker
#         out_file = OUTPUT_DIR / f"{base_ticker}.csv"
#         data.to_csv(out_file, index=False, encoding="utf-8")
#         print(f"✅ Saved: {out_file.name}\n")
#         time.sleep(SLEEP_SEC)
#     except Exception as e:
#         print(f"❌ Failed for {ticker}: {e}\n")
#         failed.append(ticker)

# print("========== DOWNLOAD SUMMARY ==========")
# print(f"Successful downloads: {len(df) - len(failed)}")
# print(f"Failed downloads: {len(failed)}")
# if failed:
#     print("Failed tickers:")
#     for t in failed:
#         print(" -", t)

# # ===================Excel-Create======================


# import pandas as pd
# from pathlib import Path
# SRC_DIR = Path("IBEX35\constituents")
# OUT_FILE = Path("IBEX35Constituents-10-25.xlsx")
# files = sorted(SRC_DIR.glob("ibex_*.csv"))
# print(f"Found {len(files)} biannual files")

# with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
#     for f in files:
#         df = pd.read_csv(f)

#         # Convert filename → sheet name
#         # ibex_2010_Jun.csv → Jun-2010
#         name = f.stem.replace("ibex_", "")
#         year, period = name.split("_")
#         sheet_name = f"{period}-{year}"

#         df.to_excel(writer, sheet_name=sheet_name, index=False)

#         print(f"Added sheet: {sheet_name}")

# print("\n✅ Excel file created:", OUT_FILE)

# ======================INDEX============================

# import yfinance as yf
# import pandas as pd
# from pathlib import Path

# INDEX_TICKERS = {
#     "NIFTY50": "^NSEI",
#     "SENSEX": "^BSESN",
#     "IBEX35": "^IBEX",
#     "OMXS30": "^OMX",
#     "ATX": "^ATX"
# }

# START_DATE = "2000-01-01"
# END_DATE   = "2026-02-05"

# OUT_DIR = Path("Other")
# OUT_DIR.mkdir(parents=True, exist_ok=True)


# for name, ticker in INDEX_TICKERS.items():
#     print(f"Downloading {name} ({ticker})")
#     try:
#         df = yf.download(
#             ticker,
#             start=START_DATE,
#             end=END_DATE,
#             progress=True,
#             auto_adjust=True
#         )
#         if df.empty:
#             print(f"⚠ No data for {name}")
#             continue
#         df.reset_index(inplace=True)
#         out_file = OUT_DIR / f"{name}.csv"
#         df.to_csv(out_file, index=False, encoding="utf-8")
#         print(f"✅ Saved {out_file.name}\n")
#     except Exception as e:
#         print(f"❌ Failed for {name}: {e}\n")


# ======================INDEX============================


# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt

# INDEX_TICKERS = {
#     "Nifty 50": "^NSEI",
#     "Sensex": "^BSESN",
#     "IBEX 35": "^IBEX",
#     "OMXS 30": "^OMX",
#     "ATX": "^ATX"
# }

# START = "2021-11-30"
# END   = "2022-12-31"

# data = {}

# for name, ticker in INDEX_TICKERS.items():
#     df = yf.download(ticker, start=START, end=END, progress=False)
#     df = df[["Close"]].dropna()
#     df["Norm"] = df["Close"] / df["Close"].iloc[0] * 100
#     df["Drawdown"] = df["Norm"] / df["Norm"].cummax() - 1
#     data[name] = df


# plt.figure(figsize=(14, 6))

# for name, df in data.items():
#     plt.plot(df.index, df["Norm"], label=name)

# plt.axvline(pd.Timestamp("2022-02-24"), linestyle="--", alpha=0.6)
# plt.text(pd.Timestamp("2022-03-01"), 80, "Ukraine War", fontsize=9)

# plt.title("Normalized Index Performance (Base = 100)")
# plt.ylabel("Index Level")
# plt.xlabel("Date")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

# for ax, (name, df) in zip(axes, data.items()):
#     ax.plot(df.index, df["Norm"])
#     ax.axvline(pd.Timestamp("2022-02-24"), linestyle="--", alpha=0.5)
#     ax.set_title(name)
#     ax.grid(alpha=0.3)

# plt.suptitle("Index-wise Performance (Independent View)", fontsize=14)
# plt.tight_layout()
# plt.show()

# dd_df = pd.DataFrame({
#     name: df["Drawdown"] * 100
#     for name, df in data.items()
# })

# plt.figure(figsize=(14, 5))
# plt.imshow(dd_df.T, aspect="auto")

# plt.yticks(range(len(dd_df.columns)), dd_df.columns)
# plt.xticks(
#     range(0, len(dd_df), 120),
#     dd_df.index.strftime("%Y-%m")[::120],
#     rotation=45
# )

# plt.colorbar(label="Drawdown (%)")
# plt.title("Drawdown Heatmap (Red = More Pain)")
# plt.show()


# plt.figure(figsize=(14, 6))

# base = data["Nifty 50"]["Norm"]

# for name, df in data.items():
#     if name != "Nifty 50":
#         relative = df["Norm"] - base
#         plt.plot(relative.index, relative, label=f"{name} vs Nifty")

# plt.axhline(0, linestyle="--", alpha=0.6)
# plt.title("Relative Performance vs Nifty 50")
# plt.ylabel("Relative Index Level")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# ==================Animate Slowdown======================

# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# INDEX_TICKERS = {
#     "Nifty 50": "^NSEI",
#     "Sensex": "^BSESN",
#     "IBEX 35": "^IBEX",
#     "OMXS 30": "^OMX",
#     "ATX": "^ATX"
# }
# START = "2021-10-30"
# END   = "2023-01-31"
# series = {}
# for name, ticker in INDEX_TICKERS.items():
#     df = yf.download(ticker, start=START, end=END, progress=False)
#     df = df[["Close"]].dropna()
#     df["Norm"] = df["Close"] / df["Close"].iloc[0] * 100
#     series[name] = df["Norm"]

# data = pd.DataFrame(series)
# fig, ax = plt.subplots(figsize=(12, 6))
# lines = {}
# for col in data.columns:
#     line, = ax.plot([], [], label=col)
#     lines[col] = line

# ax.set_xlim(data.index.min(), data.index.max())
# ax.set_ylim(data.min().min() * 0.9, data.max().max() * 1.1)
# ax.set_title("Animated Global Slowdown (2021-2023)")
# ax.set_ylabel("Normalized Index Level")
# ax.legend()
# ax.grid(alpha=0.3)

# def update(frame):
#     for col in data.columns:
#         lines[col].set_data(data.index[:frame], data[col].iloc[:frame])
#     return lines.values()

# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(data),
#     interval=20,
#     blit=False
# )
# plt.show()


# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt


# INDEX_TICKERS = {
#     "Nifty 50": "^NSEI",
#     "Sensex": "^BSESN",
#     "IBEX 35": "^IBEX",
#     "OMXS 30": "^OMX",
#     "ATX": "^ATX"
# }

# START = "2021-12-01"
# END   = "2023-01-01"

# data = {}

# for name, ticker in INDEX_TICKERS.items():
#     df = yf.download(ticker, start=START, end=END, progress=False)
#     df = df[["Close"]].dropna()
#     df["Norm"] = df["Close"] / df["Close"].iloc[0] * 100
#     df["Drawdown"] = df["Norm"] / df["Norm"].cummax() - 1
#     data[name] = df


# fig = plt.figure(figsize=(14, 10))
# gs = fig.add_gridspec(2, 2)

# ax1 = fig.add_subplot(gs[0, :])
# for name, df in data.items():
#     ax1.plot(df.index, df["Norm"], label=name)

# ax1.axvline(pd.Timestamp("2022-02-24"), linestyle="--", alpha=0.6)
# ax1.set_title("A. Normalized Index Performance (Base = 100)")
# ax1.legend()
# ax1.grid(alpha=0.3)

# ax2 = fig.add_subplot(gs[1, 0])
# for name, df in data.items():
#     ax2.plot(df.index, df["Drawdown"] * 100, label=name)

# ax2.set_title("B. Drawdowns (%)")
# ax2.set_ylabel("Drawdown (%)")
# ax2.grid(alpha=0.3)

# ax3 = fig.add_subplot(gs[1, 1])
# base = data["Nifty 50"]["Norm"]
# for name, df in data.items():
#     if name != "Nifty 50":
#         # ax3.plot(df.index, df["Norm"] - base, label=name)
#         aligned = df["Norm"].align(base, join="inner")
#         rel_perf = aligned[0] - aligned[1]
#         ax3.plot(rel_perf.index, rel_perf, label=name)

# ax3.axhline(0, linestyle="--", alpha=0.6)
# ax3.set_title("C. Relative Performance vs Nifty 50")
# ax3.legend()
# ax3.grid(alpha=0.3)

# plt.suptitle("Dashboard: 2022 Global Slowdown Across Indices", fontsize=14)
# plt.tight_layout()
# plt.show()



# =========================================================================
# =========================================================================
# ===================================Clean==================================
# =========================================================================
# =========================================================================


# # import pandas as pd
# # from pathlib import Path

# # PRICE_DIR = Path("IBEX35/prices")
# # OUT_FILE = Path("IBEX35/Master_Price_Database.csv")

# # all_dfs = []

# # for f in PRICE_DIR.glob("*.csv"):
# #     df = pd.read_csv(f)
# #     df["SourceFile"] = f.name
# #     all_dfs.append(df)

# # master_df = pd.concat(all_dfs, ignore_index=True)

# # master_df.to_csv(OUT_FILE, index=False)

# # print("Master price database created")
# # print("Total rows:", len(master_df))
# # print("Total stocks:", master_df["Company"].nunique())

# import pandas as pd
# from pathlib import Path
# import shutil

# def remove_columns_inplace(folder_path, backup=True):
#     """
#     Remove 'Company' and 'Ticker' columns from CSV files,
#     optionally creating backups
#     """
#     folder = Path(folder_path)
#     csv_files = list(folder.glob("*.csv"))
    
#     if not csv_files:
#         print("No CSV files found!")
#         return
    
#     for csv_file in csv_files:
#         try:
#             # Create backup if requested
#             if backup:
#                 backup_file = folder / f"{csv_file.stem}_backup.csv"
#                 shutil.copy2(csv_file, backup_file)
            
#             # Read and process
#             df = pd.read_csv(csv_file)
            
#             # Check and remove columns
#             removed_cols = []
#             for col in ['TickerUsed']:
#                 if col in df.columns:
#                     df.drop(columns=[col], inplace=True)
#                     removed_cols.append(col)
            
#             # Save back to original file
#             df.to_csv(csv_file, index=False)
            
#             if removed_cols:
#                 print(f"✓ {csv_file.name}: Removed {removed_cols}")
#             else:
#                 print(f"ⓘ {csv_file.name}: No columns to remove")
                
#         except Exception as e:
#             print(f"✗ {csv_file.name}: Error - {str(e)}")

# # Usage
# remove_columns_inplace("IBEX35\prices", backup=False)

# import pandas as pd
# from pathlib import Path

# def extract_headers_to_file(folder_path, output_file="all_headers.txt"):
#     """
#     Extract headers from all CSV files in a folder and save to a text file
#     """
#     folder = Path(folder_path)
#     csv_files = list(folder.glob("*.csv"))
    
#     if not csv_files:
#         print(f"No CSV files found in {folder_path}")
#         return
    
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for csv_file in csv_files:
#             try:
#                 # Read just the header (first row)
#                 df = pd.read_csv(csv_file, nrows=0)
#                 headers = list(df.columns)
                
#                 # Write to file
#                 f.write(f"=== File: {csv_file.name} ===\n")
#                 f.write(f"Number of columns: {len(headers)}\n")
#                 f.write(f"Headers: {headers}\n")
#                 f.write("-" * 50 + "\n\n")
                
#             except Exception as e:
#                 f.write(f"=== File: {csv_file.name} ===\n")
#                 f.write(f"ERROR: {str(e)}\n")
#                 f.write("-" * 50 + "\n\n")
    
#     print(f"Headers saved to {output_file}")
#     print(f"Processed {len(csv_files)} files")

# # Usage
# extract_headers_to_file("IBEX35\prices")


# import pandas as pd
# from pathlib import Path

# def rename_column_in_folder(folder_path, old_name='Price', new_name='Close'):
#     """
#     Rename specific column in all CSV files in a folder
#     """
#     folder = Path(folder_path)
    
#     for csv_file in folder.glob("*.csv"):
#         try:
#             # Read CSV
#             df = pd.read_csv(csv_file)
            
#             # Check if column exists
#             if old_name in df.columns:
#                 # Rename the column
#                 df = df.rename(columns={old_name: new_name})
                
#                 # Save back
#                 df.to_csv(csv_file, index=False)
#                 print(f"✓ {csv_file.name}: Renamed '{old_name}' to '{new_name}'")
#             else:
#                 print(f"ⓘ {csv_file.name}: Column '{old_name}' not found")
                
#         except Exception as e:
#             print(f"✗ {csv_file.name}: Error - {str(e)}")

# # Usage
# rename_column_in_folder("IBEX35/prices")

# import pandas as pd
# from pathlib import Path
# import numpy as np

# def smart_merge_close_columns(folder_path):
#     """
#     Smart merge of Close and Adj Close with detailed reporting
#     """
#     folder = Path(folder_path)
    
#     for csv_file in folder.glob("*.csv"):
#         try:
#             df = pd.read_csv(csv_file)
            
#             # Check what columns we have
#             has_close = 'Close' in df.columns
#             has_adj_close = 'Adj Close' in df.columns
            
#             if has_close and has_adj_close:
#                 print(f"\n📊 {csv_file.name}:")
#                 print(f"   {'='*40}")
                
#                 # Calculate statistics
#                 diff = df['Adj Close'] - df['Close']
#                 max_diff = diff.abs().max()
#                 mean_diff = diff.abs().mean()
#                 num_different = (diff != 0).sum()
#                 total_rows = len(df)
                
#                 print(f"   Rows with Close ≠ Adj Close: {num_different}/{total_rows} ({num_different/total_rows*100:.1f}%)")
#                 print(f"   Maximum absolute difference: {max_diff:.4f}")
#                 print(f"   Mean absolute difference: {mean_diff:.4f}")
                
#                 if num_different > 0:
#                     print(f"\n   ⚠️  Differences found! Using 'Adj Close' values (adjusted for splits/dividends)")
#                     # Create a copy of original Close for comparison
#                     original_close = df['Close'].copy()
                    
#                     # Replace Close with Adj Close values
#                     df['Close'] = df['Adj Close']
                    
#                     # Show sample of differences
#                     diff_indices = diff[diff != 0].index[:3]  # First 3 differences
#                     for idx in diff_indices:
#                         print(f"      Row {idx}: Close={original_close[idx]:.4f}, Adj Close={df.loc[idx, 'Adj Close']:.4f}")
#                 else:
#                     print(f"\n   ✓ No differences - columns are identical")
                
#                 # Drop Adj Close column
#                 df = df.drop(columns=['Adj Close'])
                
#                 # Save
#                 df.to_csv(csv_file, index=False)
#                 print(f"\n   ✅ Merged and saved")
#                 print(f"   Final columns: {list(df.columns)}")
                
#             elif has_adj_close and not has_close:
#                 # Rename Adj Close to Close
#                 df = df.rename(columns={'Adj Close': 'Close'})
#                 df.to_csv(csv_file, index=False)
#                 print(f"✓ {csv_file.name}: Only 'Adj Close' → renamed to 'Close'")
                
#         except Exception as e:
#             print(f"❌ {csv_file.name}: ERROR - {str(e)}")

# smart_merge_close_columns("IBEX35/prices")


# =========================================================================
# =========================================================================
# =====================================================================
# =========================================================================
# =========================================================================











