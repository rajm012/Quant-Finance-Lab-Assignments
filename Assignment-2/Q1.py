
"""
1. Data Verification and Cleaning
    a) Check for missing values
    b) Check for duplicate records
    c) Ensure data is properly ordered by date
    d) Verify price adjustments for splits/dividends
    e) Check continuity of trading days

"""

# source ass2/bin/activate

import pandas as pd
from pathlib import Path

DATADIR = Path("IBEX35/results/2010_6_to_2025_12")

dfs = []
for f in DATADIR.glob("*.csv"):
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Stock"] = f.stem
    dfs.append(df)

data = pd.concat(dfs)
# print(len(dfs))   # 19

# a) Missing
Nulls = data.isna().sum()
# print(Nulls)
"""
Date       0
Close      0
High       0
Low        0
Open       0
Volume     0
Company    0
Ticker     0
Stock      0
dtype: int64
"""

# b) duplicate
Duplicates = data.duplicated().sum()
# print(Duplicates)
"""
0
"""

# c) date ordered
data = data.sort_values(["Stock","Date"])
# print(data)

"""
           Date      Close        ...  ...       ...         ...     ...         ...
3981 2025-12-18   3.480000   3.503000   3.453000  ...  27606425  Telefónica  TEF.MC  Telefónica
3982 2025-12-19   3.420000   3.467000   3.395000  ...  56409660  Telefónica  TEF.MC  Telefónica
3983 2025-12-22   3.419000   3.423000   3.368000  ...  19546829  Telefónica  TEF.MC  Telefónica
3984 2025-12-23   3.436000   3.468000   3.402000  ...  12480301  Telefónica  TEF.MC  Telefónica
3985 2025-12-24   3.445000   3.459000          High        Low  ...    Volume     Company  Ticker       Stock
0    2010-06-01  11.920112  30.179079  29.081839  ...   1052145         ACS  ACS.MC         ACS
1    2010-06-02  11.863087  30.154367  29.170803  ...    618915         ACS  ACS.MC         ACS
2    2010-06-03  11.900447  30.485518  29.778734  ...    787966         ACS  ACS.MC         ACS
3    2010-06-04  11.597628  30.099998  28.716091  ...   1263824         ACS  ACS.MC         ACS
4    2010-06-07  11.499309  29.576092  28.552988  ...    910833         ACS  ACS.MC         ACS
...         ...        ...        ...3.410000  ...   4205101  Telefónica  TEF.MC  Telefónica

[75315 rows x 9 columns]
"""

# d) price adjustment by split/dividends

# splits = pd.read_csv("SplitsDivCombo.csv")
# splits["Ex-Date"] = pd.to_datetime(splits["Ex-Date"])
# StocksData = data["Stock"].unique()
# data["Return"] = data.groupby("Stock")["Close"].pct_change()
# CurrSplits = 0
# anomalies = 0

# for i, row in splits[splits["Event Type"] == "Stock Split"].iterrows():
#     stock_name = None
#     for s in StocksData:
#         if s.lower() in row['Company'].lower() or row['Company'].lower() in s.lower():
#             stock_name = s; break 
#     if not stock_name:
#         if 'IAG' in StocksData and 'Airlines Group' in row['Company']: stock_name = 'IAG'
    
#     if not stock_name: continue
#     eventDate = row['Ex-Date']
#     stockData = data[data["Stock"] == stock_name]
    
#     dayData = stockData[stockData["Date"] >= eventDate].head(1)
#     if dayData.empty: continue
#     ret = dayData.iloc[0]["Return"]
#     if pd.isna(ret): continue
#     CurrSplits += 1
#     if ret < -0.2:
#         anomalies += 1

# print(f"d) Verified {CurrSplits} stock splits across the dataset.")
# if anomalies == 0:
#     print(f"   Justification: 0 anomalies found. On the {CurrSplits} split ex-dates, no stock exhibited an artificial massive drop (e.g. -80% for a 5:1 split). This confirms the dataset's Close prices are fully adjusted for splits and dividends.")
# else:
#     print(f" Found {anomalies} unadjusted split dates.")


# e) continuity 
Cts = data.groupby("Stock")["Date"].diff().describe()
# print(Cts)

"""
count                     75296
mean     1 days 10:14:21.368465
std      0 days 20:18:55.691206
min             1 days 00:00:00
25%             1 days 00:00:00
50%             1 days 00:00:00
75%             1 days 00:00:00
max             5 days 00:00:00
Name: Date, dtype: object
"""

