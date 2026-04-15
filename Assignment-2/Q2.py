

"""
2. Exploratory Data Analysis (EDA)
    a) Plot price time series for all selected stocks
    b) Compute daily returns
    c) Plot return time series
    d) Compute descriptive statistics (mean, standard deviation, skewness, kurtosis)

"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

os.makedirs("Q2-Results/Stocks/", exist_ok=True)
os.makedirs("Q2-Results/Returns/", exist_ok=True)

def load_data():
    DATADIR = Path("IBEX35/results/2010_6_to_2025_12")

    dfs = []
    for f in DATADIR.glob("*.csv"):
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Stock"] = f.stem
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["Stock","Date"]).reset_index(drop=True)

    data["Return"] = data.groupby("Stock")["Close"].pct_change()
    data = data.dropna(subset=["Return"])

    return data

if __name__ == "__main__":
    
    data = load_data()
    
    # a) Plot price time series for all selected stocks
    for stock in data["Stock"].unique():
        subset = data[data["Stock"] == stock]
        plt.figure()
        plt.plot(subset["Date"], subset["Close"], label=stock)
        plt.legend()
        plt.title(f"{stock} Stock Price Time Series")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.savefig(f"Q2-Results/Stocks/{stock}Stock.png", dpi=300, bbox_inches="tight")
        plt.close()  # close to free memory
    

    # b) daily return
    print(data)


    # c) Plot return time series
    for stock in data["Stock"].unique():
        subset = data[data["Stock"] == stock]
        plt.figure()
        plt.plot(subset["Date"], subset["Return"], label=stock)
        plt.legend()
        plt.title(f"{stock} Daily Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.savefig(f"Q2-Results/Returns/{stock}DailyReturns.png", dpi=300, bbox_inches="tight")
        plt.close()


    # d) Compute descriptive stats (mean, std, skew, kurt)
    stats = data.groupby("Stock")["Return"].agg(
        Mean="mean",
        Std="std",
        Skewness="skew",
        Kurtosis="kurt",
        Min="min",
        Max="max"
    )

    with open("Q2-Results/stats.txt", "w") as f:
        f.write(stats.to_string())
        
    import numpy as np

    tailStats = []

    for stock in data["Stock"].unique():
        returns = data[data["Stock"] == stock]["Return"]
        mu = returns.mean()
        sigma = returns.std()
        n = len(returns)
        # Count extreme events
        be3 = np.sum(np.abs(returns - mu) > 3 * sigma)
        be4 = np.sum(np.abs(returns - mu) > 4 * sigma)
        be5 = np.sum(np.abs(returns - mu) > 5 * sigma)
        # Expected under normal
        exp3 = n * 0.0027
        exp4 = n * 0.00006
        exp5 = n * 0.0000006
        tailStats.append({
            "Stock": stock,
            "N": n,
            "Obs >3s": be3,
            "Exp >3s": exp3,
            "Obs >4s": be4,
            "Exp >4s": exp4,
            "Obs >5s": be5,
            "Exp >5s": exp5,
        })

    tailDF = pd.DataFrame(tailStats)
    tailDF.to_csv("Q2-Results/tailEvents.csv", index=False)
    print(tailDF)

    print("Done")



    """
    (ass2) rajm012@rajm012:~/Desktop/E-Drive/1. Semester 6th/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-2$ python Q2.py 
            Date      Close       High        Low       Open    Volume     Company  Ticker       Stock    Return
    0    2010-06-01  11.920112  30.179079  29.081839  30.075287   1052145         ACS  ACS.MC         ACS       NaN
    1    2010-06-02  11.863087  30.154367  29.170803  29.976437    618915         ACS  ACS.MC         ACS -0.004784
    2    2010-06-03  11.900447  30.485518  29.778734  30.258160    787966         ACS  ACS.MC         ACS  0.003149
    3    2010-06-04  11.597628  30.099998  28.716091  29.976437   1263824         ACS  ACS.MC         ACS -0.025446
    4    2010-06-07  11.499309  29.576092  28.552988  28.903908    910833         ACS  ACS.MC         ACS -0.008478
    ...         ...        ...        ...        ...        ...       ...         ...     ...         ...       ...
    3981 2025-12-18   3.480000   3.503000   3.453000   3.490000  27606425  Telefónica  TEF.MC  Telefónica -0.002865
    3982 2025-12-19   3.420000   3.467000   3.395000   3.440000  56409660  Telefónica  TEF.MC  Telefónica -0.017241
    3983 2025-12-22   3.419000   3.423000   3.368000   3.400000  19546829  Telefónica  TEF.MC  Telefónica -0.000292
    3984 2025-12-23   3.436000   3.468000   3.402000   3.402000  12480301  Telefónica  TEF.MC  Telefónica  0.004972
    3985 2025-12-24   3.445000   3.459000   3.410000   3.424000   4205101  Telefónica  TEF.MC  Telefónica  0.002619
    [75315 rows x 10 columns]
            Date      Close       High        Low       Open    Volume     Company  Ticker       Stock    Return
    1    2010-06-02  11.863087  30.154367  29.170803  29.976437    618915         ACS  ACS.MC         ACS -0.004784
    2    2010-06-03  11.900447  30.485518  29.778734  30.258160    787966         ACS  ACS.MC         ACS  0.003149
    3    2010-06-04  11.597628  30.099998  28.716091  29.976437   1263824         ACS  ACS.MC         ACS -0.025446
    4    2010-06-07  11.499309  29.576092  28.552988  28.903908    910833         ACS  ACS.MC         ACS -0.008478
    5    2010-06-08  11.367564  29.289425  28.281149  29.289425   1015136         ACS  ACS.MC         ACS -0.011457
    ...         ...        ...        ...        ...        ...       ...         ...     ...         ...       ...
    3981 2025-12-18   3.480000   3.503000   3.453000   3.490000  27606425  Telefónica  TEF.MC  Telefónica -0.002865
    3982 2025-12-19   3.420000   3.467000   3.395000   3.440000  56409660  Telefónica  TEF.MC  Telefónica -0.017241
    3983 2025-12-22   3.419000   3.423000   3.368000   3.400000  19546829  Telefónica  TEF.MC  Telefónica -0.000292
    3984 2025-12-23   3.436000   3.468000   3.402000   3.402000  12480301  Telefónica  TEF.MC  Telefónica  0.004972
    3985 2025-12-24   3.445000   3.459000   3.410000   3.424000   4205101  Telefónica  TEF.MC  Telefónica  0.002619
    [75296 rows x 10 columns]
    """
