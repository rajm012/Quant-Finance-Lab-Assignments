
"""
4. Normality tests
    a) Plot histogram of returns
    b) Create Q-Q plot
    c) Perform Jarque-Bera test
    d) Perform Shapiro-Wilk test

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro

from pathlib import Path
from Q2 import load_data

# folder for results
Path("Q3-Results/All").mkdir(parents=True, exist_ok=True)
Path("Q3-Results/Histograms").mkdir(parents=True, exist_ok=True)
Path("Q3-Results/QQPlots").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    
    data = load_data()
    
    # ==================================================
    #  Alll in one
     
    AllReturns = data["Return"].dropna()

    # Histogram
    plt.figure()
    sns.histplot(AllReturns, bins=50, kde=True)
    plt.title("Combined Return Distribution (All Stocks)")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.savefig("Q3-Results/All/allHist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # QQ plot
    plt.figure()
    stats.probplot(AllReturns, dist="norm", plot=plt)
    plt.title("Combined Q-Q Plot")
    plt.savefig("Q3-Results/All/allQQ.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Normality tests
    JBStat, JBP = jarque_bera(AllReturns)
    SHStat, SHP = shapiro(AllReturns)

    with open("Q3-Results/All/allNorms.txt", "w") as f:
        f.write("All Normality Tests\n\n")
        f.write(f"Jarque-Bera Statistic: {JBStat}\n")
        f.write(f"Jarque-Bera p-value: {JBP}\n\n")
        f.write(f"Shapiro-Wilk Statistic: {SHStat}\n")
        f.write(f"Shapiro-Wilk p-value: {SHP}\n")

    # individual stocks
    results = []

    for stock, subset in data.groupby("Stock"):
        returns = subset["Return"].dropna()

        # Histogram
        plt.figure()
        sns.histplot(returns, bins=50, kde=True)
        plt.title(f"{stock} Return Distribution")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.savefig(f"Q3-Results/Histograms/{stock}Hist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # QQ plot
        plt.figure()
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f"{stock} Q-Q Plot")
        plt.savefig(f"Q3-Results/QQPlots/{stock}QQ.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Normality tests
        JBStat, JBP = jarque_bera(returns)
        SHStat, SHP = shapiro(returns)

        results.append({
            "Stock": stock,
            "JBStat": JBStat,
            "JBPVal": JBP,
            "ShapiroStat": SHStat,
            "ShapiroPVal": SHP
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("Q3-Results/Individual.csv", index=False)
    print("Done")
    
    """
    site-packages/scipy/stats/_axis_nan_policy.py:592: UserWarning: scipy.stats.shapiro: 
    For N > 5000, computed p-value may not be accurate. Current N is 75296.
    res = hypotest_fun_out(*samples, **kwds)
    Done
    """
"""
The Jarque-Bera and Shapiro-Wilk tests strongly reject the null hypothesis of normality for all stocks in the dataset. 
The extremely small p-values indicate that the return distributions exhibit significant deviations from a Gaussian distribution. 
This behavior is consistent with well-known empirical findings in financial markets where asset returns display skewness and excess kurtosis (fat tails).
"""
