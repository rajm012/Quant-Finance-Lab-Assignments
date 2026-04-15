

"""
5. Correlation Analysis
    a) Compute correlation matrix
    b) Create correlation heatmap
    c) Identify most and least correlated stock pairs

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from Q2 import load_data


# folder for results
Path("Q4-Results").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    data = load_data()

    # a) Compute correlation matrix
    returns = data.pivot(
        index="Date",
        columns="Stock",
        values="Return"
    )
    corr = returns.corr()
    corr.to_csv("Q4-Results/CorrMatrix.csv")

    # b) Create correlation heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Stock Return Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("Q4-Results/CorrHeatmap.png", dpi=300)
    plt.close()


    # c) Identify most and least correlated stock pairs
    corrPairs = corr.unstack()
    corrPairs = corrPairs[corrPairs.index.get_level_values(0) != corrPairs.index.get_level_values(1)]
    corrPairs = corrPairs.sort_values(ascending=False)

    # most and least
    mostCorr = corrPairs.head(10)
    lestCorr = corrPairs.tail(10)

    with open("Q4-Results/CorrPairs.txt", "w") as f:
        f.write("Most Correlated Pairs\n")
        f.write(mostCorr.to_string())
        f.write("\n\nLeast Correlated Pairs\n")
        f.write(lestCorr.to_string())

    print("Done")
    
    