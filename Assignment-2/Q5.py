

"""
6. Random Portfolio Generation
    Select any two stocks from your dataset and generate random portfolios.
        a) Generate at least 5000 random portfolios
        b) Use random weights for each portfolio
        c) Compute portfolio return and risk (standard deviation)
        d) Plot risk-return scatter diagram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from Q2 import load_data

# folder for results
Path("Q5-Results").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    data = load_data()

    # return matrix
    returns = data.pivot(
        index="Date",
        columns="Stock",
        values="Return"
    )

    retPair = returns[["Inditex", "Iberdrola"]].dropna()
    meanRet = retPair.mean()
    cov = retPair.cov()
    
    N = 5000
    results = []

    # a) Generate at least 5000 random portfolios
    # b) Use random weights for each portfolio
    # c) Compute portfolio return and risk (standard deviation)
    for i in range(N):
        w = np.random.random(2)
        w = w / np.sum(w)

        ret = np.dot(w, meanRet)
        risk = np.sqrt(np.dot(w.T, np.dot(cov, w)))

        sharpe = ret / risk
        results.append([ret, risk, sharpe, w[0], w[1]])

    
    portfolios = pd.DataFrame(results, columns=["Return", "Risk", "Sharpe", "WgtInditex", "WgtIberdrola"])
    portfolios.to_csv("Q5-Results/randPortfolios.csv", index=False)

    # d) Plot risk-return scatter diagram
    plt.figure(figsize=(8,6))
    plt.scatter(
        portfolios["Risk"],
        portfolios["Return"],
        c=portfolios["Sharpe"],
        cmap="viridis",
        alpha=0.6
    )

    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Random Portfolio Risk-Return (Color Sharpe Ratio)")
    plt.tight_layout()
    plt.savefig("Q5-Results/RiskvsReturn.png", dpi=300)
    plt.close()

    print("Done")
    