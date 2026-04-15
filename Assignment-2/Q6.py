

"""
7. Markowitz Portfolio Optimization
    a) Compute expected returns
    b) Compute covariance matrix
    c) Solve minimum variance portfolio
    d) Generate portfolios for different target returns
    e) Plot random portfolios of last questions along with the efficient frontier curve
    f) Clearly indicate minimum variance portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from Q2 import load_data

Path("Q6-Results").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    
    data = load_data()
    returns = data.pivot(index="Date", columns="Stock", values="Return")

    # a) Compute expected returns
    # b) Compute covariance matrix
    retPairs = returns[["Inditex", "Iberdrola"]].dropna()
    meanRets = retPairs.mean()
    covMat = retPairs.cov()
    NAssets = len(meanRets)

    # help fxns
    def PortRet(w):
        return np.dot(w, meanRets)

    def PortRisk(w):
        return np.sqrt(np.dot(w.T, np.dot(covMat, w)))

    def MinVar(w):
        return PortRisk(w)

    const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0,1) for i in range(NAssets))

    # c) Solve minimum variance portfolio
    initGuess = np.ones(NAssets) / NAssets
    minVar = minimize(
        MinVar,
        initGuess,
        method="SLSQP",
        bounds=bounds,
        constraints=const
    )

    wMin = minVar.x
    retMin = PortRet(wMin)
    riskMin = PortRisk(wMin)

    # d) Generate portfolios for different target returns
    targetRet = np.linspace(meanRets.min(), meanRets.max(), 100)
    frontRisk = []

    for target in targetRet:
        const = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: PortRet(w) - target}
        )
        result = minimize( MinVar, initGuess, method="SLSQP", bounds=bounds, constraints=const)
        frontRisk.append(result.fun)

    frontRisk = np.array(frontRisk)
    randPort = pd.read_csv("Q5-Results/randPortfolios.csv")

    # e) Plot random portfolios of last questions along with the efficient frontier curve
    # f) Clearly indicate minimum variance portfolio

    plt.figure(figsize=(9,7))
    plt.scatter(
        randPort["Risk"], randPort["Return"],
        c=randPort["Sharpe"], cmap="viridis", alpha=0.4,
        label="Random Portfolios")

    # Only plot the efficient (upper) part of the frontier
    mask = targetRet >= retMin
    plt.plot(
        frontRisk[mask], targetRet[mask],
        color="red", linewidth=2,
        label="Efficient Frontier")

    plt.scatter(
        riskMin, retMin, color="black",
        marker="*", s=200,
        label="Minimum Variance Portfolio")

    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier vs Random Portfolios")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Q6-Results/effFront.png", dpi=300)
    plt.close()

    weights = pd.DataFrame({
        "Stock": retPairs.columns,
        "Weight": wMin
    })

    weights.to_csv("Q6-Results/minVarWgt.csv", index=False)
    print("Done")
    