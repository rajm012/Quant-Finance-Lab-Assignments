

"""

2. Use the training-period equal-weight portfolio w0 = 1/N as the reference portfolio.
    (a) Historical simulation. Computê VaR99% and̂ CVaR99% by sorting the T daily portfolio losses and applying: VaRa = L(daT e),̂ CVaRa = 1, ntail = ∑L(i), ntail = d(1 - a)T e. Report daily and annualised values in rupees for a portfolio worth Rs. 1,00,00,000.
    (b) Parametric Normal. Estimate (µP , aP) from the equal-weight portfolio and compute VaRa = -µP + za*sP , CVaRa = -µP + sP * φ(za)/(1 - a).
    (c) Parametric Student-t. Fit a Student-tv distribution to portfolio losses using scipy.stats.t.fit. Report the fitted (v, µ, a) and compute VaR99% and CVaR99% under this distribution.
    (d) Comparison table and chart. Produce a 3*4 table (rows: Historical / Normal / Student-t; columns: VaR95%, VaR99%, CVaR95%, CVaR99%). Plot the loss histogram with KDE and mark all six VaR/CVaR thresholds.

"""


# libraries
import numpy as np
import pandas as pd           
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm, t
from scipy.integrate import quad
import argparse
import os
import glob
from Q1.Q1 import LoadIBEX


# to bypass the issues while plotting
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)



# loading the data
def LoadIBEX(FolderPath="./IBEX35"):
    if FolderPath == "./IBEX35":
        FolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IBEX35")
    
    CSVFiles = glob.glob(os.path.join(FolderPath, "*.csv"))
    PricesDict = {}
    for file in CSVFiles:
        df = pd.read_csv(file, parse_dates=True, index_col=0)
        PriceCol = None
        posNames = ['Close', 'close', 'Adj Close', 'adj close', 'Price', 'price']
        for col in posNames:
            if col in df.columns:
                PriceCol = col
                break
            
        
        if PriceCol is None:
            numCols = df.select_dtypes(include=[np.number]).columns
            if len(numCols) > 0:
                PriceCol = numCols[-1]
            else:
                raise ValueError(f"No price colName in {file}")
        
        
        StockName = os.path.basename(file).replace('.csv', '')
        PricesDict[StockName] = df[PriceCol]
    
    prices = pd.DataFrame(PricesDict)
    prices = prices.ffill().dropna()
    returns = prices.pct_change().dropna()
    split = int(0.7 * len(returns))
    trainRets = returns.iloc[:split]
    testRets = returns.iloc[split:]
    return trainRets, testRets




#  Helper: Student‑t CVaR via numerical integration
def CVART(alpha, df, loc, scale):
    """Conditional VaR (expected shortfall) for Student-t distribution."""
    q = t.ppf(alpha, df, loc=loc, scale=scale)
    integrand = lambda x: x * t.pdf(x, df, loc=loc, scale=scale)
    TailProb = 1 - alpha
    CVarVal, _ = quad(integrand, q, np.inf, limit=100)
    return CVarVal / TailProb




# part(a)
def HistoricalRisk(losses, wealth=1e7, alphaLvls=[0.95, 0.99]):
    """Historical VaR and CVaR for given alpha levels."""
    lossSorted = np.sort(losses)
    T = len(lossSorted)
    ress = {}
    for alpha in alphaLvls:
        
        # VaR index from assignment formula: L(ceil(alpha * T)) using 1-based indexing.
        varRank1Based = int(np.ceil(alpha * T))
        idx = min(max(varRank1Based - 1, 0), T - 1)
        VaR = float(lossSorted[idx])

        # CVaR tail count from assignment formula: ntail = ceil((1 - alpha) * T).
        ntail = int(np.ceil((1 - alpha) * T))
        ntail = max(ntail, 1)
        tail = lossSorted[-ntail:]
        CVaR = float(np.mean(tail))
        ress[alpha] = {'VaR': VaR, 'CVaR': CVaR}
        
    return ress




# part(b)
def NormalRisk(PortRets, wealth=1e7, alphaLvls=[0.95, 0.99]):
    """Parametric normal VaR and CVaR (loss = -return)."""
    
    uPort = np.mean(PortRets)
    sigmaPort = np.std(PortRets, ddof=1)
    ress = {}
    
    for alpha in alphaLvls:
        z = norm.ppf(alpha)
        VaR = -uPort + z * sigmaPort
        CVaR = -uPort + sigmaPort * norm.pdf(z) / (1 - alpha)
        ress[alpha] = {'VaR': VaR, 'CVaR': CVaR}
        
    return ress




# part(c)
def StudentTRisk(losses, wealth=1e7, alphaLvls=[0.95, 0.99]):
    """Fit Student-t to losses, then compute VaR and CVaR."""
    
    # fit t-distribution to losses (negative returns)
    nu, locT, scaleT = t.fit(losses)
    ress = {}
    
    for alpha in alphaLvls:
        VaR = t.ppf(alpha, nu, loc=locT, scale=scaleT)
        CVaR = CVART(alpha, nu, locT, scaleT)
        ress[alpha] = {'VaR': VaR, 'CVaR': CVaR}
        
    # return fitted parameters for reporting too
    return ress, (nu, locT, scaleT)




# part(d): tables
def MakeTable(hist, normRisk, TRisk, wealth=1e7):
    """Create a 3x4 DataFrame (rows: methods, columns: VaR95%, VaR99%, CVaR95%, CVaR99%)."""
    
    data = []
    methods = ['Historical', 'Normal', 'Student-t']
    
    for _, resDict in zip(methods, [hist, normRisk, TRisk]):
        row = {
            'VaR95%': resDict[0.95]['VaR'] * wealth,
            'VaR99%': resDict[0.99]['VaR'] * wealth,
            'CVaR95%': resDict[0.95]['CVaR'] * wealth,
            'CVaR99%': resDict[0.99]['CVaR'] * wealth,
        }
        data.append(row)
        
    df = pd.DataFrame(data, index=methods)
    return df




# part(d): graphs
def PlotThresholds(losses, hist, normRisk, TRisk, wealth=1e7):
    """Histogram with KDE and vertical lines for all VaR/CVaR thresholds (a=99%)."""
    
    lossRS = losses * wealth
    plt.figure(figsize=(12,6))
    sns.histplot(lossRS, bins=50, kde=True, stat='density', alpha=0.6, label='Loss Distribution')
    
    # thresholds at a = 0.95 and a = 0.99
    thresholds = [
        hist[0.95]['VaR'] * wealth,
        hist[0.99]['VaR'] * wealth,
        
        normRisk[0.95]['VaR'] * wealth,
        normRisk[0.99]['VaR'] * wealth,
        
        TRisk[0.95]['VaR'] * wealth,
        TRisk[0.99]['VaR'] * wealth,
        
        hist[0.95]['CVaR'] * wealth,
        hist[0.99]['CVaR'] * wealth,
        
        normRisk[0.95]['CVaR'] * wealth,
        normRisk[0.99]['CVaR'] * wealth,
        
        TRisk[0.95]['CVaR'] * wealth,
        TRisk[0.99]['CVaR'] * wealth
    ]
    
    
    labels = ['Hist VaR 95', 'Hist VaR 99', 'Norm VaR 95', 'Norm VaR 99', 
              't VaR 95', 't VaR 99', 'Hist CVaR 95', 'Hist CVaR 99', 
              'Norm CVaR 95', 'Norm CVaR 99', 't CVaR 95', 't CVaR 99']
    
    colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78'   # light orange
            ]
    
    for thr, lab, col in zip(thresholds, labels, colors):
        plt.axvline(thr, color=col, linestyle='--', linewidth=1.5, label=lab)
        

    plt.xlabel('Daily Loss (Rs)')
    plt.ylabel('Density')
    plt.title('Loss Distribution with VaR/CVaR thresholds (alpha=95/99%, Rs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q2RiskThresholds.png', dpi=150)
    plt.show()
    print("Saved: Q2RiskThresholds.png")




# main fxn
def MainQ2(method="all"):
    print("Loading IBEX35 data...")
    
    trainRets, _ = LoadIBEX()
    N = len(trainRets.columns)
    wEq = np.ones(N) / N
    PortRets = trainRets @ wEq
    dailyLoss = -PortRets   # loss = negative return
    wealth = 1e7

    print(f"Training days: {len(PortRets)} days")
    print(f"Equal-weight portfolio: {N} stocks\n")
    print(f"Portfolio value used for risk reporting: Rs. {wealth:,.0f}")
    print(f"Equal-weight portfolio mean return (daily): {PortRets.mean():.6f}")
    print(f"Equal-weight portfolio volatility (daily): {PortRets.std(ddof=1):.6f}\n")
    
    # Compute risks based on selected method
    if method in ("historical", "all"):
        hist = HistoricalRisk(dailyLoss, wealth)
        print("Historical VaR/CVaR computed.")
        
    if method in ("normal", "all"):
        normRiskDict = NormalRisk(PortRets, wealth)
        print("Parametric Normal VaR/CVaR computed.")
        
    if method in ("t", "all"):
        TRiskDict, TParas = StudentTRisk(dailyLoss, wealth)
        print(f"Student-t fitted: df={TParas[0]:.2f}, loc={TParas[1]:.6f}, scale={TParas[2]:.6f}")


    # Build and print daily table
    if method == "all":
        DFDaily = MakeTable(hist, normRiskDict, TRiskDict, wealth=wealth)
        
        # Annualisation assumes square-root-of-time rule
        DFDailyAnnual = DFDaily * np.sqrt(252)
        DFDaily.to_csv('Q2RiskCompareDaily.csv')
        DFDailyAnnual.to_csv('Q2RiskCompareAnnual.csv')
        
        print("\n=== Daily VaR/CVaR (monetary units, portfolio value = 10,000,000) ===")
        print(DFDaily.round(0).to_string())
        print("\n=== Annualised VaR/CVaR (scaled by √252) ===")
        print(DFDailyAnnual.round(0).to_string())
        print("\nSaved: Q2RiskCompareDaily.csv")
        print("Saved: Q2RiskCompareAnnual.csv")
        print(f"Fitted Student-t parameters: nu={TParas[0]:.4f}, mu={TParas[1]:.6f}, sigma={TParas[2]:.6f}")
        PlotThresholds(dailyLoss, hist, normRiskDict, TRiskDict, wealth)
        
    else:
        if method == "historical":
            print("\nHistorical VaR/CVaR (daily, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {hist[alpha]['VaR']*wealth:.0f}, CVaR = {hist[alpha]['CVaR']*wealth:.0f}")
            
            print("\nHistorical VaR/CVaR (annualised, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {hist[alpha]['VaR']*wealth*np.sqrt(252):.0f}, CVaR = {hist[alpha]['CVaR']*wealth*np.sqrt(252):.0f}")
        
        elif method == "normal":
            print("\nParametric Normal VaR/CVaR (daily, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {normRiskDict[alpha]['VaR']*wealth:.0f}, CVaR = {normRiskDict[alpha]['CVaR']*wealth:.0f}")
            
            print("\nParametric Normal VaR/CVaR (annualised, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {normRiskDict[alpha]['VaR']*wealth*np.sqrt(252):.0f}, CVaR = {normRiskDict[alpha]['CVaR']*wealth*np.sqrt(252):.0f}")
        
        elif method == "t":
            print("\nStudent-t VaR/CVaR (daily, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {TRiskDict[alpha]['VaR']*wealth:.0f}, CVaR = {TRiskDict[alpha]['CVaR']*wealth:.0f}")
            
            print("\nStudent-t VaR/CVaR (annualised, in Rs.):")
            for alpha in [0.95, 0.99]:
                print(f"alpha={alpha}: VaR = {TRiskDict[alpha]['VaR']*wealth*np.sqrt(252):.0f}, CVaR = {TRiskDict[alpha]['CVaR']*wealth*np.sqrt(252):.0f}")
            
            print(f"\nFitted t parameters: mu={TParas[0]:.2f}, μ={TParas[1]:.6f}, alpha={TParas[2]:.6f}")




# calling loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 Risk Metrics for Equal-Weight Portfolio")
    parser.add_argument("--method", choices=["historical", "normal", "t", "all"], default="all", 
                        help="Which risk method to compute (default: all, produces full table and plot)")
    args = parser.parse_args()
    MainQ2(method=args.method)
    



"""
venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ python Q2.py 
Loading IBEX35 data...
Training days: 2672 days
Equal-weight portfolio: 19 stocks

Portfolio value used for risk reporting: Rs. 10,000,000
Equal-weight portfolio mean return (daily): 0.000374
Equal-weight portfolio volatility (daily): 0.014098

Historical VaR/CVaR computed.
Parametric Normal VaR/CVaR computed.
Student-t fitted: df=3.46, loc=-0.000563, scale=0.009409

=== Daily VaR/CVaR (monetary units, portfolio value = 10,000,000) ===
              VaR95%    VaR99%   CVaR95%   CVaR99%
Historical  213762.0  372993.0  326775.0  538210.0
Normal      228150.0  324225.0  287058.0  371997.0
Student-t   204378.0  379638.0  323038.0  556131.0

=== Annualised VaR/CVaR (scaled by √252) ===
               VaR95%     VaR99%    CVaR95%    CVaR99%
Historical  3393360.0  5921076.0  5187397.0  8543811.0
Normal      3621762.0  5146905.0  4556906.0  5905267.0
Student-t   3244407.0  6026560.0  5128071.0  8828300.0

Saved: Q2RiskCompareDaily.csv
Saved: Q2RiskCompareAnnual.csv
Fitted Student-t parameters: nu=3.4561, mu=-0.000563, sigma=0.009409
Saved: Q2RiskThresholds.png
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ 
"""

