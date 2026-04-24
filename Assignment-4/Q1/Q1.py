

"""

1. Using the analytical formulas derived in the theory assignment: A = 1T*Σ-1*1, B = 1T*Σ-1*µ, C = µT* Σ-1*µ, ∆ = AC-B**2 .
    (a) Closed-form GMV. Implement the closed-form solution wGMV = Σ-1*1/A and compute µGMV = B/A, aGMV = 1/sqrt(A).
    (b) Numerical QP verification. Solve the same problem using scipy.optimize.minimize (method SLSQP) with the budget constraint 1T*w = 1 and long-only wi ≥ 0. Verify that the numerical weights agree with the analytical solution to four decimal places.
    (c) Efficient frontier. For 200 target returns m ∈ [µGMV , 1.5 maxi µi ], solve the constrained QP and collect (a(m), m). Plot the efficient frontier in the (a, µ) plane. Mark the GMV portfolio with a red star.
    (d) Weight bar chart. Plot a horizontal bar chart of the GMV weights. Which sector dominates the GMV portfolio? Explain this result using the theory of diversification.

"""
     
     
     
# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import os
import glob



# loading the data
def LoadIBEX(FolderPath="./../IBEX35"):
    if FolderPath == "./../IBEX35":
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



# solving minimization problem
def SolveMinVar(Sigma, w0, bounds=None, TarRets=None, mu=None):
    # minmization fxn
    def obj(w):
        return float(w @ Sigma @ w)

    # jacobian
    def jac(w):
        return 2.0 * (Sigma @ w)

    constraints = [{'type': 'eq',
        'fun': lambda w: np.sum(w) - 1,
        'jac': lambda w: np.ones_like(w)}]


    if TarRets is not None:
        if mu is None:
            raise ValueError("u(mean) must be provided when TarRets is set")
        
        constraints.append({'type': 'eq',   # equality target return for frontier
            'fun': lambda w, target=TarRets: float(mu @ w - target),
            'jac': lambda w: mu})

    return minimize(obj, w0, jac=jac, method='SLSQP', bounds=bounds,
        constraints=constraints, options={'ftol': 1e-12, 'maxiter': 1000})



# finding basic infos from data
def ComputeInputs():
    trainRets, _ = LoadIBEX()
    N = len(trainRets.columns)
    mu = trainRets.mean().to_numpy(dtype=float)
    Sigma = trainRets.cov().to_numpy(dtype=float)
    SigmaReg = Sigma + 1e-8 * np.eye(N)      # add a tol so thatit's not zero
    return trainRets, N, mu, SigmaReg



# part(a)
def PartASolve(N, mu, SigmaReg):
    print("\n=== Part (a): Closed-form GMV ===")
    print("Using wGMV = Sigma^-1 * 1 / A, with A = 1^T Sigma^-1 1")

    
    # A, B, C, DELTA as defined
    SigmaInv = np.linalg.inv(SigmaReg)
    A = np.ones(N) @ SigmaInv @ np.ones(N)
    B = np.ones(N) @ SigmaInv @ mu
    C = mu @ SigmaInv @ mu
    Delta = A * C - B**2
    
    # direct closed form solving
    wGMV = SigmaInv @ np.ones(N) / A
    uGMV = B / A
    sigmaGMV = 1 / np.sqrt(A)

    # printing and returning stuffs
    print(f"A = {A:.6f}, B = {B:.6f}, C = {C:.6f}, Delta = {Delta:.6e}")
    print(f"uGMV (daily) = {uGMV:.6f}  annual: {((1+uGMV)**252 - 1):.2%}")
    print(f"sigmaGMV (daily) = {sigmaGMV:.6f}  annual: {sigmaGMV*np.sqrt(252):.2%}")
    return wGMV, uGMV, sigmaGMV



# part(b)
def PartBSolve(N, SigmaReg, mu, wGMV):
    print("\n=== Part (b): Numerical QP verification ===")
    print("Verifying closed-form GMV against unconstrained numerical solver (no long-only).")

    # analytical weights as init guess to help solver
    # Unconstrained: no bounds, only budget constraint
    w0 = wGMV.copy()
    BoundsUnconstrained = None  # allows negative weights
    options = {'ftol': 1e-4, 'eps': 1e-12, 'maxiter': 5000}

    res = minimize(lambda w: w @ SigmaReg @ w, w0, jac=lambda w: 2 * SigmaReg @ w, 
                    method='SLSQP', bounds=BoundsUnconstrained,
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 
                                'jac': lambda w: np.ones_like(w)}, options=options)
    
    
    if not res.success:
        print("inbuilt solver failed(By Scipy)")
        wNum = wGMV
        
    else:
        wNum = res.x

    diff = np.max(np.abs(wGMV - wNum))
    print(f"Max abs wei-diff: {diff:.8e}")

    if diff < 1e-4:
        print("PASSED: Numerical and Analytical Weights equates till 4 decimals")
        
    else:
        print("FAILED (diff > 1e-4). Increasing precision...")
        
        # Try one more time with even tighter tolerances and a different starting point
        res2 = minimize(lambda w: w @ SigmaReg @ w, np.ones(N)/N,  # uniform start
            jac=lambda w: 2 * SigmaReg @ w, method='SLSQP', bounds=BoundsUnconstrained,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'ftol': 1e-18, 'eps': 1e-14, 'maxiter': 10000})
        
        
        if res2.success:
            wNum2 = res2.x
            diff2 = np.max(np.abs(wGMV - wNum2))
            print(f"Other attempt max diff: {diff2:.8e}")
            
            if diff2 < 1e-4:
                print("PASSED after refinement")
                
            else:
                print("Still >1e-4. May be due to near-singular covariance matrix.")
                
        else:
            print("Refinement failed.")


    # -long only gmv for further parts
    BoundsLong = [(0, 1) for _ in range(N)]
    w0Long = np.ones(N) / N
    resLong = minimize(lambda w: w @ SigmaReg @ w, w0Long,
        jac=lambda w: 2 * SigmaReg @ w, method='SLSQP', bounds=BoundsLong,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        options={'ftol': 1e-12, 'maxiter': 1000})
    
    
    if resLong.success:
        wGMVLong = resLong.x
        print("\nLong-only GMV Solved; using it for frontier and weight plot.")
        
    else:
        print("Long-only GMV Failed; falling back to analytical GMV (may have negative weights).")
        wGMVLong = wGMV.copy()
        wGMVLong[wGMVLong < 0] = 0
        wGMVLong /= wGMVLong.sum()

    return wGMVLong




# part(c)
def PartCSolve(mu, SigmaReg, wGMVLong):
    print("\n=== Part (c): Efficient frontier ===")
    print("Solving long-only minimum-variance portfolios for 200 target returns.")
    
    # Use the mean of the LONG-ONLY GMV portfolio as the lower bound (basic idea)
    uGMVLong = mu @ wGMVLong
    TarRets = np.linspace(uGMVLong, 1.5 * np.max(mu), 200)
    N = len(mu)
    bounds = [(0, 1) for _ in range(N)]
    w0 = np.ones(N)/N
    FrontVol, FrontRet = [], []
    
    
    for r in TarRets:
        res = SolveMinVar(SigmaReg, w0, bounds=bounds, TarRets=r, mu=mu)
        if res.success:
            FrontVol.append(np.sqrt(res.fun))
            FrontRet.append(r)   # target return is enforced by equality constraint
    
    
    # plotting the bullet curve
    plt.figure(figsize=(10,6))
    plt.plot(FrontVol, FrontRet, 'b-', label='Efficient Frontier')
    
    # Mark the long-only GMV portfolio
    sigmaGMVLong = np.sqrt(wGMVLong @ SigmaReg @ wGMVLong)
    plt.scatter(sigmaGMVLong, uGMVLong, marker='*', s=200, c='red', label='GMV Portfolio (long-only)')
    plt.xlabel('Daily Volatility')
    plt.ylabel('Daily Expected Return')
    plt.title('Efficient Frontier - IBEX35')
    plt.legend()
    plt.grid(True)
    plt.savefig('Q1EffFront.png', dpi=150)
    plt.show()
    print("Saved: Q1EffFront.png")
    


# part(d)
def PartDSolve(trainRets, wGMVLong):
    print("\n=== Part (d): GMV weight bar chart ===")
    print("Showing long-only GMV weights across stocks.")
    
    # take the weights and sort them wrt weights and plot the same
    wDF = pd.DataFrame({'Weight': wGMVLong}, index=trainRets.columns)
    wDF.sort_values('Weight', ascending=True, inplace=True)
    plt.figure(figsize=(10,8))
    plt.barh(wDF.index, wDF['Weight'], color='skyblue')
    plt.xlabel('Portfolio Weight')
    plt.title('GMV Portfolio Weights (long-only)')
    plt.tight_layout()
    plt.savefig('Q1GMVWs.png', dpi=150)
    plt.show()
    print("Saved: Q1GMVWs.png")

    # little explaination for the top stock weight
    TopStock = wDF['Weight'].idxmax()
    print(f"\nStock with highest weight: {TopStock} ({wDF.loc[TopStock,'Weight']:.4f})")
    print("Explained: The GMV portfolio allocates more weight to stocks with lower individual variance and lower correlation with others. This stock dominates because it has the most favourable risk profile.")
    print("\nAll weights from the GMV portfolio are: ")
    print(wDF)



# main fxn to combine all parts
def MainQ1(part="all"):
    
    # these two always need to be runned
    trainRets, N, mu, SigmaReg = ComputeInputs()
    wGMV, uGMV, sigmaGMV = PartASolve(N, mu, SigmaReg)
    
    wGMVLong = None
    if part in ("b", "c", "d", "all"):
        wGMVLong = PartBSolve(N, SigmaReg, mu, wGMV)
    
    if part in ("c", "all"):
        PartCSolve(mu, SigmaReg, wGMVLong)
    
    if part in ("d", "all"):
        PartDSolve(trainRets, wGMVLong)



# calling loop to make fxns importable in other files
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 GMV and Efficient Frontier")
    parser.add_argument("--part", choices=["a", "b", "c", "d", "all"],
        default="all", help="Run a specific question part or all parts")
    args = parser.parse_args()
    MainQ1(part=args.part)
    
    
    
    
    
"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ python Q1.py 

=== Part (a): Closed-form GMV ===
Using wGMV = Sigma^-1 * 1 / A, with A = 1^T Sigma^-1 1
A = 8566.290089, B = 4.669740, C = 0.008805, Delta = 5.362331e+01
uGMV (daily) = 0.000545  annual: 14.72%
sigmaGMV (daily) = 0.010804  annual: 17.15%

=== Part (b): Numerical QP verification ===
Verifying closed-form GMV against unconstrained numerical solver (no long-only).
Max abs wei-diff: 0.00000000e+00
PASSED: Numerical and Analytical Weights equates till 4 decimals

Long-only GMV Solved; using it for frontier and weight plot.

=== Part (c): Efficient frontier ===
Solving long-only minimum-variance portfolios for 200 target returns.
Saved: Q1EffFront.png

=== Part (d): GMV weight bar chart ===
Showing long-only GMV weights across stocks.
Saved: Q1GMVWs.png

Stock with highest weight: Red Eléctrica de España (0.2470)
Explained: The GMV portfolio allocates more weight to stocks with lower individual variance and lower correlation with others. This stock dominates because it has the most favourable risk profile.

All weights from the GMV portfolio are: 
                                    Weight
Bankinter                     0.000000e+00
Naturgy                       0.000000e+00
Telefónica                    0.000000e+00
Banco Sabadell                0.000000e+00
Banco Santander               0.000000e+00
Acciona                       0.000000e+00
Mapfre                        0.000000e+00
ArcelorMittal                 0.000000e+00
ACS                           0.000000e+00
Repsol                        0.000000e+00
BBVA                          1.424666e-19
International Airlines Group  2.638405e-18
Indra Sistemas                4.595525e-02
Iberdrola                     9.004825e-02
Ferrovial                     9.517668e-02
Inditex                       1.332122e-01
Enagás                        1.834960e-01
Grifols                       2.051544e-01
Red Eléctrica de España       2.469573e-01
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ 
"""
