"""

3. Implement all three models below using the training data. Use a = 99 %, τ = rf /252 (daily), rf = 6 %, long-only wi ≥ 0, 1>w = 1, and target return m = 12 % per annum.
									Model Objective Auxiliary vars Solver
									-------------------------------------
(a) Implement each model as a separate Python function that accepts (Rtrain, µ, Σ, rf , m, a, τ ) and returns the optimal weight vector w*.
(b) For each model report: w* (bar chart), µP , sP , Sharpe Ratio (SR), VaR99%, CVaR99% on the training set.
(c) Plot all four weight vectors side-by-side as a grouped bar chart with stock tickers on the x-axis.
(d) In a Markdown cell, explain why Model 2 (CVaR) typically concentrates less than M1 (Variance) in the tail of the loss distribution.

"""


# libraries
import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize




# loading data
def LoadIBEX(FolderPath="./IBEX35"):
	if FolderPath == "./IBEX35":
		FolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IBEX35")

	CSVFiles = glob.glob(os.path.join(FolderPath, "*.csv"))
	PricesDict = {}
	for file in CSVFiles:
		df = pd.read_csv(file, parse_dates=True, index_col=0)
		PriceCol = None
		for col in ["Close", "close", "Adj Close", "adj close", "Price", "price"]:
			if col in df.columns:
				PriceCol = col
				break

		if PriceCol is None:
			numCols = df.select_dtypes(include=[np.number]).columns
			if len(numCols) == 0:
				raise ValueError(f"No price column found in {file}")
			PriceCol = numCols[-1]
   
		PricesDict[os.path.basename(file).replace(".csv", "")] = df[PriceCol]

	prices = pd.DataFrame(PricesDict).ffill().dropna()
	returns = prices.pct_change().dropna()
	split = int(0.7 * len(returns))
	return returns.iloc[:split], returns.iloc[split:]



# historic Var and CVar
def HistVarCVar(losses, alpha=0.99):
	lossesSorted = np.sort(losses)
	T = len(lossesSorted)
	idx = int(np.ceil(alpha * T)) - 1
	idx = min(max(idx, 0), T - 1)
	varA = float(lossesSorted[idx])
	ntail = max(int(np.ceil((1 - alpha) * T)), 1)
	cvarA = float(np.mean(lossesSorted[-ntail:]))
	return varA, cvarA




# model1
def SolveMinVar(Rtrain, mu, Sigma, rf, m, alpha, tau):
	N = len(mu)
	w0 = np.ones(N) / N

	def obj(w):
		return float(w @ Sigma @ w)

	def jac(w):
		return 2.0 * (Sigma @ w)

	constraints = [
		{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
		{"type": "ineq", "fun": lambda w: float(mu @ w - m), "jac": lambda w: mu}]
 
	bounds = [(0.0, 1.0) for _ in range(N)]
 
	res = minimize(obj, w0, jac=jac, method="SLSQP", bounds=bounds,
		constraints=constraints, options={"ftol": 1e-12, "maxiter": 2000})
 
	if not res.success:
		raise RuntimeError(f"Min-Var Failed: {res.message}")

	return res.x



# model2
def SolveCVar(Rtrain, mu, Sigma, rf, m, alpha, tau):
	T, N = Rtrain.shape
	nVars = N + 1 + T  # [w(0:N), zeta(N), u(N+1:)]
	zetaIdx = N
	uStart = N + 1

	c = np.zeros(nVars)
	c[zetaIdx] = 1.0
	c[uStart:] = 1.0 / ((1.0 - alpha) * T)

	Aub = []
	bub = []

	# uT >= lT - zeta, with lT = -RT w  =>  -RT*w - zeta - uT <= 0
 	# uT >= 0 enforced via bounds
	for t in range(T):
		row = np.zeros(nVars)
		row[:N] = -Rtrain[t]
		row[zetaIdx] = -1.0
		row[uStart + t] = -1.0
		Aub.append(row)
		bub.append(0.0)

	# target return: mu^T w >= m  => -mu^T w <= -m
	RowTar = np.zeros(nVars)
	RowTar[:N] = -mu
	Aub.append(RowTar)
	bub.append(-m)

	AEq = np.zeros((1, nVars))
	AEq[0, :N] = 1.0
	bEq = np.array([1.0])

	bounds = [(0.0, 1.0)] * N + [(None, None)] + [(0.0, None)] * T

	res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
		A_eq=AEq, b_eq=bEq, bounds=bounds, method="highs")
 
	if not res.success:
		raise RuntimeError(f"CVaR LP solver failed: {res.message}")

	return res.x[:N]



# model3
def SolveSemiVar(Rtrain, mu, Sigma, rf, m, alpha, tau):
	N = len(mu)
	w0 = np.ones(N) / N

	def obj(w):
		downside = np.maximum(tau - (Rtrain @ w), 0.0)
		return float(np.mean(downside**2))

	constraints = [
		{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
		{"type": "ineq", "fun": lambda w: float(mu @ w - m)}]
 
	bounds = [(0.0, 1.0) for _ in range(N)]

	res = minimize(obj, w0, method="SLSQP", bounds=bounds,
		constraints=constraints, options={"ftol": 1e-12, "maxiter": 2000})
 
	if not res.success:
		raise RuntimeError(f"Semivariance solver failed: {res.message}")

	return res.x



# other comaprision matrices
def ComputeMetrices(w, Rtrain, mu, Sigma, tau, alpha=0.99):
	PortRets = Rtrain @ w
 
	# daily vs annual
	# muPort = float(mu @ w)
	# sigmaPort = float(np.sqrt(w @ Sigma @ w))
	muPort = float(mu @ w) * 252
	sigmaPort = float(np.sqrt(w @ Sigma @ w)) * np.sqrt(252)
	RFAnnual = tau * 252
 
	sr = float((muPort - RFAnnual) / sigmaPort) if sigmaPort > 0 else np.nan
	losses = -PortRets
	var99, cvar99 = HistVarCVar(losses, alpha=alpha)
 
	return {"muPort": muPort, "sigmaPort": sigmaPort,
		"sharpe": sr, "VaR99": var99, "CVaR99": cvar99}




def PlotWBar(weights, tickers, title, FileName):
	order = np.argsort(weights)
	plt.figure(figsize=(10, 8))
	plt.barh(np.array(tickers)[order], np.array(weights)[order], color="steelblue")
	plt.title(title)
	plt.xlabel("Weight")
	plt.tight_layout()
	plt.savefig(FileName, dpi=150)
	plt.show()
	print(f"Saved: {FileName}")



# part(c)
def PlotGrpWeights(tickers, WEq, WVar, WCVar, WSemi):
	x = np.arange(len(tickers))
	width = 0.2
 
	plt.figure(figsize=(14, 6))
	plt.bar(x - 1.5 * width, WEq, width, label="Equal Weight")
	plt.bar(x - 0.5 * width, WVar, width, label="Min Variance")
	plt.bar(x + 0.5 * width, WCVar, width, label="CVaR Min")
	plt.bar(x + 1.5 * width, WSemi, width, label="Semivariance")
	plt.xticks(x, tickers, rotation=75, ha="right")
	plt.ylabel("Weight")
	plt.title("Q3: Weight vectors comparison")
	plt.legend()
	plt.tight_layout()
	plt.savefig("Q3GrpWeights.png", dpi=150)
	plt.show()
	print("Saved: Q3GrpWeights.png")




# main fxn
def MainQ3(alpha=0.99, RFAnnual=0.06, tarAnnual=0.12):
	print("Loading training data...")
	trainRets, _ = LoadIBEX()
	tickers = trainRets.columns.tolist()
	Rtrain = trainRets.to_numpy(dtype=float)
	mu = trainRets.mean().to_numpy(dtype=float)
	Sigma = trainRets.cov().to_numpy(dtype=float)

	tau = RFAnnual / 252.0
	m = (1.0 + tarAnnual) ** (1.0 / 252.0) - 1.0
	N = len(tickers)

	print(f"Universe size: {N} stocks")
	print(f"Training days: {Rtrain.shape[0]}")
	print(f"alpha={alpha}, RFDaily={tau:.8f}, tarDaily={m:.8f}")

	WEq = np.ones(N) / N

	print("\nSolving Model 1: Min-Var")
	WVar = SolveMinVar(Rtrain, mu, Sigma, RFAnnual, m, alpha, tau)

	print("Solving Model 2: CVaR")
	WCVar = SolveCVar(Rtrain, mu, Sigma, RFAnnual, m, alpha, tau)

	print("Solving Model 3: Semi-Var")
	WSemi = SolveSemiVar(Rtrain, mu, Sigma, RFAnnual, m, alpha, tau)

	# part(b): report muPort, sigmaPort, Sharpe, VaR99%, CVaR99% on training set
	metrics = {
		"Equal Weight": ComputeMetrices(WEq, Rtrain, mu, Sigma, tau, alpha),
		"Model 1 - Min Variance": ComputeMetrices(WVar, Rtrain, mu, Sigma, tau, alpha),
		"Model 2 - CVaR": ComputeMetrices(WCVar, Rtrain, mu, Sigma, tau, alpha),
		"Model 3 - Semivariance": ComputeMetrices(WSemi, Rtrain, mu, Sigma, tau, alpha),
	}

	DFMetrices = pd.DataFrame(metrics).T
	print("\n=== Q3 Metrics on Training Set ===")
	print(DFMetrices.round(6).to_string())
	DFMetrices.to_csv("Q3Metrices.csv")
	print("Saved: Q3Metrices.csv")

	# part(b): individual bar charts
	PlotWBar(WVar, tickers, "Q3 Model 1: Minvar Weights", "Q3-M1Weights.png")
	PlotWBar(WCVar, tickers, "Q3 Model 2: CVaR Weights", "Q3-M2Weights.png")
	PlotWBar(WSemi, tickers, "Q3 Model 3: SemiVar Weights", "Q3-M3Weights.png")

	# part(c): grouped chart
	PlotGrpWeights(tickers, WEq, WVar, WCVar, WSemi)

	# save weights for Q4/Q5 reuse
	pd.DataFrame({ "Ticker": tickers, "Equal": WEq, "M1-MinVar": WVar, "M2-CVaR": WCVar,
		"M3-Semivar": WSemi}).to_csv("Q3Weights.csv", index=False)
	print("Saved: Q3Weights.csv")

	# part(d): explanation cue for Markdown discussion
	print("\nPart(d) cue: CVaR minimization usually spreads tail losses more evenly than pure variance minimization, so concentration in extreme-loss states tends to reduce.")




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Q3 Portfolio Models")
	parser.add_argument("--alpha", type=float, default=0.99, help="Tail confidence level")
	parser.add_argument("--rf", type=float, default=0.06, help="Annual risk-free rate")
	parser.add_argument("--target", type=float, default=0.12, help="Annual target return")
	args = parser.parse_args()
	MainQ3(alpha=args.alpha, RFAnnual=args.rf, tarAnnual=args.target)




"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ python Q3.py 
Loading training data...
Universe size: 19 stocks
Training days: 2672
alpha=0.99, RFDaily=0.00023810, tarDaily=0.00044982

Solving Model 1: Min-Var
Solving Model 2: CVaR
Solving Model 3: Semi-Var

=== Q3 Metrics on Training Set ===
                          muPort  sigmaPort    sharpe     VaR99    CVaR99
Equal Weight            0.094166   0.223794  0.152666  0.037299  0.053821
Model 1 - Min Variance  0.135729   0.174924  0.432924  0.029459  0.040280
Model 2 - CVaR          0.142687   0.182361  0.453426  0.029401  0.038685
Model 3 - Semivariance  0.140302   0.175518  0.457517  0.029106  0.039991

Saved: Q3Metrices.csv
Saved: Q3-M1Weights.png
Saved: Q3-M2Weights.png
Saved: Q3-M3Weights.png
Saved: Q3GrpWeights.png
Saved: Q3Weights.csv

Part(d) cue: CVaR minimization usually spreads tail losses more evenly than pure variance minimization, so concentration in extreme-loss states tends to reduce.
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ 
"""
