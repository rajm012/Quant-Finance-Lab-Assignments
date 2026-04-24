

# libraries
import argparse
import itertools
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Q3 import (ComputeMetrices, LoadIBEX, SolveCVar, SolveMinVar, SolveSemiVar)       #type:ignore



# min-var objective fxn
def ObjMinVar(w, R, mu, Sigma, tau, alpha):
	return float(w @ Sigma @ w)




# CVar objective fxn
def ObjCVar(w, R, mu, Sigma, tau, alpha):
    losses = -(R @ w)
    zeta = np.quantile(losses, alpha)
    tail = losses[losses >= zeta]
    if len(tail) == 0:
        return float(zeta)

    return float(zeta + tail.mean() / (1 - alpha))




# sami-var objective fxn
def ObjSemiVar(w, R, mu, Sigma, tau, alpha):
	downside = np.maximum(tau - (R @ w), 0.0)
	return float(np.mean(downside**2))




# part(a): cardinality via complete subset enumeration
def cardSolve(ModelFxn, ObjFxn, R, mu, Sigma, K, rf, m, alpha, tau, maxSubsets=None):
	N = R.shape[1]
	combos = itertools.combinations(range(N), K)
	totSubsets = int(math.comb(N, K))
	print(f"Enumerating subsets for K={K}: total={totSubsets}")

	BestObj = np.inf
	BestWFull = None
	BestSubSet = None
	solved = 0
	tried = 0
	t0 = time.time()

	for subset in combos:
		tried += 1
		if maxSubsets is not None and tried > maxSubsets:
			break

		idx = np.array(subset)
		rSub = R[:, idx]
		MuSub = mu[idx]
		SigmaSub = Sigma[np.ix_(idx, idx)]

		try:
			wSub = ModelFxn(rSub, MuSub, SigmaSub, rf, m, alpha, tau)
   
		except Exception:
			continue

		solved += 1
		obj = ObjFxn(wSub, rSub, MuSub, SigmaSub, tau, alpha)
  
		if obj < BestObj:
			BestObj = obj
			BestSubSet = idx.copy()
			wFull = np.zeros(N)
			wFull[idx] = wSub
			BestWFull = wFull

		if tried % 5000 == 0:
			elapsed = time.time() - t0
			print(f"  tried={tried}, solved={solved}, elapsed={elapsed:.1f}s, BestObj={BestObj:.6e}")
   

	if BestWFull is None:
		raise RuntimeError("No feasible cardinality solution found.")

	elapsed = time.time() - t0
	print(f"Finished: tried={tried}, solved={solved}, elapsed={elapsed:.1f}s")
	return BestWFull, BestSubSet.tolist(), BestObj                           #type:ignore



# basic print fxn
def PrintStock(title, tickers, w, eps=1e-10):
	sel = np.where(w > eps)[0]
	print(f"\n{title}")
	for i in sel:
		print(f"  {tickers[i]}: {w[i]:.6f}")




# basic graph plotting 
def SaveSparseBar(tickers, weights, FileName, title):
	plt.figure(figsize=(10, 7))
	order = np.argsort(weights)
	plt.barh(np.array(tickers)[order], np.array(weights)[order], color="teal")
	plt.title(title)
	plt.xlabel("Weight")
	plt.tight_layout()
	plt.savefig(FileName, dpi=150)
	plt.show()
	print(f"Saved: {FileName}")




# main fxn
def MainQ4(alpha=0.99, RFAnnual=0.06, tarAnnual=0.12, maxSubsets=None):
	trainRets, _ = LoadIBEX()
	tickers = trainRets.columns.tolist()
	R = trainRets.to_numpy(dtype=float)
	mu = trainRets.mean().to_numpy(dtype=float)
	Sigma = trainRets.cov().to_numpy(dtype=float)
 
	N = len(tickers)
	Sigma = Sigma + 1e-8 * np.eye(N)			# for stability
	K = int(np.ceil(N / 3))
	tau = RFAnnual / 252.0
	m = (1.0 + tarAnnual) ** (1.0 / 252.0) - 1.0


	print("=== Q4 Cardinality-Constrained Portfolio Optimisation ===")
	print(f"N={N}, K={K}, alpha={alpha}, RFDaily={tau:.8f}, tarDaily={m:.8f}")
	if maxSubsets is not None:
		print(f"Quick mode enabled: maxSubsets={maxSubsets}")


	# Unconstrained baselines (from Task 3)
	w1U = SolveMinVar(R, mu, Sigma, RFAnnual, m, alpha, tau)
	w2U = SolveCVar(R, mu, Sigma, RFAnnual, m, alpha, tau)
	w3U = SolveSemiVar(R, mu, Sigma, RFAnnual, m, alpha, tau)
 

	# part(a): cardinality-constrained solutions via subset enumeration
	w1C, s1, o1 = cardSolve(SolveMinVar, ObjMinVar, R, mu, Sigma, K, RFAnnual, m, alpha, tau, maxSubsets=maxSubsets )
	w2C, s2, o2 = cardSolve(SolveCVar, ObjCVar, R, mu, Sigma, K, RFAnnual, m, alpha, tau, maxSubsets=maxSubsets )
	w3C, s3, o3 = cardSolve(SolveSemiVar, ObjSemiVar, R, mu, Sigma, K, RFAnnual, m, alpha, tau, maxSubsets=maxSubsets )


	# part(b): report selected K stocks and sparse weights
	PrintStock("Model 1 (budget + cardinality)", tickers, w1C)
	PrintStock("Model 2 (budget + cardinality)", tickers, w2C)
	PrintStock("Model 3 (budget + cardinality)", tickers, w3C)


	# part(c): summary comparison table (unconstrained vs cardinality)
	rows = [
		("Model 1 (unconstrained)", w1U),
		("Model 1 (budget + cardinality)", w1C),
		("Model 2 (unconstrained)", w2U),
		("Model 2 (budget + cardinality)", w2C),
		("Model 3 (unconstrained)", w3U),
		("Model 3 (budget + cardinality)", w3C),
	]

	metricesRows = []
	for name, w in rows:
		met = ComputeMetrices(w, R, mu, Sigma, tau, alpha)
		metricesRows.append(
			{"Variant": name,
			 "muP": met["muP"],
			 "sigmaP": met["sigmaP"],
			 "SR": met["sharpe"],
			 "CVaR99%": met["CVaR99"]})
  

	DFSummary = pd.DataFrame(metricesRows)
	print("\n=== Q4 Summary: Unconstrained vs Cardinality ===")
	print(DFSummary.round(6).to_string(index=False))
	DFSummary.to_csv("Q4Summary.csv", index=False)
	print("Saved: Q4Summary.csv")
 

	# Save weight table and sparse charts
	DFWeights = pd.DataFrame(
		{
			"Ticker": tickers,
			"M1Unconstrained": w1U,
			"M1Cardinality": w1C,
			"M2Unconstrained": w2U,
			"M2Cardinality": w2C,
			"M3Unconstrained": w3U,
			"M3Cardinality": w3C })
 
	DFWeights.to_csv("Q4Weights.csv", index=False)
	print("Saved: Q4Weights.csv")

	SaveSparseBar(tickers, w1C, "Q4-M1-CardinalWeights.png", "Q4 Model 1 (Cardinality) Weights")
	SaveSparseBar(tickers, w2C, "Q4-M2-CardinalWeights.png", "Q4 Model 2 (Cardinality) Weights")
	SaveSparseBar(tickers, w3C, "Q4-M3-CardinalWeights.png", "Q4 Model 3 (Cardinality) Weights")

	# Part (d): interpretation
	print("\nComment cue for report (Part d):")
	print("Cardinality often increases variance due to fewer diversification channels, but CVaR can improve when sparse selection removes assets with extreme downside tails.")




# calling loop
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Assignment 4 - Q4 Cardinality")
	parser.add_argument("--alpha", type=float, default=0.99)
	parser.add_argument("--rf", type=float, default=0.06)
	parser.add_argument("--target", type=float, default=0.12)
	parser.add_argument("--max-subsets", type=int, default=None, 
  			help="Optional quick mode for debugging; defaults to full enumeration" )
	args = parser.parse_args()
	MainQ4(alpha=args.alpha, RFAnnual=args.rf, tarAnnual=args.target, maxSubsets=args.maxSubsets)



"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ python Q4.py 
=== Q4 Cardinality-Constrained Portfolio Optimisation ===
N=19, K=7, alpha=0.99, RFDaily=0.00023810, tarDaily=0.00044982
Enumerating subsets for K=7: total=50388
  tried=5000, solved=5000, elapsed=13.2s, BestObj=1.221915e-04
  tried=10000, solved=10000, elapsed=27.9s, BestObj=1.221915e-04
  tried=15000, solved=15000, elapsed=39.0s, BestObj=1.221915e-04
  tried=20000, solved=20000, elapsed=49.2s, BestObj=1.221915e-04
  tried=25000, solved=25000, elapsed=63.8s, BestObj=1.221911e-04
  tried=30000, solved=30000, elapsed=74.4s, BestObj=1.214250e-04
  tried=35000, solved=35000, elapsed=86.4s, BestObj=1.214250e-04
  tried=40000, solved=39974, elapsed=97.6s, BestObj=1.214250e-04
  tried=45000, solved=44916, elapsed=114.9s, BestObj=1.214250e-04
  tried=50000, solved=49881, elapsed=126.8s, BestObj=1.214250e-04
Finished: tried=50388, solved=50268, elapsed=127.6s
Enumerating subsets for K=7: total=50388
  tried=5000, solved=5000, elapsed=634.6s, BestObj=3.812375e+00
  tried=10000, solved=10000, elapsed=1304.5s, BestObj=3.812375e+00
  tried=15000, solved=15000, elapsed=1843.4s, BestObj=3.812375e+00
  tried=20000, solved=20000, elapsed=2344.8s, BestObj=3.812375e+00
  tried=25000, solved=25000, elapsed=2890.0s, BestObj=3.812375e+00
  tried=30000, solved=30000, elapsed=3442.4s, BestObj=3.812375e+00
  tried=35000, solved=35000, elapsed=4111.9s, BestObj=3.812375e+00
  tried=40000, solved=39974, elapsed=4803.9s, BestObj=3.812375e+00
  tried=45000, solved=44916, elapsed=5426.6s, BestObj=3.812375e+00
  tried=50000, solved=49881, elapsed=6087.6s, BestObj=3.812375e+00
Finished: tried=50388, solved=50268, elapsed=6135.2s
Enumerating subsets for K=7: total=50388
  tried=5000, solved=5000, elapsed=170.6s, BestObj=6.234370e-05
  tried=10000, solved=10000, elapsed=351.3s, BestObj=6.234370e-05
  tried=15000, solved=15000, elapsed=561.9s, BestObj=6.234370e-05
  tried=20000, solved=20000, elapsed=784.8s, BestObj=6.234370e-05
  tried=25000, solved=25000, elapsed=972.4s, BestObj=6.234370e-05
  tried=30000, solved=30000, elapsed=1174.3s, BestObj=6.224630e-05
  tried=35000, solved=35000, elapsed=1348.0s, BestObj=6.224630e-05
  tried=40000, solved=39974, elapsed=1527.0s, BestObj=6.224630e-05
  tried=45000, solved=44916, elapsed=1643.2s, BestObj=6.224630e-05
  tried=50000, solved=49881, elapsed=1754.8s, BestObj=6.224630e-05
Finished: tried=50388, solved=50268, elapsed=1761.7s

Model 1 (budget + cardinality)
  Inditex: 0.134392
  Grifols: 0.205685
  Ferrovial: 0.095100
  Enagás: 0.184523
  Red Eléctrica de España: 0.244865
  Indra Sistemas: 0.045785
  Iberdrola: 0.089650

Model 2 (budget + cardinality)
  Inditex: 0.364182
  Grifols: 0.267990
  Mapfre: 0.004520
  Red Eléctrica de España: 0.363308

Model 3 (budget + cardinality)
  Inditex: 0.185016
  Grifols: 0.222154
  Ferrovial: 0.095016
  Enagás: 0.150030
  Red Eléctrica de España: 0.251026
  Indra Sistemas: 0.026356
  Iberdrola: 0.070401

=== Q4 Summary: Unconstrained vs Cardinality ===
                       Variant     muP  sigmaP       SR  CVaR99%
       Model 1 (unconstrained) 0.135729 0.174926 0.432920 0.040280
Model 1 (budget + cardinality) 0.135715 0.174926 0.432840 0.040281

       Model 2 (unconstrained) 0.142687 0.182363 0.453420 0.038685
Model 2 (budget + cardinality) 0.147170 0.185346 0.470312 0.038776

       Model 3 (unconstrained) 0.140302 0.175519 0.457514 0.039991
Model 3 (budget + cardinality) 0.140346 0.175499 0.457812 0.039991

Saved: Q4Summary.csv
Saved: Q4Weights.csv
Saved: Q4-M1-CardinalWeights.png
Saved: Q4-M2-CardinalWeights.png
Saved: Q4-M3-CardinalWeights.png

Comment cue for report (Part d):
Cardinality often increases variance due to fewer diversification channels, but CVaR can improve when sparse selection removes assets with extreme downside tails.
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ 
"""
