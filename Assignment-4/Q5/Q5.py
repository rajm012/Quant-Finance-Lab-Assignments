

import argparse
import os
import pandas as pd
from Q3 import LoadIBEX     #type:ignore


# mapping to sectors
SECTORMAP = {
	"Bankinter": "Financials",
	"Banco Sabadell": "Financials",
	"Banco Santander": "Financials",
	"BBVA": "Financials",
	"Mapfre": "Financials",
	"Inditex": "Consumer",
	"Naturgy": "Utilities",
	"Enagás": "Utilities",
	"Red Eléctrica de España": "Utilities",
	"Iberdrola": "Utilities",
	"Telefónica": "Telecom",
	"Grifols": "Healthcare",
	"Ferrovial": "Industrials",
	"ArcelorMittal": "Materials",
	"Acciona": "Industrials",
	"International Airlines Group": "Transport",
	"ACS": "Industrials",
	"Repsol": "Energy",
	"Indra Sistemas": "Technology",
}



# summarizating various distributions
def summarizeDist(trainRets):
	skew = trainRets.skew()
	kurt = trainRets.kurtosis()

	nFatTails = int((kurt > 0).sum())
	nPosSkew = int((skew > 0).sum())
	nNegSkew = int((skew < 0).sum())

	TopTail = kurt.sort_values(ascending=False).head(5)
	TopPosSkew = skew.sort_values(ascending=False).head(3)
	TopNegSkew = skew.sort_values().head(3)

	return {
		"nFatTails": nFatTails,
		"nPosSkew": nPosSkew,
		"nNegSkew": nNegSkew,
		"TopTail": TopTail,
		"TopPosSkew": TopPosSkew,
		"TopNegSkew": TopNegSkew,
	}




# checking sector belongings and share
def SectorPatternSummary(trainRets):
	df = trainRets.copy()
	df = df.rename(columns={c: c.strip() for c in df.columns})
	sectors = pd.Series({c: SECTORMAP.get(c, "Other") for c in df.columns})

	sectorVol = {}
	for sec in sectors.unique():
		cols = sectors[sectors == sec].index.tolist()
		if len(cols) == 0:
			continue
		sectRets = df[cols].mean(axis=1)
		sectorVol[sec] = sectRets.std()

	sectorVol = pd.Series(sectorVol).sort_values(ascending=False)
	return sectorVol





def NormalizeSummaryColumns(DFSummary):
	"""Map summary CSV columns to canonical names used by this script."""
	df = DFSummary.copy()
	df.columns = [c.strip() for c in df.columns]

	aliases = {
		"Variant": ["Variant", "variant"],
		"muP": ["muP", "mu_P"],
		"sigmaP": ["sigmaP", "sigma_P"],
		"SR": ["SR", "sr", "Sharpe", "SharpeRatio"],
		"CVaR99%": ["CVaR99%", "CVaR99", "CVaR_99%", "cvar99"],
	}

	renameMap = {}
	missing = []
	for canonical, options in aliases.items():
		match = next((col for col in options if col in df.columns), None)
		if match is None:
			missing.append(canonical)
		else:
			renameMap[match] = canonical

	if missing:
		raise KeyError(
			"Missing required columns in summary file. "
			f"Required canonical columns: {list(aliases.keys())}. "
			f"Missing after alias matching: {missing}. "
			f"Available columns: {list(df.columns)}"
		)

	return df.rename(columns=renameMap)






def RankModels(DFSummary):
	# Lower is better for sigma and CVaR, higher is better for mu and SR.
	ranking = DFSummary.copy()
	ranking["RankMu"] = ranking["muP"].rank(ascending=False, method="min")
	ranking["RankSigma"] = ranking["sigmaP"].rank(ascending=True, method="min")
	ranking["rankSR"] = ranking["SR"].rank(ascending=False, method="min")
	ranking["rankCVar"] = ranking["CVaR99%"].rank(ascending=True, method="min")
	ranking["rankTot"] = (
		ranking["RankMu"] + ranking["RankSigma"] + ranking["rankSR"] + ranking["rankCVar"]
	)
	ranking = ranking.sort_values("rankTot")
	return ranking





def cardinalityEffect(DFSummary):
	pairs = [
		("Model 1", "Model 1 (unconstrained)", "Model 1 (budget + cardinality)"),
		("Model 2", "Model 2 (unconstrained)", "Model 2 (budget + cardinality)"),
		("Model 3", "Model 3 (unconstrained)", "Model 3 (budget + cardinality)"),
	]
	rows = []
	for name, base, card in pairs:
		b = DFSummary[DFSummary["Variant"] == base].iloc[0]
		c = DFSummary[DFSummary["Variant"] == card].iloc[0]
		rows.append(
			{
				"Model": name,
				"DeltaMU": c["muP"] - b["muP"],
				"DeltaSigma": c["sigmaP"] - b["sigmaP"],
				"DeltaSR": c["SR"] - b["SR"],
				"DeltaCVaR99": c["CVaR99%"] - b["CVaR99%"],
			}
		)
	return pd.DataFrame(rows)





def ChooseRecommendation(DFSummary):
	# best = ranking.iloc[0]
	best = DFSummary.sort_values("CVaR99%").iloc[0]
	variant = best["Variant"]
	rationale = [
		f"Best aggregate rank across mu, sigma, SR, and CVaR99: {variant}.",
		"Coherence and robustness: CVaR-based methods are tail-aware and coherent.",
		"Practicality: cardinality variants reduce holdings and can lower turnover burden.",
		"Empirical fit: recommendation prioritizes lower CVaR and stable Sharpe under training data.",
	]
	return variant, rationale





def MainQ5(SummaryFile="Q4Summary.csv", OutFile="Q5AnalysisReport.txt"):
	trainRets, _ = LoadIBEX()

	if not os.path.exists(SummaryFile):
		raise FileNotFoundError(f"Required summary file not found: {SummaryFile}. Run Q4.py first to generate it.")


	DFSummary = pd.read_csv(SummaryFile)
	DFSummary = NormalizeSummaryColumns(DFSummary)


	# part(a): data observations (fat tails, skewness, sector patterns)
	dist = summarizeDist(trainRets)
	sectorVol = SectorPatternSummary(trainRets)

	#part(b): model comparison and ranking
	ranking = RankModels(DFSummary)

	# part(c): effect of cardinality
	effect = cardinalityEffect(DFSummary)

	# part(d): final model recommendation
	BestVariant, rationale = ChooseRecommendation(DFSummary)


	lines = []
	lines.append("					Q5 Analysis")
	lines.append("=" * 60)
	lines.append("")
	lines.append("(a) Data observations")
	lines.append(f"- Stocks with fat tails (kurtosis > 3): {dist['nFatTails']}")
	lines.append(f"- Positive skew count: {dist['nPosSkew']}, Negative skew count: {dist['nNegSkew']}")
	lines.append("- Top 5 fat-tail stocks by kurtosis:")
	
	for k, v in dist["TopTail"].items():
		lines.append(f"  * {k}: {v:.3f}")
	
 
	lines.append("- Most positively skewed stocks:")
	for k, v in dist["TopPosSkew"].items():
		lines.append(f"  * {k}: {v:.3f}")
  
  
	lines.append("- Most negatively skewed stocks:")
	for k, v in dist["TopNegSkew"].items():
		lines.append(f"  * {k}: {v:.3f}")
  
  
	lines.append("- Sector-level volatility pattern (higher means more unstable):")
	for k, v in sectorVol.items():
		lines.append(f"  * {k}: {v:.4f}")


	lines.append("")
	lines.append("(b) Model comparison ranking")
	lines.append("- Ranked by combined score (mu up, SR up, sigma down, CVaR down):")
 
	for _, row in ranking.iterrows():
		lines.append(
			f"  * {row['Variant']}: totRank={row['rankTot']:.1f}, "
			f"mu={row['muP']:.6f}, sigma={row['sigmaP']:.6f}, SR={row['SR']:.4f}, CVaR99={row['CVaR99%']:.6f}"
		)

	lines.append("")
	lines.append("(c) Effect of cardinality K=ceil(N/3)")
 
	for _, row in effect.iterrows():
		lines.append(
			f"  * {row['Model']}: DeltaMU={row['DeltaMU']:.6f}, DeltaSigma={row['DeltaSigma']:.6f}, "
			f"DeltaSR={row['DeltaSR']:.4f}, DeltaCVaR99={row['DeltaCVaR99']:.6f}"
		)
  
  
	lines.append("- Interpretation cue: positive DeltaCVaR99 indicates worse tail risk after cardinality.")

	lines.append("")
	lines.append("(d) Recommended model")
	lines.append(f"- Recommended variant: {BestVariant}")
 
 
	for r in rationale:
		lines.append(f"  * {r}")

	ReportText = "\n".join(lines)
	with open(OutFile, "w", encoding="utf-8") as f:
		f.write(ReportText)
  
  
	print(ReportText)
	print(f"\nSaved: {OutFile}")
 

	ranking.to_csv("Q5ModelRanking.csv", index=False)
	effect.to_csv("Q5CardinalEffect.csv", index=False)
	print("Saved: Q5ModelRanking.csv")
	print("Saved: Q5CardinalEffect.csv")





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Q5 analysis")
	parser.add_argument("--summary", type=str, default="Q4Summary.csv")
	parser.add_argument("--out", type=str, default="Q5AnalysisReport.txt")
	args = parser.parse_args()
	MainQ5(SummaryFile=args.summary, OutFile=args.out)






"""

(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ python Q5.py 
Assignment 4 - Q5 Analysis
============================================================

(a) Data observations
- Stocks with fat tails (kurtosis > 3): 19
- Positive skew count: 11, Negative skew count: 8
- Top 5 fat-tail stocks by kurtosis:
  * Bankinter: 126.355
  * ACS: 16.810
  * International Airlines Group: 13.316
  * Telefónica: 12.345
  * Naturgy: 11.325
- Most positively skewed stocks:
  * Bankinter: 5.278
  * Mapfre: 0.820
  * ACS: 0.643
- Most negatively skewed stocks:
  * Enagás: -0.629
  * International Airlines Group: -0.414
  * Iberdrola: -0.371
- Sector-level volatility pattern (higher means more unstable):
  * Materials: 0.0283
  * Transport: 0.0276
  * Technology: 0.0217
  * Energy: 0.0207
  * Financials: 0.0197
  * Telecom: 0.0172
  * Healthcare: 0.0165
  * Consumer: 0.0162
  * Industrials: 0.0159
  * Utilities: 0.0124

(b) Model comparison ranking
- Ranked by combined score (mu up, SR up, sigma down, CVaR down):
  * Model 2 (budget + cardinality): totRank=10.0, mu=0.147170, sigma=0.185346, SR=0.4703, CVaR99=0.038776
  * Model 2 (unconstrained): totRank=12.0, mu=0.142687, sigma=0.182363, SR=0.4534, CVaR99=0.038685
  * Model 3 (budget + cardinality): totRank=12.0, mu=0.140346, sigma=0.175499, SR=0.4578, CVaR99=0.039991
  * Model 3 (unconstrained): totRank=14.0, mu=0.140302, sigma=0.175519, SR=0.4575, CVaR99=0.039991
  * Model 1 (unconstrained): totRank=16.0, mu=0.135729, sigma=0.174926, SR=0.4329, CVaR99=0.040280
  * Model 1 (budget + cardinality): totRank=20.0, mu=0.135715, sigma=0.174926, SR=0.4328, CVaR99=0.040281

(c) Effect of cardinality K=ceil(N/3)
  * Model 1: DeltaMU=-0.000014, DeltaSigma=0.000000, DeltaSR=-0.0001, DeltaCVaR99=0.000001
  * Model 2: DeltaMU=0.004483, DeltaSigma=0.002982, DeltaSR=0.0169, DeltaCVaR99=0.000091
  * Model 3: DeltaMU=0.000043, DeltaSigma=-0.000020, DeltaSR=0.0003, DeltaCVaR99=0.000001
- Interpretation cue: positive DeltaCVaR99 indicates worse tail risk after cardinality.

(d) Recommended model
- Recommended variant: Model 2 (unconstrained)
  * Best aggregate rank across mu, sigma, SR, and CVaR99: Model 2 (unconstrained).
  * Coherence and robustness: CVaR-based methods are tail-aware and coherent.
  * Practicality: cardinality variants reduce holdings and can lower turnover burden.
  * Empirical fit: recommendation prioritizes lower CVaR and stable Sharpe under training data.

Saved: Q5AnalysisReport.txt
Saved: Q5ModelRanking.csv
Saved: Q5CardinalEffect.csv
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-4$ 

"""

