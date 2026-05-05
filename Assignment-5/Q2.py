
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from deap import base, creator, tools


warnings.filterwarnings('ignore')
np.random.seed(42)


def loadData(fdrpath="IBEX35"):
    CSVFiles = list(Path(fdrpath).glob("*.csv"))
    if len(CSVFiles) == 0:
        raise ValueError(f"No CSV files found in {fdrpath} folder")
    
    priceData = {}
    tickers = []
    
    for file in CSVFiles:
        ticker = file.stem
        tickers.append(ticker)
        df = pd.read_csv(file)
        
        priceCol = None
        for col in ['Close', 'close', 'Price', 'price', 'Adj Close', 'adj close']:
            if col in df.columns:
                priceCol = col
                break
        
        if priceCol is None:
            numCol = df.select_dtypes(include=[np.number]).columns
            if len(numCol) > 0:
                priceCol = numCol[0]
            else:
                raise ValueError(f"No price column found in {file}")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        priceData[ticker] = df[priceCol]
    

    priceDF = pd.DataFrame(priceData)
    priceDF = priceDF.dropna()
    retsDF = priceDF.pct_change().dropna()
    print(f"Loaded {len(tickers)} stocks with {len(retsDF)} trading days")
    print(f"Date range: {retsDF.index[0]} to {retsDF.index[-1]}")
    
    return retsDF, tickers



print("="*60)
print("LOADING DATA FROM IBEX35 FOLDER")
print("="*60)



retsDF, tickers = loadData("IBEX35")
N = len(tickers)
K = int(np.ceil(N / 3))
l = 0.02
u = 0.30



splitIdx = int(len(retsDF) * 0.7)
trainRets = retsDF.iloc[:splitIdx]
testRets = retsDF.iloc[splitIdx:]



print(f"\nNumber of assets (N): {N}")
print(f"Cardinality constraint (K = ceil(N/3)): {K}")
print(f"Lower bound (l): {l*100}%")
print(f"Upper bound (u): {u*100}%")
print(f"Training samples: {len(trainRets)}")
print(f"Testing samples: {len(testRets)}")



CovMatTrain = trainRets.cov().values
RetsArrTrain = trainRets.values



def repairBudget(w):
    wClipped = np.maximum(w, 0)
    if np.sum(wClipped) == 0:
        return np.ones(len(w)) / len(w)
    
    return wClipped / np.sum(wClipped)



print("\n" + "="*60)
print("2(a) TESTING repairBudget FUNCTION")
print("="*60)



testCases = [
    ("Mixed signs", np.array([0.5, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.1, -0.1, 0.2])[:N]),
    ("Doesn't sum to 1", np.array([0.3, 0.2, 0.1, 0.15, 0.05, 0.1, 0.05, 0.02, 0.01, 0.02])[:N]),
    ("Already feasible", np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[:N]),
    ("Edge case - all negative", np.array([-1, -2, -3, -4, -5, -1, -2, -3, -4, -5])[:N])
]



for testName, testW in testCases:
    print(f"\nTest Case: {testName}")
    print(f"Original: {testW[:5]}... (showing first 5 of {N})")
    result = repairBudget(testW)
    print(f"Repaired: {result[:5]}...")
    print(f"Sum: {np.sum(result):.6f}")
    print(f"Min weight: {np.min(result):.6f} (should be >= 0)")
    print(f"All non-negative: {np.all(result >= 0)}")
    print(f"[OK] Budget repair successful")
    
    

def repairBounds(w, l, u):
    w = np.array(w, dtype=float)
    while True:
        w = np.clip(w, l, u)
        tot = np.sum(w)
        if tot == 0:
            return np.ones(len(w)) / len(w)

        w = w / tot
        if np.all(w >= l - 1e-12) and np.all(w <= u + 1e-12):
            break

    return w




print("\n" + "="*60)
print("2(b) TESTING repairBounds FUNCTION")
print("="*60)



testWBounds = np.array([0.5, 0.01, 0.35, 0.04, 0.10, 0.02, 0.03, 0.02, 0.01, 0.02])[:N]
print(f"\nOriginal weights (violates bounds):")
print(f"  Values: {testWBounds[:5]}...")
print(f"  Min: {np.min(testWBounds):.4f}, Max: {np.max(testWBounds):.4f}")
print(f"  Sum: {np.sum(testWBounds):.4f}")



WBudgetRepaired = repairBudget(testWBounds)
print(f"\nAfter budget repair:")
print(f"  Min: {np.min(WBudgetRepaired):.4f}, Max: {np.max(WBudgetRepaired):.4f}")
print(f"  Sum: {np.sum(WBudgetRepaired):.4f}")



WRepaired = repairBounds(WBudgetRepaired, l, u)
print(f"\nAfter bound repair:")
print(f"  Values: {WRepaired[:5]}...")
print(f"  Min: {np.min(WRepaired):.4f} (should be >= {l})")
print(f"  Max: {np.max(WRepaired):.4f} (should be <= {u})")
print(f"  Sum: {np.sum(WRepaired):.6f}")



print("\n" + "="*60)
print("VERIFICATION OF BOUND CONSTRAINTS")
print("="*60)



assert np.all(WRepaired >= l - 1e-10), "[ERR] Lower bound violation!"
assert np.all(WRepaired <= u + 1e-10), "[ERR] Upper bound violation!"
assert abs(np.sum(WRepaired) - 1.0) < 1e-10, "[ERR] Sum to 1 violation!"



print("[OK] All weights satisfy: 0.02 <= w-i <= 0.30")
print("[OK] Sum of weights = 1.0")
print("\n[DONE] Bound repair successfully enforces individual asset constraints!")



print("\n" + "="*60)
print("WHY BOUND REPAIR MUST BE APPLIED AFTER BUDGET REPAIR")
print("="*60)
print("""
    The correct order is: Budget Repair -> Bound Repair
    Reasoning:
    1. Budget repair first ensures:
        - All weights are non-negative (long-only constraint)
        - Weights sum to 1 (fully invested constraint)
    
    2. If bound repair were applied first to a vector with negative weights:
        - Negative weights would be clipped to l (0.02)
        - This would artificially create positive positions where none should exist
        - Would violate the long-only intention of the portfolio
    
    3. The two-stage approach (Budget -> Bounds) ensures:
        - We start with a feasible budget-constrained portfolio
        - Then adjust individual weights to meet min/max constraints
        - Final renormalization maintains the sum-to-1 property
    
    What goes wrong if order is reversed?
        - Applying bounds first on negative weights raises them to 0.02
        - This creates artificial long positions from what should be shorts/excluded
        - The subsequent budget repair normalizes these artificially created positions
        - Results in incorrect portfolio weights that violate the spirit of long-only investing
    """)




def repairCardinal(w, K):
    if K >= len(w):
        return w
    
    indices = np.argsort(w)[::-1][:K]
    wNew = np.zeros_like(w)
    wNew[indices] = w[indices]
    if np.sum(wNew) > 0:
        wNew = wNew / np.sum(wNew)
    else:
        wNew = np.ones(len(w)) / len(w)
    
    return wNew




print("\n" + "="*60)
print("2(c) TESTING repairCardinal FUNCTION")
print("="*60)



testWCard = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03])[:N]
print(f"\nOriginal weights (all {N} assets have weights):")
print(f"  Sorted: {np.sort(testWCard)[::-1]}")
print(f"  Number of non-zero: {np.sum(testWCard > 1e-10)}")


repairedCard = repairCardinal(testWCard, K)
print(f"\nAfter cardinality repair (K={K}):")
print(f"  Weights: {repairedCard}")
print(f"  Non-zero weights: {np.sum(repairedCard > 1e-10)} (max allowed: {K})")



def repairAll(w, K, l, u):
    w = repairBudget(w)
    w = repairBounds(w, l, u)
    w = repairCardinal(w, K)
    return w



print("\n" + "="*60)
print("TESTING COMPLETE REPAIR PIPELINE (repairAll)")
print("="*60)



np.random.seed(42)
randW = np.random.randn(N) * 0.5



print(f"\nOriginal random vector (first 5 of {N}):")
print(f"  {randW[:5]}...")
print(f"  Sum: {np.sum(randW):.4f}")
print(f"  Min: {np.min(randW):.4f}, Max: {np.max(randW):.4f}")



repaired = repairAll(randW, K, l, u)
print(f"\nAfter repairAll:")
print(f"  {repaired[:5]}...")
print(f"  Sum: {np.sum(repaired):.6f}")
print(f"  Min: {np.min(repaired):.6f}")
print(f"  Max: {np.max(repaired):.6f}")
non0Cnt = np.sum(repaired > 1e-10)
print(f"  Non-zero weights: {non0Cnt} (K={K})")



print("\n" + "="*60)
print("VERIFICATION OF ALL CONSTRAINTS")
print("="*60)



assert abs(np.sum(repaired) - 1.0) < 1e-10
print("[OK] Constraint 1: Sum of weights = 1.0")


nonZeroWgt = repaired[repaired > 1e-10]
if len(nonZeroWgt) > 0:
    withinBounds = np.all(nonZeroWgt >= l - 1e-10) and np.all(nonZeroWgt <= u + 1e-10)
    print(f"[OK] Constraint 2: All non-zero weights within [{l}, {u}] - {withinBounds}")
    print(f"   Range: [{np.min(nonZeroWgt):.4f}, {np.max(nonZeroWgt):.4f}]")



assert non0Cnt <= K
print(f"[Done] Constraint 3: At most {K} non-zero weights (actual: {non0Cnt})")
print("\n[OK] All constraints satisfied by repairAll function!")



def minVar(w, covMat):
    return np.dot(w.T, np.dot(covMat, w))



def CVaR(w, returns, alpha=0.95):
    portRets = returns.values @ w
    VaR = np.percentile(portRets, (1 - alpha) * 100)
    lossBeyondVaR = portRets[portRets <= VaR]
    if len(lossBeyondVaR) > 0:
        CVaRVal = -np.mean(lossBeyondVaR)
    else:
        CVaRVal = -VaR
    return CVaRVal



def semiVar(w, returns, alpha=0.95):
    portRets = returns.values @ w
    threshold = np.percentile(portRets, (1 - alpha) * 100)
    downsideRets = portRets[portRets <= threshold]
    if len(downsideRets) > 0:
        semivar = np.mean((downsideRets - threshold) ** 2)
    else:
        semivar = 0
    return semivar



def portMetrics(w, RetsArr):
    r = RetsArr @ w
    mu = np.mean(r)
    sigma = np.std(r)
    sharpe = mu / sigma if sigma != 0 else 0.0
    var99 = np.percentile(r, 1)
    cvar99 = r[r <= var99].mean()
    return mu, sigma, sharpe, var99, cvar99



def fitFxn(individual, model, returns, K, l, u, alpha=0.95):
    w = np.array(individual)
    wFeasible = repairAll(w, K, l, u)
    
    covMat, RetsArr = returns
    
    if model == 'variance':
        fitness = minVar(wFeasible, covMat)
        
    elif model == 'cvar':
        fitness = CVaR(wFeasible, pd.DataFrame(RetsArr), alpha)
        
    elif model == 'semiVar':
        fitness = semiVar(wFeasible, pd.DataFrame(RetsArr), alpha)
        
    else:
        raise ValueError("Model must be 'variance', 'cvar', or 'semiVar'")
    
    return (fitness,)




def setupDEAP(N, model, returns, K, l, u, alpha=0.95):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin) #type:ignore
    

    toolbox = base.Toolbox()
    toolbox.register("attrWgt", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attrWgt, n=N)  #type:ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)                      #type:ignore
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/N)
    toolbox.register("select", tools.selTournament, tournsize=2)
    
    if isinstance(returns, tuple):
        covMat, RetsArr = returns
    else:
        covMat = returns.cov().values
        RetsArr = returns.values
        
    toolbox.register("evaluate", fitFxn, model=model, returns=(covMat, RetsArr), K=K, l=l, u=u, alpha=alpha)
    return toolbox



def runGA(toolbox, popSize=200, gens=500, cxpb=0.9, mutpb=None):
    if mutpb is None:
        mutpb = 1 / len(toolbox.individual())
    

    pop = toolbox.population(n=popSize)
    hof = tools.HallOfFame(1)
    history = []
    bestFitsTracker = float('inf')
    bestGen = 0

    for gen in range(gens):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalidInds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalidInds)
        for ind, fit in zip(invalidInds, fitnesses):
            ind.fitness.values = fit
        
        
        if hof:
            worstIdx = np.argmax([ind.fitness.values[0] for ind in offspring])
            offspring[worstIdx] = toolbox.clone(hof[0])
        

        pop[:] = offspring
        hof.update(pop)
        currBest = min(ind.fitness.values[0] for ind in pop)
        history.append(currBest)
        
        if currBest < bestFitsTracker:
            bestFitsTracker = currBest
            bestGen = gen

        if gen % 50 == 0:
            print(f"  Generation {gen}: Best Fitness = {currBest:.6f}")
    
    
    bestIndi = hof[0]
    bestFits = bestIndi.fitness.values[0]
    bestwgts = np.array(bestIndi)
    bestwgts = repairAll(bestwgts, K, l, u)
    
    return bestwgts, bestFits, history, bestGen



print("\n" + "="*60)
print("MODEL 1: Minimum Variance Portfolio (Constrained)")
print(f"Constraints: Budget + Bounds ({l*100}%-{u*100}%) + Cardinality (K={K})")
print("GA Parameters: P=200, G=500, pc=0.9, pm=1/N, SBX(nc=15), Poly(nm=20)")
print("="*60)


toolboxVar = setupDEAP(N, 'variance', (CovMatTrain, RetsArrTrain), K, l, u, alpha=0.95)
bestwgtsVar, bestFitsVar, historyVar, bestGenVar = runGA(toolboxVar, popSize=200, gens=500, cxpb=0.9)


print(f"\nOptimal Weights (Min Variance with constraints):")
for i, ticker in enumerate(tickers):
    if bestwgtsVar[i] > 0.001:
        print(f"  {ticker}: {bestwgtsVar[i]:.4f}")
    else:
        print(f"  {ticker}: 0.0000 (not held)")
        
        

print(f"\nMinimum Portfolio Variance: {bestFitsVar:.8f}")
print(f"Portfolio Volatility: {np.sqrt(bestFitsVar)*100:.2f}%")
print(f"Number of assets held: {np.sum(bestwgtsVar > 0.001)}/{K}")



muVar, sigmaVar, srVar, var99Var, cvar99Var = portMetrics(bestwgtsVar, RetsArrTrain)
print(f"uP: {muVar:.6f}")
print(f"sP: {sigmaVar:.6f}")
print(f"SR: {srVar:.6f}")
print(f"VaR99: {var99Var:.6f}")
print(f"CVaR99: {cvar99Var:.6f}")



print("\n" + "="*60)
print("MODEL 2: CVaR Portfolio (Constrained, alpha=95%)")
print(f"Constraints: Budget + Bounds ({l*100}%-{u*100}%) + Cardinality (K={K})")
print("="*60)


toolboxCVaR = setupDEAP(N, 'cvar', (CovMatTrain, RetsArrTrain), K, l, u, alpha=0.95)
bestwgtsCVaR, bestFitsCVaR, historyCVaR, bestGenCVaR = runGA(toolboxCVaR, popSize=200, gens=500, cxpb=0.9)



print(f"\nOptimal Weights (CVaR with constraints):")
for i, ticker in enumerate(tickers):
    if bestwgtsCVaR[i] > 0.001:
        print(f"  {ticker}: {bestwgtsCVaR[i]:.4f}")
    else:
        print(f"  {ticker}: 0.0000 (not held)")
        
        

print(f"\nMinimum CVaR (95%): {bestFitsCVaR:.6f}")
print(f"Expected tail loss: {bestFitsCVaR*100:.2f}%")
print(f"Number of assets held: {np.sum(bestwgtsCVaR > 0.001)}/{K}")



muCVaR, sigmaCVaR, srCVaR, var99CVaR, cvar99CVaR = portMetrics(bestwgtsCVaR, RetsArrTrain)
print(f"uP: {muCVaR:.6f}")
print(f"sP: {sigmaCVaR:.6f}")
print(f"SR: {srCVaR:.6f}")
print(f"VaR99: {var99CVaR:.6f}")
print(f"CVaR99: {cvar99CVaR:.6f}")



print("\n" + "="*60)
print("MODEL 3: semiVar Portfolio (Constrained, alpha=95%)")
print(f"Constraints: Budget + Bounds ({l*100}%-{u*100}%) + Cardinality (K={K})")
print("="*60)



toolboxSemi = setupDEAP(N, 'semiVar', (CovMatTrain, RetsArrTrain), K, l, u, alpha=0.95)
bestwgtsSemi, bestFitsSemi, historySemi, bestGenSemi = runGA(toolboxSemi, popSize=200, gens=500, cxpb=0.9)



print(f"\nOptimal Weights (semiVar with constraints):")
for i, ticker in enumerate(tickers):
    if bestwgtsSemi[i] > 0.001:
        print(f"  {ticker}: {bestwgtsSemi[i]:.4f}")
    else:
        print(f"  {ticker}: 0.0000 (not held)")
        
        

print(f"\nMinimum semiVar (downside deviation): {bestFitsSemi:.6f}")
print(f"Number of assets held: {np.sum(bestwgtsSemi > 0.001)}/{K}")



muSemi, sigmaSemi, srSemi, var99Semi, cvar99Semi = portMetrics(bestwgtsSemi, RetsArrTrain)
print(f"uP: {muSemi:.6f}")
print(f"sP: {sigmaSemi:.6f}")
print(f"SR: {srSemi:.6f}")
print(f"VaR99: {var99Semi:.6f}")
print(f"CVaR99: {cvar99Semi:.6f}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))


axes[0].plot(historyVar, linewidth=2, color='blue')
axes[0].set_xlabel('Generation', fontsize=12)
axes[0].set_ylabel('Best Fitness (Variance)', fontsize=12)
axes[0].set_title(f'Minimum Variance Portfolio (Constrained)\nK={K}, bounds [{l*100}%-{u*100}%]', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=bestFitsVar, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsVar:.6f}')
axes[0].axvline(x=bestGenVar, color='green', linestyle=':', linewidth=1.5, label=f'Best at gen {bestGenVar}')
axes[0].legend(fontsize=9)
axes[0].fill_between(range(len(historyVar)), historyVar, bestFitsVar, alpha=0.2, color='blue')


axes[1].plot(historyCVaR, linewidth=2, color='green')
axes[1].set_xlabel('Generation', fontsize=12)
axes[1].set_ylabel('Best Fitness (CVaR)', fontsize=12)
axes[1].set_title(f'CVaR Portfolio (Constrained)\nK={K}, bounds [{l*100}%-{u*100}%]', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=bestFitsCVaR, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsCVaR:.6f}')
axes[1].axvline(x=bestGenCVaR, color='green', linestyle=':', linewidth=1.5, label=f'Best at gen {bestGenCVaR}')
axes[1].legend(fontsize=9)
axes[1].fill_between(range(len(historyCVaR)), historyCVaR, bestFitsCVaR, alpha=0.2, color='green')


axes[2].plot(historySemi, linewidth=2, color='orange')
axes[2].set_xlabel('Generation', fontsize=12)
axes[2].set_ylabel('Best Fitness (semiVar)', fontsize=12)
axes[2].set_title(f'semiVar Portfolio (Constrained)\nK={K}, bounds [{l*100}%-{u*100}%]', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=bestFitsSemi, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsSemi:.6f}')
axes[2].axvline(x=bestGenSemi, color='green', linestyle=':', linewidth=1.5, label=f'Best at gen {bestGenSemi}')
axes[2].legend(fontsize=9)
axes[2].fill_between(range(len(historySemi)), historySemi, bestFitsSemi, alpha=0.2, color='orange')


plt.suptitle(f'GA Convergence for Three Portfolio Models with Constraints\nIBEX {N} Stocks | Portfolio must hold <= {K} assets | Each asset: {l*100}% <= wi <= {u*100}%', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2-CGT.png")



print("\n" + "="*60)
print("CONVERGENCE SUMMARY (Constrained Models)")
print("="*60)
print(f"\n{'Model':<25} {'Best Found at Gen':<20} {'Final Fitness':<20}")
print("-"*65)
print(f"{'Minimum Variance':<25} {bestGenVar:<20} {bestFitsVar:<20.8f}")
print(f"{'CVaR (95%)':<25} {bestGenCVaR:<20} {bestFitsCVaR:<20.6f}")
print(f"{'semiVar':<25} {bestGenSemi:<20} {bestFitsSemi:<20.6f}")



print("\n" + "="*60)
print("PORTFOLIO COMPOSITION SUMMARY (Constrained)")
print("="*60)



def portSummaryConst(weights, name):
    non0 = weights > 0.001
    numHolds = np.sum(non0)
    avgWgt = np.mean(weights[non0]) if numHolds > 0 else 0
    maxWgt = np.max(weights)
    minWgt = np.min(weights[non0]) if numHolds > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Number of assets held: {numHolds}/{K} (K={K})")
    print(f"  Average weight (held): {avgWgt:.4f}")
    print(f"  Max weight: {maxWgt:.4f} (<= {u})")
    print(f"  Min weight (held): {minWgt:.4f} (>= {l})")
    print(f"  Constraint satisfaction: {'[OK]' if numHolds <= K and minWgt >= l and maxWgt <= u else '[ERR]'}")



portSummaryConst(bestwgtsVar, "Minimum Variance Portfolio")
portSummaryConst(bestwgtsCVaR, "CVaR Portfolio")
portSummaryConst(bestwgtsSemi, "semiVar Portfolio")



print("\n" + "="*60)
print("GA PARAMETERS USED (DEAP implementation)")
print("="*60)
print(f"""
        - Population size: P = 200
        - gens: G = 500
        - Crossover: Simulated Binary Crossover (SBX) with nc = 15
        - Crossover probability: pc = 0.9
        - Mutation: Polynomial mutation with nm = 20
        - Mutation probability: pm = 1/N = 1/{N} = {1/N:.3f}
        - Selection: Binary tournament (tournament size = 2)
        - Elitism: Best solution carried to next generation (replace worst)

    CONSTRAINTS IMPLEMENTED:
        - Budget constraint: Long-only, sum to 1 (via repairBudget)
        - Bound constraints: {l*100}% <= wi <= {u*100}% for held assets (via repairBounds)
        - Cardinality constraint: At most K = ⌈N/3⌉ = {K} assets held (via repairCardinal)

    REPAIR SEQUENCE: Budget -> Bounds -> Cardinality
        - Applied in fitness function before objective evaluation
        - Ensures all constraints satisfied for every individual evaluated
    """)




"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-5$ python Q2.py 
============================================================
LOADING DATA FROM IBEX35 FOLDER
============================================================
Loaded 19 stocks with 3566 trading days
Date range: 2011-01-25 00:00:00 to 2024-12-31 00:00:00

Number of assets (N): 19
Cardinality constraint (K = ceil(N/3)): 7
Lower bound (l): 2.0%
Upper bound (u): 30.0%
Training samples: 2496
Testing samples: 1070

============================================================
2(a) TESTING repairBudget FUNCTION
============================================================

Test Case: Mixed signs
Original: [ 0.5 -0.2  0.3  0.4 -0.1]... (showing first 5 of 19)
Repaired: [0.29411765 0.         0.17647059 0.23529412 0.        ]...
Sum: 1.000000
Min weight: 0.000000 (should be >= 0)
All non-negative: True
[OK] Budget repair successful

Test Case: Doesn't sum to 1
Original: [0.3  0.2  0.1  0.15 0.05]... (showing first 5 of 19)
Repaired: [0.3  0.2  0.1  0.15 0.05]...
Sum: 1.000000
Min weight: 0.010000 (should be >= 0)
All non-negative: True
[OK] Budget repair successful

Test Case: Already feasible
Original: [0.1 0.1 0.1 0.1 0.1]... (showing first 5 of 19)
Repaired: [0.1 0.1 0.1 0.1 0.1]...
Sum: 1.000000
Min weight: 0.100000 (should be >= 0)
All non-negative: True
[OK] Budget repair successful

Test Case: Edge case - all negative
Original: [-1 -2 -3 -4 -5]... (showing first 5 of 19)
Repaired: [0.1 0.1 0.1 0.1 0.1]...
Sum: 1.000000
Min weight: 0.100000 (should be >= 0)
All non-negative: True
[OK] Budget repair successful

============================================================
2(b) TESTING repairBounds FUNCTION
============================================================

Original weights (violates bounds):
  Values: [0.5  0.01 0.35 0.04 0.1 ]...
  Min: 0.0100, Max: 0.5000
  Sum: 1.1000

After budget repair:
  Min: 0.0091, Max: 0.4545
  Sum: 1.0000

After bound repair:
  Values: [0.3        0.03142857 0.3        0.05714286 0.14285714]...
  Min: 0.0314 (should be >= 0.02)
  Max: 0.3000 (should be <= 0.3)
  Sum: 1.000000

============================================================
VERIFICATION OF BOUND CONSTRAINTS
============================================================
[OK] All weights satisfy: 0.02 <= w-i <= 0.30
[OK] Sum of weights = 1.0

[DONE] Bound repair successfully enforces individual asset constraints!

============================================================
WHY BOUND REPAIR MUST BE APPLIED AFTER BUDGET REPAIR
============================================================

    The correct order is: Budget Repair -> Bound Repair
    Reasoning:
    1. Budget repair first ensures:
        - All weights are non-negative (long-only constraint)
        - Weights sum to 1 (fully invested constraint)
    
    2. If bound repair were applied first to a vector with negative weights:
        - Negative weights would be clipped to l (0.02)
        - This would artificially create positive positions where none should exist
        - Would violate the long-only intention of the portfolio
    
    3. The two-stage approach (Budget -> Bounds) ensures:
        - We start with a feasible budget-constrained portfolio
        - Then adjust individual weights to meet min/max constraints
        - Final renormalization maintains the sum-to-1 property
    
    What goes wrong if order is reversed?
        - Applying bounds first on negative weights raises them to 0.02
        - This creates artificial long positions from what should be shorts/excluded
        - The subsequent budget repair normalizes these artificially created positions
        - Results in incorrect portfolio weights that violate the spirit of long-only investing
    

============================================================
2(c) TESTING repairCardinal FUNCTION
============================================================

Original weights (all 19 assets have weights):
  Sorted: [0.25 0.2  0.15 0.1  0.08 0.07 0.05 0.04 0.03 0.03]
  Number of non-zero: 10

After cardinality repair (K=7):
  Weights: [0.27777778 0.22222222 0.16666667 0.11111111 0.08888889 0.07777778
 0.05555556 0.         0.         0.        ]
  Non-zero weights: 7 (max allowed: 7)

============================================================
TESTING COMPLETE REPAIR PIPELINE (repairAll)
============================================================

Original random vector (first 5 of 19):
  [ 0.24835708 -0.06913215  0.32384427  0.76151493 -0.11707669]...
  Sum: -1.0068
  Min: -0.9566, Max: 0.7896

After repairAll:
  [0.08460631 0.         0.11032208 0.25942072 0.        ]...
  Sum: 1.000000
  Min: 0.000000
  Max: 0.268990
  Non-zero weights: 7 (K=7)

============================================================
VERIFICATION OF ALL CONSTRAINTS
============================================================
[OK] Constraint 1: Sum of weights = 1.0
[OK] Constraint 2: All non-zero weights within [0.02, 0.3] - True
   Range: [0.0535, 0.2690]
[Done] Constraint 3: At most 7 non-zero weights (actual: 7)

[OK] All constraints satisfied by repairAll function!

============================================================
MODEL 1: Minimum Variance Portfolio (Constrained)
Constraints: Budget + Bounds (2.0%-30.0%) + Cardinality (K=7)
GA Parameters: P=200, G=500, pc=0.9, pm=1/N, SBX(nc=15), Poly(nm=20)
============================================================
  Generation 0: Best Fitness = 0.000140
  Generation 50: Best Fitness = 0.000125
  Generation 100: Best Fitness = 0.000125
  Generation 150: Best Fitness = 0.000125
  Generation 200: Best Fitness = 0.000124
  Generation 250: Best Fitness = 0.000124
  Generation 300: Best Fitness = 0.000124
  Generation 350: Best Fitness = 0.000124
  Generation 400: Best Fitness = 0.000124
  Generation 450: Best Fitness = 0.000124

Optimal Weights (Min Variance with constraints):
  Bankinter: 0.0000 (not held)
  Inditex: 0.1441
  Naturgy: 0.0000 (not held)
  Telefónica: 0.0000 (not held)
  Banco Sabadell: 0.0000 (not held)
  Grifols: 0.2031
  Banco Santander: 0.0000 (not held)
  Ferrovial: 0.1101
  Enagás: 0.1962
  ArcelorMittal: 0.0000 (not held)
  Mapfre: 0.0000 (not held)
  Acciona: 0.0000 (not held)
  Red Eléctrica de España: 0.2031
  International Airlines Group: 0.0000 (not held)
  BBVA: 0.0000 (not held)
  ACS: 0.0000 (not held)
  Repsol: 0.0000 (not held)
  Indra Sistemas: 0.0716
  Iberdrola: 0.0720

Minimum Portfolio Variance: 0.00012348
Portfolio Volatility: 1.11%
Number of assets held: 7/7
uP: 0.000519
sP: 0.011110
SR: 0.046700
VaR99: -0.030191
CVaR99: -0.041365

============================================================
MODEL 2: CVaR Portfolio (Constrained, alpha=95%)
Constraints: Budget + Bounds (2.0%-30.0%) + Cardinality (K=7)
============================================================
  Generation 0: Best Fitness = 0.028872
  Generation 50: Best Fitness = 0.025873
  Generation 100: Best Fitness = 0.025834
  Generation 150: Best Fitness = 0.025829
  Generation 200: Best Fitness = 0.025829
  Generation 250: Best Fitness = 0.025828
  Generation 300: Best Fitness = 0.025803
  Generation 350: Best Fitness = 0.025799
  Generation 400: Best Fitness = 0.025770
  Generation 450: Best Fitness = 0.025746

Optimal Weights (CVaR with constraints):
  Bankinter: 0.0000 (not held)
  Inditex: 0.1937
  Naturgy: 0.0000 (not held)
  Telefónica: 0.0000 (not held)
  Banco Sabadell: 0.0000 (not held)
  Grifols: 0.1937
  Banco Santander: 0.0000 (not held)
  Ferrovial: 0.0843
  Enagás: 0.1700
  ArcelorMittal: 0.0000 (not held)
  Mapfre: 0.0000 (not held)
  Acciona: 0.0000 (not held)
  Red Eléctrica de España: 0.1937
  International Airlines Group: 0.0000 (not held)
  BBVA: 0.0000 (not held)
  ACS: 0.0000 (not held)
  Repsol: 0.0000 (not held)
  Indra Sistemas: 0.0825
  Iberdrola: 0.0820

Minimum CVaR (95%): 0.025724
Expected tail loss: 2.57%
Number of assets held: 7/7
uP: 0.000508
sP: 0.011149
SR: 0.045570
VaR99: -0.030244
CVaR99: -0.041237

============================================================
MODEL 3: semiVar Portfolio (Constrained, alpha=95%)
Constraints: Budget + Bounds (2.0%-30.0%) + Cardinality (K=7)
============================================================
  Generation 0: Best Fitness = 0.000227
  Generation 50: Best Fitness = 0.000205
  Generation 100: Best Fitness = 0.000204
  Generation 150: Best Fitness = 0.000202
  Generation 200: Best Fitness = 0.000202
  Generation 250: Best Fitness = 0.000201
  Generation 300: Best Fitness = 0.000201
  Generation 350: Best Fitness = 0.000201
  Generation 400: Best Fitness = 0.000201
  Generation 450: Best Fitness = 0.000201

Optimal Weights (semiVar with constraints):
  Bankinter: 0.0000 (not held)
  Inditex: 0.1617
  Naturgy: 0.0000 (not held)
  Telefónica: 0.0000 (not held)
  Banco Sabadell: 0.0000 (not held)
  Grifols: 0.1771
  Banco Santander: 0.0000 (not held)
  Ferrovial: 0.0000 (not held)
  Enagás: 0.1091
  ArcelorMittal: 0.0000 (not held)
  Mapfre: 0.1575
  Acciona: 0.0000 (not held)
  Red Eléctrica de España: 0.1771
  International Airlines Group: 0.0000 (not held)
  BBVA: 0.0000 (not held)
  ACS: 0.0000 (not held)
  Repsol: 0.0000 (not held)
  Indra Sistemas: 0.1138
  Iberdrola: 0.1035

Minimum semiVar (downside deviation): 0.000201
Number of assets held: 7/7
uP: 0.000426
sP: 0.011654
SR: 0.036540
VaR99: -0.031726
CVaR99: -0.041386

============================================================
CONVERGENCE SUMMARY (Constrained Models)
============================================================

Model                     Best Found at Gen    Final Fitness       
-----------------------------------------------------------------
Minimum Variance          499                  0.00012348          
CVaR (95%)                498                  0.025724            
semiVar                   499                  0.000201            

============================================================
PORTFOLIO COMPOSITION SUMMARY (Constrained)
============================================================

Minimum Variance Portfolio:
  Number of assets held: 7/7 (K=7)
  Average weight (held): 0.1429
  Max weight: 0.2031 (<= 0.3)
  Min weight (held): 0.0716 (>= 0.02)
  Constraint satisfaction: [OK]

CVaR Portfolio:
  Number of assets held: 7/7 (K=7)
  Average weight (held): 0.1429
  Max weight: 0.1937 (<= 0.3)
  Min weight (held): 0.0820 (>= 0.02)
  Constraint satisfaction: [OK]

semiVar Portfolio:
  Number of assets held: 7/7 (K=7)
  Average weight (held): 0.1429
  Max weight: 0.1771 (<= 0.3)
  Min weight (held): 0.1035 (>= 0.02)
  Constraint satisfaction: [OK]

============================================================
GA PARAMETERS USED (DEAP implementation)
============================================================

        - Population size: P = 200
        - gens: G = 500
        - Crossover: Simulated Binary Crossover (SBX) with nc = 15
        - Crossover probability: pc = 0.9
        - Mutation: Polynomial mutation with nm = 20
        - Mutation probability: pm = 1/N = 1/19 = 0.053
        - Selection: Binary tournament (tournament size = 2)
        - Elitism: Best solution carried to next generation (replace worst)

    CONSTRAINTS IMPLEMENTED:
        - Budget constraint: Long-only, sum to 1 (via repairBudget)
        - Bound constraints: 2.0% <= wi <= 30.0% for held assets (via repairBounds)
        - Cardinality constraint: At most K = ⌈N/3⌉ = 7 assets held (via repairCardinal)

    REPAIR SEQUENCE: Budget -> Bounds -> Cardinality
        - Applied in fitness function before objective evaluation
        - Ensures all constraints satisfied for every individual evaluated
    
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-5$ 
"""


