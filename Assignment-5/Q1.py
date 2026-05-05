
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


splitIdx = int(len(retsDF) * 0.7)
trainRets = retsDF.iloc[:splitIdx]
testRets = retsDF.iloc[splitIdx:]

print(f"\nNumber of assets (N): {N}")
print(f"Training samples: {len(trainRets)}")
print(f"Testing samples: {len(testRets)}")



def repairBudget(w):
    wClipped = np.maximum(w, 0)
    if np.sum(wClipped) == 0:
        return np.ones(len(w)) / len(w)

    return wClipped / np.sum(wClipped)



def miniVar(w, CovMat):
    return np.dot(w.T, np.dot(CovMat, w))



def CVaR(w, returns, alpha=0.95):
    portRets = returns.values @ w
    VaR = np.percentile(portRets, (1 - alpha) * 100)
    
    lossBeyondVar = portRets[portRets <= VaR]
    if len(lossBeyondVar) > 0:
        CVaRVal = -np.mean(lossBeyondVar)
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



def portMetrics(w, returns):
    portRets = returns.values @ w
    mu = np.mean(portRets)
    sigma = np.std(portRets)
    sharpe = mu / sigma if sigma != 0 else 0.0
    var99 = np.percentile(portRets, 1)
    CVaR99 = portRets[portRets <= var99].mean()

    return mu, sigma, sharpe, var99, CVaR99



def fitFxn(individual, model, returns, alpha=0.95):
    w = np.array(individual)
    wFeasible = repairBudget(w)
    
    # solve the chosen
    if model == 'variance':
        CovMat = returns.cov().values
        fitness = miniVar(wFeasible, CovMat)
        
    elif model == 'CVaR':
        fitness = CVaR(wFeasible, returns, alpha)
        
    elif model == 'semiVar':
        fitness = semiVar(wFeasible, returns, alpha)
        
    else:
        raise ValueError("Model must be 'variance', 'CVaR', or 'semiVar'")
    
    return (fitness,)



def setupDEAP(N, model, returns, alpha=0.95):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))         # type: ignore
    

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)      # type: ignore
    

    toolbox = base.Toolbox()
    toolbox.register("attr_weight", np.random.uniform, 0, 1)  # type: ignore
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=N)  # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)                      # type: ignore
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/N)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", fitFxn, model=model, returns=returns, alpha=alpha)
    
    return toolbox



def runGA(toolbox, popSize=200, gens=500, cxpb=0.9, mutpb=None):
    if mutpb is None:
        mutpb = 1 / len(toolbox.individual())
    
    # init pop
    pop = toolbox.population(n=popSize)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    

    hof = tools.HallOfFame(1)
    history = []

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
        
        
        if gen % 50 == 0:
            print(f"  Generation {gen}: Best Fitness = {currBest:.6f}")
            
    
    bestIndi = hof[0]
    bestFits = bestIndi.fitness.values[0]
    bestwgts = np.array(bestIndi)
    bestwgts = repairBudget(bestwgts)
    
    return bestwgts, bestFits, history




print("\n" + "="*60)
print("MODEL 1: Minimum Variance Portfolio (Unconstrained)")
print("GA Parameters: P=200, G=500, pc=0.9, pm=1/N, SBX(nc=15), Poly(nm=20)")
print("="*60)



toolboxVar = setupDEAP(N, 'variance', trainRets, alpha=0.95)
bestwgtsVar, bestFitsVar, histVar = runGA(toolboxVar, popSize=200, gens=500, cxpb=0.9)



print(f"\nOptimal Weights (Min Variance):")
for i, ticker in enumerate(tickers):
    if bestwgtsVar[i] > 0.001:
        print(f"  {ticker}: {bestwgtsVar[i]:.4f}")
        
        
        
print(f"\nMinimum Portfolio Variance: {bestFitsVar:.8f}")
print(f"Portfolio Volatility: {np.sqrt(bestFitsVar)*100:.2f}%")


muVar, sigmaVar, sharpeVaR, var99Var, CVaR99Var = portMetrics(bestwgtsVar, trainRets)
print(f"uP: {muVar:.6f}")
print(f"sP: {sigmaVar:.6f}")
print(f"Sharpe Ratio: {sharpeVaR:.6f}")
print(f"VaR99%: {var99Var:.6f}")
print(f"CVaR99%: {CVaR99Var:.6f}")



print("\n" + "="*60)
print("MODEL 2: CVaR Portfolio (Unconstrained, alpha=95%)")
print("="*60)



toolboxCVaR = setupDEAP(N, 'CVaR', trainRets, alpha=0.95)
bestwgtsCVaR, bestFitsCVaR, histCVaR = runGA(toolboxCVaR, popSize=200, gens=500, cxpb=0.9)



print(f"\nOptimal Weights (CVaR 95%):")
for i, ticker in enumerate(tickers):
    if bestwgtsCVaR[i] > 0.001:
        print(f"  {ticker}: {bestwgtsCVaR[i]:.4f}")
        
        
print(f"\nMinimum CVaR (95%): {bestFitsCVaR:.6f}")
print(f"Expected tail loss: {bestFitsCVaR*100:.2f}%")



muCVaR, sigmaCVaR, sharpeCVaR, var99CVaR, Cvar99CVaR = portMetrics(bestwgtsCVaR, trainRets)
print(f"uP: {muCVaR:.6f}")
print(f"sP: {sigmaCVaR:.6f}")
print(f"Sharpe Ratio: {sharpeCVaR:.6f}")
print(f"VaR99%: {var99CVaR:.6f}")
print(f"CVaR99%: {Cvar99CVaR:.6f}")



print("\n" + "="*60)
print("MODEL 3: semiVar Portfolio (Unconstrained, alpha=95%)")
print("="*60)



toolboxSemi = setupDEAP(N, 'semiVar', trainRets, alpha=0.95)
bestwgtsSemi, bestFitsSemi, historySemi = runGA(toolboxSemi, popSize=200, gens=500, cxpb=0.9)



print(f"\nOptimal Weights (semiVar):")
for i, ticker in enumerate(tickers):
    if bestwgtsSemi[i] > 0.001:
        print(f"  {ticker}: {bestwgtsSemi[i]:.4f}")
        
        
print(f"\nMinimum semiVar (downside deviation): {bestFitsSemi:.6f}")


muSemi, sigmaSemi, sharpeSemi, var99Semi, CVaR99Semi = portMetrics(bestwgtsSemi, trainRets)
print(f"uP: {muSemi:.6f}")
print(f"sP: {sigmaSemi:.6f}")
print(f"Sharpe Ratio: {sharpeSemi:.6f}")
print(f"VaR99%: {var99Semi:.6f}")
print(f"CVaR99%: {CVaR99Semi:.6f}")



fig, axes = plt.subplots(1, 3, figsize=(15, 5))

BestGenVaR = np.argmin(histVar)
BestGenCVaR = np.argmin(histCVaR)
BestGenSemi = np.argmin(historySemi)


axes[0].plot(histVar, linewidth=2, color='blue')
axes[0].set_xlabel('Generation', fontsize=12)
axes[0].set_ylabel('Best Fitness (Variance)', fontsize=12)
axes[0].set_title('Minimum Variance Portfolio Convergence', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=bestFitsVar, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsVar:.6f}')
axes[0].axvline(x=BestGenVaR, color='green', linestyle=':',linewidth=1.5, label=f'Best at gen {BestGenVaR}')
axes[0].legend(fontsize=10)
axes[0].fill_between(range(len(histVar)), histVar, bestFitsVar, alpha=0.2, color='blue')


axes[1].plot(histCVaR, linewidth=2, color='green')
axes[1].set_xlabel('Generation', fontsize=12)
axes[1].set_ylabel('Best Fitness (CVaR)', fontsize=12)
axes[1].set_title('CVaR Portfolio Convergence', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=bestFitsCVaR, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsCVaR:.6f}')
axes[1].axvline(x=BestGenCVaR, color='green', linestyle=':', linewidth=1.5, label=f'Best at gen {BestGenCVaR}')
axes[1].legend(fontsize=10)
axes[1].fill_between(range(len(histCVaR)), histCVaR, bestFitsCVaR, alpha=0.2, color='green')


axes[2].plot(historySemi, linewidth=2, color='orange')
axes[2].set_xlabel('Generation', fontsize=12)
axes[2].set_ylabel('Best Fitness (semiVar)', fontsize=12)
axes[2].set_title('semiVar Portfolio Convergence', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=bestFitsSemi, color='red', linestyle='--', linewidth=1.5, label=f'Final: {bestFitsSemi:.6f}')
axes[2].axvline(x=BestGenSemi, color='green', linestyle=':', linewidth=1.5, label=f'Best at gen {BestGenSemi}')
axes[2].legend(fontsize=10)
axes[2].fill_between(range(len(historySemi)), historySemi, bestFitsSemi, alpha=0.2, color='orange')


plt.suptitle('GA Convergence for Three Portfolio Models (Unconstrained) 20 stocks', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Q1-CGT.png")


print("\n" + "="*60)
print("CONVERGENCE SUMMARY")
print("="*60)
print(f"\n{'Model':<25} {'Best Found at Gen':<20} {'Final Fitness':<20}")
print("-"*65)
print(f"{'Minimum Var':<25} {BestGenVaR:<20} {bestFitsVar:<20.8f}")
print(f"{'CVaR (95%)':<25} {BestGenCVaR:<20} {bestFitsCVaR:<20.6f}")
print(f"{'semiVar':<25} {BestGenSemi:<20} {bestFitsSemi:<20.6f}")


print("\n" + "="*60)
print("PORTFOLIO COMPOSITION SUMMARY")
print("="*60)


def portSummary(weights, name):
    nonZero = weights > 0.001
    numHolds = np.sum(nonZero)
    avgWgts = np.mean(weights[nonZero]) if numHolds > 0 else 0
    maxWgt = np.max(weights)
    minWgt = np.min(weights[nonZero]) if numHolds > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Number of assets held: {numHolds}/{N}")
    print(f"  Average weight (held): {avgWgts:.4f}")
    print(f"  Max weight: {maxWgt:.4f}")
    print(f"  Min weight (held): {minWgt:.4f}")
    

portSummary(bestwgtsVar, "Minimum Variance Portfolio")
portSummary(bestwgtsCVaR, "CVaR Portfolio")
portSummary(bestwgtsSemi, "semiVar Portfolio")


print("\n" + "="*60)
print("GA PARAMETERS USED (DEAF implementation)")
print("="*60)
print(f"""
    - Population size: P = 200
    - gens: G = 500
    - Crossover: Simulated Binary Crossover (SBX) with ηc = 15
    - Crossover probability: pc = 0.9
    - Mutation: Polynomial mutation with nm = 20
    - Mutation probability: pm = 1/N = 1/{N} = {1/N:.3f}
    - Selection: Binary tournament (tournament size = 2)
    - Elitism: Best solution carried to next generation
    - Repair mechanism: Budget repair (long-only + sum to 1)
    """)


"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-5$ python Q1.py 
============================================================
LOADING DATA FROM IBEX35 FOLDER
============================================================
Loaded 19 stocks with 3566 trading days
Date range: 2011-01-25 00:00:00 to 2024-12-31 00:00:00

Number of assets (N): 19
Training samples: 2496
Testing samples: 1070

============================================================
MODEL 1: Minimum Variance Portfolio (Unconstrained)
GA Parameters: P=200, G=500, pc=0.9, pm=1/N, SBX(nc=15), Poly(nm=20)
============================================================
  Generation 0: Best Fitness = 0.000166
  Generation 50: Best Fitness = 0.000124
  Generation 100: Best Fitness = 0.000123
  Generation 150: Best Fitness = 0.000123
  Generation 200: Best Fitness = 0.000123
  Generation 250: Best Fitness = 0.000123
  Generation 300: Best Fitness = 0.000123
  Generation 350: Best Fitness = 0.000123
  Generation 400: Best Fitness = 0.000123
  Generation 450: Best Fitness = 0.000123

Optimal Weights (Min Variance):
  Inditex: 0.1470
  Grifols: 0.2036
  Ferrovial: 0.1169
  Enagás: 0.1811
  Red Eléctrica de España: 0.2425
  Indra Sistemas: 0.0459
  Iberdrola: 0.0631

Minimum Portfolio Variance: 0.00012308
Portfolio Volatility: 1.11%
uP: 0.000535
sP: 0.011092
Sharpe Ratio: 0.048234
VaR99%: -0.030042
CVaR99%: -0.041123

============================================================
MODEL 2: CVaR Portfolio (Unconstrained, alpha=95%)
============================================================
  Generation 0: Best Fitness = 0.030572
  Generation 50: Best Fitness = 0.025701
  Generation 100: Best Fitness = 0.025438
  Generation 150: Best Fitness = 0.025376
  Generation 200: Best Fitness = 0.025373
  Generation 250: Best Fitness = 0.025372
  Generation 300: Best Fitness = 0.025372
  Generation 350: Best Fitness = 0.025372
  Generation 400: Best Fitness = 0.025372
  Generation 450: Best Fitness = 0.025372

Optimal Weights (CVaR 95%):
  Inditex: 0.2259
  Grifols: 0.2496
  Ferrovial: 0.0054
  Enagás: 0.1981
  Red Eléctrica de España: 0.2653
  Iberdrola: 0.0557

Minimum CVaR (95%): 0.025372
Expected tail loss: 2.54%
uP: 0.000553
sP: 0.011254
Sharpe Ratio: 0.049170
VaR99%: -0.029360
CVaR99%: -0.040689

============================================================
MODEL 3: semiVar Portfolio (Unconstrained, alpha=95%)
============================================================
  Generation 0: Best Fitness = 0.000338
  Generation 50: Best Fitness = 0.000184
  Generation 100: Best Fitness = 0.000181
  Generation 150: Best Fitness = 0.000181
  Generation 200: Best Fitness = 0.000180
  Generation 250: Best Fitness = 0.000180
  Generation 300: Best Fitness = 0.000180
  Generation 350: Best Fitness = 0.000180
  Generation 400: Best Fitness = 0.000180
  Generation 450: Best Fitness = 0.000180

Optimal Weights (semiVar):
  Bankinter: 0.0067
  Inditex: 0.2417
  Naturgy: 0.0042
  Grifols: 0.2277
  Ferrovial: 0.0034
  Enagás: 0.0020
  Mapfre: 0.1008
  Red Eléctrica de España: 0.2046
  BBVA: 0.0028
  Repsol: 0.0084
  Indra Sistemas: 0.0702
  Iberdrola: 0.1260

Minimum semiVar (downside deviation): 0.000180
uP: 0.000484
sP: 0.011605
Sharpe Ratio: 0.041732
VaR99%: -0.030775
CVaR99%: -0.040832

============================================================
CONVERGENCE SUMMARY
============================================================

Model                     Best Found at Gen    Final Fitness       
-----------------------------------------------------------------
Minimum Var               499                  0.00012308          
CVaR (95%)                498                  0.025372            
semiVar                   499                  0.000180            

============================================================
PORTFOLIO COMPOSITION SUMMARY
============================================================

Minimum Variance Portfolio:
  Number of assets held: 7/19
  Average weight (held): 0.1429
  Max weight: 0.2425
  Min weight (held): 0.0459

CVaR Portfolio:
  Number of assets held: 6/19
  Average weight (held): 0.1667
  Max weight: 0.2653
  Min weight (held): 0.0054

semiVar Portfolio:
  Number of assets held: 12/19
  Average weight (held): 0.0832
  Max weight: 0.2417
  Min weight (held): 0.0020

============================================================
GA PARAMETERS USED (DEAF implementation)
============================================================

    - Population size: P = 200
    - gens: G = 500
    - Crossover: Simulated Binary Crossover (SBX) with ηc = 15
    - Crossover probability: pc = 0.9
    - Mutation: Polynomial mutation with nm = 20
    - Mutation probability: pm = 1/N = 1/19 = 0.053
    - Selection: Binary tournament (tournament size = 2)
    - Elitism: Best solution carried to next generation
    - Repair mechanism: Budget repair (long-only + sum to 1)
    
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-5$ 
"""
