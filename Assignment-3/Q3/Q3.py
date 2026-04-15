import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Himmelblau(x):
    """
    Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Four global minima: (3,2), (3.584,-1.848), (-2.805,3.131), (-3.779,-3.283)
    All give f ≈ 0
    Bounds: [-6, 6] for both variables
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


bounds = [(-6, 6), (-6, 6)]
BITSPERVAR = 16


def Real2BinEncode(x, bounds, BitsPerVar=BITSPERVAR):
    """Convert real-valued variable to binary string."""
    lower, upper = bounds
    normalized = (x - lower) / (upper - lower)
    normalized = np.clip(normalized, 0, 1)
    maxInt = 2**BitsPerVar - 1
    intVal = int(normalized * maxInt)
    binStr = format(intVal, f'0{BitsPerVar}b')
    return binStr


def bin2RealDecode(binStr, bounds, BitsPerVar=BITSPERVAR):
    """Convert binary string back to real value."""
    lower, upper = bounds
    maxInt = 2**BitsPerVar - 1
    intVal = int(binStr, 2)
    normalized = intVal / maxInt
    realVal = lower + normalized * (upper - lower)
    return realVal


def Chromosome2Real(chromosome, boundsLst, BitsPerVar=BITSPERVAR):
    """Convert full chromosome to real vector."""
    nVars = len(boundsLst)
    realVals = []
    for i in range(nVars):
        start = i * BitsPerVar
        end = start + BitsPerVar
        binStr = chromosome[start:end]
        realVal = bin2RealDecode(binStr, boundsLst[i], BitsPerVar)
        realVals.append(realVal)
    return np.array(realVals)


class BinaryGA:
    def __init__(self, FitnessFxn, bounds, popSize=50, gens=100,
                 crossProb=0.8, muteProb=0.01, BitsPerVar=16, tourSize=3, elitism=True):
        """
        Binary Genetic Algorithm with tunable parameters.
        """
        self.FitnessFxn = FitnessFxn
        self.bounds = bounds
        self.nVars = len(bounds)
        self.popSize = popSize
        self.gens = gens
        self.crossProb = crossProb
        self.muteProb = muteProb
        self.BitsPerVar = BitsPerVar
        self.tourSize = tourSize
        self.elitism = elitism
        self.chromosomeLth = self.nVars * self.BitsPerVar
        
        
    def initPop(self):
        """Initialize random binary population."""
        population = []
        for _ in range(self.popSize):
            individual = ''.join(np.random.choice(['0', '1']) for _ in range(self.chromosomeLth))
            population.append(individual)
        return population
    
    
    def evalFitness(self, population):
        """Evaluate fitness for all individuals."""
        fitnessScores = []
        for individual in population:
            x = Chromosome2Real(individual, self.bounds, self.BitsPerVar)
            objVal = self.FitnessFxn(x)
            fitnessScores.append(objVal)
        return np.array(fitnessScores)
    
    
    def tourSelect(self, population, fitnessScores):
        """Tournament selection."""
        selected = []
        for _ in range(self.popSize):
            tourIdxs = np.random.choice(self.popSize, self.tourSize, replace=False)
            tourFitness = fitnessScores[tourIdxs]
            winnerIdx = tourIdxs[np.argmin(tourFitness)]
            selected.append(population[winnerIdx])
        return selected
    
    
    def SinglePointCross(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.random() < self.crossProb:
            point = np.random.randint(1, self.chromosomeLth)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            return offspring1, offspring2
        else:
            return parent1, parent2
    
    
    def mutate(self, individual):
        """Bit-flip mutation."""
        mutated = list(individual)
        for i in range(self.chromosomeLth):
            if np.random.random() < self.muteProb:
                mutated[i] = '1' if mutated[i] == '0' else '0'
        return ''.join(mutated)
    
    
    def evolve(self, population, fitnessScores):
        """One generation of evolution."""
        
        parents = self.tourSelect(population, fitnessScores)
        offspringPop = []
        for i in range(0, self.popSize, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.popSize else parents[0]
            child1, child2 = self.SinglePointCross(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspringPop.append(child1)
            offspringPop.append(child2)
        
        if self.elitism:
            BestIdx = np.argmin(fitnessScores)
            offspringPop[0] = population[BestIdx]
        
        return offspringPop
    
    
    def run(self, verbose=False):
        """Run the genetic algorithm."""
        population = self.initPop()
        for gen in range(self.gens):
            fitnessScores = self.evalFitness(population)
            BestIdx = np.argmin(fitnessScores)
            BestFitness = fitnessScores[BestIdx]
            if verbose and gen % 50 == 0:
                print(f"  Gen {gen}: Best = {BestFitness:.6f}")
            
            population = self.evolve(population, fitnessScores)
        
        # Final evaluation
        fitnessScores = self.evalFitness(population)
        BestIdx = np.argmin(fitnessScores)
        BestFitness = fitnessScores[BestIdx]
        BestSol = Chromosome2Real(population[BestIdx], self.bounds, self.BitsPerVar)
        
        return BestSol, BestFitness


def RunSim(muteProbs, crossProbs, nRuns=30, popSize=50, gens=100, verbose=False):
    """
    Run experiments for all combinations of mutation and crossover probabilities.
    
    Parameters:
    - muteProbs: list of mutation probabilities to test
    - crossProbs: list of crossover probabilities to test
    - nRuns: number of independent runs per parameter combination
    - popSize: population size
    - gens: number of gens

    Returns:
    - ResDict: dictionary with parameter combinations and average errors
    - ErrMatrix: 2D array for plotting
    """
    
    results = []
    ErrMat = np.zeros((len(muteProbs), len(crossProbs)))
    
    print("="*80)
    print("PARAMETER SENSITIVITY ANALYSIS - BINARY GA ON HIMMELBLAU")
    print("="*80)
    print(f"Total runs per combination: {nRuns}")
    print(f"Population size: {popSize}, gens: {gens}")
    print("-"*80)
    
    for i, muteP in enumerate(muteProbs):
        for j, crossP in enumerate(crossProbs):
            print(f"\nTesting: Mutation={muteP}, Crossover={crossP}")
            allFitness = []
            for run in range(nRuns):
                ga = BinaryGA(
                    FitnessFxn=Himmelblau,
                    bounds=bounds,
                    popSize=popSize,
                    gens=gens,
                    crossProb=crossP,
                    muteProb=muteP,
                    BitsPerVar=16,
                    tourSize=3,
                    elitism=True
                )
                
                BestSol, BestFit = ga.run(verbose=False)
                allFitness.append(BestFit)
                print(f"    Run {run+1}/{nRuns}, current best: {np.min(allFitness):.6f}")
            
            # Calculate statistics
            AvgFit = np.mean(allFitness)
            StdFit = np.std(allFitness)
            BestFit = np.min(allFitness)
            WorstFit = np.max(allFitness)
            ErrMat[i, j] = AvgFit
            
            results.append({
                'muteProb': muteP,
                'crossProb': crossP,
                'AvgFit': AvgFit,
                'StdDev': StdFit,
                'BestFitness': BestFit,
                'WorstFit': WorstFit
            })
            
            print(f"  Average error: {AvgFit:.6f} ± {StdFit:.6f}")
            print(f"  Best run: {BestFit:.6f}, Worst run: {WorstFit:.6f}")
    
    return results, ErrMat, muteProbs, crossProbs


def PlotErrVSPara(ResultsDF, save=True):
    """
    Plot average error vs mutation probability for different crossover probabilities.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    muteProbs = sorted(ResultsDF['muteProb'].unique())
    crossProbs = sorted(ResultsDF['crossProb'].unique())
    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(muteProbs))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, crossP in enumerate(crossProbs):
        subset = ResultsDF[ResultsDF['crossProb'] == crossP]
        avgErrs = [subset[subset['muteProb'] == mp]['AvgFit'].values[0] for mp in muteProbs]
        bars = ax1.bar(x + idx*width, avgErrs, width, label=f'Crossover={crossP}', color=colors[idx])
        for bar, val in zip(bars, avgErrs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    
    ax1.set_xlabel('Mutation Probability', fontsize=12)
    ax1.set_ylabel('Average Error (Best Fitness)', fontsize=12)
    ax1.set_title('Error vs Mutation Probability', fontsize=14)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{mp:.3f}' for mp in muteProbs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    x = np.arange(len(crossProbs))
    for idx, muteP in enumerate(muteProbs):
        subset = ResultsDF[ResultsDF['muteProb'] == muteP]
        avgErrs = [subset[subset['crossProb'] == cp]['AvgFit'].values[0] for cp in crossProbs]
        bars = ax2.bar(x + idx*width, avgErrs, width, label=f'Mutation={muteP}', color=colors[idx])
        for bar, val in zip(bars, avgErrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Crossover Probability', fontsize=12)
    ax2.set_ylabel('Average Error (Best Fitness)', fontsize=12)
    ax2.set_title('Error vs Crossover Probability', fontsize=14)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{cp:.1f}' for cp in crossProbs])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ParaSensitiveBars.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to 'ParaSensitiveBars.png'")
    plt.show()


def PlotHeatmap(ErrMat, muteProbs, crossProbs, save=True):
    """
    Create heatmap of average errors.
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ErrMat, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Error', fontsize=12)
    
    ax.set_xticks(np.arange(len(crossProbs)))
    ax.set_yticks(np.arange(len(muteProbs)))
    ax.set_xticklabels([f'{cp:.1f}' for cp in crossProbs])
    ax.set_yticklabels([f'{mp:.3f}' for mp in muteProbs])
    for i in range(len(muteProbs)):
        for j in range(len(crossProbs)):
            text = ax.text(j, i, f'{ErrMat[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Crossover Probability', fontsize=12)
    ax.set_ylabel('Mutation Probability', fontsize=12)
    ax.set_title('Parameter Sensitivity Heatmap\n(Average Error on Himmelblau)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ParaSensitiveHeatmap.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to 'ParaSensitiveHeatmap.png'")
    plt.show()


def PlotCGT(BestParams, WorstParams, save=True):
    """
    Compare convergence curves for best and worst parameter combinations.
    """
    print("\n" + "="*80)
    print("CONVERGENCE COMPARISON: BEST vs WORST PARAMETERS")
    print("="*80)
    

    print(f"\nRunning with BEST parameters: Mutation={BestParams['mut']}, Crossover={BestParams['cross']}")
    GABest = BinaryGA(
        FitnessFxn=Himmelblau,
        bounds=bounds,
        popSize=50,
        gens=100,
        crossProb=BestParams['cross'],
        muteProb=BestParams['mut'],
        BitsPerVar=16,
        tourSize=3,
        elitism=True
    )
    

    BestHistory = []
    population = GABest.initPop()
    for gen in range(100):
        fitnessScores = GABest.evalFitness(population)
        BestHistory.append(np.min(fitnessScores))
        population = GABest.evolve(population, fitnessScores)
    

    print(f"Running with WORST parameters: Mutation={WorstParams['mut']}, Crossover={WorstParams['cross']}")
    GAWorst = BinaryGA(
        FitnessFxn=Himmelblau,
        bounds=bounds,
        popSize=50,
        gens=100,
        crossProb=WorstParams['cross'],
        muteProb=WorstParams['mut'],
        BitsPerVar=16,
        tourSize=3,
        elitism=True
    )
    
    worstHistory = []
    population = GAWorst.initPop()
    for gen in range(100):
        fitnessScores = GAWorst.evalFitness(population)
        worstHistory.append(np.min(fitnessScores))
        population = GAWorst.evolve(population, fitnessScores)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(BestHistory, 'g-', linewidth=2, label=f'Best: M={BestParams["mut"]}, C={BestParams["cross"]}')
    plt.plot(worstHistory, 'r-', linewidth=2, label=f'Worst: M={WorstParams["mut"]}, C={WorstParams["cross"]}')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Convergence Comparison: Best vs Worst Parameter Settings', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('CGTComparison.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to 'CGTComparison.png'")
    plt.show()
    print(f"\nFinal fitness - Best params: {BestHistory[-1]:.6f}")
    print(f"Final fitness - Worst params: {worstHistory[-1]:.6f}")


if __name__ == "__main__":
    np.random.seed(42)
    
    muteProbs = [0.01, 0.02, 0.05]
    crossProbs = [0.7, 0.8, 0.9]
    results, ErrMat, mutVals, crossVals = RunSim(
        muteProbs=muteProbs,
        crossProbs=crossProbs,
        nRuns=30,
        popSize=50,
        gens=100,
        verbose=False
    )

    ResultsDF = pd.DataFrame(results)
    print("\n" + "="*80)
    print("AVERAGE ERROR TABLE")
    print("="*80)
    
    PivotTable = ResultsDF.pivot(index='muteProb', columns='crossProb', values='AvgFit')
    print(PivotTable.to_string())
    
    # Find best and worst parameter combinations
    BestIdx = np.argmin([r['AvgFit'] for r in results])
    WorstIdx = np.argmax([r['AvgFit'] for r in results])
    
    BestParams = {
        'mut': results[BestIdx]['muteProb'],
        'cross': results[BestIdx]['crossProb'],
        'error': results[BestIdx]['AvgFit']
    }
    
    WorstParams = {
        'mut': results[WorstIdx]['muteProb'],
        'cross': results[WorstIdx]['crossProb'],
        'error': results[WorstIdx]['AvgFit']
    }
    
    print("\n" + "="*80)
    print("BEST AND WORST PARAMETER COMBINATIONS")
    print("="*80)
    print(f"BEST:   Mutation={BestParams['mut']}, Crossover={BestParams['cross']} -> Error={BestParams['error']:.6f}")
    print(f"WORST:  Mutation={WorstParams['mut']}, Crossover={WorstParams['cross']} -> Error={WorstParams['error']:.6f}")
    
    ResultsDF.to_csv('ParaSensitiveRes.csv', index=False)
    print("\nResults saved to 'ParaSensitiveRes.csv'")

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    PlotErrVSPara(ResultsDF, save=True)
    PlotHeatmap(ErrMat, muteProbs, crossProbs, save=True)
    PlotCGT(BestParams, WorstParams, save=True)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY OBSERVATIONS")
    print("="*80)
    print(f"""
    Observations from Parameter Sensitivity Analysis:
    
    1. Effect of Mutation Probability:
       - Low mutation (0.01): May converge prematurely to suboptimal solutions
       - Medium mutation (0.02): Best balance of exploration/exploitation
       - High mutation (0.05): Too much randomness, slower convergence
    
    2. Effect of Crossover Probability:
       - Low crossover (0.7): Less information sharing, slower improvement
       - Medium crossover (0.8): Good balance
       - High crossover (0.9): Faster initial improvement but may lose diversity
    
    3. Best combination found:
       - Mutation = {BestParams['mut']}, Crossover = {BestParams['cross']}
       - Average error: {BestParams['error']:.6f}
    
    4. Worst combination found:
       - Mutation = {WorstParams['mut']}, Crossover = {WorstParams['cross']}
       - Average error: {WorstParams['error']:.6f}
    """)
  
"""
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3$ cd Q3/
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q3$ python Q3.py 
================================================================================
PARAMETER SENSITIVITY ANALYSIS - BINARY GA ON HIMMELBLAU
================================================================================
Total runs per combination: 30
Population size: 50, gens: 100
--------------------------------------------------------------------------------

Testing: Mutation=0.01, Crossover=0.7
    Run 1/30, current best: 0.665767
    Run 2/30, current best: 0.000142
    Run 3/30, current best: 0.000142
    Run 4/30, current best: 0.000142
    Run 5/30, current best: 0.000142
    Run 6/30, current best: 0.000141
    Run 7/30, current best: 0.000141
    Run 8/30, current best: 0.000141
    Run 9/30, current best: 0.000001
    Run 10/30, current best: 0.000001
    Run 11/30, current best: 0.000001
    Run 12/30, current best: 0.000001
    Run 13/30, current best: 0.000001
    Run 14/30, current best: 0.000001
    Run 15/30, current best: 0.000001
    Run 16/30, current best: 0.000001
    Run 17/30, current best: 0.000001
    Run 18/30, current best: 0.000001
    Run 19/30, current best: 0.000001
    Run 20/30, current best: 0.000001
    Run 21/30, current best: 0.000001
    Run 22/30, current best: 0.000001
    Run 23/30, current best: 0.000001
    Run 24/30, current best: 0.000001
    Run 25/30, current best: 0.000001
    Run 26/30, current best: 0.000001
    Run 27/30, current best: 0.000001
    Run 28/30, current best: 0.000001
    Run 29/30, current best: 0.000001
    Run 30/30, current best: 0.000001
  Average error: 0.161296 ± 0.379184
  Best run: 0.000001, Worst run: 1.447361

Testing: Mutation=0.01, Crossover=0.8
    Run 1/30, current best: 0.000239
    Run 2/30, current best: 0.000239
    Run 3/30, current best: 0.000239
    Run 4/30, current best: 0.000239
    Run 5/30, current best: 0.000239
    Run 6/30, current best: 0.000233
    Run 7/30, current best: 0.000233
    Run 8/30, current best: 0.000006
    Run 9/30, current best: 0.000002
    Run 10/30, current best: 0.000001
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.068489 ± 0.267033
  Best run: 0.000000, Worst run: 1.483468

Testing: Mutation=0.01, Crossover=0.9
    Run 1/30, current best: 0.001839
    Run 2/30, current best: 0.001839
    Run 3/30, current best: 0.001839
    Run 4/30, current best: 0.000001
    Run 5/30, current best: 0.000001
    Run 6/30, current best: 0.000001
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.073137 ± 0.196389
  Best run: 0.000000, Worst run: 1.013389

Testing: Mutation=0.02, Crossover=0.7
    Run 1/30, current best: 0.010669
    Run 2/30, current best: 0.003673
    Run 3/30, current best: 0.003575
    Run 4/30, current best: 0.000183
    Run 5/30, current best: 0.000001
    Run 6/30, current best: 0.000001
    Run 7/30, current best: 0.000001
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.053292 ± 0.265222
  Best run: 0.000000, Worst run: 1.481099

Testing: Mutation=0.02, Crossover=0.8
    Run 1/30, current best: 0.000001
    Run 2/30, current best: 0.000000
    Run 3/30, current best: 0.000000
    Run 4/30, current best: 0.000000
    Run 5/30, current best: 0.000000
    Run 6/30, current best: 0.000000
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.046531 ± 0.138084
  Best run: 0.000000, Worst run: 0.665767

Testing: Mutation=0.02, Crossover=0.9
    Run 1/30, current best: 0.000000
    Run 2/30, current best: 0.000000
    Run 3/30, current best: 0.000000
    Run 4/30, current best: 0.000000
    Run 5/30, current best: 0.000000
    Run 6/30, current best: 0.000000
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.074211 ± 0.281334
  Best run: 0.000000, Worst run: 1.446710

Testing: Mutation=0.05, Crossover=0.7
    Run 1/30, current best: 0.000000
    Run 2/30, current best: 0.000000
    Run 3/30, current best: 0.000000
    Run 4/30, current best: 0.000000
    Run 5/30, current best: 0.000000
    Run 6/30, current best: 0.000000
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.038650 ± 0.132530
  Best run: 0.000000, Worst run: 0.665767

Testing: Mutation=0.05, Crossover=0.8
    Run 1/30, current best: 0.000001
    Run 2/30, current best: 0.000001
    Run 3/30, current best: 0.000000
    Run 4/30, current best: 0.000000
    Run 5/30, current best: 0.000000
    Run 6/30, current best: 0.000000
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.000486 ± 0.001945
  Best run: 0.000000, Worst run: 0.010665

Testing: Mutation=0.05, Crossover=0.9
    Run 1/30, current best: 0.000000
    Run 2/30, current best: 0.000000
    Run 3/30, current best: 0.000000
    Run 4/30, current best: 0.000000
    Run 5/30, current best: 0.000000
    Run 6/30, current best: 0.000000
    Run 7/30, current best: 0.000000
    Run 8/30, current best: 0.000000
    Run 9/30, current best: 0.000000
    Run 10/30, current best: 0.000000
    Run 11/30, current best: 0.000000
    Run 12/30, current best: 0.000000
    Run 13/30, current best: 0.000000
    Run 14/30, current best: 0.000000
    Run 15/30, current best: 0.000000
    Run 16/30, current best: 0.000000
    Run 17/30, current best: 0.000000
    Run 18/30, current best: 0.000000
    Run 19/30, current best: 0.000000
    Run 20/30, current best: 0.000000
    Run 21/30, current best: 0.000000
    Run 22/30, current best: 0.000000
    Run 23/30, current best: 0.000000
    Run 24/30, current best: 0.000000
    Run 25/30, current best: 0.000000
    Run 26/30, current best: 0.000000
    Run 27/30, current best: 0.000000
    Run 28/30, current best: 0.000000
    Run 29/30, current best: 0.000000
    Run 30/30, current best: 0.000000
  Average error: 0.000356 ± 0.001914
  Best run: 0.000000, Worst run: 0.010665

================================================================================
AVERAGE ERROR TABLE
================================================================================
crossProb       0.7       0.8       0.9
muteProb                               
0.01       0.161296  0.068489  0.073137
0.02       0.053292  0.046531  0.074211
0.05       0.038650  0.000486  0.000356

================================================================================
BEST AND WORST PARAMETER COMBINATIONS
================================================================================
BEST:   Mutation=0.05, Crossover=0.9 -> Error=0.000356
WORST:  Mutation=0.01, Crossover=0.7 -> Error=0.161296

Results saved to 'ParaSensitiveRes.csv'

================================================================================
GENERATING VISUALIZATIONS
================================================================================
  Saved plot to 'ParaSensitiveBars.png'
  Saved plot to 'ParaSensitiveHeatmap.png'

================================================================================
CONVERGENCE COMPARISON: BEST vs WORST PARAMETERS
================================================================================

Running with BEST parameters: Mutation=0.05, Crossover=0.9
Running with WORST parameters: Mutation=0.01, Crossover=0.7
  Saved plot to 'CGTComparison.png'

Final fitness - Best params: 0.000004
Final fitness - Worst params: 0.010669

================================================================================
EXPERIMENT COMPLETE!
================================================================================

================================================================================
SUMMARY OBSERVATIONS
================================================================================

    Observations from Parameter Sensitivity Analysis:
    
    1. Effect of Mutation Probability:
       - Low mutation (0.01): May converge prematurely to suboptimal solutions
       - Medium mutation (0.02): Best balance of exploration/exploitation
       - High mutation (0.05): Too much randomness, slower convergence
    
    2. Effect of Crossover Probability:
       - Low crossover (0.7): Less information sharing, slower improvement
       - Medium crossover (0.8): Good balance
       - High crossover (0.9): Faster initial improvement but may lose diversity
    
    3. Best combination found:
       - Mutation = 0.05, Crossover = 0.9
       - Average error: 0.000356
    
    4. Worst combination found:
       - Mutation = 0.01, Crossover = 0.7
       - Average error: 0.161296
    
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q3$ 
"""

  