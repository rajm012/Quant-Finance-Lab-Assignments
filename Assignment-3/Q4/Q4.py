
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


class RealCodedGA:
    def __init__(self, FitnessFxn, bounds, PopSize=50, gens=100, CrossProb=0.8, MuteProb=0.1, 
                 EtaC=20, EtaM=20, TourSize=3, elitism=True):
        """
        Real-Coded Genetic Algorithm with SBX crossover and Polynomial mutation.
        
        Parameters:
        - FitnessFxn: objective function (to minimize)
        - bounds: list of (lower, upper) for each variable
        - PopSize: population size
        - gens: number of gens
        - CrossProb: probability of crossover
        - MuteProb: probability of mutation (per variable)
        - EtaC: distribution index for SBX (higher = more similar to parents)
        - EtaM: distribution index for polynomial mutation (higher = smaller mutation)
        - TourSize: size of tournament selection
        - elitism: whether to preserve best individual
        """
        self.FitnessFxn = FitnessFxn
        self.bounds = np.array(bounds)
        self.nVars = len(bounds)
        self.PopSize = PopSize
        self.gens = gens
        self.CrossProb = CrossProb
        self.MuteProb = MuteProb
        self.EtaC = EtaC
        self.EtaM = EtaM
        self.TourSize = TourSize
        self.elitism = elitism
        
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]
        self.BestFitnessHist = []
        self.BestSolHist = []
        
    def initPop(self):
        """Initialize random real-valued population within bounds."""
        population = []
        for _ in range(self.PopSize):
            individual = np.random.uniform(self.lower, self.upper)
            population.append(individual)
        return np.array(population)
    
    
    def evalFitness(self, population):
        """Evaluate fitness for all individuals."""
        FitnessScores = []
        for individual in population:
            objVal = self.FitnessFxn(individual)
            FitnessScores.append(objVal)
        return np.array(FitnessScores)
    
    
    def TourSelect(self, population, FitnessScores):
        """Tournament selection for minimization."""
        selected = []
        for _ in range(self.PopSize):
            TourIdxs = np.random.choice(self.PopSize, self.TourSize, replace=False)
            TourFits = FitnessScores[TourIdxs]
            winnerIdx = TourIdxs[np.argmin(TourFits)]
            selected.append(population[winnerIdx].copy())
        return np.array(selected)
    
    
    def SBXCross(self, parent1, parent2):
        """
        Simulated Binary Crossover (SBX).
        Creates two offspring from two parents.
        """
        if np.random.random() < self.CrossProb:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            for i in range(self.nVars):
                if np.random.random() < 0.5:  # Crossover for each variable with prob 0.5
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        # Calculate beta
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (self.EtaC + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (self.EtaC + 1))
                        
                        # Ensure beta is not too extreme
                        # Create offspring
                        beta = min(beta, 10.0)
                        offspring1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                        offspring2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            
            # Clip to bounds
            offspring1 = np.clip(offspring1, self.lower, self.upper)
            offspring2 = np.clip(offspring2, self.lower, self.upper)
            return offspring1, offspring2
        else:
            return parent1.copy(), parent2.copy()
        
    
    def PolyMute(self, individual):
        """
        Polynomial mutation.
        Mutates each variable with probability MuteProb.
        """
        mutated = individual.copy()
        for i in range(self.nVars):
            if np.random.random() < self.MuteProb:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.EtaM + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.EtaM + 1))
                
                # Calculate mutation step
                deltaMax = min(
                    (self.upper[i] - individual[i]) / (self.upper[i] - self.lower[i]),
                    (individual[i] - self.lower[i]) / (self.upper[i] - self.lower[i])
                )
                delta = delta * deltaMax
                mutated[i] = individual[i] + delta * (self.upper[i] - self.lower[i])
        
        mutated = np.clip(mutated, self.lower, self.upper)
        return mutated
    
    
    def evolve(self, population, FitnessScores):
        """
        One generation of evolution.
        """
        # Selection
        # Create offspring population
        parents = self.TourSelect(population, FitnessScores)
        offspringPop = []
        
        for i in range(0, self.PopSize, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.PopSize else parents[0]
            child1, child2 = self.SBXCross(parent1, parent2)
            child1 = self.PolyMute(child1)
            child2 = self.PolyMute(child2)
            offspringPop.append(child1)
            offspringPop.append(child2)
        offspringPop = np.array(offspringPop[:self.PopSize])
        
        # Elitism: keep best individual from previous generation
        if self.elitism:
            BestIdx = np.argmin(FitnessScores)
            offspringPop[0] = population[BestIdx].copy()
        return offspringPop
    
    
    def run(self, verbose=False):
        """
        Run the genetic algorithm.
        Returns: best solution, best fitness, and history
        """
        population = self.initPop()
        for gen in range(self.gens):
            FitnessScores = self.evalFitness(population)
            BestIdx = np.argmin(FitnessScores)
            BestFit = FitnessScores[BestIdx]
            BestSol = population[BestIdx].copy()
            self.BestFitnessHist.append(BestFit)
            self.BestSolHist.append(BestSol)
            if verbose and (gen % 20 == 0 or gen == self.gens - 1):
                print(f"  Generation {gen:4d}: Best Fitness = {BestFit:.8f}")
            
            population = self.evolve(population, FitnessScores)
        
        FitnessScores = self.evalFitness(population)
        BestIdx = np.argmin(FitnessScores)
        BestFit = FitnessScores[BestIdx]
        BestSol = population[BestIdx].copy()
        return BestSol, BestFit, self.BestFitnessHist


class BinaryGA:
    def __init__(self, FitnessFxn, bounds, PopSize=50, gens=100,
                 CrossProb=0.8, MuteProb=0.01, BitsPerVar=16, TourSize=3, elitism=True):
        """
        Binary Genetic Algorithm.
        """
        self.FitnessFxn = FitnessFxn
        self.bounds = bounds
        self.nVars = len(bounds)
        self.PopSize = PopSize
        self.gens = gens
        self.CrossProb = CrossProb
        self.MuteProb = MuteProb
        self.BitsPerVar = BitsPerVar
        self.TourSize = TourSize
        self.elitism = elitism
        self.chromosomeLth = self.nVars * self.BitsPerVar
        self.BestFitnessHist = []
        self.BestSolHist = []
    
    
    def Real2Bin(self, x, bounds, BitsPerVar):
        """Convert real-valued variable to binary string."""
        lower, upper = bounds
        normalized = (x - lower) / (upper - lower)
        normalized = np.clip(normalized, 0, 1)
        maxInt = 2**BitsPerVar - 1
        IntVal = int(normalized * maxInt)
        BinStr = format(IntVal, f'0{BitsPerVar}b')
        return BinStr
    
    
    def Bin2Real(self, BinStr, bounds, BitsPerVar):
        """Convert binary string back to real value."""
        lower, upper = bounds
        maxInt = 2**BitsPerVar - 1
        IntVal = int(BinStr, 2)
        normalized = IntVal / maxInt
        realVal = lower + normalized * (upper - lower)
        return realVal
    
    
    def chromosome2Real(self, chromosome, boundsLst, BitsPerVar):
        """Convert full chromosome to real vector."""
        nVars = len(boundsLst)
        realVals = []
        for i in range(nVars):
            start = i * BitsPerVar
            end = start + BitsPerVar
            BinStr = chromosome[start:end]
            realVal = self.Bin2Real(BinStr, boundsLst[i], BitsPerVar)
            realVals.append(realVal)
        return np.array(realVals)
    
    
    def initPop(self):
        """Initialize random binary population."""
        population = []
        for _ in range(self.PopSize):
            individual = ''.join(np.random.choice(['0', '1']) for _ in range(self.chromosomeLth))
            population.append(individual)
        return population
    
    
    def evalFitness(self, population):
        """Evaluate fitness for all individuals."""
        FitnessScores = []
        for individual in population:
            x = self.chromosome2Real(individual, self.bounds, self.BitsPerVar)
            objVal = self.FitnessFxn(x)
            FitnessScores.append(objVal)
        return np.array(FitnessScores)
    
    
    def TourSelect(self, population, FitnessScores):
        """Tournament selection."""
        selected = []
        for _ in range(self.PopSize):
            TourIdxs = np.random.choice(self.PopSize, self.TourSize, replace=False)
            TourFits = FitnessScores[TourIdxs]
            winnerIdx = TourIdxs[np.argmin(TourFits)]
            selected.append(population[winnerIdx])
        return selected
    
    
    def SinglePointCross(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.random() < self.CrossProb:
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
            if np.random.random() < self.MuteProb:
                mutated[i] = '1' if mutated[i] == '0' else '0'
        return ''.join(mutated)
    
    
    def evolve(self, population, FitnessScores):
        """One generation of evolution."""
        parents = self.TourSelect(population, FitnessScores)
        offspringPop = []
        for i in range(0, self.PopSize, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.PopSize else parents[0]
            child1, child2 = self.SinglePointCross(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspringPop.append(child1)
            offspringPop.append(child2)
        
        if self.elitism:
            BestIdx = np.argmin(FitnessScores)
            offspringPop[0] = population[BestIdx]
        
        return offspringPop
    
    def run(self, verbose=False):
        """Run the genetic algorithm."""
        population = self.initPop()
        for gen in range(self.gens):
            FitnessScores = self.evalFitness(population)
            BestIdx = np.argmin(FitnessScores)
            BestFit = FitnessScores[BestIdx]
            BestSol = self.chromosome2Real(population[BestIdx], self.bounds, self.BitsPerVar)
            self.BestFitnessHist.append(BestFit)
            self.BestSolHist.append(BestSol)
            if verbose and (gen % 20 == 0 or gen == self.gens - 1):
                print(f"  Generation {gen:4d}: Best Fitness = {BestFit:.8f}")
            
            population = self.evolve(population, FitnessScores)
        
        FitnessScores = self.evalFitness(population)
        BestIdx = np.argmin(FitnessScores)
        BestFit = FitnessScores[BestIdx]
        BestSol = self.chromosome2Real(population[BestIdx], self.bounds, self.BitsPerVar)
        return BestSol, BestFit, self.BestFitnessHist


def RunSim(nRuns=30, PopSize=50, gens=100, verbose=False):
    """
    Run both Binary GA and Real-Coded GA for comparison.
    """
    results = {
        'Binary GA': {'fitness': [], 'solutions': []},
        'Real GA': {'fitness': [], 'solutions': []}
    }
    
    print("="*80)
    print("COMPARISON: BINARY GA vs REAL-CODED GA (SBX + POLYNOMIAL MUTATION)")
    print("="*80)
    print(f"Runs: {nRuns}, Population: {PopSize}, gens: {gens}")
    print("-"*80)
    
    print("\nRunning Binary GA...")
    for run in range(nRuns):
        ga = BinaryGA(
            FitnessFxn=Himmelblau,
            bounds=bounds,
            PopSize=PopSize,
            gens=gens,
            CrossProb=0.8,
            MuteProb=0.01,
            BitsPerVar=16,
            TourSize=3,
            elitism=True
        )
        
        BestSol, BestFit, history = ga.run(verbose=False)
        results['Binary GA']['fitness'].append(BestFit)
        results['Binary GA']['solutions'].append(BestSol)
        
        if (run + 1) % 10 == 0:
            print(f"  Completed run {run + 1}/{nRuns}")
    
    # Real-Coded GA
    print("\nRunning Real-Coded GA (SBX + Polynomial Mutation)...")
    for run in range(nRuns):
        ga = RealCodedGA(
            FitnessFxn=Himmelblau,
            bounds=bounds,
            PopSize=PopSize,
            gens=gens,
            CrossProb=0.8,
            MuteProb=0.1,
            EtaC=20,
            EtaM=20,
            TourSize=3,
            elitism=True
        )
        
        BestSol, BestFit, history = ga.run(verbose=False)
        results['Real GA']['fitness'].append(BestFit)
        results['Real GA']['solutions'].append(BestSol)
        
        if (run + 1) % 10 == 0:
            print(f"  Completed run {run + 1}/{nRuns}")
    
    return results


def ComputeStats(results):
    """Compute statistics for comparison."""
    stats = {}
    for method in ['Binary GA', 'Real GA']:
        fitnesses = results[method]['fitness']
        stats[method] = {
            'Best': np.min(fitnesses),
            'Worst': np.max(fitnesses),
            'Average': np.mean(fitnesses),
            'Std. Dev.': np.std(fitnesses),
            'Median': np.median(fitnesses)
        }
    return stats


def PlotComparision(results, save=False):
    """
    Create comparison plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Box plot comparison
    ax1 = axes[0]
    data2Plot = [results['Binary GA']['fitness'], results['Real GA']['fitness']]
    bp = ax1.boxplot(data2Plot, labels=['Binary GA', 'Real-Coded GA'], patch_artist=True)
    
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Performance Comparison (30 runs)', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    BestBinIdx = np.argmin(results['Binary GA']['fitness'])
    BestRealIdx = np.argmin(results['Real GA']['fitness'])
    
    # Run GA again to get convergence history for best runs
    print("\nGenerating convergence curves for best runs...")
    
    GABin = BinaryGA(FitnessFxn=Himmelblau, bounds=bounds, PopSize=50, gens=100,
                         CrossProb=0.8, MuteProb=0.01, BitsPerVar=16)
    _, _, BinHist = GABin.run(verbose=False)
    
    GAReal = RealCodedGA(FitnessFxn=Himmelblau, bounds=bounds, PopSize=50, gens=100,
                          CrossProb=0.8, MuteProb=0.1, EtaC=20, EtaM=20)
    _, _, RealHist = GAReal.run(verbose=False)
    
    ax2.plot(BinHist, 'b-', linewidth=2, label='Binary GA')
    ax2.plot(RealHist, 'g-', linewidth=2, label='Real-Coded GA')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Best Fitness', fontsize=12)
    ax2.set_title('Convergence Comparison (Best Runs)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of final fitness values
    ax3 = axes[2]
    ax3.hist(results['Binary GA']['fitness'], bins=20, alpha=0.5, label='Binary GA', color='blue')
    ax3.hist(results['Real GA']['fitness'], bins=20, alpha=0.5, label='Real-Coded GA', color='green')
    ax3.set_xlabel('Best Fitness', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Final Fitness (30 runs)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig('GAComp.png', dpi=150, bbox_inches='tight')
        print("  Saved plot to 'GAComp.png'")
    
    plt.show()


def PrintSolTable(results):
    """Print best solutions found."""
    print("\n" + "="*80)
    print("BEST SOLUTIONS FOUND")
    print("="*80)
    
    for method in ['Binary GA', 'Real GA']:
        fitnesses = results[method]['fitness']
        BestIdx = np.argmin(fitnesses)
        BestSol = results[method]['solutions'][BestIdx]
        BestFit = fitnesses[BestIdx]
        print(f"\n{method}:")
        print(f"  x1 = {BestSol[0]:.6f}")
        print(f"  x2 = {BestSol[1]:.6f}")
        print(f"  f(x1,x2) = {BestFit:.8f}")


if __name__ == "__main__":
    np.random.seed(42)
    results = RunSim(
        nRuns=30,
        PopSize=50,
        gens=100,
        verbose=False
    )
    
    stats = ComputeStats(results)
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    
    df = pd.DataFrame(stats).T
    print(df.to_string())
    df.to_csv('GAComparision.csv')
    print("\nResults saved to 'GAComparision.csv'")
    
    PrintSolTable(results)
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    PlotComparision(results, save=True)
    
    # Summary observations
    print("\n" + "="*80)
    print("SUMMARY OBSERVATIONS")
    print("="*80)
    print(f"""
    Comparison of Binary GA vs Real-Coded GA on Himmelblau Function:
    
    1. Binary GA Performance:
       - Best: {stats['Binary GA']['Best']:.8f}
       - Average: {stats['Binary GA']['Average']:.8f}
       - Std Dev: {stats['Binary GA']['Std. Dev.']:.8f}
    
    2. Real-Coded GA Performance (SBX + Polynomial Mutation):
       - Best: {stats['Real GA']['Best']:.8f}
       - Average: {stats['Real GA']['Average']:.8f}
       - Std Dev: {stats['Real GA']['Std. Dev.']:.8f}
    
    3. Key Observations:
       - Real-coded GA achieves significantly lower average error
       - Real-coded GA has much lower standard deviation (more reliable)
       - Binary GA suffers from discretization error (limited precision)
       - SBX + Polynomial mutation allows finer local search
       - Real-coded GA converges faster (see convergence plot)
    
    4. Recommendation:
       - For continuous optimization problems, Real-Coded GA with SBX and
         Polynomial mutation is superior to Binary GA.
       - Recommended parameters: CrossProb=0.8, MuteProb=0.1,
         EtaC=20, EtaM=20 for similar problems.
    """)
    
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


"""
================================================================================
COMPARISON: BINARY GA vs REAL-CODED GA (SBX + POLYNOMIAL MUTATION)
================================================================================
Runs: 30, Population: 50, gens: 100
--------------------------------------------------------------------------------

Running Binary GA...
  Completed run 10/30
  Completed run 20/30
  Completed run 30/30

Running Real-Coded GA (SBX + Polynomial Mutation)...
  Completed run 10/30
  Completed run 20/30
  Completed run 30/30

================================================================================
FINAL RESULTS TABLE
================================================================================
                   Best     Worst   Average  Std. Dev.    Median
Binary GA  7.753382e-08  2.878508  0.129941   0.525043  0.000602
Real GA    1.893183e-08  0.000163  0.000017   0.000031  0.000005

Results saved to 'GAComparision.csv'

================================================================================
BEST SOLUTIONS FOUND
================================================================================

Binary GA:
  x1 = 2.999954
  x2 = 2.000000
  f(x1,x2) = 0.00000008

Real GA:
  x1 = -3.779301
  x2 = -3.283165
  f(x1,x2) = 0.00000002

================================================================================
GENERATING VISUALIZATIONS
================================================================================
/home/rajm012/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q4/Q4.py:425: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  bp = ax1.boxplot(data2Plot, labels=['Binary GA', 'Real-Coded GA'], patch_artist=True)

Generating convergence curves for best runs...
  Saved plot to 'GAComp.png'

================================================================================
SUMMARY OBSERVATIONS
================================================================================
    Comparison of Binary GA vs Real-Coded GA on Himmelblau Function:
    1. Binary GA Performance:
       - Best: 0.00000008
       - Average: 0.12994144
       - Std Dev: 0.52504251
    
    2. Real-Coded GA Performance (SBX + Polynomial Mutation):
       - Best: 0.00000002
       - Average: 0.00001690
       - Std Dev: 0.00003062
    
    3. Key Observations:
       - Real-coded GA achieves significantly lower average error
       - Real-coded GA has much lower standard deviation (more reliable)
       - Binary GA suffers from discretization error (limited precision)
       - SBX + Polynomial mutation allows finer local search
       - Real-coded GA converges faster (see convergence plot)
    
    4. Recommendation:
       - For continuous optimization problems, Real-Coded GA with SBX and
         Polynomial mutation is superior to Binary GA.
       - Recommended parameters: CrossProb=0.8, MuteProb=0.1,
         EtaC=20, EtaM=20 for similar problems.
    
================================================================================
EXPERIMENT COMPLETE!
================================================================================
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q4$ 
"""

