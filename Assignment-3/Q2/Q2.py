import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Rastrigin(x):
    """Rastrigin function: f(x,y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))"""
    return 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))


def Rosenbrock(x):
    """Rosenbrock function: f(x,y) = 100(y - x^2)^2 + (1-x)^2"""
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def Himmelblau(x):
    """Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


BoundsDict = {
    "Rastrigin": [(-5.12, 5.12), (-5.12, 5.12)],
    "Rosenbrock": [(-5, 10), (-5, 10)],
    "Himmelblau": [(-6, 6), (-6, 6)]
}

# num of bits per variable
# Gives 2^16 = 65536 discrete levels within bounds
BITSPERVAR = 16


def Real2BinEncode(x, bounds, BitsPerVar=BITSPERVAR):
    """
    Convert real-valued variable to binary string.
    x: real value
    bounds: (lower, upper) for this variable
    """
    
    # Normalize to [0, 1]
    # Clamp to [0, 1] in case of floating point errors
    # Convert to integer in range [0, 2^bits - 1]
    # Convert to binary string with leading 
    
    lower, upper = bounds
    normalized = (x - lower) / (upper - lower)
    normalized = np.clip(normalized, 0, 1)
    maxInt = 2**BitsPerVar - 1
    intVal = int(normalized * maxInt)
    binStr = format(intVal, f'0{BitsPerVar}b')
    return binStr



def Bin2RealDecode(binStr, bounds, BitsPerVar=BITSPERVAR):
    """
    Convert binary string back to real value.
    """
    lower, upper = bounds
    maxInt = 2**BitsPerVar - 1
    intVal = int(binStr, 2)
    normalized = intVal / maxInt
    realVal = lower + normalized * (upper - lower)
    return realVal



def Chromosome2Real(chromosome, boundsList, BitsPerVar=BITSPERVAR):
    """
    Convert full chromosome (concatenated binary strings) to real vector.
    """
    
    nVars = len(boundsList)
    realVals = []
    for i in range(nVars):
        start = i * BitsPerVar
        end = start + BitsPerVar
        binStr = chromosome[start:end]
        realVal = Bin2RealDecode(binStr, boundsList[i], BitsPerVar)
        realVals.append(realVal)
        
    return np.array(realVals)



class BinaryGA:
    def __init__(self, fitnessFxn, bounds, PopSize=50, gens=100,
                 CrossProb=0.8, MuteProb=0.01, BitsPerVar=16,
                 tournamentSize=3, elitism=True):
        
        """
        Binary Genetic Algorithm.
        Parameters:
        - fitnessFxn: objective function (to minimize)
        - bounds: list of (lower, upper) for each variable
        - PopSize: population size
        - generations: number of generations
        - CrossProb: probability of crossover
        - MuteProb: probability of bit flip mutation
        - BitsPerVar: bits used per variable
        - tournamentSize: size of tournament selection
        - elitism: whether to preserve best individual
        """
        
        self.fitnessFxn = fitnessFxn
        self.bounds = bounds
        self.nVars = len(bounds)
        self.PopSize = PopSize
        self.generations = gens
        self.CrossProb = CrossProb
        self.MuteProb = MuteProb
        self.BitsPerVar = BitsPerVar
        self.tournamentSize = tournamentSize
        self.elitism = elitism
        
        # Chromosome length
        # Best tracking
        self.ChromosomeLth = self.nVars * self.BitsPerVar
        self.BestFitHistory = []
        self.BestSolHistory = []
        
        
    def initPop(self):
        """Initialize random binary population."""
        
        # Random binary string of length chromosome_length
        population = []
        for _ in range(self.PopSize):
            individual = ''.join(np.random.choice(['0', '1']) for _ in range(self.ChromosomeLth))
            population.append(individual)
        return population
    
    
    def evalFitness(self, population):
        """
        Evaluate fitness for all individuals.
        Returns: list of (fitness_value, individual)
        """
        
        fitnessScores = []
        for individual in population:
            # Decode to real values
            # Calculate objective (minimization problem)
            x = Chromosome2Real(individual, self.bounds, self.BitsPerVar)
            objVal = self.fitnessFxn(x)
            fitnessScores.append(objVal)
            
        return np.array(fitnessScores)
    
    
    def tourSelect(self, population, fitnessScores):
        """
        Tournament selection.
        Returns: selected parent
        """
        
        selected = []
        for _ in range(self.PopSize):
            # Pick random individuals for tournament
            # Winner = smallest fitness (minimization)
            tourIdxs = np.random.choice(self.PopSize, self.tournamentSize, replace=False)
            tousFitness = fitnessScores[tourIdxs]
            winnerIdx = tourIdxs[np.argmin(tousFitness)]
            selected.append(population[winnerIdx])
        return selected
    
    
    def SinglePointCross(self, parent1, parent2):
        """
        Single-point crossover with probability crossover_prob.
        """
        if np.random.random() < self.CrossProb:
            # Random crossover point (1 to length-1)
            point = np.random.randint(1, self.ChromosomeLth)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            return offspring1, offspring2
        else:
            return parent1, parent2
        
    
    def mutate(self, individual):
        """
        Bit-flip mutation with probability mutation_prob.
        """
        mutated = list(individual)
        for i in range(self.ChromosomeLth):
            if np.random.random() < self.MuteProb:
                mutated[i] = '1' if mutated[i] == '0' else '0'
        return ''.join(mutated)
    
    
    def evolve(self, population, fitnessScores):
        """
        One generation of evolution.
        """
        # Selection
        # Crossover and mutation to create offspring
        parents = self.tourSelect(population, fitnessScores)
        offspringPop = []
        
        for i in range(0, self.PopSize, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.PopSize else parents[0]
            
            # Crossover
            # Mutation
            child1, child2 = self.SinglePointCross(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspringPop.append(child1)
            offspringPop.append(child2)
        
        # Elitism: keep best individual from previous generation
        if self.elitism:
            BestIdx = np.argmin(fitnessScores)
            offspringPop[0] = population[BestIdx]
        
        return offspringPop
    
    
    def run(self, verbose=True):
        """
        Run the genetic algorithm.
        Returns: best solution, best fitness, and history
        """

        population = self.initPop()
        for gen in range(self.generations):
            # Evaluate fitness
            # Track best
            
            FitnessScores = self.evalFitness(population)
            BestIdx = np.argmin(FitnessScores)
            BestFit = FitnessScores[BestIdx]
            BestSol = Chromosome2Real(population[BestIdx], self.bounds, self.BitsPerVar)
            self.BestFitHistory.append(BestFit)
            self.BestSolHistory.append(BestSol)
            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                print(f"  Generation {gen:4d}: Best Fitness = {BestFit:.6f}")
            
            population = self.evolve(population, FitnessScores)
        
        FitnessScores = self.evalFitness(population)
        BestIdx = np.argmin(FitnessScores)
        BestFit = FitnessScores[BestIdx]
        BestSol = Chromosome2Real(population[BestIdx], self.bounds, self.BitsPerVar)
        return BestSol, BestFit, self.BestFitHistory


def RunSim(fxnName, func, bounds, nRuns=30, popSize=50, gens=100, crossProb=0.8, muteProb=0.01, verbose=False):
    """
    Run Binary GA multiple times and return statistics.
    """
    
    results = []
    BestSols = []
    print(f"\n{'='*60}")
    print(f"Binary GA on {fxnName}")
    print(f"Runs: {nRuns}, Population: {popSize}, Generations: {gens}")
    print(f"Crossover prob: {crossProb}, Mutation prob: {muteProb}")
    print(f"{'='*60}")
    
    for run in range(nRuns):
        # Create GA instance
        ga = BinaryGA(
            fitnessFxn=func,
            bounds=bounds,
            PopSize=popSize,
            gens=gens,
            CrossProb=crossProb,
            MuteProb=muteProb,
            BitsPerVar=16,
            tournamentSize=3,
            elitism=True)
        
        # Run GA
        BestSol, BestFit, history = ga.run(verbose=verbose and run == 0)
        results.append(BestFit)
        BestSols.append(BestSol)
        print(f"  Completed run {run + 1}/{nRuns}")
    
    # Compute statistics
    BestVal = np.min(results)
    WorstVal = np.max(results)
    AvgVal = np.mean(results)
    StdVal = np.std(results)
    
    print(f"\nResults for {fxnName}:")
    print(f"  Best:   {BestVal:.6f}")
    print(f"  Worst:  {WorstVal:.6f}")
    print(f"  Average: {AvgVal:.6f}")
    print(f"  Std Dev: {StdVal:.6f}")
    
    return {
        "Function": fxnName,
        "Best": BestVal,
        "Worst": WorstVal,
        "Average": AvgVal,
        "Std. Dev.": StdVal,
        "AllRuns": results
    }


def plotCGT(fxnName, func, bounds, save=True):
    """
    Plot convergence curve for a single representative run.
    """
    ga = BinaryGA(
        fitnessFxn=func,
        bounds=bounds,
        PopSize=50,
        gens=100,
        CrossProb=0.8,
        MuteProb=0.01
    )
    
    BestSol, BestFit, history = ga.run(verbose=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(f'Binary GA Convergence - {fxnName}', fontsize=14)
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f'CGT-{fxnName}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved plot to CGT-{fxnName}.png")
    
    plt.show()
    return BestSol, BestFit


if __name__ == "__main__":
    np.random.seed(42)
    fxns = [
        ("Rastrigin", Rastrigin, BoundsDict["Rastrigin"]),
        ("Rosenbrock", Rosenbrock, BoundsDict["Rosenbrock"]),
        ("Himmelblau", Himmelblau, BoundsDict["Himmelblau"])
    ]
    
    AllRes = []
    AllRunData = {}
    for name, func, bounds in fxns:
        result = RunSim(
            fxnName=name,
            func=func,
            bounds=bounds,
            nRuns=30,
            popSize=50,
            gens=100,
            crossProb=0.8,
            muteProb=0.01,
            verbose=False
        )
        AllRes.append(result)
        AllRunData[name] = result["AllRuns"]
    
    # Display final table
    print("\n" + "="*70)
    print("FINAL RESULTS TABLE - BINARY GENETIC ALGORITHM")
    print("="*70)
    df = pd.DataFrame(AllRes)
    df = df.set_index("Function")
    df = df.drop("AllRuns", axis=1)
    print(df.to_string())
    
    df.to_csv("BinaryGA.csv")
    print("\nResults saved to 'BinaryGA.csv'")
    print("\n" + "="*70)
    print("PLOTTING CONVERGENCE CURVES")
    print("="*70)
    
    for name, func, bounds in fxns:
        print(f"\nPlotting {name}...")
        bestSol, bestFit = plotCGT(name, func, bounds, save=True)
        print(f"  Best solution: x={bestSol[0]:.4f}, y={bestSol[1]:.4f}")
        print(f"  Best fitness: {bestFit:.6f}")
    

    print("\n" + "="*70)
    print("CREATING BOX PLOT COMPARISON")
    print("="*70)
    plt.figure(figsize=(10, 6))
    Data2Plot = [AllRunData[name] for name in ["Rastrigin", "Rosenbrock", "Himmelblau"]]
    bp = plt.boxplot(Data2Plot, tick_labels=["Rastrigin", "Rosenbrock", "Himmelblau"], patch_artist=True)
    

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title('Binary GA Performance Across Functions', fontsize=14)
    plt.yscale('log') 
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('BinaryGABoxPlot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nBox plot saved to 'BinaryGABoxPlot.png'")
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)


"""
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q2$ python Q2.py 

============================================================
Binary GA on Rastrigin
Runs: 30, Population: 50, Generations: 100
Crossover prob: 0.8, Mutation prob: 0.01
============================================================
  Completed run 1/30
  Completed run 2/30
  Completed run 3/30
  Completed run 4/30
  Completed run 5/30
  Completed run 6/30
  Completed run 7/30
  Completed run 8/30
  Completed run 9/30
  Completed run 10/30
  Completed run 11/30
  Completed run 12/30
  Completed run 13/30
  Completed run 14/30
  Completed run 15/30
  Completed run 16/30
  Completed run 17/30
  Completed run 18/30
  Completed run 19/30
  Completed run 20/30
  Completed run 21/30
  Completed run 22/30
  Completed run 23/30
  Completed run 24/30
  Completed run 25/30
  Completed run 26/30
  Completed run 27/30
  Completed run 28/30
  Completed run 29/30
  Completed run 30/30

Results for Rastrigin:
  Best:   0.000002
  Worst:  2.236828
  Average: 0.647305
  Std Dev: 0.690591

============================================================
Binary GA on Rosenbrock
Runs: 30, Population: 50, Generations: 100
Crossover prob: 0.8, Mutation prob: 0.01
============================================================
  Completed run 1/30
  Completed run 2/30
  Completed run 3/30
  Completed run 4/30
  Completed run 5/30
  Completed run 6/30
  Completed run 7/30
  Completed run 8/30
  Completed run 9/30
  Completed run 10/30
  Completed run 11/30
  Completed run 12/30
  Completed run 13/30
  Completed run 14/30
  Completed run 15/30
  Completed run 16/30
  Completed run 17/30
  Completed run 18/30
  Completed run 19/30
  Completed run 20/30
  Completed run 21/30
  Completed run 22/30
  Completed run 23/30
  Completed run 24/30
  Completed run 25/30
  Completed run 26/30
  Completed run 27/30
  Completed run 28/30
  Completed run 29/30
  Completed run 30/30

Results for Rosenbrock:
  Best:   0.000328
  Worst:  5.063274
  Average: 0.535484
  Std Dev: 0.986757

============================================================
Binary GA on Himmelblau
Runs: 30, Population: 50, Generations: 100
Crossover prob: 0.8, Mutation prob: 0.01
============================================================
  Completed run 1/30
  Completed run 2/30
  Completed run 3/30
  Completed run 4/30
  Completed run 5/30
  Completed run 6/30
  Completed run 7/30
  Completed run 8/30
  Completed run 9/30
  Completed run 10/30
  Completed run 11/30
  Completed run 12/30
  Completed run 13/30
  Completed run 14/30
  Completed run 15/30
  Completed run 16/30
  Completed run 17/30
  Completed run 18/30
  Completed run 19/30
  Completed run 20/30
  Completed run 21/30
  Completed run 22/30
  Completed run 23/30
  Completed run 24/30
  Completed run 25/30
  Completed run 26/30
  Completed run 27/30
  Completed run 28/30
  Completed run 29/30
  Completed run 30/30

Results for Himmelblau:
  Best:   0.000000
  Worst:  0.989043
  Average: 0.079617
  Std Dev: 0.210797

======================================================================
FINAL RESULTS TABLE - BINARY GENETIC ALGORITHM
======================================================================
                    Best     Worst   Average  Std. Dev.
Function                                               
Rastrigin   2.421852e-06  2.236828  0.647305   0.690591
Rosenbrock  3.279184e-04  5.063274  0.535484   0.986757
Himmelblau  7.753382e-08  0.989043  0.079617   0.210797

Results saved to 'BinaryGA.csv'

======================================================================
PLOTTING CONVERGENCE CURVES
======================================================================

Plotting Rastrigin...
  Generation    0: Best Fitness = 8.056258
  Generation   20: Best Fitness = 0.002250
  Generation   40: Best Fitness = 0.000002
  Generation   60: Best Fitness = 0.000002
  Generation   80: Best Fitness = 0.000002
  Generation   99: Best Fitness = 0.000002
  Saved plot to CGT-Rastrigin.png
  Best solution: x=0.0001, y=-0.0001
  Best fitness: 0.000002

Plotting Rosenbrock...
  Generation    0: Best Fitness = 1.814409
  Generation   20: Best Fitness = 0.733989
  Generation   40: Best Fitness = 0.733989
  Generation   60: Best Fitness = 0.733989
  Generation   80: Best Fitness = 0.733989
  Generation   99: Best Fitness = 0.733989
  Saved plot to CGT-Rosenbrock.png
  Best solution: x=0.1561, y=0.0391
  Best fitness: 0.733989

Plotting Himmelblau...
  Generation    0: Best Fitness = 0.718408
  Generation   20: Best Fitness = 0.016909
  Generation   40: Best Fitness = 0.016909
  Generation   60: Best Fitness = 0.016909
  Generation   80: Best Fitness = 0.016909
  Generation   99: Best Fitness = 0.016909
  Saved plot to CGT-Himmelblau.png
  Best solution: x=-3.7970, y=-3.2885
  Best fitness: 0.016909

======================================================================
CREATING BOX PLOT COMPARISON
======================================================================

Box plot saved to 'BinaryGABoxPlot.png'

======================================================================
EXPERIMENT COMPLETE!
======================================================================
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q2$ 

"""
