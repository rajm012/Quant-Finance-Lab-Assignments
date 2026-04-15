
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def Rastrigin(x):
    """
    Rastrigin function: f(x) = 10*n + sum(xI^2 - 10*cos(2*pi*xI))
    Global minimum: f(0,0) = 0 for n=2
    Bounds: usually [-5.12, 5.12]
    """
    n = len(x)
    return 10 * n + sum(xI**2 - 10 * np.cos(2 * np.pi * xI) for xI in x)


def Rosenbrock(x):
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    Global minimum: f(1,1) = 0 for n=2
    Bounds: usually [-5, 10]
    """
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def Himmelblau(x):
    """
    Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Four global minima: (3,2), (3.584,-1.848), (-2.805,3.131), (-3.779,-3.283)
    All give f ≈ 0
    Bounds: [-6, 6] for both variables
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


BoundsDict = {
    "Rastrigin": [(-5.12, 5.12), (-5.12, 5.12)],
    "Rosenbrock": [(-5, 10), (-5, 10)],
    "Himmelblau": [(-6, 6), (-6, 6)]
}


def GradOptimizeWithRestart(func, bounds, nRestarts=30, method="L-BFGS-B"):
    """
    Runs gradient-based optimization with multiple random restarts.
    
    Parameters:
    - func: objective function
    - bounds: list of tuples [(x1_min, x1_max), (x2_min, x2_max)]
    - nRestarts: number of random initializations
    - method: scipy.optimize.minimize method (default: L-BFGS-B)
    
    Returns:
    - BestSol: best x found
    - BestFit: best f(x) found
    - AllFit: list of all best fitnesses from each restart
    """
    
    dim = len(bounds)
    BestFit = np.inf
    BestSol = None
    AllFit = []
    
    for run in range(nRestarts):
        # Random initialization within bounds
        # Run gradient-based optimizer
        # Store fitness of this run
        # Update global best
        
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
        result = minimize(func, x0, method=method, bounds=bounds)        
        fitness = result.fun
        AllFit.append(fitness)
        if fitness < BestFit:
            BestFit = fitness
            BestSol = result.x
    
    return BestSol, BestFit, AllFit


def RunSim(FxnName, func, bounds, nRestarts=30, nExp=30):
    """
    Runs multiple experiments to get statistical results.
    Each experiment = multiple restarts, best result saved.
    """
    
    BestList = []
    print(f"\n{'='*50}")
    print(f"Running {FxnName} with {nRestarts} restarts per experiment")
    print(f"Total experiments: {nExp}")
    print(f"{'='*50}")
    
    # For each experiment, run multiple restarts and take best
    for exp in range(nExp):
        best_solution, best_fitness, all_fitness = GradOptimizeWithRestart(
            func, bounds, nRestarts=nRestarts
        )
        
        BestList.append(best_fitness)      
        if (exp + 1) % 5 == 0:
            print(f"  Completed experiment {exp + 1}/{nExp}")
    
    # Compute statistics
    BestVal = np.min(BestList)
    WorstVal = np.max(BestList)
    AvgVal = np.mean(BestList)
    StdVal = np.std(BestList)
    
    print(f"\nResults for {FxnName}:")
    print(f"  Best:   {BestVal:.6f}")
    print(f"  Worst:  {WorstVal:.6f}")
    print(f"  Avg:    {AvgVal:.6f}")
    print(f"  StdDev: {StdVal:.6f}")
    
    return {
        "Function": FxnName,
        "Best": BestVal,
        "Worst": WorstVal,
        "Average": AvgVal,
        "Std. Dev.": StdVal
    }


if __name__ == "__main__":
    np.random.seed(42)
    
    fxns = [
        ("Rastrigin", Rastrigin, BoundsDict["Rastrigin"]),
        ("Rosenbrock", Rosenbrock, BoundsDict["Rosenbrock"]),
        ("Himmelblau", Himmelblau, BoundsDict["Himmelblau"])
    ]
    
    results = []
    for name, func, bounds in fxns:
        result = RunSim(name, func, bounds, nRestarts=30, nExp=30)
        results.append(result)
    

    print("\n" + "="*70)
    print("FINAL RESULTS TABLE")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.set_index("Function")
    print(df.to_string())
    
    df.to_csv("Q1.csv")
    print("\nResults saved to 'Q1.csv'")


"""
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q1$ python Q1.py 

==================================================
Running Rastrigin with 30 restarts per experiment
Total experiments: 30
==================================================
  Completed experiment 5/30
  Completed experiment 10/30
  Completed experiment 15/30
  Completed experiment 20/30
  Completed experiment 25/30
  Completed experiment 30/30

Results for Rastrigin:
  Best:   0.000000
  Worst:  1.989918
  Avg:    0.895463
  StdDev: 0.470199

==================================================
Running Rosenbrock with 30 restarts per experiment
Total experiments: 30
==================================================
  Completed experiment 5/30
  Completed experiment 10/30
  Completed experiment 15/30
  Completed experiment 20/30
  Completed experiment 25/30
  Completed experiment 30/30

Results for Rosenbrock:
  Best:   0.000000
  Worst:  0.000000
  Avg:    0.000000
  StdDev: 0.000000

==================================================
Running Himmelblau with 30 restarts per experiment
Total experiments: 30
==================================================
  Completed experiment 5/30
  Completed experiment 10/30
  Completed experiment 15/30
  Completed experiment 20/30
  Completed experiment 25/30
  Completed experiment 30/30

Results for Himmelblau:
  Best:   0.000000
  Worst:  0.000000
  Avg:    0.000000
  StdDev: 0.000000

======================================================================
FINAL RESULTS TABLE
======================================================================
                    Best         Worst       Average     Std. Dev.
Function                                                          
Rastrigin   7.105427e-15  1.989918e+00  8.954632e-01  4.701993e-01
Rosenbrock  3.116479e-14  6.129918e-12  1.901001e-12  1.661418e-12
Himmelblau  1.063147e-17  1.153341e-15  4.897175e-16  3.276615e-16

Results saved to 'Q1.csv'
(venv) rajm012@rajm012:~/Desktop/6th Semester/4-Quant Finance Lab (Prof. Manoj Thakur)/Assignment-3/Q1$ 
"""

