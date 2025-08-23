import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import uniform

# Benchmark Functions (from Get_Functions_details.m)
def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += (np.sum(x[:i+1]))**2
    return o

def F4(x):
    return np.max(np.abs(x))

def F5(x):  # Rosenbrock
    dim = len(x)
    o = 0
    for i in range(dim - 1):
        o += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return o

def F6(x):  # Step function
    return np.sum((np.floor(x + 0.5))**2)

def F7(x):
    dim = len(x)
    o = np.sum(np.arange(1, dim+1) * (x**4))
    o += np.random.rand()  # noise
    return o

def F8(x):
    o = 0
    for xi in x:
        o += xi * np.sin(np.sqrt(np.abs(xi)))
    return -o

def F9(x):
    dim = len(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim

def F10(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)

def F11(x):
    dim = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1

def F12(x):
    dim = len(x)
    o = (np.pi / dim) * (10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4)))**2 +
                         np.sum((((x[:dim-1] + 1) / 4)**2) * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4)))**2)) +
                         ((x[-1] + 1) / 4)**2) + np.sum(Ufun(x, 10, 100, 4))
    return o

def F13(x):
    dim = len(x)
    o = 0.1 * ((np.sin(3 * np.pi * x[0]))**2 +
               np.sum((x[:dim-1] - 1)**2 * (1 + (np.sin(3 * np.pi * x[1:]))**2)) +
               ((x[-1] - 1)**2) * (1 + (np.sin(2 * np.pi * x[-1]))**2)) + np.sum(Ufun(x, 5, 100, 4))
    return o

def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]).T  # Transposed for columns
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j])**6)
    o = (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS)))**(-1)
    return o

def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    o = np.sum((aK - ((x[0] * (bK**2 + x[1] * bK)) / (bK**2 + x[2] * bK + x[3])))**2)
    return o

def F16(x):
    return 4*(x[0]**2) - 2.1*(x[0]**4) + (x[0]**6)/3 + x[0]*x[1] - 4*(x[1]**2) + 4*(x[1]**4)

def F17(x):
    return (x[1] - (x[0]**2)*5.1/(4*(np.pi**2)) + 5/np.pi * x[0] - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(x[0]) + 10

def F18(x):
    term1 = (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*(x[0]**2) - 14*x[1] + 6*x[0]*x[1] + 3*(x[1]**2)))
    term2 = (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*(x[0]**2) + 48*x[1] - 36*x[0]*x[1] + 27*(x[1]**2)))
    return term1 * term2

def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * ((x - pH[i])**2)))
    return o

def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * ((x - pH[i])**2)))
    return o

def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(5):
        dist_sq = np.dot((x - aSH[i]), (x - aSH[i]))
        o -= 1 / (dist_sq + cSH[i])
    return o

def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(7):
        dist_sq = np.dot((x - aSH[i]), (x - aSH[i]))
        o -= 1 / (dist_sq + cSH[i])
    return o

def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(10):
        dist_sq = np.dot((x - aSH[i]), (x - aSH[i]))
        o -= 1 / (dist_sq + cSH[i])
    return o

def Ufun(x, a, k, m):
    dim = len(x)
    o = np.zeros(dim)
    for i in range(dim):
        if x[i] > a:
            o[i] = k * ((x[i] - a)**m)
        elif x[i] < -a:
            o[i] = k * ((-x[i] - a)**m)
    return o

# Get_Functions_details
def Get_Functions_details(F):
    functions = {
        'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'F5': F5, 'F6': F6, 'F7': F7,
        'F8': F8, 'F9': F9, 'F10': F10, 'F11': F11, 'F12': F12, 'F13': F13,
        'F14': F14, 'F15': F15, 'F16': F16, 'F17': F17, 'F18': F18,
        'F19': F19, 'F20': F20, 'F21': F21, 'F22': F22, 'F23': F23
    }

    if F not in functions:
        raise ValueError(f"Function {F} not defined")

    fobj = functions[F]

    lb_dict = {
        'F1': -100, 'F2': -10, 'F3': -100, 'F4': -100, 'F5': -30, 'F6': -100,
        'F7': -1.28, 'F8': -500, 'F9': -5.12, 'F10': -32, 'F11': -600, 'F12': -50,
        'F13': -50, 'F14': -65.536, 'F15': -5, 'F16': -5, 'F17': np.array([-5, 0]),
        'F18': -2, 'F19': 0, 'F20': 0, 'F21': 0, 'F22': 0, 'F23': 0
    }
    ub_dict = {
        'F1': 100, 'F2': 10, 'F3': 100, 'F4': 100, 'F5': 30, 'F6': 100,
        'F7': 1.28, 'F8': 500, 'F9': 5.12, 'F10': 32, 'F11': 600, 'F12': 50,
        'F13': 50, 'F14': 65.536, 'F15': 5, 'F16': 5, 'F17': np.array([10, 15]),
        'F18': 2, 'F19': 1, 'F20': 1, 'F21': 10, 'F22': 10, 'F23': 10
    }
    dim_dict = {
        'F1': 30, 'F2': 30, 'F3': 30, 'F4': 30, 'F5': 30, 'F6': 30,
        'F7': 30, 'F8': 30, 'F9': 30, 'F10': 10, 'F11': 30, 'F12': 30,
        'F13': 30, 'F14': 2, 'F15': 4, 'F16': 2, 'F17': 2, 'F18': 2,
        'F19': 3, 'F20': 6, 'F21': 4, 'F22': 4, 'F23': 4
    }

    lb = lb_dict[F]
    ub = ub_dict[F]
    dim = dim_dict[F]

    return lb, ub, dim, fobj

# BBSO Algorithm
def BBSO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction):
    # Boxelder Bug Search Optimization (BBSO): Optimization Inspired by the Ecology and Behavior of Boxelder Bugs
    # This algorithm simulates the coordinated swarm behavior of Boxelder bugs searching for warm overwintering sites.
    # Key concepts: Temperature as inverse of cost, coordinated following of better positions, population reduction over time.
    #
    # Based on the paper: "Boxelder Bugs Search Optimization: A Novel Reliable Tool for Optimizing Engineering Problems Through Bio-Inspired Ecology of Boxelder Bugs"
    # Authors: Iraj Faraji Davoudkhani, Hossein Shayeghi, Abdollah Younesi
    # Neural Computing and Applications (2025) ISSN: 0941-0643
    # https://doi.org/
    # Email: faraji.iraj@gmail.com

    Nw = 0.5  # Population reduction coefficient in Eq. (12)

    class BoxelderBug:
        def __init__(self):
            self.Position = None
            self.Cost = None

    # Initialize Population Array
    pop = [BoxelderBug() for _ in range(nPop)]

    # Initialize Best Solution
    BestSol = BoxelderBug()
    BestSol.Cost = np.inf
    BestSol.Position = np.zeros(nVar)

    # Create Initial Boxelder Bugs (Random Positions within Bounds)
    for i in range(nPop):
        if np.isscalar(VarMin):
            pop[i].Position = uniform.rvs(loc=VarMin, scale=VarMax - VarMin, size=nVar)
        else:
            pop[i].Position = np.array([uniform.rvs(loc=VarMin[j], scale=VarMax[j] - VarMin[j]) for j in range(nVar)])
        pop[i].Cost = CostFunction(pop[i].Position)
        if pop[i].Cost <= BestSol.Cost:
            BestSol.Position = pop[i].Position.copy()
            BestSol.Cost = pop[i].Cost

    # Sort Initial Population by Cost (Ascending: Lower Cost = Higher Temperature)
    pop = sorted(pop, key=lambda p: p.Cost)

    # Array to Hold Best Cost Values Over Iterations (dynamic size based on estimated iterations)
    EstIt = int(np.ceil(MaxIt / (nPop * nPop / 2)))  # Approximate iterations based on inner loop complexity
    ConvergenceCurve = np.zeros(EstIt)
    CountIter = 0

    it = nPop  # Start counting evaluations from initial population
    while it < MaxIt:
        newpop = [BoxelderBug() for _ in range(nPop)]

        # Compute Average Cost (Used for Temperature Scaling) (Eq. 9)
        Sfr = sum(p.Cost for p in pop) / nPop

        for i in range(nPop):
            newpop[i].Cost = np.inf
            newsol = BoxelderBug()
            newsol.Position = np.zeros(nVar)

            # Coordinated Movement: Model Following Better Bugs Ahead (Eq. 3)
            for j in range(1, i):  # j from 1 to i-1 (0-based: 1 to i-1 for j=2:i in 1-based)
                # Scaling Factor Based on Temperature (Inverse Cost Relation) (Eq. 8)
                fr = (pop[i].Cost / Sfr)

                # Update Position: Follow j and j-1 (Better Positions), with Decay Term (Eq. 4 – Eq. 7)
                pos_j = pop[j].Position
                pos_i = pop[i].Position
                pos_jm1 = pop[j-1].Position
                newsol.Position = (pos_j +
                                   np.random.rand(nVar) * (pos_j - pos_i) +
                                   np.random.rand(nVar) * (pos_jm1 - pos_j) +
                                   (fr ** -(it / nPop)) * np.random.randn(nVar) * (pos_j / VarMax if np.isscalar(VarMax) else np.mean(pos_j / VarMax)))

                # Bound the Position
                if np.isscalar(VarMin):
                    newsol.Position = np.maximum(newsol.Position, VarMin)
                    newsol.Position = np.minimum(newsol.Position, VarMax)
                else:
                    newsol.Position = np.maximum(newsol.Position, VarMin)
                    newsol.Position = np.minimum(newsol.Position, VarMax)

                # Evaluate New Position
                newsol.Cost = CostFunction(newsol.Position)
                it += 1  # Increment Function Evaluation Counter

                # Update if Better
                if newsol.Cost <= newpop[i].Cost:
                    newpop[i].Position = newsol.Position.copy()
                    newpop[i].Cost = newsol.Cost
                    if newpop[i].Cost <= BestSol.Cost:
                        BestSol.Position = newpop[i].Position.copy()
                        BestSol.Cost = newpop[i].Cost

            # Special handling for i=0 (first bug, no followers)
            if i == 0:
                # Small perturbation around current best
                newsol.Position = pop[0].Position + 0.1 * np.random.randn(nVar) * (VarMax - VarMin if np.isscalar(VarMax) else np.mean(VarMax - VarMin))
                if np.isscalar(VarMin):
                    newsol.Position = np.maximum(newsol.Position, VarMin)
                    newsol.Position = np.minimum(newsol.Position, VarMax)
                else:
                    newsol.Position = np.maximum(newsol.Position, VarMin)
                    newsol.Position = np.minimum(newsol.Position, VarMax)
                newsol.Cost = CostFunction(newsol.Position)
                it += 1
                if newsol.Cost <= newpop[0].Cost:
                    newpop[0].Position = newsol.Position.copy()
                    newpop[0].Cost = newsol.Cost
                    if newpop[0].Cost <= BestSol.Cost:
                        BestSol.Position = newpop[0].Position.copy()
                        BestSol.Cost = newpop[0].Cost

        # Population Reduction: Simulate Missing/Destruction (Reduce by Factor Proportional to Progress)(Eq. 12)
        nPop_new = nPop - Nw * nPop * (it / MaxIt)
        nPop_new = round(max(nPop_new, nVar))  # Minimum Population Size

        # Merge Old and New Swarm (Eq. 2)
        pop.extend(newpop)

        # Sort Combined Swarm by Cost
        pop = sorted(pop, key=lambda p: p.Cost)

        # Select Top nPop_new Bugs (Gather at Best Locations) (Eq. 11)
        pop = pop[:nPop_new]
        nPop = len(pop)

        CountIter += 1

        # Store Best Cost
        if CountIter < len(ConvergenceCurve):
            ConvergenceCurve[CountIter] = BestSol.Cost

        # Show Iteration Information
        print(f'Iteration {CountIter}: Best Cost = {BestSol.Cost}')

        if it >= MaxIt:
            break

    # Fill remaining curve with last best if early termination
    if CountIter < len(ConvergenceCurve):
        ConvergenceCurve[CountIter+1:] = BestSol.Cost

    # Outputs
    BestCost = BestSol.Cost
    BestPosition = BestSol.Position
    return BestCost, BestPosition, ConvergenceCurve

# func_plot
def func_plot(func_name, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    lb, ub, dim, fobj = Get_Functions_details(func_name)

    # Define x, y ranges based on func_name
    range_dict = {
        'F1': (-100, 100, 2), 'F2': (-10, 10, 0.2), 'F3': (-100, 100, 2), 'F4': (-100, 100, 2),
        'F5': (-30, 30, 2), 'F6': (-100, 100, 2), 'F7': (-1.28, 1.28, 0.03), 'F8': (-500, 500, 10),
        'F9': (-5.12, 5.12, 0.1), 'F10': (-32, 32, 0.5), 'F11': (-600, 600, 10), 'F12': (-50, 50, 0.1),
        'F13': (-50, 50, 0.08), 'F14': (-65.536, 65.536, 2), 'F15': (-5, 5, 0.1), 'F16': (-5, 5, 0.01),
        'F17': (-5, 10, 0.1), 'F18': (-2, 2, 0.06), 'F19': (0, 1, 0.1), 'F20': (0, 1, 0.1),
        'F21': (0, 10, 0.1), 'F22': (0, 10, 0.1), 'F23': (0, 10, 0.1)
    }
    if func_name not in range_dict:
        start, end, step = -5, 5, 0.1
    else:
        start, end, step = range_dict[func_name]
        if func_name == 'F17':
            x = np.linspace(start, ub[0], int((ub[0] - start)/step + 1))
            y = np.linspace(lb[1], ub[1], int((ub[1] - lb[1])/step + 1))
        else:
            x = np.arange(start, end + step, step)
            y = x

    if func_name == 'F17':
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = fobj(np.array([X[i,j], Y[i,j]]))
    else:
        L = len(x)
        f = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                    f[i, j] = fobj(np.array([x[i], y[j]]))
                elif func_name == 'F15':
                    f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))
                elif func_name == 'F19':
                    f[i, j] = fobj(np.array([x[i], y[j], 0]))
                elif func_name == 'F20':
                    f[i, j] = fobj(np.array([x[i], y[j], 0, 0, 0, 0]))
                elif func_name in ['F21', 'F22', 'F23']:
                    f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))
        X, Y = np.meshgrid(x, y)
        Z = f

    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

    if ax is None:
        plt.show()
    return ax

# Main execution
if __name__ == "__main__":
    print('Boxelder Bug Search Optimization (BBSO)')
    print('Based on the paper: "Boxelder Bugs Search Optimization: A Novel Reliable Tool for Optimizing Engineering Problems Through Bio-Inspired Ecology of Boxelder Bugs"')
    print('Authors: Iraj Faraji Davoudkhani, Hossein Shayeghi, Abdollah Younesi')
    print('Neural Computing and Applications (2025) ISSN: 0941-0643')
    print('https://doi.org/. e-mail: faraji.iraj@gmail.com')

    # Select Benchmark Function (e.g., 'F1' for Sphere)
    Function_name = 'F2'
    Max_iteration = 300000  # Maximum number of function evaluations
    nPop = 100  # Population Size (recommended: 5*nVar)

    # Load Function Details
    lb, ub, dim, fobj = Get_Functions_details(Function_name)

    # Run BBSO
    Best_score, Best_pos, cg_curve = BBSO(nPop, Max_iteration, lb, ub, dim, fobj)

    # Visualization
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    func_plot(Function_name, ax=ax1)
    ax1.set_title('Test Function')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_zlabel(f'{Function_name}(x_1, x_2)')

    ax2 = fig.add_subplot(122)
    # Plot non-zero part of convergence curve
    valid_indices = np.where(cg_curve > 0)[0]
    if len(valid_indices) > 0:
        ax2.semilogy(valid_indices, cg_curve[valid_indices], color='b', linewidth=2)
    ax2.set_title('Convergence Curve')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Score')
    ax2.grid(True)
    ax2.legend(['BBSO'])
    plt.tight_layout()
    plt.show()

    # Display Results
    print(f'The best solution obtained by BBSO is: {Best_pos}')
    print(f'The best optimal value found by BBSO is: {Best_score}')