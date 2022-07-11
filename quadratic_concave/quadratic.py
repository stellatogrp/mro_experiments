from sklearn.cluster import KMeans
from pathlib import Path
from joblib import Parallel, delayed
import os
import mosek
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

def get_n_processes(max_n=np.inf):
    """Get number of processes from current cps number
    Parameters
    ----------
    max_n: int
        Maximum number of processes.
    Returns
    -------
    float
        Number of processes to use.
    """

    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc

def normal_returns_scaled(N, m,scale):
    """Creates scaled data
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Scales: float
        Multiplier for a single mode
    Returns:
    -------
    d: matrix
        Scaled data with a single mode
    """    
    R = np.vstack([np.random.normal(
        i*0.03*scale, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
    return (R.transpose())

def data_modes(N,m,scales):
    """Creates data scaled by given multipliers
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Scales: vector
        Multipliers of different modes
    Returns:
    -------
    d: matrix
        Scaled data with all modes
    """
    modes = len(scales)
    d = np.zeros((N+100,m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,:] = normal_returns_scaled(weights,m,scales[i])
    return d[0:N,:]

def createproblem_quad(N, m,A):
    """Create the problem in cvxpy
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    A: dict
      Matrixes for the quadratic equation
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable((N, m))
    y = cp.Variable((N,m*m))

    # OBJECTIVE #
    objective = cp.multiply(eps, lam) + s@w

    # CONSTRAINTS #
    constraints = []
    for k in range(N):
        constraints += [cp.sum([cp.quad_over_lin(A[ind]@(y[k,ind*(m):(ind+1)*m]), 2*x[ind]) for ind in range(m)]) - dat[k]@z[k] + cp.quad_over_lin(z[k], 4*lam) <= s[k]]
    constraints += [cp.sum(x) == 1]
    for k in range(N):
        constraints += [cp.sum([y[k,ind*(m):(ind+1)*m] for ind in range(m)],axis = 0)== z[k]]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, lam, dat, eps, w

def quadratic_experiment(A, Ainv, r, m, N_tot, K_nums, eps_nums, foldername):
    '''Run the experiment for multiple K and epsilon
    Parameters
    ----------
    Various inputs for combinations of experiments
    Returns
    -------
    df: dataframe
        The results of the experiments
    '''
    df = pd.DataFrame(columns = ["r","K","Epsilon","Opt_val","Eval_val","satisfy","solvetime","bound"])
    xsols = np.zeros((len(K_nums),len(eps_nums),m, R))
    d = data_modes(N_tot,m,[1,5,15,25,40])
    d2 = data_modes(N_tot,m,[1,5,15,25,40])
    for Kcount, K in enumerate(K_nums):
        kmeans = KMeans(n_clusters=K).fit(d)
        weights = np.bincount(kmeans.labels_) / N_tot
        for epscount, epsval in enumerate(eps_nums):
            problem, x, s, lam, dat, eps, w = createproblem_quadnew(K, m, Ainv)
            eps.value = epsval**2
            dat.value = kmeans.cluster_centers_
            w.value = weights
            problem.solve()
            evalvalue = np.mean(-0.5*(d2@np.sum([A[i]*x.value[i] for i in range(m)],axis = 0))@(d2.T))
            xsols[Kcount, epscount, :, r] = x.value
            L = np.linalg.norm(np.sum([A[i]*x.value[i] for i in range(m)],axis = 0),2)
            newrow = pd.Series(
                {"r":r,
                 "K": K,
                 "Epsilon": epsval,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy": evalvalue <= problem.objective.value,
                 "solvetime": problem.solver_stats.solve_time,
                 "bound": (L/(2*N_tot))*kmeans.inertia_
            })
            df = df.append(newrow,ignore_index = True)
    return xsols, df
  

if __name__ == '__main__':
    foldername = "concave/m10_K60_r20"
    N_tot = 60
    m = 10
    R = 20
    K_nums = [1,2,3,4,5,10,30,60]
    A = {}
    Ainv = {}
    for i in range(m):
        A[i] = datasets.make_spd_matrix(m)
        Ainv[i] = sc.linalg.sqrtm(np.linalg.inv(A[i])) 
    eps_nums = np.array([0.01, 0.015, 0.023, 0.036, 0.055, 0.085, 0.13, 0.20, 0.30, 0.5, 0.7,1, 1.2, 1.4, 1.43, 1.47, 1.51, 1.55, 1.58, 1.62, 1.66, 1.7, 1.73, 1.77, 1.81, 1.85, 1.88, 1.92, 1.96, 2, 2.02, 2.07, 2.11, 2.15, 2.18, 2.22, 2.26, 2.3, 2.5, 2.7,3,4,9,10])

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(quadratic_experiment)(
        A, Ainv, r, m, N_tot, K_nums, eps_nums, foldername) for r in range(R))
    
    x_sols = np.zeros((len(K_nums),len(eps_nums),m, R))
    dftemp = results[0][1]
    for r in range(R):
        x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    np.save(Path("/scratch/gpfs/iywang/mro_results/" +
            foldername + "/x_sols.npy"), x_sols)
    dftemp.to_csv('/scratch/gpfs/iywang/mro_results/' + foldername + '/df.csv')