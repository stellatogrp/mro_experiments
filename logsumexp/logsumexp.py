from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import os
import mosek
import time
import numpy as np
import cvxpy as cp
import pandas as pd
import sys
#from mro.utils import get_n_processes, cluster_data
import argparse

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

def dat_scaled(N, m,scale):
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
    R = np.vstack([np.random.uniform(0.01*i*scale,0.01*(i+1)*scale, N) for i in range(1, m+1)])
    return R.transpose()

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
    d = np.ones((N+100,m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,:] = dat_scaled(weights,m,scales[i])
    return d[0:N,:]

def createproblem_max(N, m,w):
    """Create the maximization problem to test constraint satisfaction
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    w: vector
        Weights for each data sample
    Returns:
    -------
    The cvxpy problem and parameters
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    expx = cp.Parameter(m)
    eps = cp.Parameter()
    
    u = cp.Variable((N,m))
    
    objective = cp.sum([w[k]*cp.log(u[k]@expx) for k in range(N)])
    # CONSTRAINTS #
    constraints = [cp.sum([cp.quad_over_lin(u[k]-dat[k], 1/w[k]) for k in range(N)])<= eps]
    #constraints += [u>= 0, u <= 3]

    # PROBLEM #
    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, u,expx,dat,eps

def createproblem_max1(N, m,w):
    """Create the maximization problem to test constraint satisfaction
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    w: vector
        Weights for each data sample
    Returns:
    -------
    The cvxpy problem and parameters
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    expx = cp.Parameter(m)
    eps = cp.Parameter()
    
    u = cp.Variable((N,m))
    
    objective = cp.sum([w[k]*cp.log(u[k]@expx) for k in range(N)])
    # CONSTRAINTS #
    constraints = [cp.sum([cp.quad_over_lin(u[k]-dat[k], 1/w[k]) for k in range(N)])<= eps]
    constraints += [u>= 0, u <= 4]

    # PROBLEM #
    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, u,expx,dat,eps

def createproblem_min(N, m,w,Uvals,n_planes):
    """Create minimization problem to ensure constraint satisfaction
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    w: vector
        Weights for each data sample
    Uvals: dict
        Set of uncertainty realizatoins
    n_planes: 
        Number of cutting planes added
    Returns:
    -------
    The cvxpy problem and parameters
    """
    # PARAMETERS #    
    x = cp.Variable(m)
    t = cp.Variable()
    objective = t
    
    # CONSTRAINTS #
    constraints = [cp.sum(x) >= 10, x>= 0, x <= 10]
    for index in range(n_planes):
        constraints += [cp.sum([w[k]*cp.log_sum_exp(x + Uvals[index][k]) for k in range(N)]) <= t]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x
  

def minmaxsolve(N,m,w,data,epsilon, probmin, probmax):
    """Cutting plane procedure
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    w: vector
        Weights for each data sample
    data: matrix
        Input data
    epsilon: 
        Input epsilon
    Returns:
    -------
    objs2:
        Final objective value
    x.value:
        Value for variable x
    solvetime: 
        Total solvertime 
    inds: 
        Number of cutting planes added
    """
    Uvals = {}
    inds = 0
    Uvals[inds] = np.log(data)
    inds += 1
    solvetime = 0
    problem1, x= probmin(N, m,w,Uvals,inds)
    problem1.solve()
    objs1 = problem1.objective.value
    solvetime += problem1.solver_stats.solve_time
    problem, u,expx,dat,eps= probmax(N, m,w)
    dat.value = data
    eps.value = epsilon
    expx.value = np.exp(x.value)
    problem.solve()
    solvetime += problem.solver_stats.solve_time
    Uvals[inds] = np.log(u.value)
    inds += 1
    problem1, x= probmin(N, m,w,Uvals,inds)
    problem1.solve()
    solvetime += problem1.solver_stats.solve_time
    objs2 = problem1.objective.value
    while(np.abs(objs1 - objs2)>= 0.0001 and inds <= 50):
        expx.value = np.exp(x.value)
        problem.solve()
        solvetime += problem.solver_stats.solve_time
        Uvals[inds] = np.log(u.value)
        inds += 1
        problem1, x= probmin(N, m,w,Uvals,inds)
        problem1.solve()
        solvetime += problem1.solver_stats.solve_time
        objs1 = objs2
        objs2 = problem1.objective.value
    return objs2, x.value, solvetime, inds
    
def logsumexp_experiment(r, m, N_tot, K_nums, eps_nums, foldername):
    '''Run the experiment for multiple K and epsilon
    Parameters
    ----------
    Various inputs for combinations of experiments
    Returns
    -------
    df: dataframe
        The results of the experiments
    '''
    df = pd.DataFrame(columns = ["r","K","Epsilon","Opt_val","Eval_val","satisfy","solvetime","bound","iters"])
    d = data_modes(N_tot,m,[1,3,7])
    d2 = data_modes(N_tot,m,[1,3,7])
    for Kcount, K in enumerate(K_nums):
        kmeans = KMeans(n_clusters=K).fit(d)
        weights = np.bincount(kmeans.labels_) / N_tot
        if K == N_tot:
            for epscount, epsval in enumerate(eps_nums):
                objs_val,x_val,time,iters = minmaxsolve(K,m,weights,kmeans.cluster_centers_,epsval**2,createproblem_min,createproblem_max1)
                evalvalue = cp.sum([(1/N_tot)*cp.log_sum_exp(x_val + np.log(d2[k])).value for k in range(N_tot)])
                newrow = pd.Series(
                    {"r":r,
                    "K": 9999,
                    "Epsilon": epsval,
                    "Opt_val": objs_val,
                    "Eval_val": evalvalue,
                    "satisfy": evalvalue <= objs_val,
                    "solvetime": time,
                    "bound": (1/(2*N_tot))*kmeans.inertia_,
                    "iters": iters
                })
                df = df.append(newrow,ignore_index = True)

        for epscount, epsval in enumerate(eps_nums):
            objs_val,x_val,time,iters = minmaxsolve(K,m,weights,kmeans.cluster_centers_,epsval**2,createproblem_min,createproblem_max)
            evalvalue = cp.sum([(1/N_tot)*cp.log_sum_exp(x_val + np.log(d2[k])).value for k in range(N_tot)])
            newrow = pd.Series(
                {"r":r,
                "K": K,
                "Epsilon": epsval,
                "Opt_val": objs_val,
                "Eval_val": evalvalue,
                "satisfy": evalvalue <= objs_val,
                "solvetime": time,
                "bound": (1/(2*N_tot))*kmeans.inertia_,
                "iters": iters
            })
            df = df.append(newrow,ignore_index = True)
            df.to_csv(foldername + '/df1_' + str(r) + '.csv')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    N_tot = 150
    m = 35
    R = 20
    K_nums = np.array([1,2,3,5,6,7,8,10,20,50,75,150])
    eps_nums = np.append(np.logspace(-5.2,-3.5,15),np.logspace(-3.45,1,10))
    
    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(logsumexp_experiment)(
        r, m, N_tot, K_nums, eps_nums, foldername) for r in range(R))
    
    dftemp = results[0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r].reset_index(), fill_value=0)
    dftemp = dftemp/R

    dftemp.to_csv(foldername + '/df1.csv')

