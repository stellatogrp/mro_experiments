from joblib import Parallel, delayed
import os
import mosek
import time
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import cvxpy as cp
import matplotlib.pyplot as plt
from pathlib import Path
import sys
output_stream = sys.stdout


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


def cluster_data(D_in, K):
    """Return K cluster means after clustering D_in into K clusters
    Parameters
    ----------
    D_in: array
        Input dataset, N entries
    Returns
    -------
    Dbar_in: array
        Output dataset, K entries
    weights: vector
        Vector of weights for Dbar_in
    """
    N = D_in.shape[0]
    kmeans = KMeans(n_clusters=K).fit(D_in)
    Dbar_in = kmeans.cluster_centers_
    weights = np.bincount(kmeans.labels_) / N
    return Dbar_in, weights


def createproblem_portMIP(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m, boolean=True)
    tao = cp.Variable()
    y = cp.Variable()
    # OBJECTIVE #
    objective = tao + y

    # CONSTRAINTS #
    constraints = [cp.multiply(eps, lam) + w@s <= y]
    constraints += [cp.hstack([a*tao]*N) + a*dat@x +
                    cp.hstack([cp.quad_over_lin(-a*x, 4*lam)]*N) <= s]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0,y>=0]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tao,y, lam, dat, eps, w


def port_experiment(dat, dateval, r, m, prob, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername):
    """Run the experiment for multiple K and epsilon
    Parameters
    ----------
    Various inputs for combinations of experiments
    Returns
    -------
    x_sols: array
        The optimal solutions
    df: dataframe
        The results of the experiments
    """
    x_sols = np.zeros((K_tot, eps_tot, m, R))
    df = pd.DataFrame(columns=["K", "Epsilon", "Opt_val", "Eval_val",
                               "satisfy", "solvetime", "setuptime"])
    Data = dat
    Data_eval = dateval

   ######################## solve for various K ########################
    for K_count, K in enumerate(K_nums):
        tnow = time.time()
        d_train, wk = cluster_data(Data[(N_tot*r):(N_tot*(r+1))], K)
        d_eval = Data_eval[(N_tot*r):(N_tot*(r+1))]
        assert(d_train.shape == (K, m))
        problem, x, s, tao,y, lmbda, data_train_pm, eps_pm, w_pm = prob(K, m)
        data_train_pm.value = d_train
        w_pm.value = wk
        setuptimes = time.time() - tnow

        ############## solve for various epsilons ###################
        for eps_count, eps in enumerate(eps_nums):
            eps_pm.value = eps
            problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=True, mosek_params={
                          mosek.dparam.optimizer_max_time:  1200.0})
            x_sols[K_count, eps_count, :, r] = x.value
            evalvalue = -5*np.mean(d_eval@x.value) - 5*tao.value <= y.value
            newrow = pd.Series(
                {"K": K,
                 "Epsilon": eps,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy": evalvalue,
                 "solvetime": problem.solver_stats.solve_time,
                 "setuptime": setuptimes
                 })
            df = df.append(newrow, ignore_index=True)
            df.to_csv('/scratch/gpfs/iywang/mro_results/' +
                      foldername + '/df.csv')

    return x_sols, df


if __name__ == '__main__':
    foldername = "portfolio/MIP/m50_K300_r12"
    synthetic_returns = pd.read_csv(
        '/scratch/gpfs/iywang/mro_code/portfolio/sp500_synthetic_returns.csv').to_numpy()[:, 1:]

    K_nums = np.array([1, 5, 50, 100, 150, 300])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 300
    M = 10
    R = 12           # Total times we repeat experiment 
    m = 50
    eps_min = -5    # minimum epsilon we consider
    eps_max = -3.9       # maximum epsilon we consider
    eps_nums = np.linspace(eps_min, eps_max, M)
    eps_nums = 10**(eps_nums)
    eps_tot = M

    dat = synthetic_returns[:5000, :m]
    dateval = synthetic_returns[-5000:, :m]
    njobs = get_n_processes(20)
    results = Parallel(n_jobs=njobs)(delayed(port_experiment)(
        dat, dateval, r, m, createproblem_portMIP, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername) for r in range(R))

    x_sols = np.zeros((K_tot, eps_tot, m, R))
    dftemp = results[0][1]
    for r in range(R):
        x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R

    np.save(Path("/scratch/gpfs/iywang/mro_results/" +
            foldername + "/x_sols.npy"), x_sols)
    dftemp.to_csv('/scratch/gpfs/iywang/mro_results/' + foldername + '/df.csv')



    plt.rcParams.update({
        "text.usetex":True,
        "font.size":18,
        "font.family": "serif"
    })
    
    plt.figure(figsize=(10, 6))
    for K_count in np.arange(0,len(K_nums),1):
        plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="Objective, $K = {}$".format(K_nums[K_count]),alpha = 0.6)
    plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.title("In-sample objective value")
    plt.legend(loc = "lower right")
    plt.savefig("/scratch/gpfs/iywang/mro_code/portfolio/objectives.pdf")

    plt.figure(figsize=(10, 6))
    for K_count in np.arange(0,len(K_nums),1):
        plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
    plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.legend(loc = "lower right")
    plt.title(r"$1-\beta$ (probability of constraint satisfaction)")
    plt.savefig("/scratch/gpfs/iywang/mro_code/portfolio/constraint_satisfaction.pdf")

    plt.figure(figsize=(10, 6))
    for i in np.arange(1,len(eps_nums),3):
        plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
    plt.xlabel("$K$ (Number of clusters)")
    plt.title("Time (s)")
    plt.yscale("log")
    plt.legend(loc = "lower right")
    plt.savefig("/scratch/gpfs/iywang/mro_code/portfolio/time.pdf")

    print("COMPLETE")
