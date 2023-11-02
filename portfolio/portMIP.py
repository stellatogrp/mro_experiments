import argparse
from joblib import Parallel, delayed
import os
import mosek
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import cvxpy as cp
import matplotlib.pyplot as plt
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
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w


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
    df = pd.DataFrame(columns=["R", "K", "Epsilon", "Opt_val", "Eval_val",
                               "satisfy", "solvetime", "bound"])
    Data = dat
    Data_eval = dateval

   ######################## solve for various K ########################
    for K_count, K in enumerate(K_nums):
        d_eval = Data_eval[(N_tot*r):(N_tot*(r+1))]
        kmeans = KMeans(n_clusters=K).fit(Data[(N_tot*r):(N_tot*(r+1))])
        d_train = kmeans.cluster_centers_
        wk = np.bincount(kmeans.labels_) / N_tot
        assert (d_train.shape == (K, m))
        problem, x, s, tau, lmbda, data_train_pm, eps_pm, w_pm = prob(K, m)
        data_train_pm.value = d_train
        ############## solve for various epsilons ###################
        for eps_count, eps in enumerate(eps_nums):
            w_pm.value = wk
            eps_pm.value = eps
            problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=True, mosek_params={
                mosek.dparam.optimizer_max_time:  1500.0})
            x_sols[K_count, eps_count, :, r] = x.value
            evalvalue = np.mean(
                np.maximum(-5*d_eval@x.value - 4*tau.value, tau.value)) <= problem.objective.value
            bound = np.max([np.max((d_train[k] - Data[(N_tot*r):(N_tot*(r+1))]
                           [kmeans.labels_ == k])@x.value) for k in range(K)])
            newrow = pd.Series(
                {"R": r,
                    "K": K,
                    "Epsilon": eps,
                    "Opt_val": problem.objective.value,
                    "Eval_val": np.mean(np.maximum(-5*d_eval@x.value - 4*tau.value, tau.value)),
                    "satisfy": evalvalue,
                    "solvetime": problem.solver_stats.solve_time,
                    "bound": bound
                 })
            df = df.append(newrow, ignore_index=True)
            #df.to_csv(foldername + '/df11_'+str(r)+'.csv')
    return x_sols, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    synthetic_returns = pd.read_csv(
        '/scratch/gpfs/iywang/mro_experiments/portfolio/sp500_synthetic_returns.csv').to_numpy()[:, 1:]

    K_nums = np.array([1, 4, 5, 10, 25, 50, 100, 150, 250, 500, 1000])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 1000
    M = 15
    R = 10         # Repeat experment R times in parallel
    m = 30
    eps_min = -3.5    # minimum epsilon we consider
    eps_max = -1.5       # maximum epsilon we consider
    eps_nums = np.linspace(eps_min, eps_max, M)
    eps_nums = 10**(eps_nums)
    eps_tot = M

    dat, dateval = train_test_split(
        synthetic_returns[:, :m], train_size=10000, test_size=10000, random_state=7)
    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(port_experiment)(
        dat, dateval, r, m, createproblem_portMIP, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername) for r in range(R))

    x_sols = np.zeros((K_tot, eps_tot, m, R))
    dftemp = results[0][1]
    for r in range(R):
        x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    dftemp.to_csv(foldername + '/df.csv')
