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
from mro.utils import get_n_processes, cluster_data
output_stream = sys.stdout


def createproblem_port(N, m):
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
    # for k in range(2):
    #    constraints += [cp.sum(x[k*np.ceil(m/2):(k+1)*np.ceil(m/2)]) <= 0.50]
    constraints += [lam >= 0, y >=0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tao, y, lam, dat, eps, w


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
    Data = dat
    Data_eval = dateval
    df = pd.DataFrame(columns=["K", "Epsilon", "Opt_val", "Eval_val",
                               "satisfy", "solvetime", "setuptime", "clustertime"])

   ######################## solve for various K ########################
    for K_count, K in enumerate(K_nums):
        print(r, K)

        if K == N_tot:
            d_train, wk = cluster_data(Data[(N_tot*r):(N_tot*(r+1))], K)
            clustertimes = 0
        else:
            tnow = time.time()
            d_train, wk = cluster_data(Data[(N_tot*r):(N_tot*(r+1))], K)
            clustertimes = time.time() - tnow

        d_eval = Data_eval[(N_tot*r):(N_tot*(r+1))]
        tnow = time.time()
        problem, x, s, tao,y, lmbda, data_train_pm, eps_pm, w_pm = prob(K, m)
        data_train_pm.value = d_train
        w_pm.value = wk
        setuptimes = time.time() - tnow

        ######### solve for various epsilons ############
        for eps_count, eps in enumerate(eps_nums):
            print(K,eps_count)
            eps_pm.value = eps
            problem.solve(ignore_dpp=True, solver=cp.MOSEK,mosek_params={
                          mosek.dparam.optimizer_max_time:  1000.0})
            x_sols[K_count, eps_count, :, r] = x.value
            evalvalue = -5*np.mean(d_eval@x.value) - 5*tao.value <= y.value
            newrow = pd.Series(
                {"K": K,
                 "Epsilon": eps,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy": evalvalue,
                 "solvetime": problem.solver_stats.solve_time,
                 "setuptime": setuptimes,
                 "clustertime": clustertimes
                 })
            df = df.append(newrow, ignore_index=True)

            #df.to_csv('/scratch/gpfs/iywang/mro_results/' +
            #          foldername + '/df.csv')

    return x_sols, df



if __name__ == '__main__':
    foldername = "portfolio/cont/m200_K900_r10"
    synthetic_returns = pd.read_csv(
        '/scratch/gpfs/iywang/mro_experiments/portfolio/sp500_synthetic_returns.csv').to_numpy()[:, 1:]

    K_nums = np.array([1, 5, 50, 100, 300, 500, 800,900])
    # np.array([1,10,20,50,100,500,1000]) # different cluster values we consider
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 900
    M = 15
    R = 10      # Total times we repeat experiment 
    m = 200
    eps_min = -5    # minimum epsilon we consider
    eps_max = -3.5       # maximum epsilon we consider
    eps_nums = np.linspace(eps_min, eps_max, M)
    eps_nums = 10**(eps_nums)
    eps_tot = M

    dat = synthetic_returns[-10000:, :m]
    dateval = synthetic_returns[:10000, :m]
    njobs = get_n_processes(20)
    results = Parallel(n_jobs=njobs)(delayed(port_experiment)(
        dat, dateval, r, m, createproblem_port, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername) for r in range(R))

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
