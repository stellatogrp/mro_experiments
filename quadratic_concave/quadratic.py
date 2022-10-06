from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import os
import mosek
import time
import numpy as np
import cvxpy as cp
import pandas as pd
import scipy as sc
from sklearn import datasets
import sys
from mro.utils import get_n_processes
import argparse


def normal_returns_scaled(N, m, scale):
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


def data_modes(N, m, scales):
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
    d = np.zeros((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,
          :] = normal_returns_scaled(weights, m, scales[i])
    return d[0:N, :]


def createproblem_quad(N, m, A):
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
    y = cp.Variable((N, m*m))
    #h = cp.Variable(m,boolean = True)

    # OBJECTIVE #
    objective = cp.multiply(eps, lam) + s@w

    # CONSTRAINTS #
    constraints = []
    for k in range(N):
        constraints += [cp.sum([cp.quad_over_lin(A[ind]@(y[k, ind*(m):(ind+1)*m]), 2*x[ind])
                               for ind in range(m)]) - dat[k]@z[k] + cp.quad_over_lin(z[k], 4*lam) <= s[k]]
    constraints += [cp.sum(x) == 1]
    for k in range(N):
        constraints += [cp.sum([y[k, ind*(m):(ind+1)*m]
                               for ind in range(m)], axis=0) == z[k]]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    #constraints += [x - h <= 0, cp.sum(h) <= 5]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, lam, dat, eps, w, z


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
    df = pd.DataFrame(columns=["R", "K", "Epsilon", "Opt_val",
                      "Eval_val", "satisfy", "solvetime", "bound"])
    #xsols = np.zeros((len(K_nums),len(eps_nums),m, R))
    xsols = 0
    d = data_modes(N_tot, m, [1, 5, 15, 25, 40])
    d2 = data_modes(N_tot, m, [1, 5, 15, 25, 40])
    for Kcount, K in enumerate(K_nums):
        kmeans = KMeans(n_clusters=K).fit(d)
        weights = np.bincount(kmeans.labels_) / N_tot
        problem, x, s, lam, dat, eps, w, z = createproblem_quad(K, m, Ainv)
        for epscount, epsval in enumerate(eps_nums):
            eps.value = epsval**2
            dat.value = kmeans.cluster_centers_
            w.value = weights
            problem.solve()
            evalvalue = np.mean(-0.5*(d2@np.sum([A[i]*x.value[i]
                                for i in range(m)], axis=0))@(d2.T))
            #xsols[Kcount, epscount, :, r] = x.value
            L = np.linalg.norm(np.sum([A[i]*x.value[i]
                               for i in range(m)], axis=0), 2)
            newrow = pd.Series(
                {"R": r,
                 "K": K,
                 "Epsilon": epsval,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy": evalvalue <= problem.objective.value,
                 "solvetime": problem.solver_stats.solve_time,
                 "bound": (L/(2*N_tot))*kmeans.inertia_
                 })
            df = df.append(newrow, ignore_index=True)
    return xsols, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername

    N_tot = 90
    m = 10
    R = 20
    K_nums = [1, 2, 3, 4, 5, 15, 45, 90]
    A = {}
    Ainv = {}
    for i in range(m):
        A[i] = datasets.make_spd_matrix(m, random_state=i)
        Ainv[i] = sc.linalg.sqrtm(np.linalg.inv(A[i]))

    eps_nums = np.concatenate((np.logspace(-2.2, -1, 8), np.logspace(-0.8, 0, 5),
                              np.logspace(0.1, 0.5, 20), np.array([3, 4, 7, 9, 10])))

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(quadratic_experiment)(
        A, Ainv, r, m, N_tot, K_nums, eps_nums, foldername) for r in range(R))

    dftemp = results[0][1]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    dftemp.to_csv(foldername + '/df.csv')
