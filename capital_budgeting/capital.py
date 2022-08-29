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


def createproblem(N, m, T, F, h, w):
    """Creates the problem in cvxpy"""

    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    #A = cp.Parameter((m, m))

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m, boolean=True)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable((N, m))
    y = cp.Variable((N, T*m))
    obj = cp.Variable()
    cone = cp.Variable((N, T*m))
    d_r = np.concatenate(([np.ones(m), np.zeros(m)]))
    C_r = np.vstack([np.eye(m), -np.eye(m)])
    # OBJECTIVE #
    objective = obj
    constraints = []
    constraints += [lam*eps + w@s <= obj]
    gam = cp.Variable((N, 2*m))
    t_vec = np.array([t/(t+1) for t in range(1, T+1)])
    t_vec3 = np.hstack([t**(1/(t+1)) + t**(-t/(t+1)) for t in range(1, T+1)]*m)

    for k in range(N):
        for j in range(m):
            constraints += [
                cp.constraints.power.PowCone3D(
                    -y[k, (j*T):((j+1)*T)], cp.multiply(F[j, 1:], x[j]),
                    cone[k, (j*T):((j+1)*T)], t_vec)]
    constraints += [cone <= 0]

    for k in range(N):
        constraints += [cp.sum(-y[k, :] + cp.multiply(cone[k, :], t_vec3))
                        - cp.sum(cp.multiply(F[:, 0], x)) - z[k]@dat[k] +
                        cp.quad_over_lin(z[k] + C_r.T@gam[k], 4*lam) + gam[k]@(d_r - C_r@dat[k]) <= s[k]]

        # constraints += [cp.sum([cp.sum([-y[k,(j*(T))+t-1] + cone[k,(j*(T))+t-1]*(t**(1/(t+1)) + t**(-t/(t+1))) for t in range(1,T+1)])
        #                        - F[j,0]*x[j] for j in range(m)]) - z[k]@dat[k]
        #                + cp.quad_over_lin(z[k]+ C_r.T@gam[k], 4*lam) + gam[k]@(d_r - C_r@dat[k]) <= s[k]]
        for j in range(m):
            constraints += [cp.sum(y[k, (j*(T)):(j+1)*(T)]) == z[k, j]]
    constraints += [h@x <= theta]
    constraints += [lam >= 0]
    constraints += [y <= 0, gam >= 0]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, dat, eps


def createproblem_max(N, m, T, F, h, w):
    """Creates the problem in cvxpy"""

    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    #A = cp.Parameter((m, m))

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m, boolean=True)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable((N, m))
    y = cp.Variable((N, T*m))
    obj = cp.Variable()
    cone = cp.Variable((N, T*m))
    # OBJECTIVE #
    objective = obj
    constraints = []
    constraints += [lam*eps + w@s <= obj]
    t_vec = np.array([t/(t+1) for t in range(1, T+1)])
    t_vec3 = np.hstack([t**(1/(t+1)) + t**(-t/(t+1)) for t in range(1, T+1)]*m)
    for k in range(N):
        for j in range(m):
            constraints += [
                cp.constraints.power.PowCone3D(
                    -y[k, (j*T):((j+1)*T)], cp.multiply(F[j, 1:], x[j]),
                    cone[k, (j*T):((j+1)*T)], t_vec)]

    constraints += [cone <= 0]

    for k in range(N):
        constraints += [cp.sum(-y[k, :] + cp.multiply(cone[k, :], t_vec3))
                        - cp.sum(cp.multiply(F[:, 0], x)) - z[k]@dat[k] + cp.quad_over_lin(z[k], 4*lam) <= s[k]]
        #constraints += [cp.sum([cp.sum(-y[k,j*T:(j+1)*T] + cp.multiply(cone[k,j*T:(j+1)*T],tvec_2)) - F[j,0]*x[j] for j in range(m)]) - z[k]@dat[k] + cp.quad_over_lin(z[k], 4*lam) <= s[k]]
        for j in range(m):
            constraints += [cp.sum(y[k, j*T:(j+1)*T]) == z[k, j]]
    constraints += [h@x <= theta]
    constraints += [lam >= 0]
    constraints += [y <= 0]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, dat, eps


def capital_experiment(R, r, m, N_tot, K_nums, eps_nums):
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
                      "Eval_val", "satisfy", "solvetime", "bound", "bound2"])
    xsols = np.zeros((len(K_nums), len(eps_nums), m, R))
    d = (np.vstack([np.random.uniform(0.005*i, 0.02 *
         (i+1), int(N_tot/2)) for i in range(m)])).T
    d2 = (np.vstack([np.random.uniform(0.01*i, 0.025 *
          (i+1), int(N_tot/2)) for i in range(m)])).T
    dat_val = np.vstack([d, d2])
    d = (np.vstack([np.random.uniform(0.005*i, 0.02 *
         (i+1), int(N_tot/2)) for i in range(m)])).T
    d2 = (np.vstack([np.random.uniform(0.01*i, 0.025 *
          (i+1), int(N_tot/2)) for i in range(m)])).T
    dat_eval = np.vstack([d, d2])
    for Kcount, K in enumerate(K_nums):
        kmeans = KMeans(n_clusters=K).fit(dat_val)
        weights = np.bincount(kmeans.labels_) / N_tot
        for epscount, epsval in enumerate(eps_nums):
            if (K == N_tot):
                problem, x, dat, eps = createproblem_max(
                    K, m, T, F, h, weights)
                eps.value = epsval**2
                dat.value = kmeans.cluster_centers_
                problem.solve()
                evalvalue = -np.mean([np.sum([np.sum([F[j, t]*x[j].value/(1+dat_eval[i, j])
                                     ** t for t in range(T+1)]) for j in range(m)]) for i in range(N_tot)])
                L = np.linalg.norm(
                    [np.sum([(t**2 + t) * F[j, t]*x[j].value for t in range(T+1)]) for j in range(m)])
                L2 = np.max(
                    [np.sum([(t**2 + t) * F[j, t]*x[j].value for t in range(T+1)]) for j in range(m)])
                newrow = pd.Series(
                    {"R": r,
                     "K": 9999,
                     "Epsilon": epsval,
                     "Opt_val": problem.objective.value,
                     "Eval_val": evalvalue,
                     "satisfy": evalvalue <= problem.objective.value,
                     "solvetime": problem.solver_stats.solve_time,
                     "bound": (L/(2*N_tot))*kmeans.inertia_,
                     "bound2": (L2/(2*N_tot))*kmeans.inertia_
                     })
                df = df.append(newrow, ignore_index=True)
            problem, x, dat, eps = createproblem(K, m, T, F, h, weights)
            eps.value = epsval**2
            dat.value = kmeans.cluster_centers_
            problem.solve()
            evalvalue = -np.mean([np.sum([np.sum([F[j, t]*x[j].value/(1+dat_eval[i, j])
                                 ** t for t in range(T+1)]) for j in range(m)]) for i in range(N_tot)])
            xsols[Kcount, epscount, :, r] = x.value
            L = np.linalg.norm(
                [np.sum([(t**2 + t) * F[j, t]*x[j].value for t in range(T+1)]) for j in range(m)])
            L2 = np.max(
                [np.sum([(t**2 + t) * F[j, t]*x[j].value for t in range(T+1)]) for j in range(m)])
            newrow = pd.Series(
                {"R": r,
                 "K": K,
                 "Epsilon": epsval,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy": evalvalue <= problem.objective.value,
                 "solvetime": problem.solver_stats.solve_time,
                 "bound": (L/(2*N_tot))*kmeans.inertia_,
                 "bound2": (L2/(2*N_tot))*kmeans.inertia_
                 })
            df = df.append(newrow, ignore_index=True)
            problem = cp.Problem(cp.Minimize(0))
    return xsols, df


if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    #m = 12
    #N_tot = 50
    #T = 5
    #F = np.vstack([np.random.uniform(1,5+0.04*t,m) for t in range(T+1)]).T
    # for i in range(m):
    #    F[i,0] = -10 + i*0.6
    #h = np.random.uniform(1,3,m)
    #K_nums = [1,2,5,10,25,50]
    #eps_nums = np.concatenate((np.logspace(-4.5,-2.95,15), np.logspace(-2.9,-1.9,15),np.logspace(-1.8,0,8),np.logspace(0.1,1,3)))
    #R = 20

    m = 15
    N_tot = 100
    T = 7
    F = np.vstack([np.random.uniform(1, 5+0.04*t, m) for t in range(T+1)]).T
    h = np.random.uniform(1, 3, m)
    K_nums = [1, 2, 5, 10, 25, 50, 100]
    theta = 15
    eps_nums = np.concatenate((np.logspace(-4.5, -2.95, 15),
                               np.logspace(-2.9, -1.9,
                                           15), np.logspace(-1.8, 0, 8),
                               np.logspace(0.1, 1, 3)))
    R = 20

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(capital_experiment)(
        R, r, m, N_tot, K_nums, eps_nums) for r in range(R))

    #x_sols = np.zeros((len(K_nums),len(eps_nums),m, R))
    dftemp = results[0][1]
    # for r in range(R):
    #    x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    dftemp.to_csv(foldername + '/df1.csv')

    all = pd.concat([results[r][1] for r in range(R)])
    all.to_csv(foldername + '/df_all1.csv')
