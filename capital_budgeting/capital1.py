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
# from mro.utils import get_n_processes, cluster_data
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
    t_vec = np.hstack([t/(t+1) for t in range(1, T+1)]*m)
    t_vec3 = np.hstack([t**(1/(t+1)) + t**(-t/(t+1)) for t in range(1, T+1)]*m)
    for k in range(N):
        constraints += [cp.constraints.power.PowCone3D(
            -y[k, :], cp.hstack([cp.multiply(F[j, 1:], x[j])
                                for j in range(m)]),
            cone[k, :], t_vec)]
    constraints += [cone <= 0]

    for k in range(N):
        constraints += [cp.sum(-z[k, :]) + cone[k, :]@t_vec3 - F[:, 0]@x - z[k]@dat[k] +
                        cp.quad_over_lin(z[k] + C_r.T@gam[k], 4*lam) + gam[k]@(d_r - C_r@dat[k]) <= s[k]]

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
    t_vec = np.hstack([t/(t+1) for t in range(1, T+1)]*m)
    t_vec3 = np.hstack([t**(1/(t+1)) + t**(-t/(t+1)) for t in range(1, T+1)]*m)
    for k in range(N):
        constraints += [cp.constraints.power.PowCone3D(
            -y[k, :], cp.hstack([cp.multiply(F[j, 1:], x[j])
                                for j in range(m)]),
            cone[k, :], t_vec)]

    constraints += [cone <= 0]

    for k in range(N):
        constraints += [cp.sum(-z[k, :]) + cone[k, :]@t_vec3 - F[:, 0]
                        @ x - z[k]@dat[k] + cp.quad_over_lin(z[k], 4*lam) <= s[k]]
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
                problem.solve(verbose=True)
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
            problem.solve(verbose=True)
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
            df.to_csv(foldername + '/dfv_' + str(r) + '.csv')
    return xsols, df


if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    # m = 12
    # N_tot = 50
    # T = 5
    # F = np.vstack([np.random.uniform(1, 5+0.04*t, m) for t in range(T+1)]).T
    # h = np.random.uniform(1, 3, m)
    # K_nums = [1, 2, 5, 10, 25, 50]
    # eps_nums = np.concatenate((np.logspace(-4.5, -2.95, 15),
    #                           np.logspace(-2.9, -1.9,
    #                                       15), np.logspace(-1.8, 0, 8),
    #                           np.logspace(0.1, 1, 3)))
    # R = 20
    # theta = 12

    m = 15
    N_tot = 80
    T = 6
    F = np.array([[1.88543562, 3.67048778, 1.5003693, 4.64029872, 3.97227343,
                   4.32608954, 4.95612091],
                  [2.25294218, 4.57008194, 1.92834117, 1.81739819, 1.4622979,
                   2.55112309, 1.82651784],
                  [3.37933607, 3.36363481, 4.01302427, 3.90568311, 1.73464693,
                   5.19000265, 5.11144727],
                  [1.7970513, 1.21479277, 3.61899437, 3.82024358, 4.30151879,
                   1.09787255, 2.22457537],
                  [2.59648069, 4.88036787, 1.32342337, 4.11377837, 1.68277106,
                   1.46692355, 1.83195355],
                  [3.52148473, 1.66853328, 4.15229252, 2.88080186, 1.55448922,
                   4.52033217, 4.76689795],
                  [1.81822375, 2.24923891, 2.07731721, 5.06401875, 2.64381853,
                   5.03782588, 2.94076785],
                  [4.14299893, 4.39838576, 1.89606465, 3.90719645, 4.10672585,
                   3.73877815, 1.20983229],
                  [3.87457108, 3.31007313, 1.88490476, 2.41169826, 2.66796219,
                   2.96440145, 4.3604664],
                  [2.606303, 3.56983035, 1.36577399, 1.33278638, 2.74266073,
                   2.53941326, 1.88994632],
                  [3.77514058, 4.79326704, 3.32897662, 3.52853068, 2.71548767,
                   2.50134167, 3.97703363],
                  [4.23914494, 3.98364698, 2.45747375, 3.83137351, 3.98948223,
                   1.63612536, 2.55109784],
                  [4.99478196, 3.75710909, 2.32616876, 3.66441703, 3.8925388,
                   2.01680395, 1.8762692],
                  [2.41161229, 2.38596086, 4.40727751, 2.95377216, 3.86131299,
                   4.83828471, 3.66753577],
                  [1.51797349, 1.37086407, 4.2025787, 3.77508901, 5.1057859,
                   4.90346142, 1.17422362]])*0.1
    h = np.array([2.81030099, 2.54554621, 2.89922928, 2.60889303, 1.44816106,
                  2.45265372, 1.08024165, 1.02978156, 2.11529249, 2.4284571,
                  1.71234055, 1.30026037, 1.2093349, 1.05672827, 1.75909205])
    K_nums = [1, 2, 5, 10, 25, 40, 80]
    theta = 15
    eps_nums = np.concatenate((np.logspace(-4.5, -2.95, 12),
                               np.logspace(-2.9, -1.9,
                                           4), np.logspace(-1.8, 0, 5),
                              np.logspace(0.1, 1, 3)))
    R = 1

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(capital_experiment)(
        R, r, m, N_tot, K_nums, eps_nums) for r in range(R))

    # x_sols = np.zeros((len(K_nums),len(eps_nums),m, R))
    dftemp = results[0][1]
    # for r in range(R):
    #    x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    dftemp.to_csv(foldername + '/df.csv')

    #all = pd.concat([results[r][1] for r in range(R)])
    #all.to_csv(foldername + '/df_all2.csv')
