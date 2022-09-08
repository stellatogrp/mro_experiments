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
    t_vec = np.hstack([t/(t+1) for t in range(1, T+1)]*m)
    t_vec3 = np.hstack([t**(1/(t+1)) + t**(-t/(t+1)) for t in range(1, T+1)]*m)
    for k in range(N):
        constraints += [cp.constraints.power.PowCone3D(
            -y[k, :], cp.hstack([cp.multiply(F[j, 1:(T+1)], x[j])
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
            -y[k, :], cp.hstack([cp.multiply(F[j, 1:(T+1)], x[j])
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


def capital_experiment(R, r, m, N_tot, K_nums, eps_nums, T, h, F):
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
                      "Eval_val", "satisfy", "solvetime", "bound2"])
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
        if (K == N_tot):
            problem, x, dat, eps = createproblem_max(
                K, m, T, F, h, weights)
            dat.value = kmeans.cluster_centers_
            for epscount, epsval in enumerate(eps_nums):
                eps.value = epsval**2
                problem.solve(verbose=True)
                evalvalue = -np.mean([np.sum([np.sum([F[j, t]*x[j].value/(1+dat_eval[i, j])
                                                      ** t for t in range(T+1)]) for j in range(m)]) for i in range(N_tot)])
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
                        "bound2": (L2/(2*N_tot))*kmeans.inertia_
                     })
                df = df.append(newrow, ignore_index=True)
                df.to_csv(foldername + '/df25_' + str(r) + '.csv')
        problem, x, dat, eps = createproblem(K, m, T, F, h, weights)
        dat.value = kmeans.cluster_centers_
        for epscount, epsval in enumerate(eps_nums):
            eps.value = epsval**2
            problem.solve(verbose=True)
            evalvalue = -np.mean([np.sum([np.sum([F[j, t]*x[j].value/(1+dat_eval[i, j])
                                 ** t for t in range(T+1)]) for j in range(m)]) for i in range(N_tot)])
            xsols[Kcount, epscount, :, r] = x.value
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
                 "bound2": (L2/(2*N_tot))*kmeans.inertia_
                 })
            df = df.append(newrow, ignore_index=True)
            df.to_csv(foldername + '/df25_' + str(r) + '.csv')
    return xsols, df


if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    m = 20
    N_tot = 120
    T = 5
    F = np.array([[3.04439932, 2.32878295, 3.28161728, 3.04430874, 1.30594003,
                   2.23938309, 3.72369449],
                  [2.14152551, 1.97622888, 3.68613894, 3.18439621, 4.21996254,
                   3.29721677, 1.49235931],
                  [4.75588545, 3.95598204, 2.10374412, 4.79897375, 1.98531243,
                   1.23268197, 2.17177063],
                  [2.57055513, 3.78188201, 2.16396924, 2.11955746, 4.92317748,
                   2.52478893, 4.97464476],
                  [3.65483309, 4.74925006, 4.52078127, 4.30768365, 4.66249506,
                   4.29925948, 4.73758047],
                  [1.15708367, 2.26081143, 1.98636289, 3.29077189, 2.75734394,
                   3.26784972, 3.65969981],
                  [1.01908991, 1.86107589, 4.59958978, 3.55139629, 3.51844083,
                   4.05811917, 3.51943646],
                  [3.63555713, 2.64573341, 4.88320552, 1.4141386, 2.81266139,
                   2.38232994, 4.36299005],
                  [1.88809914, 3.90343705, 4.23752268, 1.50167862, 2.60653631,
                   3.27582436, 2.43816485],
                  [2.89373618, 4.55983413, 3.21335728, 4.7805664, 1.77771294,
                   3.60727395, 2.96870004],
                  [1.63443152, 3.64037498, 1.80881327, 2.52066466, 4.74599743,
                   4.97642555, 1.67720927],
                  [2.30938815, 1.33008264, 3.95319367, 3.76421232, 2.37904068,
                   2.06281848, 1.04714323],
                  [2.0411748, 1.81070973, 1.60359384, 4.26097767, 3.69035422,
                   1.40689722, 3.27617433],
                  [3.32711473, 1.51939908, 2.22386758, 3.4038399, 4.16475583,
                   1.60825388, 4.6685136],
                  [3.86544931, 4.58342722, 4.01736716, 3.93010647, 4.5788895,
                   1.59500486, 4.17083616],
                  [2.37978645, 3.62679661, 2.88543134, 1.45656764, 3.55658878,
                   2.24882723, 1.60898515],
                  [3.9687388, 1.55918814, 4.10319924, 1.50537202, 4.45457868,
                   4.3212519, 1.48697976],
                  [2.09547383, 1.93036262, 2.88401205, 5.06669848, 1.72907132,
                   1.47723346, 1.06301078],
                  [2.88630891, 3.72070099, 2.50244754, 2.68459504, 4.11830408,
                   4.62100216, 1.5734836],
                  [2.14458888, 2.59541872, 2.16148365, 4.68587847, 3.11542053,
                   2.09627239, 1.64773358]])*0.1
    h = np.array([2.156137, 1.18725155, 2.49400183, 2.11128728, 2.69863187,
                  1.3560057, 2.4379444, 1.50789587, 2.1868894, 1.98106176,
                  1.97623634, 1.79127991, 2.37318966, 1.5157423, 1.71861499,
                  1.84303014, 2.1818837, 1.85216891, 1.72312313, 1.30040749])
    K_nums = [1, 2, 3, 10, 30, 60, 120]
    theta = 12
    eps_nums = np.concatenate((np.logspace(-5.2, -2.95, 12),
                               np.logspace(-2.9, -1.9,
                                           10), np.logspace(-1.8, 0, 4),
                              np.logspace(0.1, 1, 3)))
    R = 3

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(capital_experiment)(
        R, r, m, N_tot, K_nums, eps_nums, T, h, F) for r in range(R))

    #x_sols = np.zeros((len(K_nums),len(eps_nums),m, R))
    dftemp = results[0][1]
    # for r in range(R):
    #    x_sols += results[r][0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R
    dftemp.to_csv(foldername + '/df25.1.csv')

    #all = pd.concat([results[r][1] for r in range(R)])
    #all.to_csv(foldername + '/df_all1.csv')
