import argparse
import os
import sys

import cvxpy as cp
import joblib
import mosek
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

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


def prob_facility_separate(K, m, n):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    eps = cp.Parameter()
    d_train = cp.Parameter((K, m))
    wk = cp.Parameter(K)
    p = cp.Parameter(n)
    c = cp.Parameter(n)
    C = cp.Parameter((n, m))
    # C_r = np.vstack([-np.eye(m)])
    # d_r = np.zeros(m)

    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    lmbda = cp.Variable((n, K))
    # s = cp.Variable(K)
    s = cp.Variable((n, K))
    gam = cp.Variable((n, K*m))

    objective = cp.Minimize(cp.trace(C.T @ X) + c@x)
    # cp.Minimize(t)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]
    # for i in range(n):
    #    constraints += [cp.multiply(eps, lmbda[i]) + wk @ s[i] <= 0]
    #    constraints += [cp.hstack([-p[i]*x[i]]*K) + d_train@X[i] +
    #                    cp.hstack([cp.quad_over_lin(X[i], 4*lmbda[i])]*K) <= s[i]]
    # for i in range(n):
    #    constraints += [cp.multiply(eps, lmbda[i]) + wk @ s[i] <= 0]
    #    constraints += [cp.hstack([-p[i]*x[i]]*K) + d_train@X[i] <= s[i]]
    #    constraints += [cp.norm(X[i],2) <= lmbda[i]]

    for i in range(n):
        constraints += [wk @ s[i] <= 0]
        for k in range(K):
            constraints += [lmbda[i, k]*eps - p[i]*x[i] + d_train[k]@X[i] +
                            gam[i, (k*m):((k+1)*m)]@(d_train[k]) <= s[i, k]]
            constraints += [cp.norm(-gam[i, (k*m):((k+1)*m)
                                         ] + X[i], 2) <= lmbda[i, k]]

    constraints += [X >= 0, lmbda >= 0, gam >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, X, s, lmbda, d_train, wk, eps, p, c, C


def prob_facility_separate_max(K, m, n):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    eps = cp.Parameter()
    d_train = cp.Parameter((K, m))
    wk = cp.Parameter(K)
    p = cp.Parameter(n)
    c = cp.Parameter(n)
    C = cp.Parameter((n, m))
    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    lmbda = cp.Variable(n)

    objective = cp.Minimize(cp.trace(C.T @ X) + c@x)
    # cp.Minimize(t)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]
    for i in range(n):
        constraints += [-p[i]*x[i] +
                        cp.multiply(eps, lmbda[i]) + X[i]@(wk @ d_train) <= 0]
        constraints += [cp.norm(X[i], 2) <= lmbda[i]]

    constraints += [X >= 0, lmbda >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, X, lmbda, d_train, wk, eps, p, c, C


def generate_facility_data(n=10, m=50):
    """Generate data for one problem instance
    Parameters
    ----------
    m: int
        Number of customers
    n: int
        Number of facilities
    Returns
    -------
    c: vector
        Opening cost of each facility
    C: array
        Shipment cost between customers and facilities
    p: vector
        Production capacity of each facility
    """
    # Cost for facility
    c = np.random.uniform(30, 70, n)

    # Cost for shipment
    fac_loc = np.random.uniform(0, 15, size=(n, 2))
    cus_loc = np.random.uniform(0, 15, size=(m, 2))
    #  rho = 4

    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(fac_loc[i, :] - cus_loc[j, :])

    # Capacities for each facility
    p = np.random.randint(10, 50, n)

    # Past demands of customer (past uncertain data)
    return c, C, p


def generate_facility_demands(N, m, R):
    """Generate uncertain demand
    Parameters:
     ----------
    N: int
        Number of data samples
    m: int
        Number of facilities
    R: int
        Number of sets of data samples
    Returns:
    -------
    d_train: vector
        Demand vector
    """
    # d_train = np.maximum(1, np.random.normal(2, 0.9, (N, m, R)))
    d_train = np.random.uniform(1, 6, (N, m, R))
    return d_train


def evaluate(p, x, X, d):
    """Evaluate constraint satisfaction
    Parameters
    ----------
    p: vector
        Prices
    x: vector
        Decision variables of the optimization problem
    X: matrix
        Decision variables of the optimization problem
    d: matrix
        Validation data matrix
    Returns:
    -------
    boolean: indicator of constraint satisfaction
    """
    for ind in range(n):
        if -p.value[ind]*x.value[ind] + \
                np.reshape(np.mean(d, axis=0), (1, m))@(X.value[ind]) >= 0.001:
            return 0
    return 1


def evaluate_k(p, x, X, d):
    """Evaluate stricter constraint satisfaction
    Parameters
    ----------
    p: vector
        Prices
    x: vector
        Decision variables of the optimization problem
    X: matrix
        Decision variables of the optimization problem
    d: matrix
        Validation data matrix
    Returns:
    -------
    boolean: indicator of constraint satisfaction
    """
    maxval = np.zeros((np.shape(d)[0], np.shape(x)[0]))
    for fac in range(np.shape(x)[0]):
        for ind in range(np.shape(d)[0]):
            maxval[ind, fac] = -p.value[fac]*x.value[fac] + d[ind]@X.value[fac]
    print(np.mean(np.max(maxval, axis=1)))
    if np.mean(np.max(maxval, axis=1)) >= 0.001:
        return 0
    return 1


def facility_experiment(r, n, m, Data, Data_eval, prob_facility,
                        N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername):
    '''Run the experiment for multiple K and epsilon
    Parameters
    ----------
    Various inputs for combinations of experiments
    Returns
    -------
    x_sols: array
        The optimal solutions for x
    X_sols: array
        The optimal solutions for X
    df: dataframe
        The results of the experiments
    '''
    # X_sols = np.zeros((K_tot, eps_tot, n, m))
    # x_sols = np.zeros((K_tot, eps_tot, n))
    X_sols = 0
    x_sols = 0
    df = pd.DataFrame(columns=["R", "K", "Epsilon", "Opt_val", "Eval_val",
                               "Eval_val1", "solvetime", ])

    # solve for various K
    for K_count, K in enumerate(np.flip(K_nums)):
        d_train, wk = cluster_data(Data[:, :, r], K)
        dat_eval = Data_eval[:, :, r]

        if K == N_tot:
            problem, x, X, lmbda, data_train_pm, w_pm, eps_pm, p_pm, c_pm, C_pm = \
                prob_facility_separate_max(K, m, n)
            data_train_pm.value = d_train
            w_pm.value = wk
            p_pm.value = p
            c_pm.value = c
            C_pm.value = C

            # solve for various epsilons
            for eps_count, eps in enumerate(eps_nums):
                eps_pm.value = eps
                problem.solve(solver=cp.MOSEK, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                evalvalue = evaluate(p_pm, x, X, dat_eval)
                evalvalue1 = evaluate_k(p_pm, x, X, dat_eval)
                newrow = pd.Series(
                    {"R": r,
                     "K": 0,
                     "Epsilon": eps,
                     "Opt_val": problem.objective.value,
                     "Eval_val": evalvalue,
                     "Eval_val1": evalvalue1,
                     "solvetime": problem.solver_stats.solve_time,
                     })
                df = df.append(newrow, ignore_index=True)
        problem, x, X, s, lmbda, data_train_pm, w_pm, eps_pm, p_pm, c_pm, C_pm = prob_facility(
            K, m, n)
        data_train_pm.value = d_train
        w_pm.value = wk
        p_pm.value = p
        c_pm.value = c
        C_pm.value = C

        # solve for various epsilons
        for eps_count, eps in enumerate(np.flip(eps_nums)):
            #            print(K, eps)
            eps_pm.value = eps
            problem.solve(solver=cp.MOSEK, mosek_params={
                          mosek.dparam.optimizer_max_time:  1500.0}, verbose=True)
            # X_sols[K_count, eps_count, :, :] = X.value
            # x_sols[K_count, eps_count, :] = x.value
            evalvalue = evaluate(p_pm, x, X, dat_eval)
            evalvalue1 = evaluate_k(p_pm, x, X, dat_eval)
            newrow = pd.Series(
                {"R": r,
                 "K": K,
                 "Epsilon": eps,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "Eval_val1": evalvalue1,
                 "solvetime": problem.solver_stats.solve_time,
                 })
            df = df.append(newrow, ignore_index=True)
    return X_sols, x_sols, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="facility/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    # different cluster values we consider
    K_nums = np.array([1, 5, 10, 25, 50, 100])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 100
    M = 10
    R = 10       # Total times we repeat experiment to estimate final probabilty
    n = 5  # number of facilities
    m = 25  # number of locations
    eps_min = 1      # minimum epsilon we consider
    eps_max = 10         # maximum epsilon we consider

    eps_nums = [0.05, 0.1, 0.11, 0.13, 0.15, 0.2, 0.25, 0.3, 0.35,
                0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
    eps_tot = M
    c, C, p = generate_facility_data(n, m)
    Data = generate_facility_demands(N_tot, m, R)
    Data_eval = generate_facility_demands(N_tot, m, R)

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(
        delayed(facility_experiment)(r, n, m, Data, Data_eval,
                                     prob_facility_separate, N_tot, K_tot, K_nums,
                                     eps_tot, eps_nums, foldername) for r in range(R))

    dftemp = results[0][2]

    for r in range(1, R):
        dftemp = dftemp.add(results[r][2].reset_index(), fill_value=0)
    dftemp = dftemp/R

    dftemp.to_csv(foldername + '/df.csv')
