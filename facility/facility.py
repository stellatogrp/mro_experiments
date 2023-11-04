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

    return Dbar_in, weights, kmeans


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
    x = cp.Variable(n, boolean=True)
    X = cp.Variable((n, m))
    lmbda = cp.Variable(K)
    tau = cp.Variable()
    s = cp.Variable(K)
    gam = cp.Variable((n, K*2*m))
    a =  5
    C_r = np.vstack([-np.eye(m), np.eye(m)])
    d_r = np.hstack([-np.ones(m), np.ones(m)*6])

    objective = cp.Minimize(cp.trace(C.T @ X) + c@x)

    constraints = []
    for j in range(m):
        constraints += [cp.sum(X[:, j]) == 1]

    constraints += [tau + wk @ s <= 0]
    for k in range(K):
        for i in range(n):
            constraints += [-a*tau+ lmbda[k]*eps - a*p[i]*x[i] + a*d_train[k]@X[i] +
                            gam[i, (k*2*m):((k+1)*2*m)]@(d_r - C_r@d_train[k]) <= s[k]]
            constraints += [cp.norm(C_r.T@gam[i, (k*2*m):((k+1)*2*m)
                                            ] - a*X[i], 2) <= lmbda[k]]
        constraints += [lmbda[k]*eps <= s[k]]
    constraints += [X >= 0, lmbda >= 0, gam >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, X, s, lmbda, d_train, wk, eps, p, c, C, gam, tau


# def prob_facility_separate_max(K, m, n):
#     """Create the problem in cvxpy
#     Parameters
#     ----------
#     K: int
#         Number of data samples
#     m: int
#         Number of customers
#     n: int
#         Number of facilities
#     Returns
#     -------
#     The instance and parameters of the cvxpy problem
#     """
#     eps = cp.Parameter()
#     d_train = cp.Parameter((K, m))
#     wk = cp.Parameter(K)
#     p = cp.Parameter(n)
#     c = cp.Parameter(n)
#     C = cp.Parameter((n, m))
#     x = cp.Variable(n, boolean=True)
#     X = cp.Variable((n, m))
#     lmbda = cp.Variable(K)
#     s = cp.Variable(K)
#     a = 5
#     tau = cp.Variable()
#     objective = cp.Minimize(cp.trace(C.T @ X) + c@x)

#     constraints = []
#     for j in range(m):
#         constraints += [cp.sum(X[:, j]) == 1]

#     constraints += [tau + wk @ s <= 0]
#     for k in range(K):
#         for i in range(n):
#             constraints += [-a*tau+ lmbda[k]*eps - a*p[i]*x[i] + a*d_train[k]@X[i] <= s[k]]
#             constraints += [cp.norm(a*X[i], 2) <= lmbda[k]]

#         constraints += [lmbda[k]*eps <= s[k]]
#     constraints += [X >= 0, lmbda >= 0]

#     problem = cp.Problem(objective, constraints)

#     return problem, x, X, s, lmbda, d_train, wk, eps, p, c, C, tau


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
    np.random.seed(1)
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
    p = np.random.randint(20, 60, n)

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

    dat = np.random.normal(4,0.8,(N,m,R))
    dat2 = np.random.normal(3,0.9,(N,m,R))
    dat = np.vstack([dat, dat2])
    dat = np.minimum(dat,6)
    dat = np.maximum(dat,1)
    return dat


def evaluate(p, x, X, tau, d):
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
    maxval = np.zeros((np.shape(d)[0], np.shape(x)[0]))
    for fac in range(np.shape(x)[0]):
        for ind in range(np.shape(d)[0]):
            maxval[ind, fac] = tau.value + np.maximum(-5*p.value[fac]*x.value[fac] + 5*d[ind]@X.value[fac] - 5*tau.value, 0)
    if np.mean(np.max(maxval, axis=1)) >= 0.001:
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
        d_train, wk, kmeans = cluster_data(Data[:, :, r], K)
        dat_eval = Data_eval[:, :, r]
        # if K == N_tot:
        #     problem, x, X, s, lmbda, data_train_pm, w_pm, eps_pm, p_pm, c_pm, C_pm, tau = \
        #         prob_facility_separate_max(K, m, n)
        #     data_train_pm.value = d_train
        #     w_pm.value = wk
        #     p_pm.value = p
        #     c_pm.value = c
        #     C_pm.value = C

        #     # solve for various epsilons
        #     for eps_count, eps in enumerate(np.flip(eps_nums)):
        #         eps_pm.value = eps
        #         problem.solve(solver=cp.MOSEK, mosek_params={
        #             mosek.dparam.optimizer_max_time:  1500.0})
        #         evalvalue = evaluate(p_pm, x, X, tau, dat_eval)
        #         evalvalue1 = evaluate_k(p_pm, x, X, dat_eval)
        #         newrow = pd.Series(
        #             {"R": r,
        #              "K": 0,
        #              "Epsilon": eps,
        #              "Opt_val": problem.objective.value,
        #              "Eval_val": evalvalue,
        #              "Eval_val1": evalvalue1,
        #              "solvetime": problem.solver_stats.solve_time,
        #              "bound": np.mean([np.max([(d_train[k] - Data[:, :, r][kmeans.labels_ == k])@(-5*X[i].value) for i in range(n)],axis = 1) for k in range(K)])
        #              })
        #         df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
        #         df.to_csv(foldername + '/df.csv')
        problem, x, X, s, lmbda, data_train_pm, w_pm, eps_pm, p_pm, c_pm, C_pm, gam, tau = prob_facility(
            K, m, n)
        data_train_pm.value = d_train
        w_pm.value = wk
        p_pm.value = p
        c_pm.value = c
        C_pm.value = C

        # solve for various epsilons
        for eps_count, eps in enumerate(np.flip(eps_nums)):
            eps_pm.value = eps
            problem.solve(solver=cp.MOSEK, mosek_params={
                          mosek.dparam.optimizer_max_time:  1500.0}, verbose=True)
            evalvalue = evaluate(p_pm, x, X, tau, dat_eval)
            evalvalue1 = evaluate_k(p_pm, x, X, dat_eval)
            newrow = pd.Series(
                {"R": r,
                 "K": K,
                 "Epsilon": eps,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "Eval_val1": evalvalue1,
                 "solvetime": problem.solver_stats.solve_time,
                 "bound": np.mean([np.max([(d_train[k] - Data[:, :, r][kmeans.labels_ == k])@np.minimum(-5*X[i].value + gam.value[i, (k*2*m):((k+1)*2*m)]@C_r,0) for i in range(n)],axis = 1) for k in range(K)])
                 })
            df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
            # df.to_csv(foldername + '/df.csv')
    return X_sols, x_sols, df


if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="facility/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    # different cluster values we consider
    K_nums = np.array([1, 2, 3, 4, 5, 10, 25, 50])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 50
    M = 10
    R = 10      # Total times we repeat experiment in a run
    n = 5  # number of facilities
    m = 25  # number of locations
    eps_min = 0.05      # minimum epsilon we consider
    eps_max = 2         # maximum epsilon we consider

    eps_nums = np.concatenate([np.logspace(-3,-0.8,25), np.linspace(0.16,0.5,20), np.linspace(0.51, 0.8, 5)])
    eps_tot = M
    c, C, p = generate_facility_data(n, m)
    Data = generate_facility_demands(N_tot, m, R)
    Data_eval = generate_facility_demands(N_tot, m, R)
    C_r = np.vstack([-np.eye(m), np.eye(m)])
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
