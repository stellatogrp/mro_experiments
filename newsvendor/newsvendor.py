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


def prob_news(K, n):
    """Create the problem in cvxpy
    Parameters
    ----------
    K: int
        Number of data samples
    n: int
        Number of products
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    d_train = cp.Parameter((K, 2))
    x = cp.Variable(2)
    s = cp.Variable(K)
    buy_p = cp.Parameter(2)
    sell_p = cp.Parameter(2)
    wk = cp.Parameter(K)
    lam = cp.Variable()
    eps = cp.Parameter()
    C_r = np.vstack([-np.eye(2), np.eye(2)])
    d_r = np.hstack([np.zeros(2), np.ones(2)*40])
    gam = cp.Variable((4, K*4))

    objective = cp.Minimize(buy_p@x + eps*lam + s@wk)
    constraints = []
    # formulate constraints
    
    for k in range(K):
        constraints += [ -sell_p@x + gam[0, (k*4):((k+1)*4)]@(d_r - C_r@d_train[k]) <= s[k]]
        constraints += [-sell_p[0]*x[0] - d_train[k]@(sell_p[1]*np.array([0,1])) + gam[1, (k*4):((k+1)*4)]@(d_r - C_r@d_train[k])<= s[k]]
        constraints += [ -d_train[k]@(sell_p[0]*np.array([1,0])) -sell_p[1]*x[1] + gam[2, (k*4):((k+1)*4)]@(d_r - C_r@d_train[k])<= s[k]]
        constraints += [-d_train[k]@(sell_p) + gam[3, (k*4):((k+1)*4)]@(d_r - C_r@d_train[k])<= s[k]]
        constraints += [cp.norm(C_r.T@gam[0, (k*4):((k+1)*4)]) <= lam]
        constraints += [cp.norm(C_r.T@gam[1, (k*4):((k+1)*4)] + sell_p[1]*np.array([0,1]),2)<= lam]
        constraints += [cp.norm(C_r.T@gam[2, (k*4):((k+1)*4)] + sell_p[0]*np.array([1,0]),2)<= lam]
        constraints += [cp.norm(C_r.T@gam[3, (k*4):((k+1)*4)] + sell_p,2)<= lam]
    
    constraints += [x >= 0, gam >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, s, lam, d_train, wk, eps, gam, buy_p, sell_p


def prob_news_max(K, n):
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
    d_train = cp.Parameter((K, 2))
    x = cp.Variable(2)
    s = cp.Variable(K)
    buy_p = cp.Parameter(2)
    sell_p = cp.Parameter(2)
    wk = cp.Parameter(K)
    lam = cp.Variable()
    eps = cp.Parameter()

    objective = cp.Minimize(buy_p@x + eps*lam + s@wk)
    constraints = []
    # formulate constraints
    
    for k in range(K):
        constraints += [ -sell_p@x  <= s[k]]
        constraints += [-sell_p[0]*x[0] - d_train[k]@(sell_p[1]*np.array([0,1]))<= s[k]]
        constraints += [ -d_train[k]@(sell_p[0]*np.array([1,0])) -sell_p[1]*x[1] <= s[k]]
        constraints += [-d_train[k]@(sell_p)<= s[k]]

    constraints += [cp.norm(sell_p[1]*np.array([0,1]),2)<= lam]
    constraints += [cp.norm(sell_p[0]*np.array([1,0]),2)<= lam]
    constraints += [cp.norm(sell_p,2)<= lam]
    
    constraints += [x >= 0]

    problem = cp.Problem(objective, constraints)

    return problem, x, s, lam, d_train, wk, eps, buy_p, sell_p



def evaluate(buy_p, sell_p,x, d):
    """Evaluate constraint satisfaction
    Parameters
    ----------
    buy_p: vector
        Buying Prices
    sell_p: vector
        Selling Prices
    x: vector
        Decision variables of the optimization problem
    d: matrix
        Validation data matrix
    Returns:
    -------
    float: out of sample expected value
    """
    totval = 0
    totval = np.maximum(-sell_p[0]*x.value[0],-sell_p[0]*d[:,0]) +  np.maximum(-sell_p[1]*x.value[1],-sell_p[1]*d[:,1])
    return buy_p@x.value + np.mean(totval)

def gen_demand(N,R, seed):
    np.random.seed(seed)
    sig = np.array([[0.3,-0.1],[-0.1,0.2]])
    mu = np.array((3,2.8))
    norms = np.random.multivariate_normal(mu,sig, (N,R))
    d_train = np.exp(norms)
    d_train = np.minimum(d_train,40)
    return d_train

def news_experiment(r, n, Data, Data_eval, prob_news,
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
    # x_sols = np.zeros((K_tot, eps_tot, n))
    x_sols = 0
    df = pd.DataFrame(columns=["R", "K", "Epsilon", "Opt_val", "Eval_val",
                              "solvetime", ])

    # solve for various K
    for K_count, K in enumerate(np.flip(K_nums)):
        d_train, wk, kmeans = cluster_data(Data[:, r, :], K)
        dat_eval = Data_eval[:, r, :]

        if K == N_tot:
            problem, x, s, lam, data_train_pm, w_pm, eps_pm, buy_p, sell_p = \
                prob_news_max(K, n)
            data_train_pm.value = d_train
            w_pm.value = wk
            buy_p.value = buy_pval
            sell_p.value = sell_pval

            # solve for various epsilons
            for eps_count, eps in enumerate(np.flip(eps_nums)):
                eps_pm.value = eps
                problem.solve(solver=cp.MOSEK, mosek_params={
                    mosek.dparam.optimizer_max_time:  1500.0})
                print(problem.objective.value)
                evalvalue = evaluate(buy_pval, sell_pval, x, dat_eval)
                print(evalvalue)
                newrow = pd.Series(
                    {"R": r,
                     "K": 0,
                     "Epsilon": eps,
                     "Opt_val": problem.objective.value,
                     "Eval_val": evalvalue,
                     "satisfy" : float(evalvalue <= problem.objective.value),
                     "solvetime": problem.solver_stats.solve_time,
                     "bound": 0
                     })
                df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
        problem, x, s, lam, data_train_pm, w_pm, eps_pm, gam, buy_p, sell_p= prob_news(K,n)
        data_train_pm.value = d_train
        w_pm.value = wk
        buy_p.value = buy_pval
        sell_p.value = sell_pval

        # solve for various epsilons
        for eps_count, eps in enumerate(np.flip(eps_nums)):
            eps_pm.value = eps
            problem.solve(solver=cp.MOSEK, mosek_params={
                          mosek.dparam.optimizer_max_time:  1500.0}, verbose=True)
            evalvalue = evaluate(buy_pval, sell_pval, x, dat_eval)
            newrow = pd.Series(
                {"R": r,
                 "K": K,
                 "Epsilon": eps,
                 "Opt_val": problem.objective.value,
                 "Eval_val": evalvalue,
                 "satisfy" : float(evalvalue <= problem.objective.value),
                 "solvetime": problem.solver_stats.solve_time,
                 "bound": np.mean([np.max([(d_train[k] - Data[:, r, :][kmeans.labels_ == k])@(-zvals[i] -gam.value[i, (k*4):((k+1)*4)].T@C_r) for i in range(4)],axis = 1) for k in range(K)])
                 })
            df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
    return x_sols, df


if __name__ == '__main__':
    print("START")
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="newsvendor/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    # different cluster values we consider
    K_nums = np.array([1, 3, 5, 10, 25, 50, 100])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 100
    eps_tot = 30
    M = 10
    R = 30     # Total times we repeat experiment to estimate final probabilty
    n = 2  # number of products
    eps_nums = np.concatenate([np.logspace(-4,-0.5,30), np.linspace(0.32,2,30)])
    # eps_nums = np.linspace(0.01,3,30)
    # eps_nums = np.concatenate([np.linspace(0.01,0.25,20), np.linspace(0.3,1.2,20), np.linspace(1.25,2,15)])

    buy_pval = np.array([4.,5.])
    sell_pval = np.array([5,6.5])
    C_r = np.vstack([-np.eye(2), np.eye(2)])
    zvals = {}
    zvals[0] = 0
    zvals[1] = sell_pval[1]*np.array([0,1])
    zvals[2] = sell_pval[0]*np.array([1,0])
    zvals[3] = sell_pval

    Data = gen_demand(N_tot, R, 2)
    Data_eval = gen_demand(N_tot, R,3)

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(
        delayed(news_experiment)(r, n, Data, Data_eval,
                                     prob_news, N_tot, K_tot, K_nums,
                                     eps_tot, eps_nums, foldername) for r in range(R))

    dftemp = results[0][1]

    for r in range(1, R):
        dftemp = dftemp.add(results[r][1].reset_index(), fill_value=0)
    dftemp = dftemp/R

    dftemp.to_csv(foldername + '/df.csv')
