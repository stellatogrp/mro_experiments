import pandas as pd
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
output_stream = sys.stdout
import gurobipy as gp
from gurobipy import GRB
import time
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive","tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

synthetic_returns = pd.read_csv('sp500_synthetic_returns.csv').to_numpy()[:,1:]


def cluster_data(D_in, K):
    '''returns K cluster means after clustering D_in into K clusters'''
    N = D_in.shape[0]
    kmeans = KMeans(n_clusters=K).fit(D_in)
    Dbar_in = kmeans.cluster_centers_
    weights = np.bincount(kmeans.labels_) / N
    
    return Dbar_in, weights

def createproblem_portMIP(N, m):
    """Creates the problem in cvxpy"""
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m,boolean = True)
    # OBJECTIVE #
    objective = cp.multiply(eps, lam) + w@s

    # CONSTRAINTS #
    constraints = []
    constraints += [-dat@x + cp.quad_over_lin(x, 4*lam) <= s]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    #for k in range(2):
    #    constraints += [cp.sum(x[k*np.ceil(m/2):(k+1)*np.ceil(m/2)]) <= 0.50]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z)<=10]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, lam, dat, eps, w

def port_experiment(dat, dateval, R, m, prob, N_tot, K_tot,K_nums, eps_tot, eps_nums):
    x_sols = np.zeros((K_tot, eps_tot, m, R))
    Opt_vals = np.zeros((K_tot,eps_tot, R))
    eval_vals = np.zeros((K_tot,eps_tot, R))
    probs = np.zeros((K_tot,eps_tot, R))
    setuptimes = np.zeros((K_tot,R))
    solvetimes = np.zeros((K_tot,eps_tot,R))
    Data = dat
    Data_eval = dateval

    ######################## Repeat experiment R times ########################
    for r in range(R):
#         output_stream.write('Percent Complete %.2f%s\r' % ((r)/R*100,'%'))
#         output_stream.flush()
        
        ######################## solve for various K ########################
        for K_count, K in enumerate(K_nums):
            
           #output_stream.write('Percent Complete %.2f%s\r' % ((K_count)/K_tot*100,'%'))
           # output_stream.flush()
            
            #print(K_count)
            tnow = time.time()
            d_train, wk = cluster_data(Data[(N_tot*r):(N_tot*(r+1))], K)
            d_eval = Data_eval[(N_tot*r):(N_tot*(r+1))]
            assert(d_train.shape == (K,m))
            problem, x, s, lmbda, data_train_pm,eps_pm, w_pm = prob(K,m)
            data_train_pm.value = d_train
            w_pm.value = wk
            setuptimes[K_count,r] = time.time() - tnow

            ######################## solve for various epsilons ########################
            for eps_count, eps in enumerate(eps_nums):
                eps_pm.value = eps
                problem.solve()
                solvetimes[K_count,eps_count,r] = problem.solver_stats.solve_time
                #print(eps,K, problem.objective.value)
                x_sols[K_count, eps_count, :, r] = x.value
                evalvalue = np.mean(Data_eval@x.value)
                eval_vals[K_count, eps_count, r] = -evalvalue
                probs[K_count, eps_count, r] = -evalvalue <= problem.objective.value 
                Opt_vals[K_count,eps_count,r] = problem.objective.value

    #output_stream.write('Percent Complete %.2f%s\r' % (100,'%'))  
    
    return x_sols, Opt_vals, eval_vals, probs,setuptimes,solvetimes

    
K_nums = np.array([1,50,100,500,1000])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 1000
M = 20
R = 10           # Total times we repeat experiment to estimate final probabilty
m = 200 
eps_min = -6    # minimum epsilon we consider
eps_max = -3.5        # maximum epsilon we consider
eps_nums = np.linspace(eps_min,eps_max,M)
eps_nums = 10**(eps_nums)
eps_tot = M

dat = synthetic_returns[:10000,:m]
dateval = synthetic_returns[-10000:,:m]

x_sols, Opt_vals, eval_vals, probs,setuptimes,solvetimes = port_experiment(dat,dateval,R, m, createproblem_portMIP,N_tot, K_tot,K_nums, eps_tot,eps_nums)

