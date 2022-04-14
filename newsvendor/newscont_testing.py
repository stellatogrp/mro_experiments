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



def cluster_data(D_in, K):
    '''returns K cluster means after clustering D_in into K clusters'''
    N = D_in.shape[0]
    kmeans = KMeans(n_clusters=K).fit(D_in)
    Dbar_in = kmeans.cluster_centers_
    weights = np.bincount(kmeans.labels_) / N
    
    return Dbar_in, weights

def createproblem_news(N, m):
    """Creates the problem in cvxpy"""
    # m = 10 
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    p = cp.Parameter(m)
    a = cp.Parameter(m)
    b = cp.Parameter(m)

    # VARIABLES #
    # weights, s_i, lambda, tau
    q = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    t = cp.Variable()
    y = cp.Variable(m)
    # OBJECTIVE #
    objective = t + a@q + 0.5*a@y

    # CONSTRAINTS #
    constraints = [cp.multiply(eps, lam) + w@s <= 0]
    constraints += [cp.hstack([-t]*N) + dat@(-p) +
                    cp.hstack([cp.quad_over_lin(p, 4*lam)]*N) <= s]
    constraints += [-p@q <= t, q - b <= y, 0 <= y, a@q + 0.5*a@y <= 20, q >= 0, q<= 5*b]
    constraints += [lam >= 0]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, q, p,a,b,t, lam, dat, eps, w

def generate_news_params(m = 10):
    '''data for one problem instance of facility problem'''
    # Cost for facility
    a = np.random.uniform(0.2,0.9,m)
    b = np.random.uniform(0.1,0.7,m)
    F = np.random.normal(size = (m,5))
    sig = 0.1*F@(F.T)
    mu = np.random.uniform(-0.2,0,m)
    norms = np.random.multivariate_normal(mu,sig)
    p = np.exp(norms)
    return a,b,p, mu, sig
    
def generate_news_demands( mu,sig,N_tot, m = 10, R_samples = 30):
    norms = np.random.multivariate_normal(mu,sig,(R_samples,N_tot))
    #norms = np.random.normal(np.random.uniform(-0.2,0),0.2,(R_samples,N_tot,m))
    d_train = np.exp(norms)
    return d_train

def news_experiment(dat, dateval, R, m, a,b,p, prob, N_tot, K_tot,K_nums, eps_tot, eps_nums):
    q_sols = np.zeros((K_tot, eps_tot, m, R))
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
            
            output_stream.write('Percent Complete %.2f%s\r' % ((K_count)/K_tot*100,'%'))
            output_stream.flush()
            
            tnow = time.time()
            d_train, wk = cluster_data(Data[r], K)
            evaldat = Data_eval[r] 
            assert(d_train.shape == (K,m))
            problem, q, p_pm,a_pm,b_pm,t, lam_pm, dat_pm, eps_pm, w_pm = prob(K,m)
            setuptimes[K_count,r] = time.time() - tnow
            a_pm.value = np.array(a)
            b_pm.value = np.array(b)
            p_pm.value = np.array(p)
            dat_pm.value = d_train
            w_pm.value = wk

            ######################## solve for various epsilons ########################
            for eps_count, eps in enumerate(eps_nums):
                tnow1 = time.time()
                eps_pm.value = eps
                problem.solve()
                solvetimes[K_count,eps_count,r] = time.time() - tnow1
                #print(eps,K, problem.objective.value)
                q_sols[K_count, eps_count, :, r] = q.value
                evalvalue = np.mean(-Data_eval@p_pm.value) - t.value <= 0
                eval_vals[K_count, eps_count, r] = evalvalue
                probs[K_count, eps_count, r] = evalvalue
                Opt_vals[K_count,eps_count,r] = problem.objective.value


    #output_stream.write('Percent Complete %.2f%s\r' % (100,'%'))  
    
    return q_sols, Opt_vals, eval_vals, probs,setuptimes,solvetimes

    
K_nums = np.array([1,10,50,100,1000,5000])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 5000
M = 15
R = 20
m = 200
eps_min = -5    # minimum epsilon we consider
eps_max = 0        # maximum epsilon we consider
eps_nums = np.linspace(eps_min,eps_max,M)
eps_nums = 10**(eps_nums)
eps_tot = M
a,b,p,mu,sig = generate_news_params(m)

dat = generate_news_demands(mu,sig,N_tot, m, R)
dateval = generate_news_demands(mu,sig,N_tot, m, R)
q_sols, Opt_vals, Opt_vals1,Opt_vals2, eval_vals, probs,setuptimes,solvetimes = news_experiment(dat, dateval, R, m,a,b,p,createproblem_news, N_tot, K_tot,K_nums, eps_tot, eps_nums)

