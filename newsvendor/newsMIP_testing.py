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
from pathlib import Path  
import mosek
import os
from joblib import Parallel, delayed

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
    '''returns K cluster means after clustering D_in into K clusters'''
    N = D_in.shape[0]
    kmeans = KMeans(n_clusters=K).fit(D_in)
    Dbar_in = kmeans.cluster_centers_
    weights = np.bincount(kmeans.labels_) / N
    
    return Dbar_in, weights

def createproblem_news1(N, m):
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
    constraints += [-p@q <= t, q - b <= y, 0 <= y, a@q + 0.5*a@y <= 40, q >= 0, q<= 15*b]
    constraints += [lam >= 0]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, q, p,a,b,t, lam, dat, eps, w


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
    a_1 = 50
    b_1 = -40
    #a_1 = 1
    #b_1 = 0

    # VARIABLES #
    # weights, s_i, lambda, tau
    q = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    t = cp.Variable()
    tao = cp.Variable()
    y = cp.Variable(m)
    z = cp.Variable(m,boolean = True)
    # OBJECTIVE #
    objective = t 

    # CONSTRAINTS #
    constraints = [cp.multiply(eps, lam) + w@s <= t]
    constraints += [cp.hstack([a_1*(a@q + 0.5*a@y) + b_1*tao]*N)+ a_1*dat@(-p) +
                    cp.hstack([cp.quad_over_lin(a_1*p, 4*lam)]*N) <= s]
    constraints += [a_1*(-p@q + a@q + 0.5*a@y) + b_1*tao <= t]
    constraints += [10*tao <= t]
    constraints += [q - b <= y, 0 <= y, a@q + 0.5*a@y <= 20, q >= 0, q<= 5*b]
    constraints += [lam >= 0]
    constraints += [q - 10*z <= 0, cp.sum(z)<=10]

    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, q, y, tao, z, p,a,b,t, lam, dat, eps, w

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

def news_experiment(dat, dateval, r, m, a,b,p, prob, N_tot, K_tot,K_nums, eps_tot, eps_nums, foldername):
    q_sols = np.zeros((K_tot, eps_tot, m, R))
    Opt_vals = np.zeros((K_tot,eps_tot, R))
    probs = np.zeros((K_tot,eps_tot, R))
    setuptimes = np.zeros((K_tot,R))
    solvetimes = np.zeros((K_tot,eps_tot,R))
    clustertimes = np.zeros((K_tot,R))
    Data = dat
    Data_eval = dateval

    ######################## Repeat experiment R times ########################
        ######################## solve for various K ########################
    for K_count, K in enumerate(K_nums):
        
        #output_stream.write('Percent Complete %.2f%s\r' % ((K_count)/K_tot*100,'%'))
        #output_stream.flush()
        if K == N_tot:
            d_train, wk = cluster_data(Data[r], K)
            clustertimes[K_count,r] = 0
        else:
            tnow = time.time()
            d_train, wk = cluster_data(Data[r], K)
            clustertimes[K_count,r] = time.time() - tnow

        evaldat = Data_eval[r] 
        tnow = time.time()
        problem, q, y, tao,z, p_pm,a_pm,b_pm,t, lam_pm, dat_pm, eps_pm, w_pm = prob(K,m)
        a_pm.value = np.array(a)
        b_pm.value = np.array(b)
        p_pm.value = np.array(p)
        dat_pm.value = d_train
        w_pm.value = wk
        setuptimes[K_count,r] = time.time() - tnow

        ######################## solve for various epsilons ########################
        for eps_count, eps in enumerate(eps_nums):
            eps_pm.value = eps
            problem.solve(ignore_dpp = True,solver = cp.MOSEK, verbose = True, mosek_params = {mosek.dparam.optimizer_max_time:  1000.0})
            solvetimes[K_count,eps_count,r] = problem.solver_stats.solve_time
            q_sols[K_count, eps_count, :, r] = q.value
            evalvalue = -50*np.mean(evaldat@p_pm.value) + 50*(a@q.value + 0.5*a@y.value) -40*tao.value - t.value <= 0
            #evalvalue = -np.mean(evaldat@p_pm.value) + (a@q.value + 0.5*a@y.value) - t.value <= 0
            #evalvalue = -np.mean(evaldat@p_pm.value) <= t.value
            probs[K_count, eps_count, r] = evalvalue
            Opt_vals[K_count,eps_count,r] = problem.objective.value
            print(r, eps,K, problem.solver_stats.solve_time, problem.objective.value, evalvalue, np.sum(z.value) )
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/q"+str(r)+".npy"),q_sols)
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/Opt_vals"+str(r)+".npy"),Opt_vals)
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/solvetimes"+str(r)+".npy"),solvetimes)
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/setuptimes"+str(r)+".npy"),setuptimes)
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/clustertimes"+str(r)+".npy"),clustertimes)
            #np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/probs"+str(r)+".npy"),probs)


    plt.figure(figsize=(10, 6))
    for K_count, K in enumerate(K_nums):
        plt.plot(eps_nums, probs[:,:,r][K_count,:],linestyle='-', marker='o', color = colors[K_count], label = "$K = {}$".format(round(K,4)))
        plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.ylabel("Reliability")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/reliability'+str(r)+'.png')

    plt.figure(figsize=(10, 6))
    for K_count, K in enumerate(K_nums):
        plt.plot(eps_nums, Opt_vals[:,:,r][K_count,:],linestyle='-', marker='o', color = colors[K_count], label = "$K = {}$".format(round(K,4)))
        plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.ylabel("Optimal value")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/objs'+str(r)+'.png')

    plt.figure(figsize=(10, 6))

    for eps_count, eps in enumerate(eps_nums):
        plt.plot(K_nums,solvetimes[:,:,r][:,eps_count],linestyle='-', marker='o', label = "$\epsilon^2 = {}$".format(round(eps,6)), alpha = 0.5)
        plt.xlabel("Number of clusters (K)")

    plt.ylabel("time")
    plt.title("Solve time")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/solvetime'+str(r)+'.png')


    plt.figure(figsize=(10, 6))
    plt.plot(K_nums,  clustertimes[:,r] + setuptimes[:,r],linestyle='-', marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("time")
    plt.title("Set-up time (clustering + creating problem)")
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/setuptime'+str(r)+'.png')

    plt.figure(figsize=(10, 6))
    for eps_count, eps in enumerate(eps_nums):
        plt.plot(K_nums,clustertimes[:,r] + setuptimes[:,r] + solvetimes[:,:,r][:,eps_count],linestyle='-', marker='o', label = "$\epsilon^2 = {}$".format(round(eps,6)), alpha = 0.5)
        plt.xlabel("Number of clusters (K)")

    plt.ylabel("time")
    plt.title("totaltime")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/totaltime'+str(r)+'.png')


    #output_stream.write('Percent Complete %.2f%s\r' % (100,'%'))  
    
    return q_sols, Opt_vals, probs,setuptimes,solvetimes, clustertimes

if __name__ == '__main__':
    foldername = "newsvendor/MIP/m300_K1000_r10"
    K_nums = np.array([1,10,50,100,300,500,1000])
    K_tot = K_nums.size  # Total number of clusters we consider
    N_tot = 1000
    M = 10
    R = 10
    m = 300
    eps_min = -6    # minimum epsilon we consider
    eps_max = 0        # maximum epsilon we consider
    eps_nums = np.linspace(eps_min,eps_max,M)
    eps_nums = 10**(eps_nums)
    eps_tot = M
    a,b,p,mu,sig = generate_news_params(m)

    dat = generate_news_demands(mu,sig,N_tot, m, R)
    dateval = generate_news_demands(mu,sig,N_tot, m, R)
    njobs = get_n_processes(30)

    results = Parallel(n_jobs=njobs)(delayed(news_experiment)(dat, dateval,r, m, a,b,p,createproblem_news, N_tot, K_tot,K_nums, eps_tot, eps_nums, foldername) for r in range(R))

    q_sols = np.zeros((K_tot, eps_tot, m, R))
    Opt_vals = np.zeros((K_tot,eps_tot, R))
    probs = np.zeros((K_tot,eps_tot, R))
    setuptimes = np.zeros((K_tot,R))
    solvetimes = np.zeros((K_tot,eps_tot,R))
    clustertimes = np.zeros((K_tot,R))

    
    for r in range(R):
        q_sols += results[r][0]
        Opt_vals += results[r][1]
        probs += results[r][2]
        setuptimes += results[r][3]
        solvetimes += results[r][4]
        clustertimes +=  results[r][5]

    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/q_sols.npy"),q_sols)
    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/Opt_vals.npy"),Opt_vals)
    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/solvetimes.npy"),solvetimes)
    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/clustertimes.npy"),clustertimes)
    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/setuptimes.npy"),setuptimes)
    np.save(Path("/scratch/gpfs/iywang/mro_results/" + foldername + "/probs.npy"),probs)




    plt.figure(figsize=(10, 6))
    for K_count, K in enumerate(K_nums):
        plt.plot(eps_nums, np.mean(Opt_vals[:,:,],axis = 2)[K_count,:],linestyle='-', marker='o', color = colors[K_count], label = "$K = {}$".format(round(K,4)))
        plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.ylabel("Optimal value")
    plt.legend()
    plt.show()
    plt.savefig("/scratch/gpfs/iywang/mro_results/" + foldername + "/objs.png")

    plt.figure(figsize=(10, 6))
    for K_count, K in enumerate(K_nums):
        plt.plot(eps_nums, np.mean(probs[:,:,],axis = 2)[K_count,:],linestyle='-', marker='o', color = colors[K_count], label = "$K = {}$".format(round(K,4)))
        plt.xlabel("$\epsilon^2$")
    plt.xscale("log")
    plt.ylabel("Reliability")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/reliability.png')

    plt.figure(figsize=(10, 6))
    for eps_count, eps in enumerate(eps_nums):
        plt.plot(K_nums,np.mean(solvetimes[:,:,],axis = 2)[:,eps_count],linestyle='-', marker='o', label = "$\epsilon^2 = {}$".format(round(eps,6)), alpha = 0.5)
        plt.xlabel("Number of clusters (K)")

    plt.ylabel("time")
    plt.title("Solve time")
    plt.legend()
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/solvetime.png')

    plt.figure(figsize=(10, 6))
    plt.plot(K_nums,np.mean(clustertimes+ setuptimes,axis = 1),linestyle='-', marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("time")
    plt.title("Set-up time (clustering + creating problem)")
    plt.show()

    plt.figure(figsize=(10, 6))
    for eps_count, eps in enumerate(eps_nums):
        plt.plot(K_nums,np.mean(clustertimes+ setuptimes,axis = 1) + np.mean(solvetimes[:,:,],axis = 2)[:,eps_count],linestyle='-', marker='o', label = "$\epsilon^2 = {}$".format(round(eps,6)), alpha = 0.5)
        plt.xlabel("Number of clusters (K)")

    plt.ylabel("time")
    plt.title("Total time")
    plt.legend(fontsize=5)
    plt.show()
    plt.savefig('/scratch/gpfs/iywang/mro_results/' + foldername + '/totaltime.png')

    print("COMPLETE")

