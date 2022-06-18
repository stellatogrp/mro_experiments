from sklearn.cluster import KMeans
from pathlib import Path
from joblib import Parallel, delayed
import os
import mosek
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})


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

def lognormal_scaled(N, m,scale):
    """Creates return data, N = sample size, m = data length per sample"""
    R = np.vstack([np.random.normal(
        i*0.012*scale, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
    return (np.exp(R.transpose()))

def data_modes_log(N,m,scales):
    modes = len(scales)
    d = np.ones((N+100,m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,:] = lognormal_scaled(weights,m,scales[i])
    return d[0:N,:]

def createproblem_max(N, m,w):
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    expx = cp.Parameter(m)
    eps = cp.Parameter()
    
    u = cp.Variable((N,m))
    
    objective = cp.sum([w[k]*cp.log(u[k]@expx) for k in range(N)])
    # CONSTRAINTS #
    constraints = [cp.sum([cp.quad_over_lin(u[k]-dat[k], 1/w[k]) for k in range(N)])<= eps]

    # PROBLEM #
    problem = cp.Problem(cp.Maximize(objective), constraints)
    return problem, u,expx,dat,eps


def createproblem_min(N, m,w,Uvals,n_planes):
    # PARAMETERS #    
    x = cp.Variable(m)
    t = cp.Variable()
    objective = t
    
    # CONSTRAINTS #
    constraints = [cp.sum(x)== 1, x>= 0, x <= 1]
    for index in range(n_planes):
        constraints += [cp.sum([w[k]*cp.log_sum_exp(x + Uvals[index][k]) for k in range(N)]) <= t]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x
  

def minmaxsolve(N,m,w,data,epsilon):
    Uvals = {}
    inds = 0
    Uvals[inds] = np.log(data)
    inds += 1
    solvetime = 0
    problem1, x= createproblem_min(N, m,w,Uvals,inds)
    problem1.solve()
    objs1 = problem1.objective.value
    solvetime += problem1.solver_stats.solve_time
    problem, u,expx,dat,eps= createproblem_max(N, m,w)
    dat.value = data
    eps.value = epsilon
    expx.value = np.exp(x.value)
    problem.solve()
    solvetime += problem.solver_stats.solve_time
    Uvals[inds] = np.log(u.value)
    inds += 1
    problem1, x= createproblem_min(N, m,w,Uvals,inds)
    problem1.solve()
    solvetime += problem1.solver_stats.solve_time
    objs2 = problem1.objective.value
    iters= 1
    while(np.abs(objs1 - objs2)>= 0.0001 and iters <= 50):
        expx.value = np.exp(x.value)
        problem.solve()
        solvetime += problem.solver_stats.solve_time
        Uvals[inds] = np.log(u.value)
        inds += 1
        problem1, x= createproblem_min(N, m,w,Uvals,inds)
        problem1.solve()
        solvetime += problem1.solver_stats.solve_time
        objs1 = objs2
        objs2 = problem1.objective.value
        iters += 1
    return objs2, x.value, u.value, solvetime, iters
    
def logsumexp_experiment(r, m, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername):
    df = pd.DataFrame(columns = ["r","K","Epsilon","Opt_val","Eval_val","satisfy","solvetime","bound","iters"])
    d = data_modes_log(N_tot,m,[1,3,6])
    d2 = data_modes_log(N_tot,m,[1,3,6])
    for Kcount, K in enumerate(K_nums):
        kmeans = KMeans(n_clusters=K).fit(d)
        weights = np.bincount(kmeans.labels_) / N_tot
        for epscount, epsval in enumerate(eps_nums):
            objs_val,x_val,u_val,time,iters = minmaxsolve(K,m,weights,kmeans.cluster_centers_,epsval**2)
            evalvalue = cp.sum([(1/N_tot)*cp.log_sum_exp(x_val + np.log(d2[k])).value for k in range(N_tot)])
            newrow = pd.Series(
                {"r":r,
                "K": K,
                "Epsilon": epsval,
                "Opt_val": objs_val,
                "Eval_val": evalvalue,
                "satisfy": evalvalue <= objs_val,
                "solvetime": time,
                "bound": (1/(2*N_tot))*kmeans.inertia_,
                "iters": iters
            })
            df = df.append(newrow,ignore_index = True)

    return df


if __name__ == '__main__':
    foldername = "logsumexp/m30_K90_r50"
    N_tot = 90
    m = 30
    R = 60
    K_nums = np.append([1,2,3,4,5,10],np.append(np.arange(20, int(N_tot/2)+1,10), N_tot))
    K_tot = K_nums.size 
    eps_nums = 10**np.array([-3 , -2.79, -2.58, -2.37, -2.17,
       -1.96, -1.75, -1.55 , -1.34, -1.13, -0.92, -0.72, -0.51, -0.30, -0.1, 0])
    eps_tot = eps_nums.size

    njobs = get_n_processes(30)
    results = Parallel(n_jobs=njobs)(delayed(logsumexp_experiment)(
        r, m, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername) for r in range(R))
    
    dftemp = results[0]
    for r in range(1, R):
        dftemp = dftemp.add(results[r].reset_index(), fill_value=0)
    dftemp = dftemp/R

    dftemp.to_csv('/scratch/gpfs/iywang/mro_results/' + foldername + '/df.csv')