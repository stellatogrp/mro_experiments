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
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from sklearn.metrics import mean_squared_error
import math

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


def createproblem_portMIP(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m, boolean=True)
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w
    

def create_scenario(dat,m,num_dat):
    tau = cp.Variable()
    x = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    objective = cp.sum(tau + 5*cp.maximum(-dat@x - tau,0))/num_dat
    constraints = []
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [x - z <= 0, cp.sum(z) <= 5]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, tau
    



def port_experiment(dat, dateval, r, m, prob, N_tot, K_tot, K_nums, eps_tot, eps_nums, foldername):
    """Run the experiment for multiple K and epsilon
    Parameters
    ----------
    Various inputs for combinations of experiments
    Returns
    -------
    x_sols: array
        The optimal solutions
    df: dataframe
        The results of the experiments
    """
    x_sols = np.zeros((K_tot, eps_tot, m, R))
    df = pd.DataFrame(columns=["R", "K", "Epsilon", "Opt_val", "Eval_val",
                               "satisfy", "solvetime", "bound"])
    Data = dat
    Data_eval = dateval

    for K_count, K in enumerate(K_nums):
        d_eval = Data_eval[(N_tot*r):(N_tot*(r+1))]
        kmeans = KMeans(n_clusters=K).fit(Data[(N_tot*r):(N_tot*(r+1))])
        d_train = kmeans.cluster_centers_
        wk = np.bincount(kmeans.labels_) / N_tot
        assert (d_train.shape == (K, m))
        problem, x, s, tau, lmbda, data_train_pm, eps_pm, w_pm = prob(K, m)
        data_train_pm.value = d_train
        for eps_count, eps in enumerate(eps_nums):
            w_pm.value = wk
            eps_pm.value = eps
            problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=True, mosek_params={
                mosek.dparam.optimizer_max_time:  2000.0})
            x_sols[K_count, eps_count, :, r] = x.value
            evalvalue = np.mean(
                np.maximum(-5*d_eval@x.value - 4*tau.value, tau.value)) <= problem.objective.value
            bound = np.max([np.max((d_train[k] - Data[(N_tot*r):(N_tot*(r+1))]
                           [kmeans.labels_ == k])@x.value) for k in range(K)])
            newrow = pd.Series(
                {"R": r,
                    "K": K,
                    "Epsilon": eps,
                    "Opt_val": problem.objective.value,
                    "Eval_val": np.mean(np.maximum(-5*d_eval@x.value - 4*tau.value, tau.value)),
                    "satisfy": evalvalue,
                    "solvetime": problem.solver_stats.solve_time,
                    "bound": bound
                 })
            df = df.append(newrow, ignore_index=True)
            # df.to_csv(foldername + '/df11_'+str(r)+'.csv')
    return x_sols, df

def find_min_pairwise_distance(data):
    distances = distance.cdist(data, data)
    np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to ignore self-distances
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    return min_indices

def online_cluster_init(K,Q,data):
    k_dict = {}
    q_dict = {}
    init_num = data.shape[0]
    q_dict['cur_Q'] = np.minimum(Q,init_num)
    qmeans = KMeans(n_clusters=q_dict['cur_Q']).fit(data)
    q_dict['a'] = qmeans.cluster_centers_
    q_dict['d'] = qmeans.cluster_centers_
    q_dict['w'] = np.bincount(qmeans.labels_) / init_num
    q_dict['rmse'] = np.zeros(q_dict['cur_Q'])
    q_dict['data'] = {}
    for q in range(q_dict['cur_Q']):
        cluster_data = data[qmeans.labels_ == q]
        q_dict['data'][q] = cluster_data
        centroid_array = np.tile(q_dict['d'][q], (len(cluster_data), 1))
        rmse = np.sqrt(mean_squared_error(cluster_data, centroid_array))
        if rmse <= 1e-6:
            rmse = 0.002
        q_dict['rmse'][q] = rmse
    k_dict = {}
    k_dict['a'] = q_dict['a'][:K]
    k_dict['w'] = np.zeros(K)
    k_dict['d'] = np.zeros((K,m))
    k_dict['data'] = {}
    k_dict, time = cluster_k(K,q_dict, k_dict)
    return q_dict, k_dict

def cluster_k(K,q_dict, k_dict):
    start_time = time.time()
    cur_K = np.minimum(K,q_dict['cur_Q'])
    k_dict['K'] = cur_K
    kmeans = KMeans(n_clusters=cur_K, init=k_dict['a']).fit(q_dict['a'])
    k_dict['a'] = kmeans.cluster_centers_
    # k_dict['w'] = np.zeros(cur_K)
    # k_dict['d'] = np.zeros((cur_K,m))
    # k_dict['data'] = {}
    for k in range(cur_K):
        k_dict[k]= np.where(kmeans.labels_ == k)[0]
        d_cur = q_dict['d'][kmeans.labels_ == k]
        w_cur = q_dict['w'][kmeans.labels_ == k]
        k_dict['w'][k] = np.sum(w_cur)
        w_cur_norm = w_cur/(k_dict['w'][k])
        k_dict['d'][k] = np.sum(d_cur*w_cur_norm[:,np.newaxis],axis=0)
    total_time = time.time() - start_time
    for k in range(cur_K):
        k_dict['data'][k] = np.vstack([q_dict['data'][q] for q in k_dict[k]])
    return k_dict, total_time

def online_cluster_update(K,new_dat, q_dict, k_dict,num_dat, t, fix_time):
    new_dat = np.reshape(new_dat,(1,m))
    if t >= fix_time:
        k_dict, total_time = fixed_cluster(k_dict,new_dat,num_dat)
        return q_dict, k_dict, total_time
    start_time = time.time()
    dists = cdist(new_dat,q_dict['a'])
    min_dist = np.min(dists)
    min_ind = np.argmin(dists)
    if min_dist <= 2*q_dict['rmse'][min_ind]:
        q_dict['d'][min_ind] = (q_dict['d'][min_ind]*q_dict['w'][min_ind]*num_dat + new_dat)/(q_dict['w'][min_ind]*num_dat + 1)
        q_dict['rmse'][min_ind] = np.sqrt((q_dict['rmse'][min_ind]**2*q_dict['w'][min_ind]*num_dat + np.linalg.norm(new_dat - q_dict['d'][min_ind],2)**2)/(q_dict['w'][min_ind]*num_dat + 1))
        w_q_temp = q_dict['w']*num_dat/(num_dat+1)
        increased_w = (q_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
        q_dict['w'] = w_q_temp
        q_dict['w'][min_ind] = increased_w
        for k in range(K):
            if min_ind in k_dict[k]:
                k_dict['d'][k] = (k_dict['d'][k]*k_dict['w'][k]*num_dat + new_dat)/(k_dict['w'][k]*num_dat + 1)
                k_dict['w'][k] = (k_dict['w'][k]*num_dat + 1)/(num_dat + 1)
            else:
                k_dict['w'][k] = (k_dict['w'][k]*num_dat)/(num_dat + 1)
        total_time = time.time() - start_time
        q_dict['data'][min_ind] = np.vstack([q_dict['data'][min_ind],new_dat])
        for k in range(K):
            if min_ind in k_dict[k]:
                k_dict['data'][k] = np.vstack([k_dict['data'][k],new_dat])
    else:
        start_time = time.time()
        cur_Q = q_dict['cur_Q'] + 1
        new_a_q = np.zeros((cur_Q,m))
        new_a_q[:cur_Q-1] = q_dict['a']
        new_a_q[-1] = new_dat
        new_d_q = np.zeros((cur_Q,m))
        new_d_q[:cur_Q-1] = q_dict['d']
        new_d_q[-1] = new_dat
        new_rmse = np.zeros(cur_Q)
        new_rmse[:cur_Q-1] = q_dict['rmse']
        new_rmse[-1] = 2*np.min(q_dict['rmse'])
        new_wq = np.zeros(cur_Q)
        new_wq[:cur_Q-1] = (q_dict['w']*num_dat)/(num_dat+1)
        new_wq[-1] = 1/(num_dat+1)
        total_time = time.time() - start_time
        q_dict['data'][cur_Q-1] = new_dat
        if cur_Q > Q:
            start_time = time.time()
            q_dict['cur_Q'] = Q
            min_pair = find_min_pairwise_distance(new_a_q)
            merged_weight = np.sum(new_wq[min_pair[0]]+new_wq[min_pair[1]])
            merged_center = (new_a_q[min_pair[0]]*new_wq[min_pair[0]] + new_a_q[min_pair[1]]*new_wq[min_pair[1]])/merged_weight
            merged_centroid = (new_d_q[min_pair[0]]*new_wq[min_pair[0]] + new_d_q[min_pair[1]]*new_wq[min_pair[1]])/merged_weight
            merged_rmse = np.sqrt((new_rmse[min_pair[0]]**2*new_wq[min_pair[0]] + new_rmse[min_pair[1]]**2*new_wq[min_pair[1]])/merged_weight + (new_wq[min_pair[0]]*np.linalg.norm( new_d_q[min_pair[0]]- merged_centroid)**2 + new_wq[min_pair[1]]*np.linalg.norm(new_d_q[min_pair[1]]- merged_centroid)**2)/(merged_weight ))
            q_dict['a'] = np.zeros((Q,m))
            q_dict['d'] = np.zeros((Q,m))
            q_dict['w'] = np.zeros(Q)
            q_dict['rmse'] = np.zeros(Q)
            q_dict['a'][:Q-1] = np.delete(new_a_q,min_pair,axis=0)
            q_dict['d'][:Q-1] = np.delete(new_d_q,min_pair,axis=0)
            q_dict['w'][:Q-1] = np.delete(new_wq,min_pair)
            q_dict['rmse'][:Q-1]= np.delete(new_rmse,min_pair)
            q_dict['a'][Q-1] = merged_center
            q_dict['d'][Q-1] = merged_centroid
            q_dict['w'][Q-1] = merged_weight
            q_dict['rmse'][Q-1] = merged_rmse
            total_time += time.time() - start_time
            merged_data = np.vstack([q_dict['data'][q] for q in min_pair])
            ind = 0
            for q in range(Q+1):
                if q not in min_pair:
                    q_dict['data'][ind] = q_dict['data'][q]
                    ind += 1
            q_dict['data'][Q-1] = merged_data
        else:
            q_dict['cur_Q'] = cur_Q
            q_dict['a'] = new_a_q
            q_dict['d'] = new_d_q
            q_dict['w'] = new_wq
            q_dict['rmse'] = new_rmse
        k_dict, time_temp = cluster_k(K,q_dict,k_dict)
        total_time += time_temp
    return q_dict, k_dict, total_time

            
def fixed_cluster(k_dict, new_dat,num_dat):
    new_dat = np.reshape(new_dat,(1,m))
    start_time = time.time()
    dists = cdist(new_dat,k_dict['a'])
    min_ind = np.argmin(dists)
    k_dict['d'][min_ind] = (k_dict['d'][min_ind]*k_dict['w'][min_ind]*num_dat + new_dat)/(k_dict['w'][min_ind]*num_dat + 1)
    w_k_temp = k_dict['w']*num_dat/(num_dat+1)
    increased_w = (k_dict['w'][min_ind]*num_dat + 1)/(num_dat+1)
    k_dict['w'] = w_k_temp
    k_dict['w'][min_ind] = increased_w
    total_time = time.time() - start_time
    k_dict['data'][min_ind] = np.vstack([k_dict['data'][min_ind],new_dat])
    return k_dict, total_time


def compute_cumulative_regret(history, T,dateval):
    """
    Compute cumulative regret by comparing online decisions against optimal DRO solution in hindsight.
    At each time t, use the same samples that were available to the online policy.
    
    Args:
        history (dict): History of online decisions and parameters
        dro_params (DROParameters): Problem parameters
        online_samples (np.array): Array of observed samples
        num_eval_samples (int): Number of samples to use for SAA evaluation
        seed (int): Random seed for reproducibility
    """
    def evaluate_expected_cost(d_eval, x, tau):
        return np.mean(
            np.maximum(-5*d_eval@x - 4*tau, tau)) 
    
    regret = np.zeros(T-1)  # Instantaneous regret at each timestep
    MRO_regret = np.zeros(T-1)
    cumulative_regret = np.zeros(T-1)  # Cumulative regret up to each timestep
    MRO_cumulative_regret = np.zeros(T-1) 
    theoretical = np.zeros(T-1)

    eval_values = np.zeros(T)
    MRO_eval_values = np.zeros(T)
    DRO_eval_values = np.zeros(T)
    SA_eval_values = np.zeros(T)
    
    # Generate evaluation samples from true distribution for cost computation
    eval_samples = dateval[init_ind:(init_ind+3000)]
    
    # For each timestep t
    for t in range(T):            
        # Compute instantaneous regret at time t using true distribution
        online_cost = evaluate_expected_cost(eval_samples, history['x'][t],history['tau'][t])
        MRO_cost = evaluate_expected_cost(eval_samples, history['MRO_x'][t],history['MRO_tau'][t])
        optimal_cost = evaluate_expected_cost(eval_samples, history['DRO_x'][t],history['DRO_tau'][t])
        SA_cost = evaluate_expected_cost(eval_samples, history['SA_x'][t],history['SA_tau'][t])
        eval_values[t] = online_cost
        MRO_eval_values[t] = MRO_cost
        DRO_eval_values[t] = optimal_cost
        SA_eval_values[t] = SA_cost
        
    for t in range(T-1):
        regret[t] = history['worst_values'][t] - history['DRO_obj_values'][t+1]
        MRO_regret[t] = history['worst_values_MRO'][t] - history['DRO_obj_values'][t+1]
        
        # Update cumulative regret
        if t == 0:
            cumulative_regret[t] = regret[t]
            MRO_cumulative_regret[t] = MRO_regret[t]
            theoretical[t] = history['sig_val'][t]
        else:
            theoretical[t] = theoretical[t-1] + history['sig_val'][t]
            cumulative_regret[t] = cumulative_regret[t-1] + regret[t]
            MRO_cumulative_regret[t] = MRO_cumulative_regret[t-1] + MRO_regret[t]

    MRO_satisfy = np.array(history['MRO_obj_values'] >= MRO_eval_values).astype(float)
    satisfy = np.array(history['obj_values'] >= eval_values).astype(float)
    DRO_satisfy = np.array(history['DRO_obj_values'] >= DRO_eval_values).astype(float)
    
    return cumulative_regret, regret, eval_values, MRO_eval_values, DRO_eval_values, SA_eval_values, theoretical, MRO_cumulative_regret, MRO_regret, satisfy, MRO_satisfy, DRO_satisfy

def plot_regret_analysis(cumulative_regret, regret, theo, MRO_cumulative_regret, MRO_regret):
    """Plot regret analysis results with LaTeX formatting and log scales."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure with 2x2 subplots

    T = len(cumulative_regret)
    t_range = np.arange(T)
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, cumulative_regret, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_cumulative_regret, 'r-', linewidth=2, label = "reclustering")
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Cumulative Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_cumulative.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, regret, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_regret, 'r-', linewidth=2, label = "reclustering")
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Instantaneous Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_inst.pdf', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1,1, figsize=(9, 4), dpi=300)
    ax.plot(t_range, cumulative_regret, 'b-', linewidth=2, label = "actual cumulative regret")
    ax.plot(t_range, theo, 'r-', linewidth=2, label = "theoretical regret")
    # axins = zoomed_inset_axes(ax, 6, loc="lower right")
    # axins.set_xlim(3700, 4000)
    # axins.set_ylim(7, 10)
    # axins.plot(t_range, cumulative_regret, 'b-',linewidth=2)
    # axins.set_xticks(ticks=[])
    # axins.set_yticks(ticks=[])
    # mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
    ax.set_xlabel(r'Time step $(t)$')
    ax.set_ylabel(r'Cumulative Regret')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'regret_analysis_comp.pdf', bbox_inches='tight', dpi=300)


def plot_eval(eval, MRO_eval, DRO_eval,SA_eval, history):
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16
    })
    T = len(eval)
    t_range = np.arange(T)
    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, eval, 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, MRO_eval, 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, SA_eval, 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, DRO_eval, color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Evaluation value (out of sample)')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'eval_analysis.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, history['obj_values'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_obj_values'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['SA_obj_values'], 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, history['DRO_obj_values'], color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Objective value (in sample)')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'obj_analysis.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(7, 4), dpi=300)
    plt.plot(t_range, history['obj_values'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_obj_values'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['DRO_obj_values'], color ='black', linewidth=2, label = "DRO")
    plt.plot(t_range, history['SA_obj_values'], 'g-', linewidth=2, label = "SAA")
    plt.plot(t_range, eval,  'b', linewidth=2, linestyle='-.')
    plt.plot(t_range, MRO_eval, 'r', linewidth=2, linestyle='-.')
    plt.plot(t_range, DRO_eval, color ='black', linestyle='-.')
    plt.plot(t_range, SA_eval,'g', linestyle='-.')
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Objective value and evaluation value')
    plt.grid(True, alpha=0.3)
    plt.savefig(foldername+'obj_eval_analysis.pdf', bbox_inches='tight', dpi=300)



def plot_results(history):
    """Plot results with LaTeX formatting."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure with higher DPI
    plt.figure(figsize=(11, 4), dpi=300)

    # Plot 2: Epsilon Evolution
    plt.subplot(121)
    plt.plot(history['epsilon'], 'r-', linewidth=2)
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'$\epsilon$')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Ball Weights
    plt.subplot(122)
    plt.plot(np.array(history['weights']), linewidth=2)
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Ball Weights')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    plt.savefig(foldername+'radius.pdf', bbox_inches='tight', dpi=300)


def plot_computation_times(history):
    """Plot computation time analysis with LaTeX formatting."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    
    # Create figure
    plt.figure(figsize=(15, 3), dpi=300)
    
    # Prepare data for boxplot
    data = [
        history['online_computation_times']['total_iteration'], history['MRO_computation_times']['total_iteration'],history['DRO_computation_times']['total_iteration'] 
    ]
    # np.save("online",history['online_computation_times']['total_iteration'])
    # np.save("mro",history['MRO_computation_times']['total_iteration'])
    # np.save("dro",history['DRO_computation_times']['total_iteration'])

    # Create boxplot
    bp = plt.boxplot(data, labels=[

        r'online clustering', r'reclustering', r'DRO' 
    ])
    
    # Customize boxplot colors
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['caps'], color='blue')
    plt.setp(bp['medians'], color='red')

    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'Compuation time')
    plt.yscale("log")
    plt.savefig(foldername+'time.pdf', bbox_inches='tight', dpi=300)

def plot_computation_times_iter(history):
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22
    })
    t_range = np.arange(len( history['online_computation_times']['total_iteration']))
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(t_range, history['online_computation_times']['total_iteration'], 'b-', linewidth=2, label = "online clustering")
    plt.plot(t_range, history['MRO_computation_times']['total_iteration'], 'r-', linewidth=2, label = "reclustering")
    plt.plot(t_range, history['DRO_computation_times']['total_iteration'], color ='black', linewidth=2, label = "DRO")
    plt.legend()
    plt.xlabel(r'Time step $(t)$')
    plt.ylabel(r'Compuation time')
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig(foldername+'time_iters.pdf', bbox_inches='tight', dpi=300)

def calc_cluster_val(K,k_dict, num_dat,x):
    mean_val = 0
    square_val = 0
    sig_val = 0
    for k in range(K):
        centroid = k_dict['d'][k]
        for dat in k_dict['data'][k]:
            cur_val = np.linalg.norm(dat-centroid,2)
            mean_val += cur_val
            square_val += cur_val**2
            sig_val = np.maximum(sig_val,(dat-centroid)@x)
    return mean_val/num_dat, square_val/num_dat, sig_val

def port_experiments(r,K,T,N_init, eps_init):
    dfs = {}
    dat, dateval = train_test_split(
        synthetic_returns[:, :m], train_size=10000, test_size=10000, random_state=r)
    
    for epsnum in range(len(eps_init)):
      init_eps = eps_init[epsnum]
      num_dat = N_init
      q_dict, k_dict = online_cluster_init(K,Q)
      new_k_dict = None
      assert k_dict["K"] == K

      online_problem, online_x, online_s, online_tau, online_lmbda, data_train, eps_train, w_train = createproblem_portMIP(K, m)

      # Initialize solutions
      x_current = np.zeros(30)

      # History for analysis
      history = {
          'x': [],
          'tau': [],
          'obj_values': [],
          'MRO_x': [],
          'MRO_tau': [],
          'MRO_obj_values': [],
          'DRO_x': [],
          'DRO_tau': [],
          'DRO_obj_values': [],
          'worst_values': [],
          'worst_values_MRO':[],
          'epsilon': [],
          'weights': [],
          'weights_q': [],
          'online_computation_times': {
              'weight_update': [],
              'min_problem': [],
              'total_iteration': []
          },
          'MRO_computation_times':{
          'clustering': [],
          'min_problem': [],
          'total_iteration':[]
          },
          'DRO_computation_times':{
          'total_iteration':[]
          },
          'distances':[],
          'mean_val':[],
          'square_val': [],
          'sig_val': [],
          'mean_val_MRO':[],
          'square_val_MRO': [],
          'sig_val_MRO': [],
          'SA_computation_times':[],
          'SA_obj_values':[],
          'SA_x': [],
          'SA_tau':[],
          "satisfy":[],
          "MRO_satisfy":[],
          "DRO_satisfy":[]
      }


      for t in range(T):
          print(f"\nTimestep {t+1}/{T}")
          
          radius = init_eps*(1/(num_dat**(1/(2*m))))
          running_samples = dat[init_ind:(init_ind+num_dat)]

          # solve online MRO problem
          if t % interval == 0 or ((t-1) % interval == 0) :
              data_train.value = k_dict['d']
              eps_train.value = radius
              w_train.value = k_dict['w']
              online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                  mosek.dparam.optimizer_max_time:  2000.0})
              x_current = online_x.value
              tau_current = online_tau.value
              min_obj = online_problem.objective.value
              min_time = online_problem.solver_stats.solve_time

          # Store timing information
          history['online_computation_times']['min_problem'].append(min_time)

          if t % interval == 0 or ((t-1) % interval == 0) :
              # solve MRO problem with new clusters
              start_time = time.time()
              if new_k_dict is not None:
                  kmeans = KMeans(n_clusters=K, init=new_k_dict['d']).fit(running_samples)
              else:
                  kmeans = KMeans(n_clusters=K).fit(running_samples)
              new_centers = kmeans.cluster_centers_
              wk = np.bincount(kmeans.labels_) / num_dat
              cluster_time = time.time()-start_time
              new_k_dict = {}
              new_k_dict['data'] = {}
              for k in range(K):
                  new_k_dict['data'][k] = running_samples[kmeans.labels_==k]
              new_k_dict['d'] = new_centers

              data_train.value = new_centers
              w_train.value = wk
              # eps_train.value = new_radius
              online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                  mosek.dparam.optimizer_max_time:  2000.0})
              MRO_x_current = online_x.value
              MRO_tau_current = online_tau.value
              MRO_min_obj = online_problem.objective.value
              MRO_min_time = online_problem.solver_stats.solve_time
              mean_val_mro, square_val_mro, sig_val_mro = calc_cluster_val(K, new_k_dict,num_dat,MRO_x_current)
          
          history['MRO_computation_times']['min_problem'].append(MRO_min_time)
          history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time)
          history['MRO_computation_times']['clustering'].append(cluster_time)

          if t % interval == 0 or ((t-1) % interval == 0) :
          # solve DRO problem 
              DRO_problem, DRO_x, DRO_s, DRO_tau, DRO_lmbda, DRO_data, DRO_eps, DRO_w = createproblem_portMIP(num_dat,m)
              DRO_data.value = running_samples
              DRO_w.value = (1/num_dat)*np.ones(num_dat)
              DRO_eps.value = radius
              DRO_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                  mosek.dparam.optimizer_max_time:  2000.0})
              DRO_x_current = DRO_x.value
              DRO_tau_current = DRO_tau.value
              DRO_min_obj = DRO_problem.objective.value
              DRO_min_time = DRO_problem.solver_stats.solve_time
          history['DRO_computation_times']['total_iteration'].append(DRO_min_time)

          if t % interval == 0 or ((t-1) % interval == 0) :
              # compute online MRO worst value (wrt non clustered data)
              orig_cons = DRO_problem.constraints
              orig_obj = DRO_problem.objective
              new_cons = orig_cons + [DRO_x == x_current, DRO_tau == tau_current]
              new_problem = cp.Problem(orig_obj, new_cons)
              new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                  mosek.dparam.optimizer_max_time:  2000.0})
              new_worst = new_problem.objective.value

              orig_cons = DRO_problem.constraints
              orig_obj = DRO_problem.objective
              new_cons = orig_cons + [DRO_x == MRO_x_current, DRO_tau == MRO_tau_current]
              new_problem_MRO = cp.Problem(orig_obj, new_cons)
              new_problem_MRO.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                  mosek.dparam.optimizer_max_time:  2000.0})
              new_worst_MRO = new_problem_MRO.objective.value

              mean_val, square_val, sig_val = calc_cluster_val(K, k_dict,num_dat,x_current)
              q_lens = [len(q_dict['data'][i]) for i in range(q_dict['cur_Q'])]
              k_lens = [len(k_dict['data'][i]) for i in range(k_dict['K'])]
              # print("Q nums", q_lens, np.sum(q_lens), num_dat)
              # print("K nums", k_lens, np.sum(k_lens), num_dat)

          history['worst_values'].append(new_worst)
          history['worst_values_MRO'].append(new_worst_MRO)

          if t % interval == 0 or ((t-1) % interval == 0) :
              s_prob, s_x, s_tau = create_scenario(running_samples,m,num_dat)
              s_prob.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
                      mosek.dparam.optimizer_max_time:  2000.0})
              SA_x_current = s_x.value
              SA_tau_current = s_tau.value
              SA_obj_current = s_prob.objective.value
              SA_time = s_prob.solver_stats.solve_time

          history['SA_computation_times'].append(SA_time)
          history['SA_x'].append(SA_x_current)
          history['SA_tau'].append(SA_tau_current)
          history['SA_obj_values'].append(SA_obj_current)

          # New sample
          new_sample = dat[init_ind+num_dat]
          q_dict, k_dict, weight_update_time = online_cluster_update(K,new_sample, q_dict, k_dict,num_dat, t, fixed_time)
          num_dat += 1
          history['online_computation_times']['weight_update'].append(weight_update_time)
          history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)
          
          history['mean_val'].append(mean_val)
          history['sig_val'].append(sig_val)
          history['square_val'].append(square_val)
          history['mean_val_MRO'].append(mean_val_mro)
          history['sig_val_MRO'].append(sig_val_mro)
          history['square_val_MRO'].append(square_val_mro)
          history['x'].append(x_current)
          history['tau'].append(tau_current)
          history['obj_values'].append(min_obj)
          history['MRO_x'].append(MRO_x_current)
          history['MRO_tau'].append(MRO_tau_current)
          history['MRO_obj_values'].append(MRO_min_obj)
          history['DRO_x'].append(DRO_x_current)
          history['DRO_tau'].append(DRO_tau_current)
          history['DRO_obj_values'].append(DRO_min_obj)
          history['epsilon'].append(radius)
          history['weights'].append(k_dict['w'].copy())
          history['weights_q'].append(q_dict['w'].copy())
          
          print(f"Current allocation: {x_current}")
          print(f"Current epsilon: {radius}")
          print(f"Weight sum: {np.sum(k_dict['w'])}")
          # print(f"Weights: {q_dict['w'], np.sum(q_dict['w']) }")
          

      cumulative_regret, instantaneous_regret, eval, MRO_eval, DRO_eval, SA_eval,theo, MRO_cum_regret, MRO_regret,satisfy, MRO_satisfy, DRO_satisfy = compute_cumulative_regret(
          history,T,dateval)

      dfs[epsnum] = pd.DataFrame({
      'x': history['x'],
      'tau': np.array(history['tau']),
      'obj_values': np.array(history['obj_values']),
      'MRO_x': history['MRO_x'],
      'MRO_tau':np.array(history['MRO_tau']),
      'MRO_obj_values': np.array(history['MRO_obj_values']),
      'DRO_x': history['DRO_x'],
      'DRO_tau': np.array(history['DRO_tau']),
      'DRO_obj_values': np.array(history['DRO_obj_values']),
      'epsilon': np.array(history['epsilon']),
      'weights':  history['weights'],
      'weights_q': history['weights_q'],
      'online_time':  np.array(history['online_computation_times']['total_iteration']),
      'MRO_time':  np.array(history['MRO_computation_times']['total_iteration']),
      'DRO_time':  np.array(history['DRO_computation_times']['total_iteration']),
      'MRO_mean_val': np.array(history['mean_val_MRO']),
      'MRO_square_val': np.array(history['square_val_MRO']),
      'MRO_sig_val': np.array(history['sig_val_MRO']),
      'mean_val': np.array(history['mean_val']),
      'square_val': np.array(history['square_val']),
      'sig_val': np.array(history['sig_val']),
      'cumulative_regret': np.array(np.concatenate([cumulative_regret,np.zeros(1)])),
      'regret': np.array(np.concatenate([instantaneous_regret,np.zeros(1)])),
      'eval': np.array(eval),
      'MRO_eval': np.array(MRO_eval),
      'DRO_eval': np.array(DRO_eval),
      'theoretical': np.array(np.concatenate([theo,np.zeros(1)])),
      'MRO_cumulative_regret': np.array(np.concatenate([MRO_cum_regret,np.zeros(1)])),
      "MRO_regret": np.array(np.concatenate([MRO_regret,np.zeros(1)])),
      "satisfy": satisfy,
      "MRO_satisfy": MRO_satisfy,
      "DRO_satisfy": DRO_satisfy
      })
    # df.to_csv('df.csv')
    df =  pd.concat([dfs[i] for i in range(len(eps_init))],ignore_index=True)

     # Plot regret analysis
    plot_regret_analysis(
          cumulative_regret, 
          instantaneous_regret,theo,MRO_cum_regret,MRO_regret
      )

      # After all other plots
    plot_computation_times(history)

    plot_eval(eval, MRO_eval, DRO_eval, SA_eval, history)

    plot_computation_times_iter(history)
  
    return df
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str,
                        default="/scratch/gpfs/iywang/mro_results/", metavar='N')
    arguments = parser.parse_args()
    foldername = arguments.foldername
    datname = '/scratch/gpfs/iywang/mro_experiments/portfolio_time/sp500_synthetic_returns.csv'
    synthetic_returns = pd.read_csv(datname
                                    ).to_numpy()[:, 1:]
    
    T = 3
    fixed_time = 1500
    interval = 100
    m = 30
    N_init = 50
    K = 5
    Q = 500
    init_ind = 0
    njobs = get_n_processes(30)
    R = 10
    eps_init = [0.008,0.006,0.005,0.004,0.0035,0.003,0.0025,0.002,0.001]
    
    results = Parallel(n_jobs=njobs)(delayed(port_experiments)(
        r,K,T,N_init, eps_init) for r in range(R))
    
    for r in range(R):
        results[r].to_csv(foldername + '/df_' + str(r) +'.csv')
    print("DONE")