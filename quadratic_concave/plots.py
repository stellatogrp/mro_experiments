import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="quadratic_concave/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername + 'df.csv')
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})

N_tot = 90
m = 10
R = 5
K_nums = [1, 2, 3, 4, 5, 15, 45, 90]

eps_nums = np.concatenate((np.logspace(-2.2, -1, 8), np.logspace(-0.8, 0, 5),
                           np.logspace(0.1, 0.5, 20), np.array([3, 4, 7, 9, 10])))


def normal_returns_scaled(N, m, scale):
    """Creates scaled data
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Scales: float
        Multiplier for a single mode
    Returns:
    -------
    d: matrix
        Scaled data with a single mode
    """
    R = np.vstack([np.random.normal(
        i*0.03*scale, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
    return (R.transpose())


def data_modes(N, m, scales):
    """Creates data scaled by given multipliers
    Parameters:
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Scales: vector
        Multipliers of different modes
    Returns:
    -------
    d: matrix
        Scaled data with all modes
    """
    modes = len(scales)
    d = np.zeros((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights,
          :] = normal_returns_scaled(weights, m, scales[i])
    return d[0:N, :]

d = data_modes(N_tot, m, [1, 5, 15, 25, 40])
vals = []
for K in np.arange(1,45):
    kmeans = KMeans(n_clusters=K, n_init='auto').fit(d)
    weights = np.bincount(kmeans.labels_) / N_tot
    vals.append(kmeans.inertia_/N_tot)
plt.figure(figsize = (8,2.5))
plt.plot(np.arange(1,45),vals)
plt.yscale("log")
plt.xlabel("$K$ (number of clusters)")
plt.ylabel("$D(K)$")
plt.tight_layout()
plt.savefig(foldername + "quad_k.pdf")
plt.show()

styles = ["o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(14, 4.5))

styles = ["o", 's', "^", "v", "<", ">"]
j = 0
for K_count in [0, 1, 4, 6, 7]:
    ax1.plot(np.sort(eps_nums),
             dftemp.sort_values(
                 ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"],
             linestyle='-', marker=styles[j],
             label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot(np.sort(eps_nums),
             dftemp.sort_values(
                 ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"],
             color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
ax1.set_xlabel(r"$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")

j = 0
for K_count in [0, 1, 4, 6, 7]:
    ax21.plot(1 - dftemp.sort_values(
        ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],
        dftemp.sort_values(["K", "Epsilon"])[
        K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
        linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_xlim([-0.025, 10**(-0.25)])
ax21.set_ylim([-102, -90])
ax21.legend(bbox_to_anchor=(1, 0.35), fontsize=14)
plt.tight_layout()
plt.savefig("quadratictop.pdf")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
for i in [8, 10, 18, 24]:
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"] - gnval)
    ax31.plot(K_nums[:-1], dif[:-1], label=r"$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-1], (dftemp.sort_values(["Epsilon", "K"])[
              i*len(K_nums):(i+1)*len(K_nums)]["bound"])[:-1], linestyle='--',
              color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - \bar{g}^N$")

j = 0
for i in [8, 10, 18, 24]:
    ax4.plot(K_nums,
             dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"],
             linestyle="-", marker=styles[j], color=colors[j],
             label=r"$\epsilon = {}$".format(np.round(eps_nums[i], 3)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc="lower right", bbox_to_anchor=(1.33, 0.2), fontsize=14)

plt.tight_layout()
plt.savefig("quadraticbot.pdf")
plt.show()
