import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="logsumexp/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername
dftemp = pd.read_csv(foldername + 'df.csv')


def dat_scaled(N, m, scale):
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
    R = np.vstack([np.random.uniform(0.01*i*scale, 0.01*(i+1)*scale, N)
                  for i in range(1, m+1)])
    return R.transpose()


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
    d = np.ones((N+100, m))
    weights = int(np.ceil(N/modes))
    for i in range(modes):
        d[i*weights:(i+1)*weights, :] = dat_scaled(weights, m, scales[i])
    return d[0:N, :]


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})
N_tot = 90
m = 30
K_nums = np.array([1, 2, 3, 5, 6, 7, 8, 10, 20, 40, 90])
eps_nums = np.append(np.logspace(-5.2, -4, 15), np.logspace(-3.9, 1, 10))

d = data_modes(90, 30, [1, 3, 7])
vals = []
for K in np.arange(1, 40):
    kmeans = KMeans(n_clusters=K, n_init='auto').fit(d)
    weights = np.bincount(kmeans.labels_) / N_tot
    vals.append(kmeans.inertia_/N_tot)
plt.figure(figsize=(8, 2.5))
plt.plot(np.arange(1, 40), vals)
plt.yscale("log")
plt.xlabel("$K$ (number of clusters)")
plt.ylabel("$D(K)$")
plt.tight_layout()
plt.savefig(foldername + "log_k.pdf")
plt.show()

colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(14, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
j = 0
for K_count in [0, 1, 2, 9, len(K_nums)-1]:
    ax1.plot((np.sort(eps_nums))[10:],
             dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
             ["Opt_val"][10:], linestyle='-',
             marker=styles[j],
             label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot((np.sort(eps_nums))[10:],
             dftemp.sort_values(
        ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][10:],
        color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
ax1.set_xlabel(r"$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")

j = 0
for K_count in [0, 1, 2, 9, 10]:
    ax21.plot(1 - dftemp.sort_values(
        ["K", "Epsilon"]
    )[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],
        dftemp.sort_values(["K", "Epsilon"])[
        K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
        linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
# ax21.set_ylim([2.735, 2.755])
ax21.legend(bbox_to_anchor=(1.3, .8), fontsize=14)
plt.tight_layout()
plt.savefig(foldername + "logtop.pdf")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
for i in [13, 18, 20, 21]:
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)-1]["Opt_val"] - gnval)
    ax31.plot(K_nums[:-1], dif, label=r"$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-1], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)
              * len(K_nums)-1]["bound2"]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - g^N$")

j = 0

for i in [13, 18, 20, 21]:
    ax4.plot(K_nums[:],
             dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][:],
             linestyle="-", marker=styles[j], color=colors[j],
             label=r"$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc="lower right", bbox_to_anchor=(1.38, 0.2), fontsize=14)
plt.tight_layout()
plt.savefig(foldername + "logbot.pdf")
plt.show()
