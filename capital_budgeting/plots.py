import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="capital_budgeting/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername+'df.csv')


m = 12
N_tot = 50
K_nums = [1, 2, 5, 10, 25, 50, 9999]
eps_nums = np.concatenate((np.logspace(-4, -2.95, 10), np.logspace(-2.9, -
                          1.9, 15), np.logspace(-1.8, 0, 8), np.logspace(0.1, 1, 3)))


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "serif"
})

d = (np.vstack([np.random.uniform(0.005*i, 0.02 *
                                  (i+1), int(N_tot/2)) for i in range(m)])).T
d2 = (np.vstack([np.random.uniform(0.01*i, 0.025 *
      (i+1), int(N_tot/2)) for i in range(m)])).T
dat_val = np.vstack([d, d2])
d = (np.vstack([np.random.uniform(0.005*i, 0.02 *
                                  (i+1), int(N_tot/2)) for i in range(m)])).T
d2 = (np.vstack([np.random.uniform(0.01*i, 0.025 *
      (i+1), int(N_tot/2)) for i in range(m)])).T
dat_eval = np.vstack([d, d2])
vals = []
for K in np.arange(1, 25):
    kmeans = KMeans(n_clusters=K, n_init='auto').fit(dat_val)
    weights = np.bincount(kmeans.labels_) / N_tot
    vals.append(kmeans.inertia_/N_tot)
plt.figure(figsize=(8, 2.5))
plt.plot(np.arange(1, 25), vals)
plt.yscale("log")
plt.xlabel("$K$ (number of clusters)")
plt.ylabel("$D(K)$")
plt.tight_layout()
plt.savefig(foldername + "capital_k.pdf")


fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(13, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
j = 0
for K_count in [0, 1, 3, 5, 6]:
    ax1.plot(np.array(eps_nums),
             dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
                 eps_nums)]["Opt_val"],
             linestyle='-', marker=styles[j],
             label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot(np.array(eps_nums),
             dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
                 eps_nums)]["Eval_val"],
             color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
ax1.set_xlabel(r"$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")
axins = zoomed_inset_axes(ax1, 6, loc="upper left")
axins.set_xlim(10**(-3.5), 10e-4)
axins.set_ylim(-12.3, -11.9)
j = 0
for K_count in [0, 1, 3, 5, 6]:
    axins.plot(np.array(eps_nums), dftemp.sort_values(["K", "Epsilon"])[
               K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], color=colors[j])
    axins.plot(np.array(eps_nums),
               dftemp.sort_values(["K", "Epsilon"])[
               K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"],
               linestyle=':', color=colors[j])
    j += 1
axins.set_xticks(ticks=[])
axins.set_yticks(ticks=[])
mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")

j = 0
for K_count in [0, 1, 3, 5]:
    ax21.plot(1 - dftemp.sort_values(
        ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],
        dftemp.sort_values(["K", "Epsilon"])[
        K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
        linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
K_count = 6
ax21.plot(1 - dftemp.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],
    dftemp.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
    linestyle='-',
    label="$K = 50^*$", marker=styles[j], alpha=0.7)
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_ylim([-12.3, -11.9])
ax21.legend(bbox_to_anchor=(1, 0.75), fontsize=14)
plt.tight_layout()
plt.savefig(foldername + "capitaltop.pdf")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
for i in np.arange(0, len(eps_nums)-3, 8):
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-2]
    dif = (dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)-2]["Opt_val"] - gnval)
    ax31.plot(K_nums[:-2], dif, label=r"$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-2], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1) *
              len(K_nums)]["bound2"][:-2]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - \bar{g}^N$")


j = 0
for i in np.arange(0, len(eps_nums)-3, 8):
    ax4.plot(K_nums[0:-1],
             dftemp.sort_values(
                 ["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:-1],
             linestyle="-", marker=styles[j],
             color=colors[j], label=r"$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc="lower right", bbox_to_anchor=(1.36, 0.2), fontsize=14)

plt.tight_layout()
plt.savefig(foldername + "capitalbot.pdf")
plt.show()
