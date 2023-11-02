import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import argparse
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername + 'df_supp.csv')


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})
styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

K_nums = np.array([0,1, 3, 5, 10, 25, 50, 100])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 100
eps_tot = 30
M = 10
R = 1     # Total times we repeat experiment to estimate final probabilty
n = 2  # number of products
eps_nums = np.concatenate([np.logspace(-4,-0.5,30), np.linspace(0.32,2,30)])
fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(13, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
j = 0
for K_count in [1, 3, 4,7]:
    ax1.plot(np.array(eps_nums)[::5], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"][::5], color=colors[j],linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot(np.array(eps_nums)[::5], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"][::5], color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
K_count = 0
ax1.plot(np.array(eps_nums)[::5], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"][::5],
        linestyle='-', color=colors[j],marker=styles[j], label="$K = 100^*$", alpha=0.7)
ax1.plot(np.array(eps_nums)[::5], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"][::5], color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    
ax1.set_xlabel("$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")

j = 0
for K_count in [1, 3, 4,7]:
    ax21.plot(1 - dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1], dftemp.sort_values(["K", "Epsilon"])[
              K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1], linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
K_count = 0
ax21.plot(1 - dftemp.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],
    dftemp.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
    linestyle='-',
    label="$K = 100^*$", marker=styles[j], alpha=0.7)

ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_ylim([-35, -10])
ax21.legend(bbox_to_anchor=(1, 0.75), fontsize=14)
plt.tight_layout()
# plt.savefig(foldername + "newstop_supp")
plt.savefig(foldername + "newstop_supp.pdf")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))
j = 0
for i in [2, 10, 20,30]:
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (gnval - dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])
    ax31.plot(K_nums[1:-1], dif[1:-1], label="$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[1:-1], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1) *
              len(K_nums)]["bound"][1:-1]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^N - \bar{g}^K$")


j = 0
for i in [2, 10, 20,30]:
    ax4.plot(K_nums[1:], dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][1:],
             linestyle="-", marker=styles[j], color=colors[j], label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
# ax4.set_yticks([10e-2, 10e-1, 10e0, 10e1, 10e2])
# ax4.grid()
ax4.legend(loc="lower right", bbox_to_anchor=(1.36, 0.2), fontsize=14)

plt.tight_layout()
# plt.savefig(foldername + "newsbot_supp")
plt.savefig(foldername + "newsbot_supp.pdf")

plt.show()
