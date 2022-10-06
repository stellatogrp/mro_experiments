import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername+'df.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})

K_nums = np.array([0, 1, 5, 10, 25, 50])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 50
n = 5  # number of facilities
m = 25  # number of locations


eps_nums = [0.05, 0.1, 0.11, 0.13, 0.15, 0.2, 0.25, 0.3, 0.35,
            0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8]


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    "font.family": "serif"

})

plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 4)

ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[:2, 2:])
ax3 = plt.subplot(gs[2:4, :2])
ax4 = plt.subplot(gs[2:4, 2:])

for K_count in [0, 1, 2, 4, 5]:
    ax1.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"], linestyle='-', marker='v', label="$K = {}$".format(K_nums[K_count]), alpha=0.6)
ax1.set_xlabel("$\epsilon$")
ax1.set_xscale("log")
ax1.set_title("Objective value")


ax2.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[0*len(eps_nums):(1)*len(
    eps_nums)]["Eval_val"], label="$K = 1^*$", linestyle='-', marker='v', alpha=0.5)
for K_count in [1, 2, 4, 5]:
    ax2.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"], label="$K = {}$".format(K_nums[K_count]), linestyle='-', marker='v', alpha=0.5)
ax2.set_xlabel("$\epsilon$")
ax2.set_title(r"$1-\beta$ (probability of constraint satisfaction)*")
ax2.set_xscale("log")
ax2.legend()

for K_count in [0, 1, 2, 4, 5]:
    ax3.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val1"], label="$K = {}$".format(K_nums[K_count]), linestyle='-', marker='v', alpha=0.5)
ax3.set_xlabel("$\epsilon$")
ax3.set_title(r"$1-\beta$ (probability of constraint satisfaction)**")

ord = 0
for i in [0, 10, 18, 22]:
    ax4.plot(K_nums[:], dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][:],
             linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), zorder=ord)
    ord += 1
    ax4.scatter(K_nums[0], np.array(dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):((i+1)*len(K_nums))]
                                    ["solvetime"])[0], marker="s", color="black", zorder=ord)
    ord += 1
ax4.scatter(K_nums[0], np.array(dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):((i+1)*len(K_nums))]
                                ["solvetime"])[0], marker="s", color="black", zorder=ord, label="$K = 1^*$")
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(fontsize=13)

plt.tight_layout()
plt.savefig(foldername + "facilitytop.pdf")
