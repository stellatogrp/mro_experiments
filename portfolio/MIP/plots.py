import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername + 'df_redo1.csv')
for r in range(2, 7):
    dftemp = dftemp.add(pd.read_csv(foldername + 'df_redo' +
                                    str(r) + '.csv').reset_index(), fill_value=0)
dftemp = dftemp/6


#df = pd.read_csv(foldername + 'df_all3.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})
styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

K_nums = np.array([1, 4, 5, 10, 25, 50, 100, 150, 250, 500, 1000])
N_tot = 1000
M = 15
R = 12
m = 30
eps_min = -3.5
eps_max = -1.5
eps_nums = np.linspace(eps_min, eps_max, M)
eps_nums = (10**(eps_nums))

plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 4)

ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[:2, 2:])
ax3 = plt.subplot(gs[2:4, :4])
j = 0
for K_count in [0, 2, 3, 6, 9, 10]:
    ax1.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"], linestyle='-', marker=styles[j], color=colors[j], label="$K = {}$".format(K_nums[K_count]), alpha=0.6)
    ax1.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"], linestyle=':', color=colors[j], alpha=0.6)
    j += 1
ax1.set_xlabel("$\epsilon$")
ax1.set_xscale("log")
ax1.set_title("In-sample objective and %\n out of sample expected values")

j = 0
for K_count in [0, 2, 3, 6, 9, 10]:
    ax2.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["satisfy"], label="$K = {}$".format(K_nums[K_count]), linestyle='-', marker=styles[j], alpha=0.5)
    j += 1
ax2.set_xlabel("$\epsilon$")
ax2.set_xscale("log")
ax2.legend(loc="lower right")
ax2.legend(fontsize=13)
ax2.set_title(r"$1-\beta$ (probability of constraint satisfaction)")
j = 0
for i in [2, 4, 6, 7, 8]:
    ax3.plot(K_nums[:], np.array(dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):((i+1)*len(K_nums))]
             ["solvetime"])[:], linestyle='-', marker=styles[j], label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
    j += 1
ax3.set_xlabel("$K$ (Number of clusters)")
ax3.set_yscale("log")
ax3.set_yticks([10e-2, 10e-1, 10e0, 10e1, 10e2])
ax3.set_ylabel("Time (s)")
ax3.grid()
ax3.legend(fontsize=13)
plt.tight_layout()
plt.savefig("portMIP_redo4.pdf")
plt.savefig("portMIP_redo4.png")


fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(13, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
j = 0
for K_count in [0, 2, 3, 9, 10]:
    ax1.plot(np.array(eps_nums), dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"], linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot(np.array(eps_nums), dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"], color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
ax1.set_xlabel("$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")

j = 0
for K_count in [0, 2, 3, 9, 10]:
    ax21.plot(1 - dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1], dftemp.sort_values(["K", "Epsilon"])[
              K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1], linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_ylim([0, 0.04])
ax21.legend(bbox_to_anchor=(1, 0.75), fontsize=14)
plt.tight_layout()
plt.savefig("portMIPtop.pdf")
plt.savefig("portMIPtop.png")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))
j = 0
for i in [2, 4, 6, 7, 8]:
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (gnval - dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])
    ax31.plot(K_nums[:-1], dif[:-1], label="$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-1], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1) *
              len(K_nums)]["bound"][:-1]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^N - \bar{g}^K$")


j = 0
for i in [2, 4, 6, 7, 8]:
    ax4.plot(K_nums[2:], dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][2:],
             linestyle="-", marker=styles[j], color=colors[j], label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.set_yticks([10e-2, 10e-1, 10e0, 10e1, 10e2])
# ax4.grid()
ax4.legend(loc="lower right", bbox_to_anchor=(1.36, 0.2), fontsize=14)

plt.tight_layout()
plt.savefig("portMIPbot2.pdf")
plt.savefig("portMIPbot2.png")
plt.show()
