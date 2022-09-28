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
dftemp = pd.read_csv(foldername + 'df.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})
N_tot = 90
m = 30
K_nums = np.array([1, 2, 3, 5, 6, 7, 8, 10, 20, 40, 90])
eps_nums = np.append(np.logspace(-5.2, -4, 15), np.logspace(-3.9, 1, 10))


colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(14, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
j = 0
for K_count in [0, 1, 2, 9, len(K_nums)-1]:
    ax1.plot((np.sort(eps_nums))[10:], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
             ["Opt_val"][10:], linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
    ax1.plot((np.sort(eps_nums))[10:], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"][10:], color=colors[j], linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
    j += 1
ax1.set_xlabel("$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")

j = 0
for K_count in [0, 1, 2, 9, 10]:
    ax21.plot(1 - dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1], dftemp.sort_values(["K", "Epsilon"])[
              K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1], linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
#ax21.set_ylim([2.735, 2.755])
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
    ax31.plot(K_nums[:-1], dif, label="$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-1], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)
              * len(K_nums)-1]["bound2"]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - g^N$")

j = 0

for i in [13, 18, 20, 21]:
    ax4.plot(K_nums[:], dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][:],
             linestyle="-", marker=styles[j], color=colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j += 1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc="lower right", bbox_to_anchor=(1.38, 0.2), fontsize=14)
plt.tight_layout()
plt.savefig(foldername + "logbot.pdf")
plt.show()
