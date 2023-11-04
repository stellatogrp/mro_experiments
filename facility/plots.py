import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="facility/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername+'df.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})

# K_nums = np.array([1,2, 5,7,10, 25, 50])
K_nums = np.array([1,2, 3, 4,5,10, 25, 50])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 50 
n = 5  # number of facilities
m = 25  # number of locations


# eps_nums = [0.05, 0.1, 0.11, 0.13, 0.15, 0.2, 0.25, 0.3, 0.35,
#             0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
eps_nums = np.concatenate([np.logspace(-3,-0.8,25), np.linspace(0.16,0.5,20), np.linspace(0.51, 0.8, 5)])

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(13, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
j = 0
for K_count in [0,1, 4,5,7]:
    ax1.plot(eps_nums[::3], dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"][::3],
        linestyle='-', marker=styles[j], label="$K = {}$".format(K_nums[K_count]), alpha=0.7)
    j+=1
# K_count = 0
# ax1.plot(eps_nums, dftemp.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
#         eps_nums)]["Opt_val"],
#         linestyle='-', marker=styles[j], label="$K = 50^*$", alpha=0.6)
ax1.set_xlabel(r"$\epsilon$")
ax1.set_xscale("log")
ax1.set_title("Objective value")

j = 0
for K_count in [0,1,4,5,7]:
    ax21.plot(1 - dftemp.sort_values(
        ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][0:-1:1],
        dftemp.sort_values(["K", "Epsilon"])[
        K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
        linestyle='-', label="$K = {}$".format(K_nums[K_count]), marker=styles[j], alpha=0.7)
    j += 1
# K_count = 0
# ax21.plot(1 - dftemp.sort_values(
#     ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val1"][0:-1:1],
#     dftemp.sort_values(
#     ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],
#     linestyle='-',
#     label="$K = 50^*$", marker=styles[j], alpha=0.7)
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
# ax21.set_ylim([303, 320])
# ax21.set_xlim([-0.05, 0.8])
ax21.legend(bbox_to_anchor=(1, 0.75), fontsize=14)
plt.tight_layout()
# plt.savefig(foldername + "facilitytop")
plt.savefig(foldername + "facilitytop1.pdf")

plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
for i in [10,15,25,37,39]:
    gnval = np.array(dftemp.sort_values(["Epsilon", "K"])[
                     i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (gnval - dftemp.sort_values(["Epsilon", "K"])[
           i*len(K_nums):(i+1)*len(K_nums)-1]["Opt_val"])
    ax31.plot(K_nums[:-1], dif, label=r"$\epsilon = {}$".format(
        np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color=colors[j])
    ax31.plot(K_nums[:-1], (dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1) *
              len(K_nums)]["bound"][:-1]), linestyle='--', color=colors[j], label="Upper bound")
    j += 1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"Obj($N$) - Obj($K$)")


ord = 0
j = 0
for i in [10,15,25,37, 39]:
    ax4.plot(K_nums[:],
             dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][:],
             linestyle='-',
             marker='o', label=r"$\epsilon = {}$".format(np.round(eps_nums[i], 5)), zorder=ord)
    ord += 1
#     ax4.scatter(K_nums[-1],
#                 np.array(dftemp.sort_values(
#                     ["Epsilon", "K"])[i*len(K_nums):((i+1)*len(K_nums))]["solvetime"])[0],
#                 marker="s",  color=colors[j], zorder=ord)
#     j+=1
#     ord += 1
# ax4.scatter(K_nums[-1], np.array(
#             dftemp.sort_values(["Epsilon", "K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"])[0],
#             marker="s", color=colors[j-1], zorder=ord, label="$K = 50^*$")
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc="lower right", bbox_to_anchor=(1.38, 0.2), fontsize=14)

plt.tight_layout()
# plt.savefig(foldername + "facilitybot")
plt.savefig(foldername + "facilitybot1.pdf")

plt.show()
