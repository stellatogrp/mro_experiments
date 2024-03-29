import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                    default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

df = pd.read_csv(foldername + 'df.csv')

dfout = pd.read_csv(foldername + 'df_out.csv')

dfsupp = pd.read_csv(foldername + 'df_supp.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "font.family": "serif"
})
styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown",
          "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown",
          "tab:pink", "tab:grey", "tab:olive"]

K_nums = np.array([0, 1, 3, 5, 10, 25, 50, 100])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 100
eps_tot = 30
M = 10
R = 1     # Total times we repeat experiment to estimate final probabilty
n = 2  # number of products
eps_nums = np.concatenate([np.logspace(-4, -0.5, 30), np.linspace(0.32, 2, 30)])
fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(13, 4.5))

styles = ["o", 's', "^", "v", "<", ">", "o", 's', "^", "v", "<", ">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown",
          "tab:pink", "tab:grey", "tab:olive",
          "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown",
          "tab:pink", "tab:grey", "tab:olive"]
j = 0
K_count = 7
ax1.plot(np.array(eps_nums)[::5],
         df.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
             eps_nums)]["Opt_val"][::5], color=colors[j],
         linestyle='-', marker=styles[j],
         label="Objective,$K={}$".format(K_nums[K_count]), alpha=0.7)
ax1.plot(np.array(eps_nums)[::5], df.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"][::5], color=colors[j],
    linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
j += 1
ax1.plot(np.array(eps_nums)[::5],
         dfout.sort_values(["K", "Epsilon"])[K_count*len(eps_nums):
                                             (K_count+1)*len(
             eps_nums)]["Opt_val"][::5], color=colors[j], linestyle='-',
         marker=styles[j], label="Objective, $K = {}$".format(
    K_nums[K_count]), alpha=0.7)
ax1.plot(np.array(eps_nums)[::5], dfout.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Eval_val"][::5], color=colors[j],
    linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
j += 1
ax1.plot(np.array(eps_nums)[::5],
         dfsupp.sort_values(["K", "Epsilon"])[
    K_count*len(eps_nums):(K_count+1)*len(
        eps_nums)]["Opt_val"][::5], color=colors[j],
    linestyle='-', marker=styles[j],
    label="Objective, $K = {}$".format(K_nums[K_count]), alpha=0.7)
ax1.plot(np.array(eps_nums)[::5],
         dfsupp.sort_values(["K", "Epsilon"])
         [K_count*len(eps_nums):(K_count+1)*len(
             eps_nums)]["Eval_val"][::5], color=colors[j],
         linestyle=':', label="Expectation, $K = {}$".format(K_nums[K_count]))
j += 1

ax1.set_xlabel(r"$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values, $K=100$")
ax1.set_xscale("log")

j = 0
ax21.plot(1 - df.sort_values(["K", "Epsilon"])
          [K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
          ["satisfy"][0:-1:1], df.sort_values(["K", "Epsilon"])[
    K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
    ["Opt_val"][0:-1:1], linestyle='-',
    label="ROB-MRO", marker=styles[j], alpha=0.7)
j += 1
ax21.plot(1 - dfout.sort_values(
    ["K", "Epsilon"])[K_count*len(eps_nums):
                      (K_count+1)*len(eps_nums)]
    ["satisfy"][0:-1:1], dfout.sort_values(["K", "Epsilon"])[
    K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
    ["Opt_val"][0:-1:1], linestyle='-', label="MRO",
    xsmarker=styles[j], alpha=0.7)
j += 1
ax21.plot(1 - dfout.sort_values(["K", "Epsilon"])
          [K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
          ["satisfy"][0:-1:1], dfsupp.sort_values(["K", "Epsilon"])[
    K_count*len(eps_nums):(K_count+1)*len(eps_nums)]
    ["Opt_val"][0:-1:1], linestyle='-',
    label="AUG-MRO", marker=styles[j], alpha=0.7)
j += 1

ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
# ax21.set_ylim([-35, -10])
ax21.legend(bbox_to_anchor=(1.4, 0.75), fontsize=14)
plt.tight_layout()
# plt.savefig(foldername + "newscomp100")
plt.savefig(foldername + "newscomp100.pdf")
plt.show()
