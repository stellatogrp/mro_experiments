import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername+'df.csv')
df = pd.read_csv(foldername+'df_all.csv')

plt.rcParams.update({
"text.usetex":True,
"font.size":18,
"font.family": "serif"
})

K_nums = np.array([1, 5, 10, 50, 100, 200])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 200
M = 10
n = 12  # number of facilities
m = 60  # number of locations
R = 10
eps_min = 1      # minimum epsilon we consider
eps_max = 10         # maximum epsilon we consider
eps_nums = np.linspace(eps_min, eps_max, M)**2
eps_tot = M

plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="$K = {}$".format(K_nums[K_count]),alpha = 0.6)
plt.xlabel("$\epsilon$")
plt.title("In-sample objective value")
plt.legend(loc = "lower right")
plt.savefig("objectives.pdf")

plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
plt.xlabel("$\epsilon$")
plt.legend(loc = "lower right")
plt.title(r"$1-\beta$ (probability of constraint satisfaction)(7)")
plt.savefig("constraint_satisfaction.pdf")

plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val1"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
plt.xlabel("$\epsilon$")
plt.legend(loc = "lower right")
plt.title(r"$1-\beta$ (probability of constraint satisfaction)(8)")
plt.savefig("constraint_satisfaction_strict.pdf")

plt.figure(figsize=(10, 6))
for i in np.arange(0,len(eps_nums),3):
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i]**0.5, 5)))
plt.xlabel("$K$ (Number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend(loc = "lower right")
plt.savefig("time.pdf")



plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    "font.family" : "serif"

})

import matplotlib.gridspec as gridspec

plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 4)

ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[:2, 2:])
ax3 = plt.subplot(gs[2:4, :2])
ax4 = plt.subplot(gs[2:4, 2:])

for K_count, K in enumerate(K_nums):
    ax1.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="$K = {}$".format(K_nums[K_count]),alpha = 0.6)
ax1.set_xlabel("$\epsilon$")
ax1.set_title("Objective value")


for K_count, K in enumerate(K_nums):
    ax2.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
ax2.set_xlabel("$\epsilon$")
ax2.set_title(r"$1-\beta$ (probability of constraint satisfaction)")

for K_count, K in enumerate(K_nums):
    ax3.plot(eps_nums**0.5, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val1"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
ax3.set_xlabel("$\epsilon$")
ax3.set_title(r"$1-\beta$ (probability of constraint satisfaction)")
ax3.legend()


labelprint = 1
for i in np.arange(0,len(eps_nums),3):
    if labelprint == 1:
        ax4.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.2, label = "0.25 to 0.75 quantiles")
    else:
        ax4.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.2)
    labelprint = 0
    ax4.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i]**0.5, 5)))

ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(fontsize = 13)

plt.tight_layout()
plt.savefig("facilitytop1.pdf")