import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername+'df.csv')

plt.rcParams.update({
"text.usetex":True,
"font.size":18,
"font.family": "serif"
})

K_nums = np.array([1, 5, 10, 50, 100])
K_tot = K_nums.size  # Total number of clusters we consider
N_tot = 100
M = 10
n = 10  # number of facilities
m = 50  # number of locations
eps_min = 5      # minimum epsilon we consider
eps_max = 30         # maximum epsilon we consider
eps_nums = np.linspace(eps_min, eps_max, M)
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