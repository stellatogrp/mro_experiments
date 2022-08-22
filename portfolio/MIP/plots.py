import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername + 'df.csv')
df = pd.read_csv(foldername + 'df_all.csv')


plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})

K_nums = np.array([1, 5, 50, 100, 150, 300])
N_tot = 300
M = 10
R = 12           
m = 50
eps_min = -5  
eps_max = -3.5 
#eps_min = -5   
#eps_max = -3.9       
eps_nums = np.linspace(eps_min, eps_max, M)
eps_nums = (10**(eps_nums))**0.5

plt.figure(figsize=(10, 6))
labelprint =1
for K_count in np.arange(0,len(K_nums),1):
    if labelprint == 1:
        plt.fill_between(eps_nums, np.quantile([df[df["R"]==r].sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"] for r in range(R)],0.75,axis = 0),label="0.25 to 0.75 quantiles", alpha = 0.05)
    else: 
        plt.fill_between(eps_nums, np.quantile([df[df["R"]==r].sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"] for r in range(R)],0.75,axis = 0), alpha = 0.05)
    labelprint = 0
    plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="$K = {}$".format(K_nums[K_count]),alpha = 0.6)
plt.xlabel("$\epsilon$")
plt.xscale("log")
plt.title("In-sample objective value")
plt.legend(loc = "lower right")
plt.savefig("objectives.pdf")

plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
plt.xlabel("$\epsilon$")
plt.xscale("log")
plt.legend(loc = "lower right")
plt.title(r"$1-\beta$ (probability of constraint satisfaction)")
plt.savefig("constraint_satisfaction.pdf")


plt.figure(figsize=(10, 6))
labelprint = 1
for i in np.arange(5,len(eps_nums),1):
    if labelprint == 1:
        plt.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.1, label = "0.25 to 0.75 quantiles")
    else:
        plt.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.1)
    labelprint = 0
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
plt.xlabel("$K$ (Number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend(loc = "lower right")
plt.savefig("time.pdf")




import matplotlib.gridspec as gridspec

plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 4)

ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[:2, 2:])
ax3 = plt.subplot(gs[2:4, :4])

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    "font.family" : "serif"
})

for K_count in [0,1,2,4,5]:
    ax1.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="$K = {}$".format(K_nums[K_count]),alpha = 0.6)
ax1.set_xlabel("$\epsilon$")
ax1.set_xscale("log")
ax1.set_title("In-sample objective value")
#ax1.legend(loc = "lower right")

for K_count in [0,1,2,4,5]:
    ax2.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
ax2.set_xlabel("$\epsilon$")
ax2.set_xscale("log")
ax2.legend(loc = "lower right")
ax2.set_title(r"$1-\beta$ (probability of constraint satisfaction)")

labelprint = 1
for i in np.arange(5,len(eps_nums)-2,1):
    if labelprint == 1:
        ax3.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.1, label = "0.25 to 0.75 quantiles")
    else:
        ax3.fill_between(K_nums, np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"] for r in range(2)],0.75,axis = 0), alpha = 0.1)
    labelprint = 0
    ax3.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
i = len(eps_nums)-2
ax3.fill_between(K_nums[0:2], np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2] for r in range(2)],0.75,axis = 0), alpha = 0.1)
ax3.plot(K_nums[0:2], dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
i = len(eps_nums)-1
ax3.fill_between(K_nums[0:2], np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2] for r in range(R)],0.25,axis = 0), np.quantile([df[df["R"]==r].sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2] for r in range(2)],0.75,axis = 0), alpha = 0.1)
ax3.plot(K_nums[0:2], dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][0:2], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5))) 
ax3.set_xlabel("$K$ (Number of clusters)")
ax3.set_ylabel("Time (s)")
ax3.set_yscale("log")
ax3.legend(loc = "lower right")

plt.tight_layout()
plt.savefig("portMIP1.pdf")
