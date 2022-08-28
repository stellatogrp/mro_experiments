import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

dftemp = pd.read_csv(foldername + 'df.csv')

plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})

N_tot = 60
m = 10
R = 30
K_nums = [1,2,3,4,5,10,30,60]
eps_nums = [0.01, 0.015, 0.023, 0.036, 0.055, 0.085, 0.13, 0.20, 0.30, 0.5, 0.7,1, 1.2, 1.4, 1.43, 1.47, 1.51, 1.55, 1.58, 1.62, 1.66, 1.7, 1.73, 1.77, 1.81, 1.85, 1.88, 1.92, 1.96, 2, 2.02, 2.07, 2.11, 2.15, 2.18, 2.22, 2.26, 2.3, 2.5, 2.7,3,4,9,10]
#eps_nums = np.concatenate((np.logspace(-2,0.4,20), np.array([3,4,7,9,10])))


styles = ["o",'s',"^","v","<",">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive","tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,1,3,4,7]:
    plt.plot((np.sort(eps_nums))[:-1], dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][:-1], linestyle='-', marker=styles[j], label="$K = {}$".format(K_nums[K_count]),alpha = 0.7)
    plt.plot((np.sort(eps_nums))[:-1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][:-1],color = colors[j], linestyle=':')
    j+=1
plt.xlabel("$\epsilon$")
plt.title("In-sample objective and %\n out-of-sample expected values")
plt.legend()
#plt.savefig("objectives.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,1,3,4,7]:
    plt.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K = {}$".format(K_nums[K_count]),marker = styles[j], alpha = 0.7)
    j += 1
plt.xlabel(r"$\beta$ (probability of constraint violation)")
plt.xlim([-0.025,10**(-0.25)])
#plt.ylim([-230,-220])
#plt.title("Objective value")
plt.legend()
#plt.savefig("constraint_satisfaction.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0

for i in [0,15,25,32]:
    gnval = np.array(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"]- gnval)
    plt.plot(K_nums,dif,label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color = colors[j])
    plt.plot(K_nums,(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["bound"]), linestyle='--', color = colors[j],label = "Upper bound")
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.yscale("log")
plt.title(r"$\bar{g}^K - \bar{g}^N$")
plt.legend()
#plt.savefig("upper_bound_diff.pdf")
plt.show()


plt.figure(figsize=(10, 6))
j = 0
for i in [0,15,25,32]:
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend()
plt.savefig("time.pdf")
plt.show()


import matplotlib.gridspec as gridspec

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(14, 4.5))

#gs = gridspec.GridSpec(4, 4)
#ax1 = plt.subplot(gs[:2, :2])
#ax21 = plt.subplot(gs[:2, 2:])
#ax31 = plt.subplot(gs[2:4, :2])
#ax4 = plt.subplot(gs[2:4, 2:])
styles = ["o",'s',"^","v","<",">"]
j = 0
for K_count in [0,1,3,4,7]:
    ax1.plot(np.array(eps_nums), dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]),alpha = 0.7)
    ax1.plot(np.array(eps_nums),dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"],color = colors[j], linestyle=':', label = "Expectation, $K = {}$".format(K_nums[K_count]))
    j+=1
ax1.set_xlabel("$\epsilon$")
#ax1.set_ylim([-330,-50])
#ax2.set_ylim([-330,-50])
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
#ax2.set_ylabel("Out-of-sample expected value")
#plt.ylim([objs2[0]-1, objs2[0]+1])
#ax1.legend(fontsize = 13, loc = "lower right")

j = 0
for K_count in [0,1,3,4,7]:
    ax21.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K = {}$".format(K_nums[K_count]),marker = styles[j], alpha = 0.7)
    j += 1
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_xlim([-0.025,10**(-0.25)])
ax21.set_ylim([-150,-130])
#plt.xscale("log")
#plt.yscale("log")
ax21.legend(bbox_to_anchor=(1, 0.35),fontsize = 14)
plt.tight_layout()
plt.savefig("5modestop.pdf")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
for i in [0,15,25, 32]:
    gnval = np.array(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"]- gnval)
    ax31.plot(K_nums,dif,label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color = colors[j])
    ax31.plot(K_nums,(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["bound"]), linestyle='--', color = colors[j],label = "Upper bound")
    j+=1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_ylim([10**-2,2*10**2])
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - \bar{g}^N$")

#ax32.legend()
#ax31.legend(loc = "lower right")

j = 0
for i in [0,15,25,32]:
    ax4.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(eps_nums[i], 3)))
    j+=1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc = "lower right", bbox_to_anchor=(1.33, 0.2), fontsize = 14)

plt.tight_layout()
plt.savefig("5modesbot.pdf")
plt.show()


