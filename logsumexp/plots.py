import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername 

dftemp = pd.read_csv(foldername+ 'df_final.csv')

plt.rcParams.update({
"text.usetex":True,
"font.size":18,
"font.family": "serif"
})
N_tot = 90
m = 30
K_nums = np.array([1,2,3,5,6,7,8,10,20,40,90,9999])
eps_nums = np.append(np.logspace(-5.2,-4,15),np.logspace(-3.9,1,10))
#eps_nums = np.append(np.logspace(-4.5,-3.5,10),np.logspace(-3.48,1,10))


styles = ["o",'s',"^","v","<",">","o",'s',"^","v","<",">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive","tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,1,2,10,11]:
    plt.plot((np.sort(eps_nums))[:-1], dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][:-1], linestyle='-', marker=styles[j], label="$K = {}$".format(K_nums[K_count]),alpha = 0.7)
    plt.plot((np.sort(eps_nums))[:-1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][:-1],color = colors[j], linestyle=':')
    j+=1
plt.xlabel("$\epsilon$")
plt.title("In-sample objective and %\n out-of-sample expected values")
plt.xscale("log")
plt.legend()
#plt.savefig("objectives.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,1,2,10,11]:
    plt.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K = {}$".format(K_nums[K_count]),marker = styles[j], alpha = 0.7)
    j += 1
plt.xlabel(r"$\beta$ (probability of constraint violation)")
plt.ylim([2.7,3])
plt.title("Objective value")
plt.legend()
#plt.savefig("constraint_satisfaction.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for i in np.arange(0,len(eps_nums),5):
    gnval = np.array(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"]- gnval)
    plt.plot(K_nums,dif,label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color = colors[j])
    plt.plot(K_nums,(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["bound"]), linestyle='--', color = colors[j],label = "Upper bound")
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.yscale("log")
plt.title(r"$\bar{g}^K - g^N$")
plt.legend()
#plt.savefig("upper_bound_diff.pdf")
plt.show()


plt.figure(figsize=(10, 6))
j = 0
for i in np.arange(0,len(eps_nums),5):
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend()
#plt.savefig("time.pdf")
plt.show()

import matplotlib.gridspec as gridspec
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive","tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]

fig, (ax1, ax21) = plt.subplots(1, 2, figsize=(14, 4.5))

styles = ["o",'s',"^","v","<",">","o",'s',"^","v","<",">"]
j = 0
for K_count in [0,1,2,10,len(K_nums)-1]:
    ax1.plot((np.sort(eps_nums))[10:], dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][10:], linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]),alpha = 0.7)
    ax1.plot((np.sort(eps_nums))[10:],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][10:],color = colors[j], linestyle=':', label = "Expectation, $K = {}$".format(K_nums[K_count]))
    j+=1
ax1.set_xlabel("$\epsilon$")
ax1.set_title("In-sample objective and %\n out-of-sample expected values")
ax1.set_xscale("log")

j = 0
for K_count in [0,1,2,10]:
    ax21.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K = {}$".format(K_nums[K_count]),marker = styles[j], alpha = 0.7)
    j += 1
K_count = len(K_nums)-1
ax21.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K=90^*$",marker = styles[j], alpha = 0.7)
ax21.set_xlabel(r"$\beta$ (probability of constraint violation)")
ax21.set_title("Objective value")
ax21.set_ylim([2.735,2.755])
ax21.legend(bbox_to_anchor=(1.3, .8),fontsize = 14)
plt.tight_layout()
plt.savefig("logtop.pdf")
plt.savefig("logtop.png")
plt.show()


fig, (ax31, ax4) = plt.subplots(1, 2, figsize=(14, 4.5))

j = 0
#or i in np.arange(13,len(eps_nums)-2,4):
for i in [13,18,20,21]:
    gnval = np.array(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-2]
    dif = (dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)-2]["Opt_val"]- gnval)
    ax31.plot(K_nums[:-2],dif,label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color = colors[j])
    ax31.plot(K_nums[:-2],(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)-2]["bound"]), linestyle='--', color = colors[j],label = "Upper bound")
    j+=1
ax31.set_xlabel("$K$ (number of clusters)")
ax31.set_yscale("log")
ax31.set_title(r"$\bar{g}^K - g^N$")

j = 0
for i in [13,18,20,21]:
#for i in np.arange(13,len(eps_nums)-2,4):
    ax4.plot(K_nums[:-1], dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"][:-1], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j+=1
ax4.set_xlabel("$K$ (number of clusters)")
ax4.set_title("Time (s)")
ax4.set_yscale("log")
ax4.legend(loc = "lower right", bbox_to_anchor=(1.38, 0.2), fontsize = 14)
#ax4.legend()
plt.tight_layout()
plt.savefig("logbot.pdf")
plt.savefig("logbot.png")
plt.show()
