import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, default="/scratch/gpfs/iywang/mro_results/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername 

dftemp = pd.read_csv(foldername+ 'df.csv')
plt.rcParams.update({
"text.usetex":True,
"font.size":18,
"font.family": "serif"
})
N_tot = 90
m = 30
R = 30
K_nums = np.append([1,2,3,5,6,7,8,10],np.append(np.arange(20, int(N_tot/2)+1,10), N_tot))
eps_nums = np.append(np.logspace(-5.5,-4,20),np.logspace(-3.9,1,10))


styles = ["o",'s',"^","v","<",">"]
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
plt.savefig("objectives.pdf")
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
plt.savefig("constraint_satisfaction.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for i in [6,12,18,24,28]:
    gnval = np.array(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"])[-1]
    dif = (dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["Opt_val"]- gnval)
    plt.plot(K_nums,dif,label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)), linestyle='-', marker=styles[j], color = colors[j])
    plt.plot(K_nums,(dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["bound"]), linestyle='--', color = colors[j],label = "Upper bound")
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.yscale("log")
plt.title(r"$\bar{g}^K - \bar{g}^N$")
plt.legend()
plt.savefig("upper_bound_diff.pdf")
plt.show()


plt.figure(figsize=(10, 6))
j = 0
for i in [6,12,18,24,28]:
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend()
plt.savefig("time.pdf")
plt.show()
