import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl

dftemp = pd.read_csv('df.csv')
plt.rcParams.update({
"text.usetex":True,
"font.size":18,
"font.family": "serif"
})


styles = ["o",'s',"^","v","<",">"]
colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive","tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,2,4,6,8,9]:
    plt.plot((np.sort(eps_nums))[:-1], dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][:-1], linestyle='-', marker=styles[j], label="Objective, $K = {}$".format(K_nums[K_count]),alpha = 0.7)
    plt.plot((np.sort(eps_nums))[:-1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Eval_val"][:-1],color = colors[j], linestyle=':', label = "Expectation, $K = {}$".format(K_nums[K_count]))
    j+=1
plt.xlabel("$\epsilon$")
plt.title("In-sample objective and %\n out-of-sample expected values")
plt.xscale("log")
plt.legend()
plt.savefig("objectives.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for K_count in [0,2,5,8,9]:
    plt.plot(1 - dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"][0:-1:1],dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"][0:-1:1],linestyle='-',label="$K = {}$".format(K_nums[K_count]),marker = styles[j], alpha = 0.7)
    j += 1
plt.xlabel(r"$\beta$ (probability of constraint violation)")
plt.title("Objective value")
plt.legend()
plt.savefig("constraint_satisfaction.pdf")
plt.show()

plt.figure(figsize=(10, 6))
j = 0
for i in [2,6,9,11,15]:
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
for i in [2,6,9,11,15]:
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle="-", marker=styles[j],color = colors[j], label="$\epsilon = {}$".format(np.round(np.sort(eps_nums)[i], 5)))
    j+=1
plt.xlabel("$K$ (number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend()
plt.savefig("time.pdf")
plt.show()
