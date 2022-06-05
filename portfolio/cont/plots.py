import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl

dftemp = pd.read_csv('dftemp.csv')

plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})

K_nums = np.array([1, 5, 50, 100, 300, 500, 800,900])
# np.array([1,10,20,50,100,500,1000]) 
K_tot = K_nums.size 
N_tot = 900
M = 15
R = 10        
m = 200
eps_min = -5    
eps_max = -3.5   
eps_nums = np.linspace(eps_min, eps_max, M)
eps_nums = 10**(eps_nums)
eps_tot = M


plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["Opt_val"], linestyle='-', marker = 'o', label="Objective, $K = {}$".format(K_nums[K_count]),alpha = 0.6)
plt.xlabel("$\epsilon^2$")
plt.xscale("log")
plt.title("In-sample objective value")
plt.legend(loc = "lower right")
plt.save("objectives.pdf")

plt.figure(figsize=(10, 6))
for K_count in np.arange(0,len(K_nums),1):
    plt.plot(eps_nums, dftemp.sort_values(["K","Epsilon"])[K_count*len(eps_nums):(K_count+1)*len(eps_nums)]["satisfy"], label="$K = {}$".format(K_nums[K_count]),linestyle='-', marker='o', alpha=0.5)
plt.xlabel("$\epsilon^2$")
plt.xscale("log")
plt.legend(loc = "lower right")
plt.title(r"$1-\beta$ (probability of constraint satisfaction)")
plt.savefig("constraint_satisfaction.pdf")

plt.figure(figsize=(10, 6))
for i in np.arange(0,len(eps_nums),3):
    plt.plot(K_nums, dftemp.sort_values(["Epsilon","K"])[i*len(K_nums):(i+1)*len(K_nums)]["solvetime"], linestyle='-', marker='o', label="$\epsilon = {}$".format(np.round(eps_nums[i], 5)))
plt.xlabel("$K$ (Number of clusters)")
plt.title("Time (s)")
plt.yscale("log")
plt.legend(loc = "lower right")
plt.savefig("time.pdf")


