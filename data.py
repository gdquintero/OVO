import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def l(t,a,b,c):
    aux = a * t - c
    aux = aux * np.exp(-b * t)
    aux = aux + c
    return aux

def F(t,a,b,c):
    aux = (a/b) * t * np.exp(-b * t)
    aux = aux + (1.0/b) * ((a/b) - c) * (np.exp(-b * t) - 1.0)
    aux = aux - c * t
    aux = 1.0 - np.exp(aux)
    return aux

age = np.array([
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,21,23,25,27,29,31,33,35,40,45,55,65
])

sero_measles = np.array([
    0.207,0.301,0.409,0.589,0.757,0.669,0.797,0.818,0.866,0.859,0.908,0.923,0.889,0.936,0.889,\
    0.898,0.959,0.957,0.937,0.918,0.939,0.967,0.973,0.943,0.967,0.946,0.961,0.968,0.968
])

sero_mumps = np.array([
    0.115,0.147,0.389,0.516,0.669,0.768,0.786,0.798,0.878,0.861,0.844,0.881,0.895,0.882,0.869,\
    0.895,0.911,0.920,0.915,0.950,0.909,0.873,0.880,0.915,0.906,0.933,0.917,0.898,0.839
])

sero_rubella = np.array([
    0.126,0.171,0.184,0.286,0.400,0.503,0.524,0.634,0.742,0.664,0.735,0.815,0.768,0.842,0.760,\
    0.869,0.844,0.852,0.907,0.935,0.921,0.896,0.890,0.949,0.899,0.955,0.937,0.933,0.917
])

sol_measles = np.array([0.197,0.287,0.021])
sol_mumps = np.array([0.156,0.250,0.0])
sol_rubella = np.array([0.0628,0.178,0.020])

t = np.linspace(0,70,1000)

# plt.ylim([0,1.1])
# plt.plot(age,sero_measles,"o",ls=":")
# plt.savefig("sero_measles.pdf",bbox_inches = "tight") 
# plt.close()

# plt.ylim([0,1.1])
# plt.plot(age,sero_mumps,"o",ls=":")
# plt.savefig("sero_mumps.pdf",bbox_inches = "tight") 
# plt.close()

# plt.ylim([0,1.1])
# plt.plot(age,sero_rubella,"o",ls=":") 
# plt.savefig("sero_rubella.pdf",bbox_inches = "tight")  
# plt.show()

sns.boxplot(x=sero_measles)
plt.show()

outliers = np.empty((3,5))
outliers[0,0] = F(16,*sol_measles) - random.uniform(0.2,0.3)
outliers[0,1] = F(18,*sol_measles) - random.uniform(0.2,0.3)
outliers[0,2] = F(20,*sol_measles) - random.uniform(0.2,0.3)
outliers[0,3] = F(22,*sol_measles) - random.uniform(0.2,0.3)
outliers[0,4] = F(24,*sol_measles) - random.uniform(0.2,0.3)

outliers[1,0] = F(16,*sol_mumps) - random.uniform(0.2,0.3)
outliers[1,1] = F(18,*sol_mumps) - random.uniform(0.2,0.3)
outliers[1,2] = F(20,*sol_mumps) - random.uniform(0.2,0.3)
outliers[1,3] = F(22,*sol_mumps) - random.uniform(0.2,0.3)
outliers[1,4] = F(24,*sol_mumps) - random.uniform(0.2,0.3)

outliers[2,0] = F(16,*sol_rubella) - random.uniform(0.2,0.3)
outliers[2,1] = F(18,*sol_rubella) - random.uniform(0.2,0.3)
outliers[2,2] = F(20,*sol_rubella) - random.uniform(0.2,0.3)
outliers[2,3] = F(22,*sol_rubella) - random.uniform(0.2,0.3)
outliers[2,4] = F(24,*sol_rubella) - random.uniform(0.2,0.3)

age = np.insert(age,15,16)
age = np.insert(age,17,18)
age = np.insert(age,19,20)
age = np.insert(age,21,22)
age = np.insert(age,23,24)

sero_measles = np.insert(sero_measles,15,outliers[0,0])
sero_measles = np.insert(sero_measles,17,outliers[0,1])
sero_measles = np.insert(sero_measles,19,outliers[0,2])
sero_measles = np.insert(sero_measles,21,outliers[0,3])
sero_measles = np.insert(sero_measles,23,outliers[0,4])

sero_mumps = np.insert(sero_mumps,15,outliers[1,0])
sero_mumps = np.insert(sero_mumps,17,outliers[1,1])
sero_mumps = np.insert(sero_mumps,19,outliers[1,2])
sero_mumps = np.insert(sero_mumps,21,outliers[1,3])
sero_mumps = np.insert(sero_mumps,23,outliers[1,4])

sero_rubella = np.insert(sero_rubella,15,outliers[2,0])
sero_rubella = np.insert(sero_rubella,17,outliers[2,1])
sero_rubella = np.insert(sero_rubella,19,outliers[2,2])
sero_rubella = np.insert(sero_rubella,21,outliers[2,3])
sero_rubella = np.insert(sero_rubella,23,outliers[2,4])

samples = len(age)

with open("output/measles.txt","w") as f:
    f.write("%i\n" % samples)
    for i in range(samples):
        f.write("%i %f\n" % (age[i],sero_measles[i]))

with open("output/mumps.txt","w") as f:
    f.write("%i\n" % samples)
    for i in range(samples):
        f.write("%i %f\n" % (age[i],sero_mumps[i]))

with open("output/rubella.txt","w") as f:
    f.write("%i\n" % samples)
    for i in range(samples):
        f.write("%i %f\n" % (age[i],sero_rubella[i]))

# plt.plot(age,sero_measles,"o")
# plt.plot(t,F(t,*sol_measles))
# plt.show()

# plt.plot(age,sero_mumps,"o")
# plt.plot(t,F(t,*sol_mumps))
# plt.show()

# plt.plot(age,sero_rubella,"o")
# plt.plot(t,F(t,*sol_rubella))
# plt.show()

# plt.plot(t,l(t,*sol_measles))
# plt.show()
