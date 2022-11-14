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

sol_measles = np.array([0.197,0.287,0.021])
t = np.linspace(0,70,1000)

# sns.boxplot(x=sero_measles)
# plt.show()

outliers = []
outliers.append(F(16,*sol_measles) - random.uniform(0.2,0.3))
outliers.append(F(18,*sol_measles) - random.uniform(0.2,0.3))
outliers.append(F(20,*sol_measles) - random.uniform(0.2,0.3))
outliers.append(F(22,*sol_measles) - random.uniform(0.2,0.3))
outliers.append(F(24,*sol_measles) - random.uniform(0.2,0.3))

age = np.insert(age,15,16)
sero_measles = np.insert(sero_measles,15,outliers[0])

age = np.insert(age,17,18)
sero_measles = np.insert(sero_measles,17,outliers[1])

age = np.insert(age,19,20)
sero_measles = np.insert(sero_measles,19,outliers[2])

age = np.insert(age,21,22)
sero_measles = np.insert(sero_measles,21,outliers[3])

age = np.insert(age,23,24)
sero_measles = np.insert(sero_measles,23,outliers[4])

samples = len(age)

with open("output/measles.txt","w") as f:
    f.write("%i\n" % samples)
    for i in range(samples):
        f.write("%i %f\n" % (age[i],sero_measles[i]))

plt.plot(age,sero_measles,"o")
plt.plot(t,F(t,*sol_measles))
plt.show()

plt.plot(t,l(t,*sol_measles))
plt.show()
