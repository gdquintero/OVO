import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

seropositive = np.array([
    0.207,0.301,0.409,0.589,0.757,0.669,0.797,0.818,0.866,0.859,0.908,0.923,0.889,0.936,0.889,\
    0.898,0.959,0.957,0.937,0.918,0.939,0.967,0.973,0.943,0.967,0.946,0.961,0.968,0.968
])

solution = np.array([0.197,0.287,0.021])
t = np.linspace(0,70,1000)

# sns.boxplot(x=seropositive)
# plt.show()

outliers = []
outliers.append(F(16,*solution) - 0.3)
outliers.append(F(18,*solution) - 0.3)
outliers.append(F(20,*solution) - 0.3)
outliers.append(F(22,*solution) - 0.3)
outliers.append(F(24,*solution) - 0.3)

age = np.insert(age,15,16)
seropositive = np.insert(seropositive,15,outliers[0])

# age = np.insert(age,26,36)
# seropositive = np.insert(seropositive,26,outlier2)

# age = np.insert(age,-1,60)
# seropositive = np.insert(seropositive,-1,outlier3)

samples = len(age)

with open("output/measles.txt","w") as f:
    f.write("%i\n" % samples)
    for i in range(samples):
        f.write("%i %f\n" % (age[i],seropositive[i]))

plt.plot(age,seropositive,"o")
# plt.plot(t,F(t,*solution))
# plt.plot(t,l(t,*solution))
plt.show()
