import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

with open("output/xstar_measles.txt") as f:
    lines = f.readlines()
    xdata = [line.split()[0] for line in lines]

x_measles = np.zeros(len(xdata))

for i in range(len(xdata)):
    x_measles[i] = float(xdata[i])

df1 = pd.read_table("output/measles_outliers.txt",delimiter=" ",header=None,skiprows=1)
df2 = pd.read_table("output/measles_only_outliers.txt",delimiter=" ",header=None,skiprows=0)

t = np.linspace(0,70,1000)
sol_measles = np.array([0.197,0.287,0.021])

plt.plot(df1[0].values,df1[1].values,"o")
plt.plot(df2[0].values,df2[1].values,'ro',mfc='none',ms=10)
plt.plot(t,F(t,*x_measles),label="OVO")
plt.plot(t,F(t,*sol_measles),label="Least Squares")
plt.legend()
plt.show()


plt.plot(t,l(t,*x_measles),label="OVO")
plt.plot(t,l(t,*sol_measles),label="Least Squares")
plt.legend()
plt.show()

