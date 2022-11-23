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

def plot_solutions(i):
    ind = 2*i-1
    plt.plot(df_sero_outliers[ind-1].values,df_sero_outliers[ind].values,"o")
    plt.plot(df_only_outliers[ind-1].values,df_only_outliers[ind].values,'ro',mfc='none',ms=10)
    plt.show()

df_sero_outliers = pd.read_table("output/seropositives_outliers.txt",delimiter=" ",header=None,skiprows=2)
df_only_outliers = pd.read_table("output/seropositives_only_outliers.txt",delimiter=" ",header=None,skiprows=0)
df_solutions_ovo = pd.read_table("output/solutions_ovo.txt",delimiter=" ",header=None,skiprows=0)
df_solutions_ls  = pd.read_table("output/solutions_ls.txt",delimiter=" ",header=None,skiprows=0)

solutions_farrington = np.array([
    [0.197,0.287,0.021],
    [0.156,0.250,0.0],
    [0.0628,0.178,0.020]
])

t = np.linspace(0,70,1000)

plot_solutions(1)
plot_solutions(2)
plot_solutions(3)

# plt.plot(t,F(t,*x_measles),label="OVO")
# plt.plot(t,F(t,*sol_measles),label="Farrington")
# plt.plot(t,F(t,*df3.values),label="Least Squares")
# plt.legend()

# plt.plot(t,l(t,*x_measles),label="OVO")
# plt.plot(t,l(t,*sol_measles),label="Farrington")
# plt.plot(t,l(t,*df3.values),label="Least Squares")
# plt.legend()
# plt.show()

