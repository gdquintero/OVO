import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models

# 'ro',mfc='none',ms=10)

def plot_solutions(i,df_seropositives,df_sol_ovo,sero_outliers):
    t = np.linspace(0,70,1000)
    plt.plot(df_seropositives[0].values,df_seropositives[i].values,"o")
    plt.plot(t,models.F(t,*solutions_farrington[i-1]),label="Farrington")
    plt.plot(t,models.F(t,*df_sol_ovo.iloc[i-1].values),label="OVO")
    plt.plot(sero_outliers[0],sero_outliers[1],'ro',mfc='none',ms=10)
    plt.legend()
    plt.show()

df_seropositives = pd.read_table("output/seropositives.txt",delimiter=" ",header=None,skiprows=1)
df_solutions_ovo = pd.read_table("output/solutions_ovo.txt",delimiter=" ",header=None,skiprows=0)
df_solutions_ls  = pd.read_table("output/solutions_ls.txt",delimiter=" ",header=None,skiprows=0)

with open("output/outliers.txt") as f:
    lines = f.readlines()
    xdata = [line.split()[0] for line in lines]

noutliers = int(xdata[0])

outliers = np.empty(3*noutliers,dtype=int)

for i in range(3*noutliers):
    outliers[i] = int(xdata[i+1])

measles_outliers = np.empty((2,noutliers))
mumps_outliers   = np.empty((2,noutliers))
rubella_outliers = np.empty((2,noutliers))

for i in range(noutliers):
    measles_outliers[0,i] = df_seropositives[0].values[outliers[i]-1]
    measles_outliers[1,i] = df_seropositives[1].values[outliers[i]-1]

    mumps_outliers[0,i] = df_seropositives[0].values[outliers[noutliers+i]-1]
    mumps_outliers[1,i] = df_seropositives[2].values[outliers[noutliers+i]-1]

    rubella_outliers[0,i] = df_seropositives[0].values[outliers[2*noutliers+i]-1]
    rubella_outliers[1,i] = df_seropositives[3].values[outliers[2*noutliers+i]-1]


solutions_farrington = np.array([
    [0.197,0.287,0.021],
    [0.156,0.250,0.0],
    [0.0628,0.178,0.020]
])

# Plotamos las soluciones 1:Measles, 2:Mumps, 3:Rubella
plot_solutions(1,df_seropositives,df_solutions_ovo,measles_outliers)
plot_solutions(2,df_seropositives,df_solutions_ovo,mumps_outliers)
plot_solutions(3,df_seropositives,df_solutions_ovo,rubella_outliers)

