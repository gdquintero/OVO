import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models

# def plot_solutions(i):
#     ind = 2*i-1
#     plt.plot(df_sero_outliers[ind-1].values,df_sero_outliers[ind].values,"o")
#     plt.plot(df_only_outliers[ind-1].values,df_only_outliers[ind].values,'ro',mfc='none',ms=10)
#     plt.plot(t,F(t,*solutions_farrington[i-1]),label="Farrington")
#     plt.plot(t,F(t,*df_solutions_ovo.iloc[i-1].values),label="OVO")
#     plt.plot(t,F(t,*df_solutions_ls.iloc[i-1].values),label="Least Squares")
#     plt.legend()
#     plt.show()
#     plt.plot(t,l(t,*solutions_farrington[i-1]),label="Farrington")
#     plt.plot(t,l(t,*df_solutions_ovo.iloc[i-1].values),label="OVO")
#     plt.plot(t,l(t,*df_solutions_ls.iloc[i-1].values),label="Least Squares")
#     plt.legend()
#     plt.show()

df_seropositives= pd.read_table("output/seropositives.txt",delimiter=" ",header=None,skiprows=1)
df_solutions_ovo = pd.read_table("output/solutions_ovo.txt",delimiter=" ",header=None,skiprows=0)
df_solutions_ls  = pd.read_table("output/solutions_ls.txt",delimiter=" ",header=None,skiprows=0)

solutions_farrington = np.array([
    [0.197,0.287,0.021],
    [0.156,0.250,0.0],
    [0.0628,0.178,0.020]
])

t = np.linspace(0,70,1000)

plt.plot(df_seropositives[0].values,df_seropositives[1].values,"o")
plt.plot(t,models.F(t,*solutions_farrington[0]),label="Farrington")
plt.plot(t,models.F(t,*df_solutions_ovo.iloc[0].values),label="OVO")
plt.show()

# Plotamos las soluciones 1:Measles, 2:Mumps, 3:Rubella
# plot_solutions(1)
# plot_solutions(2)
# plot_solutions(3)
