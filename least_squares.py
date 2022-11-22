import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def F(t,a,b,c):
    aux = (a/b) * t * np.exp(-b * t)
    aux = aux + (1.0/b) * ((a/b) - c) * (np.exp(-b * t) - 1.0)
    aux = aux - c * t
    aux = 1.0 - np.exp(aux)

    return aux

df = pd.read_table("output/seropositives_outliers.txt",delimiter=" ",header=None,skiprows=2)

popt_measles, pcov_measles  = curve_fit(F,df[0].values,df[1].values,bounds=(0.,np.inf * np.ones(3)))
popt_mumps, pcov_mumps      = curve_fit(F,df[0].values,df[2].values,bounds=(0.,np.inf * np.ones(3)))
popt_rubella, pcov_rubella  = curve_fit(F,df[0].values,df[3].values,bounds=(0.,np.inf * np.ones(3)))

with open("output/solutions_ls.txt","w") as f:
    f.write("%f %f %f\n" % (popt_measles[0],popt_measles[1],popt_measles[2]))
    f.write("%f %f %f\n" % (popt_mumps[0],popt_mumps[1],popt_mumps[2]))
    f.write("%f %f %f\n" % (popt_rubella[0],popt_rubella[1],popt_rubella[2]))
    
