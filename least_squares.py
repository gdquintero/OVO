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

df = pd.read_table("output/measles_outliers.txt",delimiter=" ",header=None,skiprows=1)
popt, pcov = curve_fit(F,df[0].values,df[1].values,bounds=(0.,np.inf * np.ones(3)))

with open("output/ls_measles.txt","w") as f:
    f.write("%f\n" % popt[0])
    f.write("%f\n" % popt[1])
    f.write("%f" % popt[2])