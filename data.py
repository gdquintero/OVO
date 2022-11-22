import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_seropositive(sero,x,y,outliers):
    plt.ylim([0,1.1])
    t = np.linspace(0,70,1000)

    plt.plot(x,y,"o",ls=":")
    plt.plot(outliers[0,:],outliers[1,:],'ro',mfc='none',ms=10)

    if sero == "measles":
        plt.savefig("sero_measles.pdf",bbox_inches = "tight") 
    elif sero == "mumps":
        plt.savefig("sero_mumps.pdf",bbox_inches = "tight")
    else:
        plt.savefig("sero_rubella.pdf",bbox_inches = "tight")

    plt.show()

def pollute_data(age,sero,n_outliers):
    inf = 0.2
    sup = 0.3
    outliers = np.empty((2,n_outliers))
    pollute_age = np.copy(age)
    pollute_sero = np.copy(sero)
    
    # Lista con posibles valores para las edades
    free_ages = []

    for i in range(1,pollute_age[-1]):
        if i not in age:
            free_ages.append(i)

    for i in range(n_outliers):
        # Escogemos una edad de forma aleatoria
        new_age = random.sample(free_ages,1)[0]

        # Encontramos el indice de la edad mas cercana a la nueva
        ind = np.where(pollute_age < new_age)[0][-1]

        # Insertamos la nueva edad y su seropositivo correspondiente
        pollute_age = np.insert(pollute_age,ind+1,new_age)
        outlier = F(pollute_age[ind],*sol_measles) - random.uniform(inf,sup)
        pollute_sero = np.insert(pollute_sero,ind+1,outlier)

        outliers[0,i] = new_age 
        outliers[1,i] = outlier

        # Eliminamos la edad recien ingresada en age
        free_ages.remove(new_age)

    return pollute_age,pollute_sero,outliers

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


# Solucion exacta (del paper farrington)
sol_measles = np.array([0.197,0.287,0.021])
sol_mumps = np.array([0.156,0.250,0.0])
sol_rubella = np.array([0.0628,0.178,0.020])

# Contaminamos los datos con n_outliers valores atipicos
n_outliers = 5
pollute_age_measles,pollute_sero_measles,outliers_measles = pollute_data(age,sero_measles,n_outliers)
pollute_age_mumps,pollute_sero_mumps,outliers_mumps = pollute_data(age,sero_mumps,n_outliers)
pollute_age_rubella,pollute_sero_rubella,outliers_rubella = pollute_data(age,sero_rubella,n_outliers)

samples = len(pollute_age_measles)

with open("output/measles_outliers.txt","w") as f:
    f.write("%i\n" % samples)
    f.write("%i\n" % n_outliers)
    for i in range(samples):
        f.write("%i %f\n" % (pollute_age_measles[i],pollute_sero_measles[i]))

with open("output/mumps_outliers.txt","w") as f:
    f.write("%i\n" % samples)
    f.write("%i\n" % n_outliers)
    for i in range(samples):
        f.write("%i %f\n" % (pollute_age_mumps[i],pollute_sero_mumps[i]))

with open("output/rubella_outliers.txt","w") as f:
    f.write("%i\n" % samples)
    f.write("%i\n" % n_outliers)
    for i in range(samples):
        f.write("%i %f\n" % (pollute_age_rubella[i],pollute_sero_rubella[i]))

with open("output/measles_only_outliers.txt","w") as f:
    for i in range(n_outliers):
        f.write("%i %f\n" % (outliers_measles[0,i],outliers_measles[1,i]))

with open("output/mumps_only_outliers.txt","w") as f:
    for i in range(n_outliers):
        f.write("%i %f\n" % (outliers_mumps[0,i],outliers_mumps[1,i]))

with open("output/rubella_only_outliers.txt","w") as f:
    for i in range(n_outliers):
        f.write("%i %f\n" % (outliers_rubella[0,i],outliers_rubella[1,i]))

# Graficamos y guardamos 
plot_seropositive("measles",pollute_age_measles,pollute_sero_measles,outliers_measles)
plot_seropositive("mumps",pollute_age_mumps,pollute_sero_mumps,outliers_mumps)
plot_seropositive("rubella",pollute_age_rubella,pollute_sero_rubella,outliers_rubella)

