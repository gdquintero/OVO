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

n_outliers = 11
inf = 0.1
sup = 0.3

outliers = np.empty((2,n_outliers))

# Lista con posibles valores para las edades
free_ages = []

for i in range(1,age[-1]):
    if i not in age:
        free_ages.append(i)

for i in range(n_outliers):
    # Escogemos una edad de forma aleatoria
    new_age = random.sample(free_ages,1)[0]

    # Encontramos el indice de la edad mas cercana a la nueva
    ind = np.where(age < new_age)[0][-1]

    # Insertamos la nueva edad y su seropositivo correspondiente
    age = np.insert(age,ind+1,new_age)
    outlier = F(age[ind],*sol_measles) - random.uniform(inf,sup)
    sero_measles = np.insert(sero_measles,ind+1,outlier)

    outliers[0,i] = new_age 
    outliers[1,i] = outlier

    # Eliminamos la edad recien ingresada en age
    free_ages.remove(new_age)

# plt.plot(age,sero_measles,"o")
# plt.plot(outliers[0,:],outliers[1,:],'ro',mfc='none',ms=10)
# plt.show()


# Graficar cada una de las proporciones de seropositivos
plt.ylim([0,1.1])
plt.plot(age,sero_measles,"o",ls=":")
plt.plot(outliers[0,:],outliers[1,:],'ro',mfc='none',ms=10)
# plt.savefig("sero_measles.pdf",bbox_inches = "tight") 
# plt.close()
plt.show()

# plt.ylim([0,1.1])
# plt.plot(age,sero_mumps,"o",ls=":")
# plt.savefig("sero_mumps.pdf",bbox_inches = "tight") 
# plt.close()

# plt.ylim([0,1.1])
# plt.plot(age,sero_rubella,"o",ls=":") 
# plt.savefig("sero_rubella.pdf",bbox_inches = "tight")  
# plt.show()

# samples = len(age)

# with open("output/measles.txt","w") as f:
#     f.write("%i\n" % samples)
#     for i in range(samples):
#         f.write("%i %f\n" % (age[i],sero_measles[i]))

# with open("output/mumps.txt","w") as f:
#     f.write("%i\n" % samples)
#     for i in range(samples):
#         f.write("%i %f\n" % (age[i],sero_mumps[i]))

# with open("output/rubella.txt","w") as f:
#     f.write("%i\n" % samples)
#     for i in range(samples):
#         f.write("%i %f\n" % (age[i],sero_rubella[i]))

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
