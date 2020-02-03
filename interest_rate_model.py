import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import math
import random
import matplotlib.pyplot as plt


def vasi_mean(r, t1, t2):
    """Gives the mean under Vasicek model. Note that t2 > t1, r is the
    interest rate from the beginning of the period"""
    return np.exp(-alpha*(t2-t1))*r+b*(1-np.exp(-alpha*(t2-t1)))


def vasi_var(t1, t2):
    """Gives the variance under Vasicek model. Note that t2 > t1"""
    return (sigma**2)*(1-np.exp(-2*alpha*(t2-t1)))/(2*alpha)


# Parameters
r0 = 0.05
alpha = 0.2
b = 0.08
sigma = 0.025

# Simlating interest rate paths in 10 years
# NB short rates are simulated on an annual basis
np.random.seed(0)
n_years = 10
n_simulations = 10
t = np.array(range(0, n_years+1))
Z = norm.rvs(size=[n_simulations, n_years])
r_sim = np.zeros([n_simulations, n_years+1])
r_sim[:, 0] = r0  # Set the initial value of each simulation
vasi_mean_vector = np.zeros(n_years+1)

for i in range(n_years):
    r_sim[:, i+1] = vasi_mean(r_sim[:, i], t[i], t[i+1]) + \
        np.sqrt(vasi_var(t[i], t[i+1]))*Z[:, i]

s_mean = r0*np.exp(-alpha*t)+b*(1-np.exp(-alpha*t))

# Plotting the result
t_graph = np.ones(r_sim.shape)*t
plt.plot(np.transpose(t_graph), np.transpose(r_sim*100), 'r')
plt.plot(t, s_mean*100)
plt.xlabel("Year")
plt.ylabel("Short Rate")
plt.show()
