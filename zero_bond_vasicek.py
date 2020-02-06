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


# Analtical bond price
def A(t1, t2):
    return (1-np.exp(-alpha*(t2-t1)))/alpha


def D(t1, t2):
    val1 = (t2-t1-A(t1, t2))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1, t2)**2/(4*alpha)
    return val1-val2


def bond_price(r, t, T):
    return np.exp(-A(t, T)*r+D(t, T))


# Functions for means, variances and correlatiions
def Y_mean(Y,r,t1,t2):
    return Y+(t2-t1)*b+(r-b)*A(t1,t2)


def Y_var(t1,t2):
    return sigma**2*(t2-t1-A(t1,t2)-alpha*A(t1,t2)**2/2)/(alpha**2)


def rY_var(t1,t2):
    return sigma**2*(A(t1,t2)**2)/2


def rY_rho(t1,t2):
    return rY_var(t1,t2)/np.sqrt(vasi_var(t1,t2)*Y_var(t1,t2))


# Parameters
r0 = 0.05
alpha = 0.2
b = 0.08
sigma = 0.025

Y0=0
np.random.seed(0)
n_years = 10
n_simulations = 10
t = np.array(range(0, n_years+1))
Z_mont1 = norm.rvs(size=[n_simulations, n_years])
Z_mont2 = norm.rvs(size=[n_simulations, n_years])
r_simtemp = np.zeros([n_simulations, n_years+1])
Y_simtemp = np.zeros([n_simulations, n_years+1])
r_simtemp[:,0]=r0
Y_simtemp[:,0]=Y0
correlations=rY_rho(t[0:-1],t[1:])
Z_mont2=correlations*Z_mont1+np.sqrt(1-correlations**2)*Z_mont2

# Method 1: Vasicek
for i in range(n_years):
    r_simtemp[:,i+1]=vasi_mean(r_simtemp[:,i],t[i],t[i+1])+np.sqrt(vasi_var(t[i],t[i+1]))*Z_mont1[:,i]
    Y_simtemp[:,i+1]=Y_mean(Y_simtemp[:,i],r_simtemp[:,i],t[i],t[i+1])+np.sqrt(Y_var(t[i],t[i+1]))*Z_mont2[:,i]

ZCB_prices = np.mean(np.exp(-Y_simtemp),axis=0) # mean Yt by each year

# Method 2: Alternative to simulate Yt using only the rt, equation (3.4)
# Yt estimates
r_mat=np.cumsum(r_simtemp[:,0:-1],axis=1)*(t[1:]-t[0:-1])
r_mat2=np.cumsum(r_simtemp[:,0:-1]+r_simtemp[:,1:],axis=1)/2*(t[1:]-t[0:-1])
squad_prices = np.ones(n_years+1) # At T0, bond price = 1
trap_prices = np.ones(n_years+1)
squad_prices[1:] = np.mean(np.exp(-r_mat),axis=0)
trap_prices[1:] = np.mean(np.exp(-r_mat2),axis=0)

# Close-form bond prices
bond_vec=bond_price(r0,0,t)

# Determining yields
bond_yield = -np.log(bond_vec[1:])/t[1:]
mont_yield = -np.log(ZCB_prices[1:])/t[1:]
squad_yield = -np.log(squad_prices[1:])/t[1:]
trap_yield = -np.log(trap_prices[1:])/t[1:]

# Plotting bond prices
plt.subplot(311)
plt.title('Bond Prices')
plt.plot(t, bond_vec) # Analytical solution
plt.plot(t, ZCB_prices, '.') # Simulated Yt and rt
plt.plot(t, squad_prices, 'x') # Simulated rt and estimated Yt
plt.plot(t, trap_prices, '^') # Simulated rt and estimated Yt

# Ploting bond yield
plt.subplot(313)
plt.title('Bond Yield')
plt.plot(t[1:], bond_yield*100) 
plt.plot(t[1:], mont_yield*100, '.') 
plt.plot(t[1:], squad_yield*100, 'x') 
plt.plot(t[1:], trap_yield*100, '^') 

plt.show()
