import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import math
import random
import matplotlib.pyplot as plt


S0 = 100
sigma = 0.3
gamma = 0.75  # TODO: how to select?
beta = gamma + 1
r = 0.1
T = 3

#Strikes array to test volatility
test_strikes = np.linspace(80,120,41)

# Call price under CEV
def C(t, K):
    kappa = 2*r/(sigma**2*(2-beta)*(np.exp(r*(2-beta)*t)-1)) 
    x = kappa*S0**(2-beta)*np.exp(r*(2-beta)*t) 
    y = kappa*K**(2-beta) 
    zb = 2 + 2/(2-beta)
    return S0*(1-(ncx2.cdf(2*y,zb,2*x)))-K*(1-(1-ncx2.cdf(2*x,zb-2,2*y)))*np.exp(-r*t)
    
#Estimating partial derivatives
delta_t = 0.01
delta_K = 0.01
dC_dT = (C(T+delta_t, test_strikes)-C(T-delta_t,test_strikes))/(2*delta_t)
dC_dK = (C(T,test_strikes+delta_K)-C(T,test_strikes-delta_K))/(2*delta_K)
d2C_dK2 = (C(T,test_strikes+2*delta_K)-2*C(T,test_strikes+delta_K)+C(T,test_strikes))/(delta_K**2)

#Estimating local volatility - Dupire Equation
vol_est = np.sqrt(2)/test_strikes*np.sqrt((dC_dT+r*test_strikes*dC_dK)/d2C_dK2)
z1=C(T,K)/400

#Plotting closed-form and Dupire equation
plt.subplot(211)
plt.plot(test_strikes, sigma*test_strikes**(gamma-1), label="Closed-form")
plt.grid(True)
plt.legend()
plt.subplot(212)
plt.plot(test_strikes, z1, lw=3, label="Dupire Equation")
# plt.plot(test_strikes, vol_est, lw=3, label="Dupire Equation")
plt.grid(True)
plt.legend()
plt.show()

