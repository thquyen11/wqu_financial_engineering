import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import math
import random
import matplotlib.pyplot as plt


# Call price under CEV
def C(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    z = 2 + 1/(1-gamma)
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)

def a_path(S, r, sigma, Z, dT):
    return S*np.exp(np.cumsum((r-0.5*sigma**2)*dT+sigma*np.sqrt(dT)*Z))

def priceCallCEV(s, strike, r, T):
    return max(0, s-strike)*np.exp(-r*T)

#Stock properties
S0 = 100
sigma = 0.3
gamma = 0.75  # TODO: how to select?
beta = gamma + 1
r = 0.03
T = 1
n_steps = 12
dT = 1/n_steps
K = 100

#Sampling the price paths with 1000, 2000,...50000 sample
opt_est = [0]*50
for i in range(1, 51):
    s_path = np.zeros(1000*i)
    c_path = np.zeros(1000*i)
    
    for j in range(1000*i):
        Z = norm.rvs(size=n_steps)
        s_path[j] = a_path(S0, r, sigma, Z, dT)[-1]
        c_path[j] = priceCallCEV(s_path[j], K, r, T)
    opt_est[i-1] = c_path.mean()

plt.plot(opt_est)
plt.show()

# Strikes array to test volatility
# test_strikes = np.linspace(80, 120, 41)

# Estimating partial derivatives
# delta_t = 0.01
# delta_K = 0.01
# dC_dT = (C(T+delta_t, test_strikes)-C(T-delta_t, test_strikes))/(2*delta_t)
# dC_dK = (C(T, test_strikes+delta_K)-C(T, test_strikes-delta_K))/(2*delta_K)
# d2C_dK2 = (C(T, test_strikes+2*delta_K)-2 *
#            C(T, test_strikes+delta_K)+C(T, test_strikes))/(delta_K**2)

# # Estimating local volatility - Dupire Equation
# vol_est = np.sqrt(2)/test_strikes*np.sqrt((dC_dT+r*test_strikes*dC_dK)/d2C_dK2)

# Plotting closed-form and Dupire equation
# plt.subplot(211)
# plt.plot(test_strikes, sigma*test_strikes**(gamma-1), label="Closed-form")
# plt.grid(True)
# plt.legend()
# plt.subplot(212)
# plt.plot(test_strikes, vol_est, lw=3, label="Dupire Equation")
# plt.grid(True)
# plt.legend()
# plt.show()
