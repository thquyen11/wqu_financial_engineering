import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from assignment7_interest_rate import discount_factor_libor
from assignment7_interest_rate import calibrate


def a_path(S0, f0, r, sigma, Z, dT, steps, gamma):
    stock_cev_path = np.zeros([2,steps])
    stock_cev_path[0,0] = S0
    stock_cev_path[1,0] = f0
    for i in range(1, steps):
        sigma_dT = sigma*(stock_cev_path[:,i-1]**(gamma-1))
        stock_cev_path[:,i] = stock_cev_path[:,i-1]*np.exp(((r-0.5*sigma_dT**2)*dT+sigma*np.sqrt(dT)*Z[:,i-1]))
    return stock_cev_path


def UpOutCall(p, K, L, discount_factor, T):
    if max(p) > L:
        return 0
    else:
        return max(0, p[-1]-K)*discount_factor*T


# Stock info
r = 0.08
S0 = 100
sigma_s = 0.3
T = 1
K = 100
barrier = 150

# Firm info
init_f = 200
sigma_f = 0.25
debt = 175
recovery = 0.25
corr = 0.2
corr_mat = np.array([[1, corr], [corr, 1]])
L = np.linalg.cholesky(corr_mat)
n_steps = 12
dT = 1/n_steps
S = np.array([[S0], [init_f]])
sigma = np.array([[sigma_s], [sigma_f]])

# CEV model parameters
gamma = 0.75
beta = gamma + 1

# Calibration parameteres
alpha = 0.2
b = 0.08
sigma = 0.025

# Pricing Up-Out Barrier Call Option
np.random.seed(0)
n_simulations = 100000

opt_est = None
opt_std = None
d_opt_est = None
d_opt_std = None
cva_est = None
cva_std = None
p_array = np.zeros(n_simulations)
l_array = np.zeros(n_simulations)

opt_alpha,opt_b,opt_sigma = calibrate(alpha,b,sigma)

for j in range(n_simulations):
    Z = np.matmul(L, norm.rvs(size=(2, n_steps)))
    price_paths = a_path(S0, init_f, r, sigma_s, Z, dT, n_steps, gamma)
    stock_prices_path = price_paths[0]
    firm_values_path = price_paths[1]
    df_array = discount_factor_libor(r, n_simulations,opt_alpha,opt_b,opt_sigma)
    df_one_year = df_array[-1]
    p_array[j] = UpOutCall(stock_prices_path, K, barrier, df_one_year, T)
    l_array[j] = df_one_year*T*(1-recovery)*(firm_values_path[-1] < debt)*p_array[j]
    
opt_est = p_array.mean()
opt_std = p_array.std()/np.sqrt(n_simulations)
cva_est = l_array.mean()
cva_std = l_array.std()/np.sqrt(n_simulations)
d_opt_est = opt_est-cva_est
d_opt_std = np.std(p_array-l_array)/np.sqrt(n_simulations)

print("Option Price (default free) ",opt_est)
print("Credit Valuation Adjustment ",cva_est)
print("Option Price with CVA ",d_opt_est)



