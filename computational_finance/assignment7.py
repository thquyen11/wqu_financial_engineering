import math
import numpy as np
from scipy.stats import norm
import scipy.optimize
import matplotlib.pyplot as plt


def UpOutCall(p, K, L, discount_factor, T):
    price = (p[:,-1]-K)*discount_factor
    condition_one = p[:,-1] < K
    condition_two = np.amax(p,axis=1) > L
    price[condition_one+condition_two] = 0
    return price


def forward_rate_libor(r0, n_simulations, market_bond_prices):
    mc_forward = np.ones([n_simulations, n_steps-1])*(market_bond_prices[:-1]-market_bond_prices[1:])/(dT*market_bond_prices[1:])
    predcorr_forward = np.ones([n_simulations, n_steps-1])*(market_bond_prices[:-1]-market_bond_prices[1:])/(dT*market_bond_prices[1:])
    mc_capfac = np.ones([n_simulations, n_steps])
    predcorr_capfac = np.ones([n_simulations, n_steps])
    
    for i in range(1, n_steps):
        Z = norm.rvs(size=[n_simulations, 1])

        # Explicit Monte Carlo simulation
        muhat = np.cumsum(dT*mc_forward[:, i:]*sigmaj **
                          2/(1+dT*mc_forward[:, i:]), axis=1)
        mc_forward[:, i:] = mc_forward[:, i:] * \
            np.exp((muhat-sigmaj**2/2) *
                   dT+sigmaj*np.sqrt(dT)*Z)

        # Predictor-Corrector Monte Carlo simulation
        mu_initial = np.cumsum(dT*predcorr_forward[:, i:] *
                               sigmaj**2/(1+dT*predcorr_forward[:, i:]), axis=1)
        for_temp = predcorr_forward[:, i:]*np.exp(
            (mu_initial-sigmaj**2/2)*dT+sigmaj*np.sqrt(dT)*Z)
        mu_term = np.cumsum(dT*for_temp*sigmaj**2 /
                            (1+dT*for_temp), axis=1)
        predcorr_forward[:, i:] = predcorr_forward[:, i:]*np.exp(
            (mu_initial+mu_term-sigmaj**2)*dT/2+sigmaj*np.sqrt(dT)*Z)


    # Implying capitalisation factors from the forward rates
    mc_capfac[:, 1:] = np.cumprod(1+dT*mc_forward, axis=1)
    predcorr_capfac[:, 1:] = np.cumprod(1+dT*predcorr_forward, axis=1)

    # Inverting the capitalisation factors to imply bond prices (discount factors)
    mc_price = mc_capfac**(-1)
    predcorr_price = predcorr_capfac**(-1)

    return (np.mean(mc_price, axis=0), np.mean(predcorr_price, axis=0))


# Stock info
r0 = 0.08
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
correlation = 0.2
n_months = 12
dT = 1/n_months
months = np.array(range(13))

# CEV model parameters
gamma = 0.75
beta = gamma + 1
sigmaj = 0.2
n_simulations = 100000

# Calibration parameteres
t = np.linspace(0,1,13)
n_steps=len(t)
market_bonds = np.array([100,99.38,98.76,98.15,97.54,96.94,96.34,95.74,95.16,94.57,93.99,93.42,92.85])

# Find and calibrate the forward rate
np.random.seed(0)
mc_price_mean, predcorr_price_mean = forward_rate_libor(r0, n_simulations, market_bonds)
plt.plot(months, market_bonds/100, label = 'Market Bonds')
plt.plot(months, mc_price_mean, 'x', label = 'Simple Monte Carlo')
plt.plot(months, predcorr_price_mean, 'o', label = 'Predictor-Corrector')
plt.xlabel('Maturity(month)')
plt.ylabel('Bond price')
plt.legend()
plt.show()

forward_rate = (market_bonds[:-1]-market_bonds[1:])/(dT*market_bonds[1:])
mc_forward_rate = (mc_price_mean[:-1]-mc_price_mean[1:])/(dT*mc_price_mean[1:])
predcorr_forward_rate = (predcorr_price_mean[:-1]-predcorr_price_mean[1:])/(dT*predcorr_price_mean[1:])
compound_rate = np.log(1+forward_rate*dT)/dT
mc_compound_rate = np.log(1+mc_forward_rate*dT)/dT
predcorr_compound_rate = np.log(1+predcorr_forward_rate*dT)/dT
discount_oneyear = 1/(1+dT*predcorr_compound_rate).prod()

# Pricing Up-Out Barrier Call Option
opt_est = None
opt_std = None
d_opt_est = None
d_opt_std = None
cva_est = None
cva_std = None
S_array = np.zeros([n_simulations,n_months+1])
F_array = np.zeros([n_simulations,n_months+1])
stock_prices_path = np.zeros([n_simulations,n_months+1])
firm_values_path = np.zeros([n_simulations,n_months+1])
stock_prices_path[:,0] = S0
firm_values_path[:,0] = init_f

Z1 = norm.rvs(size = [n_simulations, n_months])
Z2 = norm.rvs(size = [n_simulations, n_months])
Z2 = correlation*Z1 + np.sqrt(1-correlation**2)*Z2

for j in range(n_months):
    sigma_s = 0.3*(stock_prices_path[:,j]**(gamma-1))
    sigma_f = 0.3*(firm_values_path[:,j]**(gamma-1))
    stock_prices_path[:,j+1] = stock_prices_path[:,j]*np.exp(dT*(predcorr_compound_rate[j]-0.5*(sigma_s**2))+sigma_s*np.sqrt(dT)*Z1[:,j])   
    firm_values_path[:,j+1] = firm_values_path[:,j]*np.exp(dT*(predcorr_compound_rate[j]-0.5*(sigma_f**2))+sigma_f*np.sqrt(dT)*Z2[:,j])   
    S_array = UpOutCall(stock_prices_path, K, barrier, discount_oneyear, T)
    F_array = discount_oneyear*T*(1-recovery)*(firm_values_path[:,-1] < debt)*S_array
    
opt_est = S_array.mean()
opt_std = S_array.std()/np.sqrt(n_simulations)
cva_est = F_array.mean()
cva_std = F_array.std()/np.sqrt(n_simulations)
d_opt_est = opt_est-cva_est
d_opt_std = np.std(S_array-F_array)/np.sqrt(n_simulations)

print("Option Price (default free) ",opt_est)
print("Credit Valuation Adjustment ",cva_est)
print("Option Price with CVA ",d_opt_est)
print("")
print("Option Std Deviation (default free) ",opt_std)
print("Credit Valuation Adjustment Std Deviation",cva_std)
print("Option Price with CVA Std Deviation",d_opt_std)

plt.plot(stock_prices_path[0,:], label = 'Stock price')
plt.plot([K]*13, label = 'Strike price')
plt.plot([barrier]*13, label = 'Option barrier')

plt.plot(firm_values_path[0,:], label = 'Firm value')
plt.plot([debt]*13, label = 'Counterparty\'s debt')

plt.xlabel('Maturity (month)')
plt.ylabel('Price')
plt.legend()
plt.show()



