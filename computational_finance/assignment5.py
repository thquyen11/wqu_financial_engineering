import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import math
import random
import matplotlib.pyplot as plt


"""
Returns:
    number -- price of call option
"""def priceCall(s, strike, r, T):
    return max(0, s-strike)*np.exp(-r*T)

"""
Calculate the stock price at each step using CEV model
Returns:
    number list -- stock price at each step
"""
def a_path(S0, r, sigma, Z, dT, steps, gamma):
    a_path = np.zeros(steps)
    a_path[0] = S0
    for i in range(1, steps):
        sigma_dT = sigma*(a_path[i-1]**(gamma-1))
        a_path[i] = a_path[i-1] * \
            np.exp(((r-0.5*sigma_dT**2)*dT+sigma*np.sqrt(dT)*Z[i-1]))
    return a_path


def log_char(u):
    return np.exp(C(u)+D(u)*v0+1j*u*np.log(S0))


def adj_char(u):
    return log_char(u-1j)/log_char(-1j)


def b(u):
    return kappa - rho*sigma*1j*u


def c(u):
    return -(u**2+1j*u)/2


def d(u):
    return np.sqrt(b(u)**2-4*a*c(u))


def xminus(u):
    return (b(u)-d(u))/(2*a)


def xplus(u):
    return (b(u)+d(u))/(2*a)


def g(u):
    return xminus(u)/xplus(u)


def C(u):
    val1 = T*xminus(u)-np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return r*T*1j*u + theta*kappa*val1


def D(u):
    val1 = 1-np.exp(-T*d(u))
    val2 = 1-g(u)*np.exp(-T*d(u))
    return (val1/val2)*xminus(u)

"""
Price the option (call/put) using Black-Scholes formula
Returns:
    number -- option price
"""
def priceBSM(type, initial_stock_price, strike_price, risk_free_rate, sigma, maturity):
    d1 = (np.log(initial_stock_price/strike_price) + (risk_free_rate +
                                                      sigma**2/2)*maturity)/(sigma*np.sqrt(maturity))
    d2 = d1 - sigma*np.sqrt(maturity)
    if type == 'call':
        return initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*strike_price*np.exp(-risk_free_rate*maturity)
    elif type == 'put':
        return -initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*strike_price*np.exp(-risk_free_rate*maturity)


# Stock specific information
S0 = 100
sigma = 0.3
r = 0.03
T = 1
K = 100

# Heston model parameters
v0 = 0.06
kappa = 9  # TODO: how to select
theta = 0.06  # TODO: how to select
rho = -0.4  # TODO: how to select
k_log = np.log(K)
t_max = 30  # TODO: how to select
N = 100
# Heston characteristic functions
a = sigma**2/2
delta_t = t_max/N
from_1_to_N = np.linspace(1, N, N)
t_n = (from_1_to_N-1/2)*delta_t

# CEV model parameters
n_steps = 12
dT = 1/n_steps
gamma = 0.75  # TODO: how to select?
beta = gamma + 1


# Question 1
# Calculate the Fourier estimate
first_integral = sum(
    (((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum(
    (((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)
fourier_call_val = S0*(0.5 + first_integral/np.pi) - \
    np.exp(-r*T)*K*(0.5 + second_integral/np.pi)
print("Call price by Heston model: ", fourier_call_val)

# Cross check with Black-Scholes call price
call_bsm = priceBSM('call', S0, K, r, sigma, T)
print("Call price by Black-Scholes model: ", call_bsm)


# Question 2 + 3
# Calculate the CEV call price using MonteCarlo simulation
# Sampling the price paths with 1000, 2000,...50000 sample
call_cev_price = [None]*50  # call price by CEV model of each simulation
call_cev_std = [None]*50  # call price by CEV model of each simulation

for i in range(1, 51):
    sT_array = np.zeros(1000*i)  # Stock prices at maturity of all samples
    call_array = np.zeros(1000*i)  # Call prices of all samples
    for j in range(1000*i):
        Z = norm.rvs(size=n_steps)
        sT_array[j] = a_path(S0, r, sigma, Z, dT, n_steps, gamma)[-1]
        call_array[j] = priceCall(sT_array[j], K, r, T)
    call_cev_price[i-1] = call_array.mean()
    call_cev_std[i-1] = call_array.std()/np.sqrt(1000*i)

print("Call price by CEV model: ", call_cev_price)


# Question 4
plt.subplot(311)
plt.plot([fourier_call_val]*50, 'blue', label='heston')
plt.plot([call_bsm]*50, 'orange', label='black-scholes')
plt.ylim(9, 17)
plt.ylabel('Call price estimate')
plt.legend()

plt.subplot(313)
plt.plot(call_cev_price, '.', label='cev')
plt.plot(call_cev_price+3*np.array(call_cev_std), label='std deviation')
plt.plot(call_cev_price-3*np.array(call_cev_std), label='std deviation')
plt.ylim(9, 17)
plt.ylabel('Call price estimate')
plt.xlabel('Sample Sizes')
plt.legend()

plt.show()
