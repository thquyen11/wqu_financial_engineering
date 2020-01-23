import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import math
import random
import matplotlib.pyplot as plt


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

def log_char(u):
    return np.exp(C(u)+D(u)*v0+1j*u*np.log(S0))

def adj_char(u):
    return log_char(u-1j)/log_char(-1j)

#Share specific information
S0 = 100
v0 = 0.06
kappa = 9 #TODO: how to select
theta = 0.06 #TODO: how to select
r = 0.03
sigma = 0.5
rho = -0.4 #TODO: how to select

#Call option specific information
K = 105
T = 0.5
k_log = np.log(K)

#Approximation information
t_max = 30 # TODO: how to select
N = 100

#Characteristic function code
a = sigma**2/2

delta_t = t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n=(from_1_to_N-1/2)*delta_t

#Calculate the Fourier estimate
first_integral = sum(
            (((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum(
            (((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)
fourier_call_val = S0*(0.5 + first_integral/np.pi) - np.exp(-r*T)*K*(0.5 + second_integral/np.pi)
print(fourier_call_val)

plt.title("Value of additional terms in integrand estimate")
plt.plot((((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t, label="First integrand")
plt.plot((((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t, label="Second integrand")
plt.xlabel("t")
plt.ylabel("Integrand value")
plt.legend()
plt.show()