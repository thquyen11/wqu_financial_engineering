import numpy as np


import matplotlib.pyplot as plt


from scipy.stats import ncx2

S0 = 100
sigma = 0.3
gamma = 0.75
r = 0.1
T = 3


   
test_strikes = np.linspace(80,120,41)

K = np.linspace(80,120,41)

delta_t = 0.01
delta_K = 0.01



beta = gamma+1

zb = 2 + 2/(2-beta)

def C_h(t,K):  
  kappa = 2*r/(sigma**2*(2-beta)*(np.exp(r*(2-beta)*t)-1)) 
  x = kappa*S0**(2-beta)*np.exp(r*(2-beta)*t) 
  y = kappa*K**(2-beta) 
  return S0*(1-(ncx2.cdf(2*y,zb,2*x)))-K*(1-(1-ncx2.cdf(2*x,zb-2,2*y)))*np.exp(-r*t)



C_h(T,K)

z1=C_h(T,K)/400


plt.subplot(211)
plt.plot(test_strikes, sigma*test_strikes**(gamma-1), label="Closed-form")
plt.grid(True)
plt.legend()
plt.subplot(212)
plt.plot(test_strikes, z1, lw=3, label="Dupire Equation")
plt.grid(True)
plt.legend()
