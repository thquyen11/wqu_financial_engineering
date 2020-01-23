import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

S0 = 100
sigma = 0.3
gamma = 0.75
r = 0.1
T = 3

z = 2 + 1/(1-gamma)
def C(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)
    
test_strikes = np.linspace(80,120,41)

delta_t = 0.01
delta_K = 0.01
dC_dT = (C(T+delta_t,test_strikes)-C(T-delta_t,test_strikes))/(2*delta_t)
dC_dK = (C(T,test_strikes+delta_K)-C(T,test_strikes-delta_K))/(2*delta_K)
d2C_dK2 = (C(T,test_strikes+2*delta_K)-2*C(T,test_strikes+delta_K)+C(T,test_strikes))/(delta_K**2)

vol_est = np.sqrt(2)/test_strikes*np.sqrt((dC_dT+r*test_strikes*dC_dK)/d2C_dK2)
