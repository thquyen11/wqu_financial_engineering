import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


r = 0.08
init_s = 100
sigma_s = 0.3
T = 1
K = 100
barrier = 150
init_f = 200
sigma_f = 0.25
debt = 175
recovery = 0.25
corr = 0.2
corr_mat = np.array([[1, corr], [corr, 1]])
L = np.linalg.cholesky(corr_mat)
n_steps = 12  # monthly
dT = 1/n_steps
S = np.array([[init_s], [init_f]])
sigma = np.array([[sigma_s], [sigma_f]])


def a_path(S, r, sigma, Z, dT):
    return S*np.exp(np.cumsum((r-0.5*sigma*sigma)*dT+sigma*np.sqrt(dT)*Z, 1))


def UpOutCall(p, K, L, r, T):
    if max(p) > L:
        return 0
    else:
        return max(0, p[-1]-K)*np.exp(-r*T)


np.random.seed(10)
opt_est = [None]*50
opt_std = [None]*50
d_opt_est = [None]*50
d_opt_std = [None]*50
cva_est = [None]*50
cva_std = [None]*50
for i in range(1, 51):
    p_array = np.zeros(1000*i)
    l_array = np.zeros(1000*i)
    for j in range(1000*i):
        Z = np.matmul(L, norm.rvs(size=(2, 12)))
        p_path = a_path(S, r, sigma, Z, dT)
        s_path = p_path[0]
        f_path = p_path[1]
        p_array[j] = UpOutCall(s_path, K, barrier, r, T)
        l_array[j] = np.exp(-r*T)*(1-recovery)*(f_path[-1] < debt)*p_array[j]
    opt_est[i-1] = p_array.mean()
    opt_std[i-1] = p_array.std()/np.sqrt(1000*i)
    cva_est[i-1] = l_array.mean()
    cva_std[i-1] = l_array.std()/np.sqrt(1000*i)
    d_opt_est[i-1] = opt_est[i-1]-cva_est[i-1]
    d_opt_std[i-1] = np.std(p_array-l_array)/np.sqrt(1000*i)

plt.plot(opt_est, '.')
plt.plot(opt_est+3*np.array(opt_std))
plt.plot(opt_est-3*np.array(opt_std))
plt.ylabel('Price')
plt.xlabel('N*1000 Sample Size')
plt.title('Option Price (Default Free)')
plt.show()
plt.plot(cva_est, '.')
plt.plot(cva_est+3*np.array(cva_std))
plt.plot(cva_est-3*np.array(cva_std))
plt.ylabel('Price')
plt.xlabel('N*1000 Sample Size')
plt.title('Credit Valuation Adjustment')
plt.show()
plt.plot(d_opt_est, '.')
plt.plot(d_opt_est+3*np.array(d_opt_std))
plt.plot(d_opt_est-3*np.array(d_opt_std))
plt.ylabel('Price')
plt.xlabel('N*1000 Sample Size')
plt.title("Option Price with CVA")
plt.show()


