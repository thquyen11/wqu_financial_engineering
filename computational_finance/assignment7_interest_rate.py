import numpy as np
from scipy.stats import norm
import random
import scipy.optimize



# Problem parameters
t = np.array(range(12))
n_steps = len(t)
sigmaj = 0.2
r0 = 0.08

np.random.seed(0)
market_bond_prices = np.array([99.38,98.76,98.15,97.54,96.94,96.34,95.74,95.16,94.57,93.99,93.42,92.85])


def calibrate(alpha,b,sigma):
    bnds = ((0,alpha),(0,b),(0,sigma))
    opt_val = scipy.optimize.fmin_slsqp(F,(0.3,0.05,0.03),bounds=bnds)
    opt_alpha = opt_val[0]
    opt_b = opt_val[1]
    opt_sig = opt_val[2]
    return (opt_alpha,opt_b,opt_sig)


def discount_factor_libor(r0, n_simulations,alpha,b,sigma):
    # Applying the algo
    vasi_bond = bond_price(r0, 0, t, alpha, b, sigma)

    mc_forward = np.ones([n_simulations, n_steps-1]) * \
        (vasi_bond[:-1]-vasi_bond[1:])/(2*vasi_bond[1:])
    predcorr_forward = np.ones(
        [n_simulations, n_steps-1])*(vasi_bond[:-1]-vasi_bond[1:])/(2*vasi_bond[1:])
    predcorr_capfac = np.ones([n_simulations, n_steps])

    delta = np.ones([n_simulations, n_steps-1])*(t[1:]-t[:-1])
    
    for i in range(1, n_steps):
        Z = norm.rvs(size=[n_simulations, 1])

        # Explicit Monte Carlo simulation
        muhat = np.cumsum(delta[:, i:]*mc_forward[:, i:]*sigmaj **
                          2/(1+delta[:, i:]*mc_forward[:, i:]), axis=1)
        mc_forward[:, i:] = mc_forward[:, i:] * \
            np.exp((muhat-sigmaj**2/2) *
                   delta[:, i:]+sigmaj*np.sqrt(delta[:, i:])*Z)

        # Predictor-Corrector Monte Carlo simulation
        mu_initial = np.cumsum(delta[:, i:]*predcorr_forward[:, i:] *
                               sigmaj**2/(1+delta[:, i:]*predcorr_forward[:, i:]), axis=1)
        for_temp = predcorr_forward[:, i:]*np.exp(
            (mu_initial-sigmaj**2/2)*delta[:, i:]+sigmaj*np.sqrt(delta[:, i:])*Z)
        mu_term = np.cumsum(delta[:, i:]*for_temp*sigmaj**2 /
                            (1+delta[:, i:]*for_temp), axis=1)
        predcorr_forward[:, i:] = predcorr_forward[:, i:]*np.exp(
            (mu_initial+mu_term-sigmaj**2)*delta[:, i:]/2+sigmaj*np.sqrt(delta[:, i:])*Z)


    # Implying capitalisation factors from the forward rates
    predcorr_capfac[:, 1:] = np.cumprod(1+delta*predcorr_forward, axis=1)

    # Inverting the capitalisation factors to imply bond prices (discount factors)
    predcorr_price = predcorr_capfac**(-1)

    return np.mean(predcorr_price, axis=0)


# Analytical Vasicek bond price
def A(t1, t2, alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha


def D(t1, t2, alpha, b, sigma):
    val1 = (t2-t1-A(t1, t2, alpha))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1, t2, alpha)**2/(4*alpha)
    return val1-val2


def bond_price(r, t, T, alpha, b, sigma):
    return np.exp(-A(t, T, alpha)*r+D(t, T, alpha, b, sigma))


def F(x):
    alpha = x[0]
    b = x[1]
    sigma = x[2]
    return sum(np.abs(bond_price(r0,0,t,alpha,b,sigma)-market_bond_prices))
