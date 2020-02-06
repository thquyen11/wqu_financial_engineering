import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random


# Parameters
# r0 = 0.05
alpha = 0.2
b = 0.08
sigma = 0.025

# Problem parameters
# t = np.linspace(0, 40, 21)
t = np.linspace(0,12,13)
sigmaj = 0.2


def forward_rate_libor(r0):
    vasi_bond = bond_price(r0, 0, t)

    # Applying the algo
    np.random.seed(0)
    n_simulations = 100000
    n_steps = len(t)

    mc_forward = np.ones([n_simulations, n_steps-1])*(vasi_bond[:-1]-vasi_bond[1:])/(2*vasi_bond[1:])
    predcorr_forward=np.ones([n_simulations, n_steps-1])*(vasi_bond[:-1]-vasi_bond[1:])/(2*vasi_bond[1:])
    mc_capfac=np.ones([n_simulations, n_steps])
    predcorr_capfac=np.ones([n_simulations, n_steps])

    delta=np.ones([n_simulations, n_steps-1])*(t[1:]-t[:-1])

    for i in range(1, n_steps):
        Z=norm.rvs(size = [n_simulations, 1])

        # Explicit Monte Carlo simulation
        muhat=np.cumsum(delta[:, i:]*mc_forward[:, i:]*sigmaj**
                        2/(1+delta[:, i:]*mc_forward[:, i:]), axis = 1)
        mc_forward[:, i:]=mc_forward[:, i:]*np.exp((muhat-sigmaj**2/2)*delta[:, i:]+sigmaj*np.sqrt(delta[:, i:])*Z)

        # Predictor-Corrector Monte Carlo simulation
        mu_initial=np.cumsum(delta[:, i:]*predcorr_forward[:, i:] * \
                            sigmaj**2/(1+delta[:, i:]*predcorr_forward[:, i:]), axis = 1)
        for_temp=predcorr_forward[:, i:]*np.exp(
            (mu_initial-sigmaj**2/2)*delta[:, i:]+sigmaj*np.sqrt(delta[:, i:])*Z)
        mu_term=np.cumsum(delta[:, i:]*for_temp*sigmaj**2 / \
                        (1+delta[:, i:]*for_temp), axis = 1)
        predcorr_forward[:, i:]=predcorr_forward[:, i:]*np.exp(
            (mu_initial+mu_term-sigmaj**2)*delta[:, i:]/2+sigmaj*np.sqrt(delta[:, i:])*Z)

    # Implying capitalisation factors from the forward rates
    # mc_capfac[:, 1:]=np.cumprod(1+delta*mc_forward, axis = 1)
    # predcorr_capfac[:, 1:]=np.cumprod(1+delta*predcorr_forward, axis = 1)

    # # Inverting the capitalisation factors to imply bond prices (discount factors)
    # mc_price=mc_capfac**(-1)
    # predcorr_price=predcorr_capfac**(-1)

    # # Taking averages
    # mc_final=np.mean(mc_price, axis = 0)
    # predcorr_final=np.mean(predcorr_price, axis = 0)

    return predcorr_forward


# Vasicek bond price
def A(t1, t2):
    return (1-np.exp(-alpha*(t2-t1)))/alpha


def D(t1, t2):
    val1 = (t2-t1-A(t1, t2))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1, t2)**2/(4*alpha)
    return val1-val2


def bond_price(r, t, T):
    return np.exp(-A(t, T)*r+D(t, T))



