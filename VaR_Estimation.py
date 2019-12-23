import numpy as np
import numpy.matlib
from scipy.stats import norm, uniform
import math, random
import matplotlib.pyplot as plt
from Option_Pricing import Option_Pricing


#General share information
S0=np.array([[100],[95],[50]])
sigma=np.array([[0.15],[0.2],[0.3]])
cor_mat=np.array([[1, 0.2, 0.4],[0.2, 1, 0.8],[0.4, 0.8,1]])
L=np.linalg.cholesky(cor_mat) #Cholesky decomposition
r=0.1
T=1

#Applying Monte Carlo estimation of VaR
np.random.seed(0)
t_simulations=10000
alpha=0.05

#Current portfolio value
portval_current=np.sum(S0)


