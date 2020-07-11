import numpy as np
import scipy as ps 
import math


def priceFuture(present_value, effective_period_rate, number_period = 1, continuously_compounded = 0):
    return present_value*(1+effective_period_rate)**number_period if continuously_compounded == 0 else present_value*math.exp(effective_period_rate*number_period)


def priceFXFuture(present_fx_rate, number_period, domestic_deposit_rate, foreign_deposit_rate):
    return present_fx_rate*math.exp((domestic_deposit_rate - foreign_deposit_rate)*number_period)


print(priceFuture(100, 0.08/2, 4, continuously_compounded = 1))
print(50e6)
    