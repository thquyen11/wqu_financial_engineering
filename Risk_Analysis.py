import numpy as np
import numpy.matlib
import math, random


def stockMonteCarlo(no_shares = 1, confidence_level = 99, inital_price = 0, future_prices = []):
    portfolio_value = np.sort(np.array(future_prices) - inital_price)*no_shares
    return portfolio_value[int(((100-confidence_level)/100)*portfolio_value.size-1)]


def optionMonteCarlo(no_options = 1, confidence_level = 99, premium = 0, future_return = []):
    portfolio_value = np.sort(np.array(future_return) - premium)*no_options
    return portfolio_value[int(((100-confidence_level)/100)*portfolio_value.size-1)]


def optionCVA()