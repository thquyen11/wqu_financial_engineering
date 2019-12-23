import numpy as np
from scipy.stats import norm
import math, random
import matplotlib.pyplot as plt


def pricePutOptionMC(terminal_stockprice, strike_price, risk_free_rate, maturity):
    return np.exp(-risk_free_rate*maturity)*np.maximum(strike_price-terminal_stockprice, 0)


def terminalStockPrice(initial_stock_price, risk_free_rate, sigma, Z, maturity):
    return initial_stock_price*np.exp((risk_free_rate-sigma**2/2)*maturity + sigma*np.sqrt(maturity)*Z)


def pricePutOptionBSM(initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
    d1 = (math.log(initial_stock_price/strike_price) + (risk_free_rate+sigma**2/2)*maturity)/(sigma*math.sqrt(maturity))
    d2 = d1 - sigma*math.sqrt(maturity)
    return -initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*strike_price*math.exp(-risk_free_rate*maturity)


if __name__ == "__main__":
    random.seed(0)
    initial_stock_price = 100
    risk_free_rate = 0.1
    sigma = 0.3
    strike_price = 110
    maturity = 0.5
    put_mean_MC = [None]*50
    put_std_MC = [None]*50
    put_estimate_BSM = 0

    # Monte Carlo for Vanilla European Put Option
    for i in range(1, 51):
        norm_array = norm.rvs(size=1000*(i))
        put_value_array = pricePutOptionMC(terminalStockPrice(initial_stock_price, risk_free_rate, sigma, norm_array, maturity), strike_price, risk_free_rate, maturity)
        put_mean_MC[i-1] = np.mean(put_value_array)
        put_std_MC[i-1] = np.std(put_value_array)/np.sqrt((i)*1000)

    # BSM for Vanilla European Put Option
    put_estimate_BSM = pricePutOptionBSM(initial_stock_price, strike_price, risk_free_rate, maturity, sigma)

plt.xlabel("Sample Size")
plt.ylabel("Monte Carlo Estimate")
plt.plot(put_mean_MC, ".")
plt.plot([put_estimate_BSM]*50)
plt.plot(put_estimate_BSM+3*np.array(put_std_MC), 'r')
plt.plot(put_estimate_BSM-3*np.array(put_std_MC), 'r')
plt.show()

