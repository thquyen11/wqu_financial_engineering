import numpy as np
from scipy.stats import norm
import math, random
import matplotlib.pyplot as plt


class Option_Pricing:
    option_mean_MC = [None]*50
    option_std_MC = [None]*50
    option_estimate_BSM = 0

    def __init__(self, initial_stock_price,risk_free_rate,sigma,strike_price,maturity):
        self.initial_stock_price = initial_stock_price
        self.risk_free_rate = risk_free_rate
        self.sigma = sigma
        self.strike_price = strike_price
        self.maturity = maturity


    def priceOption(self, option='call', method='MC'):
        random.seed(0)
        if method == 'MC':
            for i in range(1, 51):
                norm_array = norm.rvs(size=1000*(i))
                if option == 'put':
                    option_value_array = self.pricePutOptionMC(self.terminalStockPrice(self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                elif option == 'call':
                    option_value_array = self.priceCallOptionMC(self.terminalStockPrice(self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                self.option_mean_MC[i-1] = np.mean(option_value_array)
                self.option_std_MC[i-1] = np.std(option_value_array)/np.sqrt((i)*1000)
        elif method == 'BSM':
            self.option_estimate_BSM = self.priceCallOptionBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma) if option == 'call' else self.pricePutOptionBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma)
        
        return self.option_mean_MC if method == 'MC' else [self.option_estimate_BSM]


    def terminalStockPrice(self, initial_stock_price, risk_free_rate, sigma, Z, maturity):
        return initial_stock_price*np.exp((risk_free_rate-sigma**2/2)*maturity + sigma*np.sqrt(maturity)*Z)


    def priceCallOptionMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(terminal_stockprice-strike_price, 0)


    def pricePutOptionMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(strike_price-terminal_stockprice, 0)


    def priceCallOptionBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.calcualte_d1(initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.calculate_d2(d1, sigma, maturity)
        return initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*strike_price*math.exp(-risk_free_rate*maturity)


    def pricePutOptionBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.calcualte_d1(initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.calculate_d2(d1, sigma, maturity)
        return -initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*strike_price*math.exp(-risk_free_rate*maturity)


    def calcualte_d1(self, initial_stock_price, strike_price, risk_free_rate, sigma, maturity):
        return (math.log(initial_stock_price/strike_price) + (risk_free_rate+sigma**2/2)*maturity)/(sigma*math.sqrt(maturity))


    def calculate_d2(self, d1, sigma, maturity):
        return d1 - sigma*math.sqrt(maturity)


    def estimationGraph(self):
        plt.xlabel("Sample Size")
        plt.ylabel("Monte Carlo Estimate")
        plt.plot(self.option_mean_MC, ".")
        plt.plot([self.option_estimate_BSM]*50)
        plt.plot(np.mean(self.option_mean_MC)+3*np.array(self.option_std_MC), 'r')
        plt.plot(np.mean(self.option_mean_MC)-3*np.array(self.option_std_MC), 'r')
        plt.show()


if __name__ == "__main__":
    price_tool = Option_Pricing(100, 0.1, 0.3, 110, 0.5)
    price_tool.priceOption('put', 'MC')
    price_tool.priceOption('put', 'BSM')
    price_tool.estimationGraph()


