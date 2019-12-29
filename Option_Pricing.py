import numpy as np
from scipy.stats import norm
import math, random
import matplotlib.pyplot as plt
import VaR_Calculation


class Vanila_Option:
    _option_mean_MC = [None]*50
    _option_std_MC = [None]*50
    _option_estimate_BSM = 0

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
                sample_size = 1000*i
                norm_array = norm.rvs(size=sample_size)
                if option == 'put':
                    option_value_array = self.pricePutMC_List(self.__terminalStockPrice(self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                elif option == 'call':
                    option_value_array = self.priceCallMC_List(self.__terminalStockPrice(self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                self._option_mean_MC[i-1] = np.mean(option_value_array)
                self._option_std_MC[i-1] = np.std(option_value_array)/np.sqrt(sample_size)
        elif method == 'BSM':
            self._option_estimate_BSM = self.priceCallBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma) if option == 'call' else self.pricePutBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma)
        
        return self._option_mean_MC if method == 'MC' else [self._option_estimate_BSM]


    def __terminalStockPrice(self, initial_stock_price, risk_free_rate, sigma, Z, maturity):
        return initial_stock_price*np.exp((risk_free_rate-sigma**2/2)*maturity + sigma*np.sqrt(maturity)*Z)


    def priceCallMC_List(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(terminal_stockprice-strike_price, 0)
    
    
    def priceCallMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.max(terminal_stockprice-strike_price, 0)


    def pricePutMC_List(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(strike_price-terminal_stockprice, 0)
    
    
    def pricePutMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.max(strike_price-terminal_stockprice, 0)


    def priceCallBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.__calcualte_d1(initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.__calculate_d2(d1, sigma, maturity)
        return initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*strike_price*math.exp(-risk_free_rate*maturity)


    def pricePutBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.__calcualte_d1(initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.__calculate_d2(d1, sigma, maturity)
        return -initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*strike_price*math.exp(-risk_free_rate*maturity)


    def __calcualte_d1(self, initial_stock_price, strike_price, risk_free_rate, sigma, maturity):
        return (math.log(initial_stock_price/strike_price) + (risk_free_rate+sigma**2/2)*maturity)/(sigma*math.sqrt(maturity))


    def __calculate_d2(self, d1, sigma, maturity):
        return d1 - sigma*math.sqrt(maturity)
                                            

    def estimationGraph(self, bsm = True):
        plt.xlabel("Sample Size")
        plt.ylabel("Monte Carlo Estimate")
        plt.plot(self._option_mean_MC, ".")
        plt.plot(np.mean(self._option_mean_MC)+3*np.array(self._option_std_MC), 'r')
        plt.plot(np.mean(self._option_mean_MC)-3*np.array(self._option_std_MC), 'r')
        if bsm:
            plt.plot([self._option_estimate_BSM]*50)
        plt.show()


class Barrier_Option(Vanila_Option):
    def __init__(self, initial_stock_price, risk_free_rate, sigma, strike_price, maturity, barrier_price):
        super(Barrier_Option, self).__init__(initial_stock_price, risk_free_rate, sigma, strike_price, maturity)
        self.barrier_price = barrier_price
    
    def priceOption(self, option = 'call', steps = 1):
        dT = self.maturity/steps
        random.seed(0)
        vaR = [None]*50

        for i in range(1, 51):
            sample_size = 1000*i
            terminal_stock_prices = [self.initial_stock_price]*sample_size
            # norm_array = norm.rvs(size=sample_size)

            for _ in range(steps):
                norm_array = norm.rvs(size=sample_size)
                terminal_stock_prices = super()._Vanila_Option__terminalStockPrice(terminal_stock_prices, self.risk_free_rate, self.sigma, norm_array, dT)
                terminal_stock_prices = [0 if price >= self.barrier_price else price for price in terminal_stock_prices]
                # norm_array = norm.rvs(size=len(terminal_stock_prices))

            option_value_array = super().priceCallMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate, self.maturity) if option == 'call' else super().pricePutMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate, self.maturity)
            super()._option_mean_MC[i-1] = np.mean(option_value_array)
            super()._option_std_MC[i-1] = np.std(option_value_array)/np.sqrt(option_value_array.size)
            vaR[i-1] = VaR_Calculation.optionMonteCarlo(1, 99, 5, option_value_array)

        return (super()._option_mean_MC, super()._option_std_MC, vaR)


if __name__ == "__main__":
    # price_tool = Vanila_Option(100, 0.1, 0.3, 110, 0.5)
    # print(price_tool.priceOption('put', 'MC'))
    # price_tool.priceOption('put', 'BSM')
    # price_tool.estimationGraph()

    price_tool = Barrier_Option(100, .08, 0.03, 100, 1, 110) #TODO: revert barrier price to 105
    mean, std, vaR = price_tool.priceOption(option='call', steps = 12)
    print(mean)
    print(std)
    print(vaR)
    price_tool.estimationGraph(bsm = False)


