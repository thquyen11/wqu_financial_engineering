import numpy as np
from scipy.stats import norm
import math
import random
import matplotlib.pyplot as plt


class Vanila_Option:
    def __init__(self, initial_stock_price, risk_free_rate, sigma, strike_price, maturity):
        self.initial_stock_price = initial_stock_price
        self.risk_free_rate = risk_free_rate
        self.sigma = sigma
        self.strike_price = strike_price
        self.maturity = maturity

    def valueWithoutCVA(self, option='call', method='MC', simulation=0):
        random.seed(0)
        option_mean_MC = [None]*simulation
        option_std_MC = [None]*simulation
        # There are 2 methods: MC a.k.a MonteCarlo and BSM a.k.a Black Scholes
        if method == 'MC':
            for i in range(1, simulation + 1):
                sample_size = 1000*i
                norm_array = norm.rvs(size=sample_size)
                if option == 'put':
                    option_value_array = self.pricePutMC_List(self.__terminalStockPrice(
                        self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                elif option == 'call':
                    option_value_array = self.priceCallMC_List(self.__terminalStockPrice(
                        self.initial_stock_price, self.risk_free_rate, self.sigma, norm_array, self.maturity), self.strike_price, self.risk_free_rate, self.maturity)
                option_mean_MC[i-1] = np.mean(option_value_array)
                option_std_MC[i-1] = np.std(option_value_array) / \
                    np.sqrt(sample_size)
            return (option_mean_MC, option_std_MC)
        elif method == 'BSM':
            if option == 'call':
                return (self.priceCallBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma), self.sigma)
            else:
                return (self.pricePutBSM(self.initial_stock_price, self.strike_price, self.risk_free_rate, self.maturity, self.sigma), self.sigma)

    # Calcualte future stock price at maturity
    def __terminalStockPrice(self, initial_stock_price, risk_free_rate, sigma, Z, maturity):
        return initial_stock_price*np.exp((risk_free_rate-sigma**2/2)*maturity + sigma*np.sqrt(maturity)*Z)

    # Pricing Call Option via MonteCarlo, return a list
    def priceCallMC_List(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(terminal_stockprice-strike_price, 0)

    # Pricing Call Option via MonteCarlo, return a value
    def priceCallMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.max(terminal_stockprice-strike_price, 0)

    # Pricing Put Option via MonteCarlo, return a list
    def pricePutMC_List(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.maximum(strike_price-terminal_stockprice, 0)

    # Pricing Put Option via MonteCarlo, return a value
    def pricePutMC(self, terminal_stockprice, strike_price, risk_free_rate, maturity):
        return np.exp(-risk_free_rate*maturity)*np.max(strike_price-terminal_stockprice, 0)

    # Pricing Call Option via Black Scholes Model
    def priceCallBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.__calcualte_d1(
            initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.__calculate_d2(d1, sigma, maturity)
        return initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*strike_price*math.exp(-risk_free_rate*maturity)

    # Pricing Put Option via Black Scholes Model
    def pricePutBSM(self, initial_stock_price, strike_price, risk_free_rate, maturity, sigma):
        d1 = self.__calcualte_d1(
            initial_stock_price, strike_price, risk_free_rate, sigma, maturity)
        d2 = self.__calculate_d2(d1, sigma, maturity)
        return -initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*strike_price*math.exp(-risk_free_rate*maturity)

    def __calcualte_d1(self, initial_stock_price, strike_price, risk_free_rate, sigma, maturity):
        return (math.log(initial_stock_price/strike_price) + (risk_free_rate+sigma**2/2)*maturity)/(sigma*math.sqrt(maturity))

    def __calculate_d2(self, d1, sigma, maturity):
        return d1 - sigma*math.sqrt(maturity)

    def estimationGraph(self, mean_mc, std_mc, mean_bsm):
        plt.xlabel("Sample Size")
        plt.ylabel("Monte Carlo Estimate")
        plt.plot(mean_mc, ".")
        plt.plot(np.mean(mean_mc)+3*np.array(std_mc), 'r')
        plt.plot(np.mean(mean_mc)-3*np.array(std_mc), 'r')
        plt.show()


class Barrier_Option(Vanila_Option):
    def __init__(self, initial_stock_price, risk_free_rate, sigma, strike_price, maturity, barrier_price):
        super(Barrier_Option, self).__init__(initial_stock_price,
                                             risk_free_rate, sigma, strike_price, maturity)
        self.barrier_price = barrier_price

    # Pricing barrier option via MonteCarlo
    def valueWithoutCVA(self, option='call', simulation=0, steps=1):
        random.seed(0)
        dT = self.maturity/steps
        option_mean_MC = [None]*simulation
        option_std_MC = [None]*simulation

        for i in range(1, simulation + 1):
            sample_size = 1000*i
            terminal_stock_prices = [self.initial_stock_price]*sample_size

            for _ in range(steps):
                norm_array = norm.rvs(size=sample_size)
                terminal_stock_prices = super()._Vanila_Option__terminalStockPrice(
                    terminal_stock_prices, self.risk_free_rate, self.sigma, norm_array, dT)
                terminal_stock_prices = [
                    0 if price >= self.barrier_price else price for price in terminal_stock_prices]

            option_value_array = super().priceCallMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate,
                                                          self.maturity) if option == 'call' else super().pricePutMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate, self.maturity)
            option_mean_MC[i-1] = np.mean(option_value_array)
            option_std_MC[i-1] = np.std(option_value_array) / \
                np.sqrt(option_value_array.size)

        return (option_mean_MC, option_std_MC)

    # Pricing barrier option via MonteCarlo, considering the CVA
    def valueWithCVA(self, option='call', simulation=0, steps=1,  firm_sigma=0, firm_debt=0, recovery_rate=0, firm_corr=0, firm_initial_value=100):
        random.seed(0)
        dT = self.maturity/steps
        option_mean_MC = [None]*simulation
        option_std_MC = [None]*simulation
        cva_estimates = [None]*simulation
        cva_std = [None]*simulation
        corr_tested = [
            firm_corr]*simulation if firm_corr != 0 else np.linspace(-1, 1, simulation)

        for i in range(1, simulation + 1):
            sample_size = 1000*i
            terminal_stock_prices = [self.initial_stock_price]*sample_size
            correlation = corr_tested[i-1]

            if correlation == -1 or correlation == 1:
                norm_vec_0 = norm.rvs(size=sample_size)
                norm_vec_1 = correlation*norm_vec_0
                corr_norm_matrix = np.array([norm_vec_0, norm_vec_1])
            else:
                corr_matrix = np.array([[1, correlation], [correlation, 1]])
                norm_matrix = norm.rvs(size=np.array([2, sample_size]))
                corr_norm_matrix = np.matmul(
                    np.linalg.cholesky(corr_matrix),  norm_matrix)

            for _ in range(steps):
                terminal_stock_prices = super()._Vanila_Option__terminalStockPrice(
                    terminal_stock_prices, self.risk_free_rate, self.sigma, corr_norm_matrix[0, ], dT)
                terminal_stock_prices = [
                    0 if price >= self.barrier_price else price for price in terminal_stock_prices]

            option_value_array = super().priceCallMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate,
                                                          self.maturity) if option == 'call' else super().pricePutMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate, self.maturity)
            term_firm_val = super()._Vanila_Option__terminalStockPrice(firm_initial_value,
                                                                       self.risk_free_rate, firm_sigma, corr_norm_matrix[1, ], self.maturity)
            amount_lost = np.exp(-self.risk_free_rate*self.maturity) * \
                (term_firm_val < firm_debt) * \
                option_value_array*(1 - recovery_rate)
            cva_estimates[i-1] = np.mean(amount_lost)
            cva_std[i-1] = np.std(amount_lost)/np.sqrt(sample_size)
            # option_mean_MC[i-1] = np.mean(option_value_array) - cva_estimates[i-1]
            option_mean_MC[i-1] = np.mean(option_value_array)
            option_std_MC[i-1] = np.std(option_value_array) / \
                np.sqrt(option_value_array.size)

        return (option_mean_MC, option_std_MC, cva_estimates, cva_std)

    # Calculate Value at Risk for barrier option
    def calculateVaR(self, option='call', simulation=0, steps=1):
        random.seed(0)
        dT = self.maturity/steps
        VaR = [None]*simulation
        option_mean_MC = [None]*simulation
        option_std_MC = [None]*simulation

        for i in range(1, simulation + 1):
            sample_size = 1000*i
            terminal_stock_prices = [self.initial_stock_price]*sample_size

            for _ in range(steps):
                norm_array = norm.rvs(size=sample_size)
                terminal_stock_prices = super()._Vanila_Option__terminalStockPrice(
                    terminal_stock_prices, self.risk_free_rate, self.sigma, norm_array, dT)
                terminal_stock_prices = [
                    0 if price >= self.barrier_price else price for price in terminal_stock_prices]

            option_value_array = super().priceCallMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate,
                                                          self.maturity) if option == 'call' else super().pricePutMC_List(np.array(terminal_stock_prices), self.strike_price, self.risk_free_rate, self.maturity)
            option_mean_MC[i-1] = np.mean(option_value_array)
            option_std_MC[i-1] = np.std(option_value_array) / \
                np.sqrt(option_value_array.size)
            VaR[i-1] = self.optionMonteCarlo(1, 99, 5, option_value_array)

        return VaR

    # Return option VaR via MonteCarlo

    def optionMonteCarlo(self, no_options=1, confidence_level=99, premium=0, future_return=[]):
        portfolio_value = np.sort(np.array(future_return) - premium)*no_options
        return portfolio_value[int(((100-confidence_level)/100)*portfolio_value.size-1)]


if __name__ == "__main__":
    # price_tool = Vanila_Option(100, 0.1, 0.3, 110, 0.5)
    # (mean, std) = price_tool.valueWithoutCVA('put', 'MC', 50)
    # (mean_bsm, std_bsm) = price_tool.valueWithoutCVA('put', 'BSM')
    # print(mean)
    # price_tool.estimationGraph(mean, std, mean_bsm)

    price_tool = Barrier_Option(100, .08, 0.3, 100, 1, 150)
    # (mean, std) = price_tool.valueWithoutCVA(option='call', simulation = 50, steps=12)
    # print(mean[:10])
    (mean, std, cva_mean, cva_std) = price_tool.valueWithCVA(
        'call', 50, 12, 0.25, 175, 0.25, 0.2, 200)
    print(mean[:10])
    print(cva_mean[:10])

