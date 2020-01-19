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

    def priceCOS(self):
        k_log = np.log(self.strike_price)
        t_max = 20
        N = 100
        delta_t = t_max/N
        from_1_to_N = np.linspace(1, N, N)
        t_n = (from_1_to_N-1/2)*delta_t

        # Approximate integral estimates
        first_integral = sum(
            (((np.exp(-1j*t_n*k_log)*self.c_M2(t_n)).imag)/t_n)*delta_t)
        second_integral = sum(
            (((np.exp(-1j*t_n*k_log)*self.c_M1(t_n)).imag)/t_n)*delta_t)
        return self.initial_stock_price*(0.5 + first_integral/np.pi) - np.exp(-self.risk_free_rate*self.maturity)*self.strike_price*(0.5+second_integral/np.pi)

    def priceCOS_Exp(self, N, b2, b1):
        price = self.v_n(self.strike_price, b2, b1, 0)*self.logchar_func(
            0, self.initial_stock_price, self.risk_free_rate, self.sigma, self.strike_price, self.maturity)/2
        for n in range(1, N):
            price += self.logchar_func(n*np.pi/(b2-b1), self.initial_stock_price, self.risk_free_rate, self.sigma,
                                       self.strike_price, self.maturity)*np.exp(-1j*n*np.pi*b1/(b2-b1))*self.v_n(self.strike_price, b2, b1, n)
        return price.real*np.exp(-self.risk_free_rate*self.maturity)

    def v_n(self, strike_price, b2, b1, n):
        return 2*strike_price*(self.upsilon_n(b2, b1, b2, 0, n)-self.psi_n(b2, b1, b2, 0, n))/(b2-b1)

    def logchar_func(self, u, initial_stock_price, risk_free_rate, sigma, strike_price, maturity):
        return np.exp(1j*u*(np.log(initial_stock_price/strike_price)+(risk_free_rate-sigma**2/2)*maturity)-(sigma**2)*maturity*(u**2)/2)

    # Fourier charateristic function
    def c_M1(self, t):
        return np.exp(1j*t*(np.log(self.initial_stock_price)+(self.risk_free_rate-self.sigma**2/2)*self.maturity)-0.5*(self.sigma**2)*self.maturity*(t**2))

    # Fourier charateristic function
    def c_M2(self, t):
        return np.exp(1j*t*self.sigma**2*self.maturity)*self.c_M1(t)

    def upsilon_n(self, b2, b1, d, c, n):
        npi_d = np.pi*n*(d-b1)/(b2-b1)
        npi_c = np.pi*n*(c-b1)/(b2-b1)
        val_one = (np.cos(npi_d)*np.exp(d)-np.cos(npi_c)*np.exp(c))
        val_two = (n*np.pi*(np.sin(npi_d)*np.exp(d) -
                            np.sin(npi_c)*np.exp(c))/(b2-b1))
        return (val_one+val_two)/(1+(n*np.pi/(b2-b1))**2)

    def psi_n(self, b2, b1, d, c, n):
        return d-c if n == 0 else (b2-b1)*(np.sin(n*np.pi*(d-b1)/(b2-b1))-np.sin(n*np.pi*(c-b1)/(b2-b1)))/(n*np.pi)

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
    def priceCallBSM(self):
        d1 = self.__calcualte_d1(
            self.initial_stock_price, self.strike_price, self.risk_free_rate, self.sigma, self.maturity)
        d2 = self.__calculate_d2(d1, self.sigma, self.maturity)
        return self.initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*self.strike_price*math.exp(-self.risk_free_rate*self.maturity)

    # Pricing Put Option via Black Scholes Model
    def pricePutBSM(self):
        d1 = self.__calcualte_d1(
            self.initial_stock_price, self.strike_price, self.risk_free_rate, self.sigma, self.maturity)
        d2 = self.__calculate_d2(d1, self.sigma, self.maturity)
        return -self.initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*self.strike_price*math.exp(-self.risk_free_rate*self.maturity)

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
    price_tool = Vanila_Option(100, 0.06, 0.3, 110, 1)
    # (mean, std) = price_tool.valueWithoutCVA('put', 'MC', 50)
    # (mean_bsm, std_bsm) = price_tool.valueWithoutCVA('put', 'BSM')
    print(price_tool.priceCallBSM())
    print(price_tool.priceCOS())

    c1 = 0.06
    c2 = 1*0.3**2
    c4 = 0
    L = 10
    b1 = c1-L*np.sqrt(c2-np.sqrt(c4))
    b2 = c1+L*np.sqrt(c2-np.sqrt(c4))
    COS_callprice = [None]*50
    for i in range(1, 51):
        COS_callprice[i-1] = price_tool.priceCOS_Exp(i, b2, b1)
    plt.plot(COS_callprice)
    plt.plot([price_tool.priceCallBSM()]*50)
    plt.xlabel("N")
    plt.ylabel("Call Price")
    plt.show()

    # price_tool.estimationGraph(mean, std, mean_bsm)

    # price_tool = Barrier_Option(100, .08, 0.3, 100, 1, 150)
    # # (mean, std) = price_tool.valueWithoutCVA(option='call', simulation = 50, steps=12)
    # # print(mean[:10])
    # (mean, std, cva_mean, cva_std) = price_tool.valueWithCVA(
    #     'call', 50, 12, 0.25, 175, 0.25, 0.2)
    # print(mean[:10])
    # print(cva_mean[:10])
