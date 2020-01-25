import numpy as np
from scipy.stats import norm
import math
import random
import matplotlib.pyplot as plt


class EuropeanOption:
    def __init__(self, initial_stock_price, risk_free_rate, sigma, strike_price, maturity):
        self.initial_stock_price = initial_stock_price
        self.risk_free_rate = risk_free_rate
        self.sigma = sigma
        self.strike_price = strike_price
        self.maturity = maturity

    def priceDFT(self, N, t_max, type='call'):
        k_log = np.log(self.strike_price)
        delta_t = t_max/N
        from_1_to_N = np.linspace(1, N, N)
        t_n = (from_1_to_N-1/2)*delta_t

        # Approximate integral estimates
        first_integral = sum(
            (((np.exp(-1j*t_n*k_log)*self.c_M2(t_n)).imag)/t_n)*delta_t)
        second_integral = sum(
            (((np.exp(-1j*t_n*k_log)*self.c_M1(t_n)).imag)/t_n)*delta_t)
        if type == 'call':
            return self.initial_stock_price*(0.5 + first_integral/np.pi) - np.exp(-self.risk_free_rate*self.maturity)*self.strike_price*(0.5 + second_integral/np.pi)
        elif type == 'put':
            return self.initial_stock_price*(first_integral/np.pi - 0.5) - np.exp(-self.risk_free_rate*self.maturity)*self.strike_price*(second_integral/np.pi - 0.5)

    def priceDFT_COS(self, N, type='call'):
        c1 = self.risk_free_rate
        c2 = (self.sigma**2)*self.maturity
        c4 = 0
        L = 10  # TODO: how to get L value?
        b1 = c1-L*np.sqrt(c2-np.sqrt(c4))
        b2 = c1+L*np.sqrt(c2-np.sqrt(c4))
        price = self.v_n(self.strike_price, b2, b1, 0, type)*self.logchar_func(
            0, self.initial_stock_price, self.risk_free_rate, self.sigma, self.strike_price, self.maturity, 'cos')/2
        for n in range(1, N):
            price += self.logchar_func(n*np.pi/(b2-b1), self.initial_stock_price, self.risk_free_rate, self.sigma,
                                       self.strike_price, self.maturity, 'cos')*np.exp(-1j*n*np.pi*b1/(b2-b1))*self.v_n(self.strike_price, b2, b1, n, type)
        return price.real*np.exp(-self.risk_free_rate*self.maturity)

    def priceFFT(self, b, n, delta, alpha, log_strike):
        x = np.exp(1j*b*n*delta)*self.c_func(n*delta, alpha, log_strike)*delta
        x[0] = x[0]*0.5
        x[-1] = x[-1]*0.5
        # xhat = np.fft.fft(x).real
        xhat = self.fft(x).real
        return np.exp(-alpha*log_strike)*xhat/np.pi

    # Can use numpy fft function instead
    def fft(self, x):
        N = len(x)
        if N == 1:
            return x
        else:
            ek = self.fft(x[:-1:2])
            ok = self.fft(x[1::2])
            m = np.array(range(int(N/2)))
            okm = ok*np.exp(-1j*2*np.pi*m/N)
            return np.concatenate((ek+okm, ek-okm))

    def c_func(self, v, alpha, log_strike):
        val1 = np.exp(-self.risk_free_rate*self.maturity) * \
            self.logchar_func(v-(alpha+1)*1j, self.initial_stock_price,
                              self.risk_free_rate, self.sigma, log_strike, self.maturity, 'fft')
        val2 = alpha**2+alpha-v**2+1j*(2*alpha+1)*v
        return val1/val2

    def v_n(self, strike_price, b2, b1, n, type):
        if type == 'call':
            return 2*strike_price*(self.upsilon_n(b2, b1, b2, 0, n)-self.psi_n(b2, b1, b2, 0, n))/(b2-b1)
        elif type == 'put':
            return 2*strike_price*(self.psi_n(b2, b1, 0, b1, n) - self.upsilon_n(b2, b1, 0, b1, n))/(b2-b1)

    def logchar_func(self, u, initial_stock_price, risk_free_rate, sigma, strike_price, maturity, type):
        if type == 'fft':
            return np.exp(1j*u*(np.log(initial_stock_price)+(risk_free_rate-sigma**2/2)*maturity)-(sigma**2)*maturity*(u**2)/2)
        elif type == 'cos':
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

    def setStrikePrice(self, new_strike):
        try:
            self.strike_price = new_strike
            return True
        except Exception as e:
            print(e)
            return False

    # Pricing Call Option via Black Scholes Model
    def priceBSM(self, type):
        d1 = (np.log(self.initial_stock_price/self.strike_price) + (self.risk_free_rate +
                                                                    self.sigma**2/2)*self.maturity)/(self.sigma*np.sqrt(self.maturity))
        d2 = d1 - self.sigma*np.sqrt(self.maturity)
        if type == 'call':
            return self.initial_stock_price*norm.cdf(d1) - norm.cdf(d2)*self.strike_price*np.exp(-self.risk_free_rate*self.maturity)
        elif type == 'put':
            return -self.initial_stock_price*norm.cdf(-d1) + norm.cdf(-d2)*self.strike_price*np.exp(-self.risk_free_rate*self.maturity)


if __name__ == "__main__":
    r = 0.1
    K = 100
    S0 = 120
    T = 2
    sigma = .25

    option = EuropeanOption(S0, r, sigma, K, T)
    putprice_analytical = option.priceBSM('put')
    print("put price Black-Scholes: ", putprice_analytical)

    N = 200
    t_max = 40
    putprice_dft = option.priceDFT(N, t_max, 'put')
    print("put price Fourier Transform: ", putprice_dft)

    putprice_cos = [None]*50
    for i in range(1, 51):
        putprice_cos[i-1] = option.priceDFT_COS(i, 'put')
    print("option price, Fourier-Cosine: ", putprice_cos)
    plt.subplot(311)
    plt.plot(putprice_cos)
    plt.plot([putprice_analytical]*50)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel("Number of integral rectangles")
    plt.ylabel("Put Price")
    plt.title("Fourier-Cosine Option Pricing")

    N = 2**10
    delta = 0.25
    alpha = -1.5  # Call > 1, Put < -1
    n = np.array(range(N))
    delta_k = 2*np.pi/(N*delta)
    b = delta_k*(N-1)/2
    log_strike = np.linspace(-b, b, N)

    putprice_fft = option.priceFFT(b, n, delta, alpha, log_strike)
    print("option price, Fast Fourier Transform: ", putprice_fft)

    option.setStrikePrice(np.exp(log_strike))
    putprice_analytical = option.priceBSM('put')
    print("option price, Black-Scholes: ", putprice_analytical)
    plt.subplot(313)
    plt.plot(np.exp(log_strike), putprice_fft, 'blue')
    plt.plot(np.exp(log_strike), putprice_analytical, 'red')
    plt.xlim(0, 100)
    plt.ylim(0, 5)
    plt.xlabel("Strike")
    plt.ylabel("Put Price")
    plt.title("Fast Fourier Option Pricing")
    plt.show()
