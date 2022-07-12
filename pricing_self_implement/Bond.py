import math
from tkinter import Y


class Bond:
    def __init__(self, nominal=100, coupon_pct=0, compounding_freq=1, full_coupon_period=1, broken_period=0):
        self.nominal=nominal
        self.coupon_pct=coupon_pct
        self.coupon=coupon_pct/100
        self.compounding_freq=compounding_freq
        self.full_coupon_period=full_coupon_period
        # self.ytm=ytm_pct/100
        self.broken_period=broken_period
        self.accrual = (self.coupon_pct/self.compounding_freq)*(1-self.broken_period)/100

    @classmethod
    def from_zcb(cls, nominal=1, compounding_freq=1, full_coupon_period=1):
        return cls(nominal=nominal, coupon_pct=0, compounding_freq=compounding_freq, full_coupon_period=full_coupon_period)

    def get_coupon_flow(self):
        return [self.nominal*(self.coupon/self.compounding_freq)]*int((self.compounding_freq*self.full_coupon_period))

    def get_capital_flow(self):
        capital_flow = [0]*int((self.compounding_freq*self.full_coupon_period))
        capital_flow[-1]=self.nominal
        return capital_flow

    def dirty_price(self, ytm_pct, type=""):
        return self.dirty_price_approx(ytm_pct) if type == "approx" or type == "approximate" else self.dirty_price_formula(ytm_pct)

    def dirty_price_approx(self, ytm_pct):
        c = self.nominal*(self.coupon_pct/self.compounding_freq)
        ytm = ytm_pct/100
        k = self.compounding_freq
        t = self.full_coupon_period
        fv = self.nominal

        interest_flow_discount = (c/ytm)*(1-1/math.pow(1+ytm/k,k*t))
        capital_flow_discount = fv/math.pow(1+ytm/k,k*t)

        return interest_flow_discount + capital_flow_discount

    def dirty_price_formula(self,ytm_pct):
        C = self.coupon_pct
        ytm = ytm_pct/100
        h = self.compounding_freq
        n = self.full_coupon_period
        N = 100
        dt = self.broken_period
        interest_flow_discount = 0
        capital_flow_discount = 0
        broken_period_flow_discount = 0

        # zero coupon bond skip this step
        if C != 0:
            for i in range(n+1):
                # if no broken period, skip coupon payment at t=0
                if i == 0 and dt == 0:
                    continue
                else:
                    interest_flow_discount += (C/h)/math.pow(1+ytm/h,i)
            
        capital_flow_discount = N/math.pow(1+ytm/h,n)
        broken_period_flow_discount = 1/math.pow(1+ytm/h,dt)
        
        return broken_period_flow_discount*(interest_flow_discount + capital_flow_discount)

    def clean_price(self, ytm_pct):
        return self.dirty_price(ytm_pct) - self.accrual

    def YTM(self, price):
        C = self.coupon_pct
        N = 100
        P = price
        n = self.broken_period + self.full_coupon_period
        return (C + (N-P)/n)/((N+P)/2)

class PerpetualBond:
    def __init__(self, nominal, coupon):
        self.nominal = nominal
        self.coupon = coupon

    def price(interest_rate, growth_rate=0):
        return self.nominal*self.coupon/(interest_rate+growth_rate)


# zcb1=Bond.from_zcb(50000,2,3.5*2)
# price1=zcb1.dirty_price(ytm_pct=2)*zcb1.nominal/100

# bond1=Bond(nominal=100,coupon_pct=6,full_coupon_period=3,broken_period=0.5)
# dirty_price1=bond1.dirty_price(ytm_pct=4)










