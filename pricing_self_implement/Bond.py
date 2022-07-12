import math
from tkinter import Y


class Bond:
    def __init__(self, nominal, coupon, compounding, maturity, YTM, accrual=0):
        self.nominal=nominal
        self.coupon=coupon/100
        self.compounding=compounding
        self.mauturity=maturity
        self.YTM=YTM/100
        self.accrual=accrual

    def get_coupon_flow(self):
        return [self.nominal*(self.coupon/self.compounding)]*int((self.compounding*(self.mauturity+self.accrual)))

    def get_capital_flow(self):
        capital_flow = [0]*int((self.compounding*(self.mauturity+self.accrual)))
        capital_flow[-1]=self.nominal
        return capital_flow

    def get_cashflow(self):
        print("cash flow coupon:")
        print(self.get_coupon_flow())
        print("cash flow capital:")
        print(self.get_capital_flow())

    def get_clean_price(self, type="formula"):
        return self.get_clean_price_formula() if type == "formula" else self.get_clean_price_discounting_flows()

    def get_clean_price_formula(self):
        c = self.nominal*(self.coupon/self.compounding)
        ytm = self.YTM
        k = self.compounding
        t = self.mauturity
        fv = self.nominal

        coupon_flow_discount = (c/ytm)*(1-1/math.pow(1+ytm/k,k*t))
        capital_flow_discount = fv/math.pow(1+ytm/k,k*t)
        return coupon_flow_discount + capital_flow_discount

    def get_clean_price_discounting_flows(self):
        c = self.nominal*(self.coupon/self.compounding)
        ytm = self.YTM
        k = self.compounding
        t = self.mauturity
        fv = self.nominal
        accr = self.accrual

        count_payments = int(k*(t+accr))
        clean_price = 0
        for i in range(count_payments):
            if i < count_payments - 1:
                clean_price += c/math.pow(1+ytm/k,k*t)
            else:
                clean_price += c/math.pow(1+ytm/k,k*t) + fv/math.pow(1+ytm/k,k*t)

        return clean_price

    def get_accrual_amt(self):
        return self.nominal*self.accrual*(self.coupon/self.compounding)

    def get_dirty_price(self):
        return self.get_clean_price() + self.get_accrual_amt()


bond=Bond(nominal=150000,coupon=5,compounding=2,maturity=8.593,YTM=2,accrual=0.407)
print(bond.get_coupon_flow())
print(bond.get_capital_flow())
print(bond.get_clean_price())
print(bond.get_accrual_amt())
print(bond.get_dirty_price())




