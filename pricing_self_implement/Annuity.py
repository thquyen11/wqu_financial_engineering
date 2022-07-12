import math


def FV(cash_flow, compounding_rate, maturity, defer_freq=0):
    """compute future value

    Args:
        PV (float): present value
        compounding_rate (float): interest rate (%)
        times (float): fraction of time

    Returns:
        float: future value
    """
    A = cash_flow
    y = compounding_rate
    # h = compounding_freq
    T = maturity
    M = defer_freq
    return (A/y)*(math.pow(1+y,T-M)-1)


def PV(cash_flow, compounding_rate, maturity, defer_freq=0):
    """compute present value

    Args:
        FV (float): future value
        compounding_rate (float): interest rate (%)
        times (float): fraction of time

    Returns:
        float: present value
    """
    A = cash_flow
    y = compounding_rate
    # h = compounding_freq
    T = maturity
    M = defer_freq
    return (A/y/math.pow(1+y,M))*(1-1/math.pow(1+y,T-M))


PV=PV(25000,0.055,35,15)

FV=FV(5000,0.04,10)


