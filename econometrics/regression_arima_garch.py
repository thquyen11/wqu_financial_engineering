#%%
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from statsmodels.stats.stattools import jarque_bera
from sklearn.model_selection import train_test_split

#%%
def _get_best_model(timeseries):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_range = range(5)
    d_range = range(2)
    for p in pq_range:
        for d in d_range:
            for q in pq_range:
                try:
                    tmp_mdl = ARIMA(timeseries, order=(p, d, q)).fit(
                        method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_mdl = tmp_mdl
                except:
                    continue

    print(f"aic: {best_aic}, order: {best_order}")
    return best_aic, best_order, best_mdl

#%%
def _check_heteroskedastic_behavior(residuals):
    score, pvalue_jb, _, _ = jarque_bera(residuals)
    lbvalue, pvalue_lb = acorr_ljungbox(
        residuals**2, lags=[20], boxpierce=False)

    if pvalue_jb < 0.05:
        print("We have reason to suspect that the residuals are not normally distributed")
    else:
        print("The residuals seem normally distributed")

    if pvalue_lb < 0.05:
        print("We have reason to suspect that the residuals are autocorrelated")
        return True
    else:
        print("The residuals seem like white noise")
        return False

#%%
def _get_garch_model(residuals):
    pq_range = range(5)
    o_range = range(2)
    best_aic = np.inf
    best_model = None
    best_order = None

    for p in pq_range:
        for o in o_range:
            for q in pq_range:
                try:
                    tmp_model = arch_model(residuals, p=p, o=o, q=q, dist='StudentsT').fit(
                        update_freq=5, disp='off')
                    tmp_aic = tmp_model.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_model = tmp_model
                        best_order = (p, o, q)
                except:
                    continue

    print(f'aic: {best_aic}, order: {best_order}')
    return best_aic, best_order, best_model

#%%
if __name__ == "__main__":
    start_date = '2000-01-01'
    end_date = '2002-11-30'
    # end_date = '2016-11-30'
    symbol = '^GSPC'
    windowLength = 252

    data = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
    lrets = np.log(data/data.shift(1)
                   ).replace([np.inf, -np.inf], np.nan).dropna()
    foreLength = len(lrets)-windowLength
    signal = 0*lrets[-foreLength:]

#%%
    for d in range(foreLength):
        TS = lrets[(1+d):(windowLength+d)]
        res_setup = _get_best_model(lrets)
        order_arimia = res_setup[1]
        mdl_arimia = res_setup[2]

        if _check_heteroskedastic_behavior(mdl_arimia.resid):
            res_garch = _get_garch_model(mdl_arimia.resid)
            mdl_garch = res_garch[2]
            # print(mdl_garch.summary())

            # Generate a forecast of next day return
            out = mdl_garch.forecast(horizon=1, start=None, align='origin')
            signal.iloc[d] = np.sign(out.mean['h.1'].iloc[-1])

#%%
returns = pd.DataFrame(index=signal.index, columns=[
                       'Buy and Hold', 'Strategy'])
returns['Buy and Hold'] = lrets[-foreLength:]
returns['Strategy'] = signal[symbol]*returns['Buy and Hold']

eqCurves = pd.DataFrame(index=signal.index,
                        columns=['Buy and Hold', 'Strategy'])
eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
eqCurves['Strategy'].plot(figsize=(10, 8))
eqCurves['Buy and Hold'].plot()
plt.legend()
plt.show()
