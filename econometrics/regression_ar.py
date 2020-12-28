#%%
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.stattools import jarque_bera
from sklearn.model_selection import train_test_split


# RETREIVE TIMESERIES DATA
# np.random.seed(42)
# x=w=np.random.normal(size=int(1000))
# a=0.6

# for t in range(len(x)):
#     x[t] = a*x[t-1]+w[t]

start_date='2010-01-01'
end_date='2016-12-31'
ticker='AAPL'

data = yf.download(ticker, start=start_date, end=end_date)
data=data.resample('M').mean()
x = data['Adj Close'].diff().dropna()


# FIND THE AR(p) MODEL FOR THE DATA
#%%
sns.tsplot(data=x)
plot_acf(x, lags=20, alpha=0.05)
plot_pacf(x, lags=20, alpha=0.05)
# plt.show()

#%%
N=10
AIC = np.zeros((N,1))

for lag in range(N):
    model = ARMA(x, order=(lag,0)).fit()
    AIC[lag] = model.aic

AIC_min = np.min(AIC)
model_p = np.argmin(AIC)
ar_model=ARMA(x, order=(model_p, 0)).fit()
print(ar_model.summary())

#%%
# CHECK IF RESIDUAL IS WHITE NOISE
residuals = ar_model.resid
score, p_value, _,_ = jarque_bera(residuals)
lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(residuals, lags=[model_p], boxpierce=True)

if p_value< 0.05:
    print("We have reason to suspect that the residuals are not normally distributed")
else:
    print("The residuals seem normally distributed")

if pvalue < 0.05:
    print("We have reason to suspect that the residuals are autocorrelated")
else:
    print("The residuals seem like white noise")





