#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Cointegration - BTCETH'))
	print(os.getcwd())
except:
	pass

#%%
from matplotlib import gridspec
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import seaborn as sns


#%%
btc_raw = pd.read_csv("BTC-USD.csv", index_col=0, parse_dates=True, dayfirst=True)


#%%
btc_raw['Returns'] = np.log(btc_raw['Adj Close'].astype(np.float)/btc_raw['Adj Close'].shift(1).astype(float))


#%%
btc_raw = btc_raw[datetime(2018,1,1):datetime(2019,7,13)]


#%%
btc_raw_insample = btc_raw[datetime(2018,1,1):datetime(2018,12,31)]


#%%
btc_raw_outsample = btc_raw[datetime(2019,1,1):datetime(2019,7,13)]


#%%
btc_raw = btc_raw_insample[1:]


#%%
eth_raw = pd.read_csv("ETH-USD.csv", index_col=0, parse_dates=True, dayfirst=True)


#%%
eth_raw['Returns'] = np.log(eth_raw['Adj Close'].astype(np.float)/eth_raw['Adj Close'].shift(1).astype(float))


#%%
eth_raw = eth_raw[datetime(2018,1,1):datetime(2019,7,13)]


#%%
eth_raw_outsample = eth_raw[datetime(2019,1,1):datetime(2019,7,13)]


#%%
eth_raw_insample = eth_raw[datetime(2018,1,1):datetime(2018,12,31)]


#%%
eth_raw = eth_raw_insample[1:]


#%%
returns1 = btc_raw['Returns'].values


#%%
returns2 = eth_raw['Returns'].values


#%%
Y1_t = btc_raw['Adj Close']


#%%
Y2_t = eth_raw['Adj Close']


#%%
Y1_t_Series = pd.Series(Y1_t, name='Bitcoin')


#%%
Y2_t_Series = pd.Series(Y2_t, name='Ethereum')


#%%
returns1_os = btc_raw_outsample['Returns'].values


#%%
returns2_os = eth_raw_outsample['Returns'].values


#%%
Y1_t_os = btc_raw_outsample['Adj Close']


#%%
Y2_t_os = eth_raw_outsample['Adj Close']


#%%
Y1_t_Series_os = pd.Series(Y1_t_os, name='Bitcoin')


#%%
Y2_t_Series_os = pd.Series(Y2_t_os, name='Ethereum')


#%%
_ = Y1_t_Series.plot()
_ = Y2_t_Series.plot()
_ = plt.xlabel('Time')
_ = plt.legend(['Bitcoin', 'Ethereum'], loc='upper left')


#%%
dY1_t = pd.Series(Y1_t, name='Δ Bitcoin').diff().dropna()


#%%
dY2_t = pd.Series(Y2_t, name='Δ Ethereum').diff().dropna()


#%%
_ = dY1_t.plot()
_ = dY2_t.plot()
_ = plt.xlabel('Time')
_ = plt.legend(['Δ Bitcoin', 'Δ Ethereum'], loc='upper left')


#%%
print('Loading all VAR/OLS/Optimal Lag functions')
get_ipython().run_line_magic('run', 'Cointegration.py')
print('Additional functions loaded')


#%%
data = pd.concat([btc_raw['Returns'], eth_raw['Returns']], axis = 1, keys = ['Bitcoin Returns', 'Ethereum Returns'])


#%%
Yt = np.vstack((Y1_t, Y2_t))


#%%
Yr = np.vstack((returns1, returns2))


#%%
dY = np.vstack((dY1_t, dY2_t))


#%%
maxlags = int(round(12*(len(Yr)/100.)**(1/4)))


#%%
print('Maxlags to test: %d' % maxlags)


#%%
maxlagOptimumVectorAR = GetOptimalLag(Yr, maxlags, modelType='VectorAR')


#%%
print(maxlagOptimumVectorAR)


#%%
model = VAR(Yr.T)


#%%
results = model.fit(maxlags, method='ols', ic='aic', trend='c', verbose=True)


#%%
results.summary()


#%%
# Stability Check
resultGetADFuller = GetADFuller(Y=dY1_t, maxlags = 0, regression='c')
roots = resultGetADFuller['roots']
IsStable(roots)


#%%
# Result from custom implementation

print("ADF Statistic: %f" % resultGetADFuller['adfstat'])


#%%
# Verify result from statsmodel implementation

resultadfuller = adfuller(dY1_t, maxlag=0, regression='c', autolag=None, regresults=True)
print(resultadfuller)


#%%
# Engle-Granger Self Implementation

Y2_t_d = np.vstack((np.ones(len(Y2_t)), Y2_t))
resultGetOLS = GetOLS(Y=Y1_t.values, X=Y2_t_d)

a_hat = resultGetOLS['beta_hat'][0,0]
beta2_hat = resultGetOLS['beta_hat'][0,1]

et_hat = Y1_t - np.dot(beta2_hat, Y2_t) - a_hat

result_et_hat_adf = GetADFuller(Y=et_hat, maxlags=1, regression='nc')
print('ADF Statistic: %f' % result_et_hat_adf['adfstat'])


#%%
# Verifying above with statsmodel

sm_result_et_hat_adf = adfuller(et_hat, maxlag=1, regression='nc', autolag=None, regresults=True)
print(sm_result_et_hat_adf)

resultols = OLS(Y1_t.T, Y2_t_d.T).fit()

resultols.summary2()


#%%
# Plot OLS Fit

# Generate equally spaced X values between the true X range
x_axis = np.linspace(Y2_t.min(), Y2_t.max(), 100)

#Plot the estimated dependent variable
Y1_t_hat = a_hat + beta2_hat * x_axis

# Plot own fit on top of seaborrn scatter + fit
plt.title('Cointegrating Regression: Bitcoin and Ethereum')
ax = sns.regplot(x=Y2_t_Series, y=Y1_t_Series, fit_reg=False)
ax.plot(x_axis, Y1_t_hat, 'r')
plt.legend(['OLS Fit', 'Real Values'], loc='lower right')


#%%
plt.figure(1, figsize=(15,20))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5])

et_hat_series = pd.Series(et_hat, name = 'Spread')

plt.subplot(gs[0])
plt.title('Cointegrating Spread $\hat{e}_t$ (Bitcoin and Ethereum)')
et_hat_series.plot()
plt.axhline(et_hat_series.mean(), color='red', linestyle='--')
plt.legend(['$\hat{e}_t$', 'mean={0:0.2g}'.format(et_hat_series.mean())], loc='lower right')
plt.xlabel('')


#%%
# Spread Histogram

plt.subplot(gs[1])

from scipy import stats

ax = sns.distplot(et_hat_series, bins=20, kde=False, fit=stats.norm)
plt.title('Distribution of Cointegrating Spread For Bitcoin And Ethereum')

# Get the fitted parameters used by Seaborn
(mu, sigma) = stats.norm.fit(et_hat_series)
print ('mu={%f}, sigma={%f}' % (mu, sigma))

# Legend and Labels

plt.legend(["Normal Dist. Fit ($\mu \sim${0}, $\sigma=${1:.2f})".format(0, sigma),'$\hat{e}_t$'])
plt.xlabel('Value')
plt.ylabel('Frequency')


#%%

from statsmodels.graphics.tsaplots import plot_pacf

ax = plt.subplot(gs[2])
plot_pacf(et_hat_series, lags=50, alpha=0.01, ax=ax)
plt.title('')
plt.xlabel('Lags')
plt.ylabel('PACF')


#%%
from statsmodels.tsa.ar_model import AR

resultGetVectorAR = GetVectorAR(et_hat[None,:], maxlags=1, trend='c')
resultGetAR = AR(et_hat).fit(maxlag=3, trend='c', method='cmle')
print('Is AR({%d}) model stable: {%s}' % (resultGetAR.k_ar, str(IsStable(resultGetAR.roots))))
print('Is VectorAR({%s}) model stable: {%s}' % (resultGetVectorAR['maxlags'], str(IsStable(resultGetVectorAR['roots']))))
print('NOTE THAT VECTOR_AR[1] IS *NOT* STABLE')


#%%
tau = 1.0 / 252.0
print(resultGetVectorAR['sigma_hat'][0])
C = resultGetVectorAR['beta_hat'][0][0]
B = resultGetVectorAR['beta_hat'][0][1]
theta = - np.log(B) / tau
mu_e = C / (1.0 - B)
sigma_ou = np.sqrt((2 * theta / 1 - np.exp(-2 * theta * tau))) * resultGetVectorAR['sigma_hat'][0]
sigma_e = sigma_ou / np.sqrt(2 * theta)
halflife = np.log(2) / theta


#%%
print(' AR({%f}): C={%f}, B={%f}, tau={%f}, theta={%f}, mu_e={%f}, sigma_ou={%f}, sigma_e={%f}, halflife={%f}' % (resultGetVectorAR['maxlags'], C, B, tau, theta, mu_e, sigma_ou, sigma_e, halflife))


#%%
# AR(3)

sm_C = resultGetAR.params[0]
sm_B = resultGetAR.params[1]
sm_theta = - np.log(sm_B) / tau
sm_mu_e = sm_C / (1. - sm_B)
sm_sigma_ou = np.sqrt((2 * sm_theta / (1 - np.exp(-2 * sm_theta * tau))) * resultGetAR.sigma2)
sm_sigma_e = sm_sigma_ou / np.sqrt(2 * abs(sm_theta))
sm_halflife = np.log(2) / sm_theta
print('SM AR({%f}): sm_C={%f}, sm_B={%f}, tau={%f}, sm_theta={%f}, sm_mu_e={%f}, sm_sigma_ou={%f}, sm_sigma_e={%f}, sm_halflife={%f}' % (resultGetAR.k_ar, sm_C, sm_B, tau, sm_theta, sm_mu_e, sm_sigma_ou, sm_sigma_e, sm_halflife))


#%%
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

Y_e = et_hat_series.diff()[1:]
X_e = et_hat_series.shift(1)[1:]
X_e = add_constant(X_e)
ols = OLS(Y_e, X_e).fit()
X_e = X_e.iloc[:, 1]

ols.summary2()


#%%
et_hat_series.plot(label='$\hat{e}_t$', figsize=(15, 7))

plt.title('Trading Bounds For Cointegrated Spread (Bitcoin and Ethereum)')
plt.axhline(0, color='grey', linestyle='-') # axis line
plt.axhline(mu_e, color='green', linestyle=':', label='AR(1) OU $\mu_e \pm \sigma_{eq}$')
plt.axhline(sm_mu_e, color='red', linestyle='--', label='AR(3) OU $\mu_e \pm \sigma_{eq}$')
plt.axhline(sigma_e, color='green', linestyle=':')
plt.axhline(-sigma_e, color='green', linestyle=':')
plt.axhline(sm_sigma_e, color='red', linestyle='--')
plt.axhline(-sm_sigma_e, color='red', linestyle='--')
plt.legend(loc='lower right')


#%%
et_hat_n = GetZScore(et_hat, mean=sm_mu_e, sigma=sm_sigma_e)
et_hat_n_series = pd.Series(et_hat_n, name='et_hat_n')
et_hat_n_series.plot()
plt.axhline(1, color='red', linestyle='--')
plt.axhline(-1, color='green', linestyle='--')
plt.legend(['z-score $\hat{e}_t$', '+1', '-1'], loc='upper right')
plt.axhline(0, color='grey')


#%%
pnl_is = Get_Pnl_DF(et_hat_series, mean=sm_mu_e, sigma=sm_sigma_e)
pnl_is.tail()


#%%
pnl_is.loc[pnl_is['pnl'].isnull(), 'pnl']


#%%
get_ipython().run_line_magic('run', 'getpyfolio.py')
# Not running properly for me on this machine because I need C++ build tools to get NumPy/Lapack going - can someone else give this a run?


#%%
plot_drawdown_periods(pnl_is['pnl'], top=5)


#%%
# Buy-and-Hold Bitcoin PnL
btc_return = (btc_raw_insample['Adj Close'] - btc_raw_insample['Adj Close'].shift(1)).dropna().cumsum()
btc_return[-1] = -btc_raw_insample.iloc[0]['Adj Close'] + btc_raw_insample.iloc[-1]['Adj Close']
plt.plot(btc_return)
plt.show()


#%%
# OUT OF SAMPLE TESTING

# Construct the out-of-sample spread

et_hat_os = Y1_t_os - np.dot(beta2_hat, Y2_t_os) - a_hat

# Normalise to OU bounds
et_hat_os_norm = GetZScore(et_hat_os, mean=sm_mu_e, sigma=sm_sigma_e)

et_hat_os_norm = pd.Series(et_hat_os_norm, name = 'et_hat_os_norm')
et_hat_os_norm.plot()

plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['z-score $\hat{e}_t$', '+1', '-1'], loc='upper left')
plt.axhline(0, color='grey')


#%%
pnl_os = Get_Pnl_DF(et_hat_os, mean=sm_mu_e, sigma=sm_sigma_e)
pnl_os.tail()


#%%
pnl_is[:-1]['pnl_cum'][-1]


#%%
get_ipython().run_line_magic('run', 'getpyfolio.py')


#%%
pnl_is.index[-1]


#%%
df_temp = pnl_is[:-1]['pnl_cum']
k = df_temp[-1]
df_temp.plot()
plot_drawdown_periods(pnl_os['pnl'], k=k, top=5)
plt.axvline(df_temp.index[-1], color='black', linestyle='--')
plt.legend(['in-sample', 'out-of-sample', 'boundary'], loc='upper left')


#%%



