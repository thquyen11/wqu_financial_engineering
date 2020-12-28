import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
import matplotlib.pyplot as plt
import plotly
from plotly.plotly import plot_mpl



plotly.tools.set_credentials_file(username='thquyen11', api_key='DEU7LZZR5QP9QfznXjRO')

jpm_stock='JPM.csv'
sp500='^GSPC.csv'
caseshiller='CSUSHPINSA.csv'

jpm_df=pd.read_csv(jpm_stock, header=0, index_col='Date', parse_dates=True)
sp500_df=pd.read_csv(sp500, header=0, index_col='Date', parse_dates=True)
shiller_df=pd.read_csv(caseshiller, header=0, index_col='DATE', parse_dates=True)

print(jpm_df.info())
print('Average stock value ' + str(jpm_df['Adj Close'].mean()))
print('Stock volatility ' + str(jpm_df['Adj Close'].std()))
print('Daily stock return ' + str(jpm_df['Adj Close'].pct_change()))


# Two-variable regression
dfx=sm.add_constant(sp500_df['Adj Close'])
jpm_sp500_regression = sm.OLS(jpm_df['Adj Close'], dfx).fit()

print(jpm_sp500_regression.summary())
print('Pearson covariance: ' + str((jpm_df['Adj Close']).corr(sp500_df['Adj Close'])))


# ---------------------------------------------
# Forecast S&P/Case-Schiller: unit root exist
shiller_df=shiller_df.resample('M').last()
# shiller_df.plot()
# plt.show()
adf_shiller = adfuller(shiller_df['CSUSHPINSA'])
print('ADF test p-value ' + str(adf_shiller[1]))

# Trend, Seasonal check
figure = seasonal_decompose(shiller_df, model='multiplicative').plot()
plot_mpl(figure)


# ACF PACF: proposed ARIMA(0,1,4)
# fig, axes = plt.subplots(4, 3)

# shiller_df['CSUSHPINSA'].plot(ax=axes[0,0], title='Original Series')
# plot_acf(shiller_df['CSUSHPINSA'], ax=axes[0,1])
# plot_pacf(shiller_df['CSUSHPINSA'], ax=axes[0,2])

# shiller_df['CSUSHPINSA'].diff().dropna().plot(ax=axes[1,0], title='1st Order Differencing')
# plot_acf(shiller_df['CSUSHPINSA'].diff().dropna(), ax=axes[1,1])
# plot_pacf(shiller_df['CSUSHPINSA'].diff().dropna(), ax=axes[1,2])

# shiller_df['CSUSHPINSA'].diff().diff().dropna().plot(ax=axes[2,0], title='2nd Order Differencing')
# plot_acf(shiller_df['CSUSHPINSA'].diff().diff().dropna(), ax=axes[2,1])
# plot_pacf(shiller_df['CSUSHPINSA'].diff().diff().dropna(), ax=axes[2,2])

# shiller_df['CSUSHPINSA'].diff().diff().diff().dropna().plot(ax=axes[3,0], title='3rd Order Differencing')
# plot_acf(shiller_df['CSUSHPINSA'].diff().diff().diff().dropna(), ax=axes[3,1])
# plot_pacf(shiller_df['CSUSHPINSA'].diff().diff().diff().dropna(), ax=axes[3,2])

# plt.show()


# Build Seasonal ARIMA automatically
stepwise_model=auto_arima(shiller_df['CSUSHPINSA'], start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, start_Q=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warning=True, stepwise=True)
print(stepwise_model.aic())


# Build ARIMA(1,1,3) manually
# arima_model=ARIMA(shiller_df['CSUSHPINSA'], order=(1,1,4)).fit()
# print(arima_model.summary())

# arima_model=ARIMA(shiller_df['CSUSHPINSA'], order=(1,1,3)).fit()
# print(arima_model.summary())


# Plot residuals
# residuals=pd.DataFrame(arima_model.resid)
# fig, axes = plt.subplots(1,2)
# residuals.plot(title='Residuals', ax=axes[0])
# residuals.plot(kind='kde', title='Density', ax=axes[1])
# plt.show()


# Actual vs Fitted
# arima_model.plot_predict(dynamic=False)
# plt.show()
