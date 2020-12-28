import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
# from pyramid.arima import auto_arima
import matplotlib.pyplot as plt
import plotly
from plotly.plotly import plot_mpl



plotly.tools.set_credentials_file(username='thquyen11', api_key='DEU7LZZR5QP9QfznXjRO')
appl_daily_stockprice = 'AAPL.csv'

appl_df = pd.read_csv(appl_daily_stockprice, header=0, parse_dates=True, index_col='Date')
aapl_return_df = appl_df['Adj Close'].diff()

aapl_return_df.plot()
plt.show()