import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from statsmodels.stats.stattools import jarque_bera
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn


def find_cointegrated_pairs(data, columns, plot=False):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pvalue_min = np.inf
    alpha = 0.05
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            return_S1 = data[columns[i]]
            return_S2 = data[columns[j]]
            res_coint = coint(return_S1, return_S2)
            score_matrix[i, j] = res_coint[0]
            pvalue_matrix[i, j] = res_coint[1]
            print(
                f'pvalue: {res_coint[1]} of pairs: {(columns[i], columns[j])}')
            if res_coint[1] < alpha and res_coint[1] < pvalue_min:
                pvalue_min = res_coint[1]
                pairs = [columns[i], columns[j]]

    if plot is True:
        # Illustrate the co-integrated pairs
        seaborn.heatmap(pvalue_matrix, xticklabels=keys,
                        yticklabels=keys, cmap='RdYlGn_r',
                        mask=(pvalue_matrix >= 0.98))
        plt.show()

    return score_matrix, pvalue_matrix, pairs


def get_arima_model(timeseries):
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

    print(f"ARIMA - aic: {best_aic}, order: {best_order}")
    return best_aic, best_order, best_mdl


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


def get_garch_model(residuals):
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

    print(f'GARCH - aic: {best_aic}, order: {best_order}')
    return best_aic, best_order, best_model


def predict_future_trend(data, plot=False):
    windowLength = 50
    foreLength = len(data) - windowLength
    signal = 0*data[-foreLength:]
    forecast_output = []
    forecast = pd.DataFrame(index=signal.index, columns=['Forecast'])

    for d in range(foreLength):
        TS = data[(1+d):(windowLength+d)]
        # TS = data[d:(windowLength+d)]
        out = forecast_nextday(TS)
        forecast_output.append(out)
        signal.iloc[d] = np.sign(out)

    if plot is True:
        returns = pd.DataFrame(index=signal.index, columns=[
                               'Buy and Hold', 'Strategy'])
        returns['Buy and Hold'] = return_S1[-foreLength:]
        returns['Strategy'] = signal*returns['Buy and Hold']
        eqCurves = pd.DataFrame(index=signal.index, columns=[
                                'Buy and Hold', 'Strategy'])
        eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
        eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        eqCurves['Strategy'].plot(figsize=(10, 8))
        eqCurves['Buy and Hold'].plot()
        plt.legend()
        plt.show()

    forecast['Forecast'] = forecast_output
    return forecast['Forecast']


def forecast_nextday(timeseries):
    res_setup = get_arima_model(timeseries)
    aic_arima = res_setup[0]
    order_arima = res_setup[1]
    mdl_arima = res_setup[2]

    if _check_heteroskedastic_behavior(mdl_arima.resid):
        res_garch = get_garch_model(mdl_arima.resid)
        aic_garch = res_garch[0]
        order_garch = res_garch[1]
        mdl_garch = res_garch[2]
        # print(mdl_garch.summary())
        out = mdl_garch.forecast(horizon=1, start=None, align='origin')
        return out.mean['h.1'].iloc[-1]

    else:
        return mdl_arima.forecast()[0]


def initiate_trading_signal(spread, series_1, series_2, plot=False, series1_label='Series 1', series2_label='Series 2'):
    mean = spread.mean()
    upper_threshold = mean + spread.std()
    lower_threshold = mean - spread.std()
    buy_signal = spread.copy()
    sell_signal = spread.copy()
    close_signal = spread.copy()
    buy_signal[spread > lower_threshold] = 0
    sell_signal[spread < upper_threshold] = 0
    close_signal[spread <= mean - (upper_threshold - lower_threshold)/4] = 0
    close_signal[spread >= mean + (upper_threshold - lower_threshold)/4] = 0

    trading_signal = 0*spread.copy()
    trading_signal[buy_signal != 0] = 1
    trading_signal[sell_signal != 0] = -1
    trading_signal[close_signal != 0] = 0.5

    # Graph of return spread
    if plot is True:
        plt.figure(figsize=(15, 7))
        plt.plot(spread)
        plt.axhline(0, color='black')
        plt.axhline(upper_threshold, color='red', linestyle='--')
        plt.axhline(lower_threshold, color='green', linestyle='--')
        plt.legend(['Difference of Logarithm Return', 'Mean',
                    'Upper Threshold', 'Lower Threshold'])
        plt.show()

    if plot is True:
        buyR = 0*series_1.copy()
        sellR = 0*series_1.copy()
        # When buying the ratio, buy series_1 and sell series_2
        buyR[buy_signal != 0] = series_1[buy_signal != 0]
        sellR[buy_signal != 0] = series_2[buy_signal != 0]
        # When selling the ratio, sell series_1 and buy series_2
        buyR[sell_signal != 0] = series_2[sell_signal != 0]
        sellR[sell_signal != 0] = series_1[sell_signal != 0]

        series_1.plot(color='b')
        series_2.plot(color='c')
        buyR.plot(color='g', linestyle='None', marker='^')
        sellR.plot(color='r', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, min(series_1.min(), series_2.min()),
                  max(series_1.max(), series_2.max())))
        plt.legend([series1_label, series2_label, 'Buy Signal', 'Sell Signal'])
        plt.show()

    return trading_signal


def get_log_return(data):
    return np.log(data/data.shift(1)
                  ).replace([np.inf, -np.inf], np.nan).dropna()


def compare_PnL(series1, series2, trading_signal, plot=False):
    ratios = series1/series2
    money = 0
    money_cum = pd.Series(index=ratios.index)
    countseries1 = 0
    countseries2 = 0
    for i in range(len(ratios)):
        # Buy long, buy Y and sell ratios*X
        if trading_signal[i] == 1:
            money += -series1[i] + ratios[i]*series2[i]
            countseries1 += 1
            countseries2 -= ratios[i]
        # Sell short, sell Y and buy ratios*X
        elif trading_signal[i] == -1:
            money += series1[i] - ratios[i]*series2[i]
            countseries1 -= 1
            countseries2 += ratios[i]
        # Clear position
        elif trading_signal[i] == 0.5:
            money += series1[i]*countseries1 + series2[i]*countseries2
            countseries1 = 0
            countseries2 = 0
        money_cum[i] = money

    print(f'Strategy PnL {money}')

    # Buy and Hold PnL
    money_BnH = -(series1[0] + series2[0]) + (series1[-1] + series2[-1])
    print(f'Buy-Hold PnL {money_BnH}')

    money_BnH_cum = pd.Series(index=ratios.index)
    initial_investment = -(series1[0] + series2[0])
    for j in range(len(ratios)):
        money_BnH_cum[j] = (series1[j] + series2[j]) + initial_investment
    if plot is True:
        fit, ax = plt.subplots()
        ax.plot(money_cum, label='Trading Strategy profit')
        ax.plot(money_BnH_cum, label='Buy and Hold profit')
        ax.legend(loc='lower left')
        plt.show()

    return True


##########################################################################
#################### SCRIPT START ########################################
if __name__ == '__main__':
    # Retrieve train data
    start_date = '2018-03-01'
    end_date = '2018-08-31'
    tickers = ['BTC-USD', 'ETH-USD', 'EOS-USD', 'LTC-USD', 'XMR-USD',
               'NEO-USD', 'ZEC-USD', 'BNB-USD', 'TRX-USD']
    train_data = data.DataReader(
        tickers, 'yahoo', start_date, end_date)['Adj Close']
    keys = train_data.keys()

    # Retrieve test data
    start_date = '2018-09-03'
    end_date = '2018-10-31'
    test_data = data.DataReader(
        tickers, 'yahoo', start_date, end_date)['Adj Close']
    keys_test = test_data.keys()

    # Find the best co-integrated pairs
    scores, pvalues, pairs = find_cointegrated_pairs(
        train_data, keys, plot=True)
    print(f'The best cointegrated pairs is: {pairs}')
    # Plot the historical price of conintergrated pair
    fig, ax = plt.subplots()
    ax.plot(train_data[pairs[0]], color='b', label=pairs[0])
    ax.plot(train_data[pairs[1]], color='r', label=pairs[1])
    ax.legend(loc='upper right')
    plt.show()

    # BACKTESTING
    S1 = train_data[pairs[0]][1:]
    S2 = train_data[pairs[1]][1:]
    return_S1 = get_log_return(train_data[pairs[0]])
    return_S2 = get_log_return(train_data[pairs[1]])

    # VERIFY TRADING STRATEGY WITHOUT PREDICTION MODEL ARIMA-GARCH
    print("Comparision between Buy-n-Hold and Trading Strategy without prediction model ARIMA-GARCH")
    trading_signal = initiate_trading_signal(
        return_S1 - return_S2, S1, S2, True, pairs[0], pairs[1])
    compare_PnL(S1, S2, trading_signal, True)

    # VERIFY TRADING STRATEGY WITH PREDICTION MODEL ARIMA-GARCH
    spread = return_S1 - return_S2
    forecast_spread = predict_future_trend(spread)
    # Plot forecast_spread vs original spread
    fig, ax = plt.subplots()
    ax.plot(spread, color='blue', label='spread')
    ax.plot(forecast_spread, color='green', label='forecast spread')
    ax.legend(loc='upper right')
    # ax.title('Train Data: Spread vs Predicted Spread by ARIMA-GARCH')
    plt.show()

    print("Comparision between Buy-n-Hold and Trading Strategy with prediction model ARIMA-GARCH")
    trading_signal = initiate_trading_signal(
        forecast_spread, S1, S2, True, pairs[0], pairs[1])
    compare_PnL(S1, S2, trading_signal, True)

    # FOWARD TESTING
    return_S1 = get_log_return(test_data[pairs[0]])
    return_S2 = get_log_return(test_data[pairs[1]])
    S1 = test_data[pairs[0]][1:]
    S2 = test_data[pairs[1]][1:]
    spread = return_S1 - return_S2
    forecast_spread = predict_future_trend(spread)
    # Plot forecast_spread vs original spread
    fig, ax = plt.subplots()
    ax.plot(spread, color='blue', label='spread')
    ax.plot(forecast_spread, color='green', label='forecast spread')
    ax.legend(loc='upper right')
    # plt.title('Test Data: Spread vs Predicted Spread by ARIMA-GARCH')
    plt.show()

    print("Comparision between Buy-n-Hold and Trading Strategy with prediction model ARIMA-GARCH")
    trading_signal = initiate_trading_signal(forecast_spread, S1, S2, True, pairs[0], pairs[1])
    compare_PnL(S1, S2, trading_signal, True)
    print('Script end')


########################################################################
#################### SCRIPT END ########################################
