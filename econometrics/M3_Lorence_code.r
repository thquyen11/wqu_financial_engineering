load.packages('quantmod')
load.packages('PerformanceAnalytics')
load.packages('ggplot2')
load.packages('forecast')
load.packages('tseries')
load.packages('FitAR')

data <- new.env()

getSymbols('JPM', src = 'yahoo', env = data, auto.assign = T)

JPM = JPM["2018-02-01/2018-12-30"]
JPM = JPM[, "JPM.Adjusted"]

plot(JPM)

# Calculating the average stock price of JPM
jpm_avg <- sum(JPM)/length(JPM)
jpm_avg

# Calculating the standard deviation of the JPM stock price
jpm_std <- StdDev(JPM)
jpm_std

# Calculating the daily JPM stock return
jpm_log_ret <- diff(log(JPM))
jpm_log_ret

# ---------------- #

getSymbols("^GSPC", src = 'yahoo', env = data, auto.assign = T)

GSPC <- GSPC["2018-02-01/2018-12-30"]
GSPC <- GSPC[, "GSPC.Adjusted"]

two_reg_data <- cbind(JPM, GSPC)

scatter.smooth(x=two_reg_data$JPM.Adjusted, y=two_reg_data$GSPC.Adjusted, main="JPM ~ GSPC", xlab= "JPM Closing Price", ylab= "S&P500 Closing Price")  # scatterplot

# Producing a linear model with S&P500 close prices as the explanatory variable.
linearModel <- lm(JPM.Adjusted ~ GSPC.Adjusted, data = two_reg_data)
summary(linearModel)

# Reading in the CSUSHPINSA monthly data 
csushpinsa <- read.csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CSUSHPINSA&scale=left&cosd=1987-01-01&coed=2019-03-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2019-06-17&revision_date=2019-06-17&nd=1987-01-01")
csushpinsa = csushpinsa$CSUSHPINSA

# Confirm that we now have a simple list of doubles for values
typeof(csushpinsa)

csushpinsa = ts(csushpinsa, start = c(1987,1), frequency = 12)

csushpinsa

# Augmented Dickey-Fuller Test
df <- adf.test(csushpinsa)
df # The p-value of 0.4609 here means that we accept the null hypothesis of non-stationarity

# Using Box-Jenkins to explore the parameters for an ARIMA model
T = length(csushpinsa)

csush0 = csushpinsa[-1]
csush1 = csushpinsa[-T]

lag_csush = cbind(csush0, csush1)

plot(lag_csush, main = "T[x] vs T[x-1] for Case Shiller Index Values")

cor(csush0, csush1)

acf(csushpinsa, lag.max = 20, plot = TRUE) # Linear decay => non-stationary

pacf(csushpinsa, lag.max = 20, plot = TRUE)

d_csush = diff(csushpinsa)
d_csush

plot(d_csush) # More stationary than the original data

acf(d_csush, lag.max = 20, plot = TRUE) # Far quicker decay than the original data
pacf_dcsush <- pacf(d_csush, lag.max = 20, plot = TRUE) # Cuts off after lag 1.

# Using auto-fit to detect best model to use via AIC and MLE minimisation
auto.arima(csushpinsa, trace = TRUE) 

# The above suggests ARIMA(3,1,2) with drift. No lower AIC can be found.
model <- arima(csushpinsa, order = c(3, 1, 2))
model

# Determine the confidence intervals for the parameters of the ARIMA(3,1,2) model.
confint(model, level = 0.995)

# Calculate the residuals from the model - checking for goodness-of-fit.
model_resid = resid(model)
model_resid

# Residuals seem relatively stationary.
plot(csushpinsa, model_resid, ylab="Residuals", xlab="Observations")
abline(0,0)

# Residual differentials look a lot like white noise, gaining intensity later on.
resid_diff = diff(model_resid) 
plot(resid_diff) 

# Residual differentials appear to mostly fall along a normal distribution.
qqnorm(resid_diff)

plot(acf(model_resid), main = "Autocorrelation of ARIMA(3,1,2) Residuals")
plot(pacf(model_resid), main = "Partial autocorrelation of ARIMA(3,1,2) Residuals")

boxresult = LjungBoxTest(model_resid,k=2,lag.max =5,StartLag=1)
plot(boxresult[,3],main= "Ljung-Box Q Test", ylab= "P-values", xlab= "Lag") # All well above 0.05, indicating non-significance.

# Having checked that the residuals appear to be normally distributed, we go on to produce a forecast using the model.
predict(model,n.ahead = 12)
plot(forecast(model,h=12, level=c(99.5)))

# The prediction for April 2019 (one month after the latest data) has:
# Forecast  208.0465, Lo99.5  207.1655, Hi99.5  208.9276.

# Potential exogenous variables to improve forecasting ability: mortgage rate, personal income, delinquency rate on mortgages, home ownership rate.