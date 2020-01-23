#Plotting closed-form and Dupire equation
plt.plot(test_strikes,sigma*test_strikes**(gamma-1))
plt.plot(test_strikes,vol_est,'.')