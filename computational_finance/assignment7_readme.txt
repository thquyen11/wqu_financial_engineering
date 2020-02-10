METHOD:
Sample size = 100000
Monthly path for 1 year maturity, n_steps = 12
For each sample, iterate from 1 to 12 (12 months). 
- In each step, calculate the local volatility in (t - (t-1)) by CEV, similar approaching as assignment5.
- Calculate the stock price and firm value in each month

Then, calculate the forward rate of each month by LIBOR rate model => find the 1 year forward rate
- Calibrate the alpha, b, sigma in LIBOR model by using the given market zero-coupon-bond prices (lines 18 to 24 in file assignment7_interest_rate.py)

Use 1 year forward rate to find the discount factor => price the up-and-out call option
Calculate the CVA and call price after CVA


CODE STRUCTURE:
Main code run in assignment7.py
assignment7.py call function discount_factor_libor in assignment7_interest_rate.py to find discount factor
Separate discount factor to another file to keep code clean,and easier to add code to calibrate forward rate later




