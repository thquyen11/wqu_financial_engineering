import numpy as np
from scipy.stats import uniform
import random, math
from statistics import mean
import matplotlib.pyplot as plt

# Question 1.2
# Calculate the integral of cos(x)dx in (0, 2) interval
random.seed(0)
f_estimates = [None]*50
f_std = [None]*50

for i in range(1, 51):
    random_array = uniform.rvs(size = i*1000, scale=2)
    f_values_array = np.cos(random_array)*2
    f_estimates[i-1] = np.mean(f_values_array)
    f_std[i-1] = np.std(f_values_array/np.sqrt(i*1000))

# Question 1.3
analytical_value = math.sin(2) - math.sin(0)

plt.plot(f_estimates, ".")
plt.plot(mean(f_estimates)+np.array(f_std)*3)
plt.plot(mean(f_estimates)-np.array(f_std)*3)
plt.plot([analytical_value]*50, "x")
plt.show()

