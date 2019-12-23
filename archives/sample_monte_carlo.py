import numpy as np
from scipy.stats import uniform
import random
import matplotlib.pyplot as plt


# Calculate the integral of sin(x)dx in (0, pi) interval
random.seed(0)
f_estimates = [None]*50
f_std = [None]*50

for i in range(50):
    random_array = uniform.rvs(size = i*1000)*np.pi
    f_values_array = np.sin(random_array)*np.pi
    f_estimates[i] = np.mean(f_values_array)
    f_std[i] = np.std(f_values_array/np.sqrt(i*1000))

plt.plot(f_estimates, ".")
plt.plot(2+np.array(f_std)*3)
plt.plot(2-np.array(f_std)*3)
plt.show()