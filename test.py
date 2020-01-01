import numpy as np
from scipy.stats import norm

corr_matrix = np.array([[1, 0.2], [0.2, 1]])
norm_matrix = norm.rvs(size = np.array([2, 50000]))
corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), norm_matrix)
print(corr_norm_matrix[:, :20])