import maxdiv
import numpy as np
import matplotlib.pylab as plt

t = np.arange(0, 10, 0.1)
X = np.vstack([np.sin(t), np.cos(t)]) + 0.01*np.random.randn(2, len(t))
X[0, 20:40] *= 0.0

K = maxdiv.calc_normalized_gaussian_kernel(X)
a, b = maxdiv.find_extreme_interval_kldivergence(K, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=5.0)
print ("Extreme interval detected at: {} to {}".format(a, b))

plt.figure()
plt.plot(t, X[0,:])
plt.plot(t, X[1,:])
plt.show()

maxdiv.plot_matrix_with_interval(K, a, b)
