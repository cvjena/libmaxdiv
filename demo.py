import maxdiv
import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)

t = np.arange(0, 10, 0.1)
X = np.vstack([np.sin(t), np.cos(t)]) + 0.01*np.random.randn(2, len(t))
X[0, 20:40] *= 0.0

K = maxdiv.calc_normalized_gaussian_kernel(X)
a, b, score = maxdiv.find_extreme_interval_kldivergence(K, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=1.0)
a, b, score = maxdiv.find_extreme_interval_kldivergence(K, extint_min_len=10, extint_max_len=30, mode="LAMBDA", alpha=0.5)
print ("Extreme interval detected at {} to {} with scores {}".format(a, b, score))

#plt.figure()
#plt.plot(t, X[0,:])
#plt.plot(t, X[1,:])
#plt.show()

#maxdiv.plot_matrix_with_interval(K, a, b)
