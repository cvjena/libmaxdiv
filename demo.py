import maxdiv
import numpy as np
import matplotlib.pylab as plt

run_parzen = True

np.random.seed(0)

t = np.arange(0, 100, 0.1)
print ("Length of the time series: {}".format(len(t)))
X = np.vstack([np.sin(t), np.cos(t)]) + 0.01*np.random.randn(2, len(t))
X[0, 20:40] *= 3.0

if len(t)>5000:
    run_parzen = False

if run_parzen:
    K = maxdiv.calc_normalized_gaussian_kernel(X)
    a, b, score = maxdiv.maxdiv_parzen(K, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=1.0)
    print ("Parzen: Extreme interval detected at {} to {} with scores {}".format(a, b, score))

print ("Running MDR Gaussian case")
a, b, score = maxdiv.maxdiv_gaussian(X, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=1.0)
print ("Parzen: Extreme interval detected at {} to {} with scores {}".format(a, b, score))


#plt.figure()
#plt.plot(t, X[0,:])
#plt.plot(t, X[1,:])
#plt.show()

#maxdiv.plot_matrix_with_interval(K, a, b)
