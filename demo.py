#!/usr/bin/env python
# -*- coding: utf8 -*-
""" Demo program for finding extreme intervals using maximally divergent regions """

import maxdiv
import numpy as np
import matplotlib.pylab as plt

__author__ = "Erik Rodner"

run_parzen = True

# set random seed to zero to have comparable results
np.random.seed(0)

# generative some boring time series
t = np.arange(0, 100, 0.001)
print ("Length of the time series: {}".format(len(t)))
X = np.vstack([np.sin(t), np.cos(t)]) + 0.01*np.random.randn(2, len(t))
# integrate a boring defect
X[0, 20:40] *= 3.0

# if we have more than 5000 data points, using the Parzen version
# of the algorithm should not be done
if len(t)>5000:
    run_parzen = False

# run the Parzen version of the algorithm
if run_parzen:
    print ("Running MDR Parzen case")
    # compute kernel matrix first (Gaussian kernel)
    K = maxdiv.calc_normalized_gaussian_kernel(X)
    # obtain the interval [a,b] of the extreme event with score score
    a, b, score = maxdiv.maxdiv_parzen(K, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=1.0)
    print ("Parzen: Extreme interval detected at {} to {} with scores {}".format(a, b, score))

# run the Gaussian version of the algorithm
print ("Running MDR Gaussian case")
a, b, score = maxdiv.maxdiv_gaussian(X, extint_min_len=10, extint_max_len=30, mode="I_OMEGA", alpha=1.0)
print ("Parzen: Extreme interval detected at {} to {} with scores {}".format(a, b, score))


#plt.figure()
#plt.plot(t, X[0,:])
#plt.plot(t, X[1,:])
#plt.show()
#maxdiv.plot_matrix_with_interval(K, a, b)
