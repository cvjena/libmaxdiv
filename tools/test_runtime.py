import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
import time

from maxdiv import maxdiv, maxdiv_util

# ensure reproducable results
np.random.seed(0)

def sample_gp(length, sigma, n=1, noise=0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    X = np.linspace(0, 1, length, False).reshape([1, length])
    zeroY = np.zeros(length)
    K = maxdiv_util.calc_gaussian_kernel(X, sigma / length) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(zeroY, K, n)

def sample_interval(n, minlen, maxlen):
    """ sample the bounds of an interval """
    defect_start = int(np.random.randint(0,n-minlen))
    defect_end = int(np.random.randint(defect_start+minlen,min(defect_start+maxlen,n)))
    return defect_start, defect_end

def sample_gp_with_meanshift(length, sigma = 5.0, n=1, noise=0.001, shift=1.0):
    
    gp = sample_gp(length, sigma, n, noise)
    start, end = sample_interval(length, 20, 50)
    gp[0,start:end] -= shift
    return gp


METHODS = maxdiv.get_available_methods() # available probability density estimators
PREPROC = 'td'
MODE = 'I_OMEGA'
m = 10 # number of time series per length

# Measure runtime of various methods for different lengths of time series
times = { method : [] for method in METHODS }
lengths = []
for n in range(25, 1001, 25):
    
    print('-- n = {} --'.format(n))
    
    for method in METHODS:
        times[method].append(0.0)
    
    for i in range(m):
        gp = sample_gp_with_meanshift(n)
        for method in METHODS:
            start_time = time.time()
            maxdiv.maxdiv(gp, method = method, preproc = PREPROC, mode = MODE)
            stop_time = time.time()
            times[method][-1] += stop_time - start_time
    
    lengths.append(n)
    for method in METHODS:
        times[method][-1] /= m

# Plot results
markers = ['x', 'o', '*', 'v', '^', '<', '>']
fig = plt.figure()
sp = fig.add_subplot(111, xlabel = 'Length of time series', ylabel = 'Runtime in seconds')
for i, (method, t) in enumerate(times.items()):
    sp.plot(lengths, t, marker = markers[i % len(markers)], label = method)
sp.legend(loc = 'upper left')
fig.savefig('runtime.svg')
plt.show()