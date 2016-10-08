""" Compares the performance of different variants of the MDI algorithm on time-series of varying dimensionality. """

import sys
sys.path.append('..')

import numpy as np
from time import time
import csv

from maxdiv import maxdiv, maxdiv_util, libmaxdiv_wrapper

# ensure reproducable results
np.random.seed(0)

def sample_gp(length, dim = 1, sigma = 0.02, noise = 0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    X = np.arange(0, length / 250.0, 0.004)
    X = np.reshape(X, [1, len(X)])
    meany = np.zeros(X.shape[1])
    K = maxdiv_util.calc_gaussian_kernel(X, sigma) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, dim)


if (len(sys.argv) < 2) or (sys.argv[1].lower() == 'noplot'):
    
    # Parameters
    min_len = 10
    max_len = 100
    N = 1000
    dims = np.arange(1, 26)
    times = np.ndarray((len(dims), 3), dtype = np.float64)

    # Prepare libmaxdiv pipelines
    params = libmaxdiv_wrapper.maxdiv_params_t()
    libmaxdiv_wrapper.libmaxdiv.maxdiv_init_params(params)
    params.min_size[0] = min_len
    params.max_size[0] = max_len
    params.preproc.embedding.kt = 1
    params.erph.num_hist = 100
    params.erph.num_bins = 5

    params.estimator = libmaxdiv_wrapper.enums['MAXDIV_GAUSSIAN']
    pipeline_gaussian = libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params)

    params.estimator = libmaxdiv_wrapper.enums['MAXDIV_KDE']
    pipeline_parzen = libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params)
    
    params.estimator = libmaxdiv_wrapper.enums['MAXDIV_ERPH']
    pipeline_erph = libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params)

    # Measure runtimes and write them to timing.csv
    with open('timing_multvar.csv', 'w') as outFile:
        outFile.write('Dimensions,Gaussian,KDE,ERPH\n')
        
        for i, dim in enumerate(dims):
            gps = sample_gp(N, dim)
            
            start = time()
            libmaxdiv_wrapper.maxdiv_exec(gps, pipeline_gaussian, None)
            stop = time()
            times[i, 0] = stop - start
            
            start = time()
            libmaxdiv_wrapper.maxdiv_exec(gps, pipeline_parzen, None)
            stop = time()
            times[i, 1] = stop - start
            
            start = time()
            libmaxdiv_wrapper.maxdiv_exec(gps, pipeline_erph, None)
            stop = time()
            times[i, 2] = stop - start
            
            outFile.write('{},{:.3f},{:.3f},{:.3f}\n'.format(dim, *times[i,:]))
            outFile.flush()

else:

    dims = []
    times = []
    with open(sys.argv[1]) as inFile:
        for i, d in enumerate(csv.reader(inFile)):
            if i > 0:
                dims.append(int(d[0]))
                times.append([float(x) for x in d[1:4]])
    times = np.array(times)


# Plot results
if (len(sys.argv) < 2) or (sys.argv[1].lower() != 'noplot'):
    import matplotlib.pylab as plt
    plt.plot(dims, times[:, 0], 'b-', label = 'Gaussian')
    plt.plot(dims, times[:, 1], 'r-', label = 'KDE')
    plt.plot(dims, times[:, 2], 'g-', label = 'ERPH')
    plt.xlabel('Dimensionality of Time Series')
    plt.ylabel('Algorithm Run-Time in Seconds')
    plt.grid(True)
    plt.legend(loc = 'upper left')
    plt.show()
