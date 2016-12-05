""" Compares the run-time of the Python and the libmaxdiv implementation of the MDI algorithm on time-series of varying length. """

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
    N = np.arange(200, 2501, 100)
    times = np.ndarray((len(N), 4), dtype = np.float64)

    # Prepare libmaxdiv pipelines
    params = libmaxdiv_wrapper.maxdiv_params_t()
    libmaxdiv_wrapper.libmaxdiv.maxdiv_init_params(params)
    params.min_size[0] = min_len
    params.max_size[0] = max_len
    params.preproc.embedding.kt = 3
    params.preproc.embedding.temporal_borders = libmaxdiv_wrapper.enums['MAXDIV_BORDER_POLICY_CONSTANT']

    params.estimator = libmaxdiv_wrapper.enums['MAXDIV_GAUSSIAN']
    pipeline_gaussian = libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params)

    params.estimator = libmaxdiv_wrapper.enums['MAXDIV_KDE']
    pipeline_parzen = libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params)

    # Measure runtimes and write them to timing.csv
    with open('timing.csv', 'w') as outFile:
        outFile.write('Length,Gaussian (Python),KDE (Python),Gaussian (libmaxdiv),KDE (libmaxdiv)\n')
        
        for i, n in enumerate(N):
            gps = sample_gp(n)
            
            start = time()
            maxdiv.maxdiv(gps, 'gaussian_cov', None, 'dense', useLibMaxDiv = False, mode = 'I_OMEGA', preproc = 'td', extint_min_len = min_len, extint_max_len = max_len)
            stop = time()
            times[i, 0] = stop - start
            
            start = time()
            maxdiv.maxdiv(gps, 'parzen', None, 'dense', useLibMaxDiv = False, mode = 'I_OMEGA', preproc = 'td', extint_min_len = min_len, extint_max_len = max_len)
            stop = time()
            times[i, 1] = stop - start
            
            start = time()
            libmaxdiv_wrapper.maxdiv_exec(gps, pipeline_gaussian, None)
            stop = time()
            times[i, 2] = stop - start
            
            start = time()
            libmaxdiv_wrapper.maxdiv_exec(gps, pipeline_parzen, None)
            stop = time()
            times[i, 3] = stop - start
            
            outFile.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(n, *times[i,:]))
            outFile.flush()

else:

    N = []
    times = []
    with open(sys.argv[1]) as inFile:
        for i, d in enumerate(csv.reader(inFile)):
            if i > 0:
                N.append(int(d[0]))
                times.append([float(x) for x in d[1:5]])
    times = np.array(times)


# Plot results
if (len(sys.argv) < 2) or (sys.argv[1].lower() != 'noplot'):
    import matplotlib.pylab as plt
    plt.plot(N, times[:, 0] * 1000, 'b--', label = 'Gaussian (Python)')
    plt.plot(N, times[:, 1] * 1000, 'r--', label = 'KDE (Python)')
    plt.plot(N, times[:, 2] * 1000, 'b-',  label = 'Gaussian (libmaxdiv)')
    plt.plot(N, times[:, 3] * 1000, 'r-',  label = 'KDE (libmaxdiv)')
    plt.xlabel('Length of Time Series')
    plt.ylabel('Algorithm Run-Time')
    plt.yscale('log')
    ticks = [10 ** e for e in range(1, 6, 1)]
    plt.yticks(ticks, ['{:.0f} s'.format(l / 1000) if l >= 1000 else '{:.0f} ms'.format(l) for l in ticks])
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.show()