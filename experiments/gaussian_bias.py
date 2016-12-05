""" Create an example for the small-interval bias of the KL divergence. """

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt

from maxdiv.maxdiv import maxdiv
from maxdiv.eval import plotDetections


# Create synthetic time series with one/two huge flat region(s)
mean = 5.0
sd = 2.0
ts_len = 500
window_center1 = 350
window_center2 = 100
window_sd = 10

np.random.seed(0)
gauss_window1 = np.exp(-0.5 * ((np.arange(0.0, ts_len) - window_center1) ** 2) / (window_sd ** 2))
gauss_window2 = np.exp(-0.5 * ((np.arange(0.0, ts_len) - window_center2) ** 2) / (window_sd ** 2))
ts = mean + np.random.randn(ts_len) * sd
ts1 = gauss_window1 * ts + 0.1 * np.random.randn(ts_len)
ts2 = (gauss_window1 + gauss_window2) * ts + 0.1 * np.random.randn(ts_len)

gt = [(window_center1 - 3 * window_sd, window_center1 + 3 * window_sd + 1), (window_center2 - 3 * window_sd, window_center2 + 3 * window_sd + 1)]

# Apply MDI Gaussian on different scenarios
print('--- OMEGA_I on single extremum ---')
det = maxdiv(ts1.reshape((1, ts_len)), 'gaussian_cov', None, mode = 'OMEGA_I', extint_min_len = 10, extint_max_len = 8 * window_sd, preproc = 'td')
plotDetections(ts1.reshape((1, ts_len)), det, [gt[0]], silent = False)

print('--- I_OMEGA on single extremum ---')
det = maxdiv(ts1.reshape((1, ts_len)), 'gaussian_cov', None, mode = 'I_OMEGA', extint_min_len = 10, extint_max_len = 8 * window_sd, preproc = 'td')
plotDetections(ts1.reshape((1, ts_len)), det, [gt[0]], silent = False)

print('--- I_OMEGA on two extrema ---')
det = maxdiv(ts2.reshape((1, ts_len)), 'gaussian_cov', None, mode = 'I_OMEGA', extint_min_len = 10, extint_max_len = 8 * window_sd, preproc = 'td')
plotDetections(ts2.reshape((1, ts_len)), det, gt, silent = False)
