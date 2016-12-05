""" Compares several deseasonalization methods on pre-defined time-series from the Yahoo! dataset. """

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from maxdiv import maxdiv_util, preproc
import datasets


def sample_gp(X, meany, sigma, n=1, noise=0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    K = maxdiv_util.calc_gaussian_kernel(X, sigma) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)


def periods2time(periods, n):
    times = float(n) / periods
    return ['{:.0f}h'.format(t) if t <= 48 else '{:.1f}d'.format(t / 24.0) for t in times]


# Load some time series with seasonal patterns
ids = ['real_3', 'real_9', 'real_12', 'real_13', 'real_14', 'real_15', 'real_17', 'real_18',
       'real_21', 'real_24', 'real_26', 'real_27', 'real_28', 'real_29',
       'real_30', 'real_34', 'real_36', 'real_38', 'real_39', 'real_44', 'real_46', 'real_47', 'real_49',
       'real_50', 'real_51', 'real_52', 'real_54', 'real_55', 'real_56', 'real_57', 'real_60', 'real_65']
gt_period = 24 # our data has a true period of 24 hours
data = { func['id']: func for func in datasets.loadYahooDataset(subset = 'real')['A1Benchmark'] if func['id'] in ids }


# Joint de-seasonalization by SVD
"""
print('-- SVD --')
minSVDLen = 1420
numLongFuncs = sum(1 for func in data.values() if func['ts'].shape[1] >= minSVDLen)
# Build model matrix
mm = np.ndarray((numLongFuncs, minSVDLen))
row = 0
rowAssoc = {}
for id, func in data.items():
    if func['ts'].shape[1] >= minSVDLen:
        mm[row,:] = func['ts'][0,:minSVDLen]
        rowAssoc[id] = row
        row += 1
# Perform SVD
mm_u, mm_s, mm_vt = np.linalg.svd(mm, full_matrices = False)
# Plot first right singular vectors
fig = plt.figure()
for i in range(6):
    ax = fig.add_subplot(3, 2, i + 1, title = '{}. right-singular vector for singular value {}'.format(i+1, mm_s[i]))
    ax.plot(mm_vt[i,:])
    freq = np.fft.fft(mm_vt[i,:])
    ps = (freq * freq.conj()).real
    period = ps[1:(len(ps)//2)+1].argmax() + 1
    print('Period of {}. right singular vector: {} -> {}'.format(i+1, period, float(minSVDLen) / period))
plt.show()
# Remove some leading singular values
try:
    rsRem = raw_input('Enter number of right-singular vectors to remove: ')
except:
    rsRem = input('Enter number of right-singular vectors to remove: ')
rsRem = int(rsRem)
mm_s[:rsRem] = 0.0
mm_norm = mm_u.dot(np.diag(mm_s).dot(mm_vt))
"""


# Individual De-seasonalization by DFT, Hourly Z Score and OLS
print('-- DFT, Hourly Z Score, OLS --')
for id in ids:
    func = preproc.normalize_time_series(data[id]['ts']).ravel()

    # Search non-trivial peak in power-spectrum
    freq = np.fft.fft(func)
    ps = (freq * freq.conj()).real
    ps[0] = 0
    th = np.mean(ps) + 3 * np.std(ps)
    period = (ps > th)
    period[0:7] = False
    period[-6:] = False
    period_ind = np.where(period)[0]
    print('{}: period = {} -> {}'.format(id, period_ind[:len(period_ind)//2], periods2time(period_ind[:len(period_ind)//2], len(func))))
    
    # Remove seasonal frequency and reconstruct deseasonalized time series
    freq[period] = 0
    norm_func_dft = np.fft.ifft(freq).real
    
    # Normalize each hour separately by Hourly Z Score
    norm_func_z = func.copy()
    for h in range(gt_period):
        hourly_values = func[h::gt_period]
        norm_func_z[h::gt_period] -= np.mean(hourly_values)
        norm_func_z[h::gt_period] /= np.std(hourly_values)
    
    # Model seasonal influence and linear trends by OLS and take residuals as deseasonalization:
    # x_t = a_0 + b_0 * t + a_j + b_j * t/period + e_t
    A = np.zeros((len(func), 2 * gt_period + 2))
    A[:,0] = 1                          # intercept term a_0
    A[:,1] = np.arange(0.0, len(func))  # linear term b*t
    for t in range(len(func)):          # seasonal terms
        A[t, 2 + (t % gt_period)] = 1
        A[t, 2 + gt_period + (t % gt_period)] = float(t) / gt_period
    ols_seasonality = np.linalg.lstsq(A, func)[0]
    ols_seasonal_ts = A.dot(ols_seasonality)
    norm_func_ols = func - ols_seasonal_ts
    
    # Plot
    funcs = [(func, 'Original Time Series with OLS Seasonal Model'),
             (ps, 'Power Spectrum'),
             (norm_func_z, 'Deseasonalized by Hourly Z Score'),
             (norm_func_dft, 'Deseasonalized by DFT'),
             (norm_func_ols, 'Deseasonalized and detrended by OLS')]
    #if id in rowAssoc:
    #    funcs.append((mm_norm[rowAssoc[id],:].T, 'De-seasonalized by SVD'))
    fig = plt.figure()
    fig.canvas.set_window_title('De-seasonalization of {}'.format(id))
    for row, (f, title) in enumerate(funcs):
        ax = fig.add_subplot(len(funcs), 1, row + 1, title = title)
        ax.plot(f, 'g' if row == 1 else 'b')
        if row == 0:
            ax.plot(ols_seasonal_ts, 'r')
        elif row == 1:
            ax.plot([0, len(func)-1], [th, th], '--r')
            ax.plot(period_ind, ps[period], 'r.')
    fig.subplots_adjust(0.06, 0.04, 0.96, 0.96, hspace = 0.3)
    plt.show()


"""
# Synthetic multivariate example: Seasonal correlations
print('-- Multivariate --')
mv_period = 20
np.random.seed(0)
X = np.linspace(0.0, 1.0, 250)
X = np.reshape(X, [1, len(X)])
gp = sample_gp(X, np.zeros(X.shape[1]), 0.01)
func = np.vstack((gp, gp * np.sin(X * mv_period)))
func += np.random.randn(*func.shape) * 0.1

# Normalize by multivariate Z Score
norm_func = func.copy()
for h in range(mv_period):
    hourly_values = func[:,h::mv_period]
    mu = np.mean(hourly_values, axis = 1)
    cov = np.cov(hourly_values)
    zeromean_X = (hourly_values.T - mu).T
    norm_func[:,h::mv_period] = np.dot(np.linalg.inv(cov), zeromean_X)

fig = plt.figure()
ax = fig.add_subplot(211, title = 'Multivariate TS with seasonal correlations')
ax.plot(func.T)
ax = fig.add_subplot(212, title = 'De-seasonalized by Z Score')
ax.plot(norm_func.T)
plt.show()
"""