import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import pyeemd
from maxdiv import preproc
import datasets


def m_estimation(A, b, k = 1.345):
    
    # Obtain initial estimate
    x = np.linalg.lstsq(A, b)[0]
    
    # Iteratively Re-weighted Least Squares
    for it in range(1000):
    
        # Determine weights according to Huber's function
        w = np.abs(b - A.dot(x)).ravel()
        el = (w <= k)
        w[el] = 1
        w[~el] = k / w[~el]
        
        # Weighted Least Squares
        x_old = x.copy()
        W = np.diag(np.sqrt(w))
        x = np.linalg.lstsq(W.dot(A), W.dot(b))[0]
        
        if np.linalg.norm(x - x_old) < 1e-6:
            break

    return x


def lmeds(A, b, outlier_ratio = 0.8):
    
    # Determine number of trials
    trials = int(np.ceil(np.log(0.005) / np.log(1 - (1 - outlier_ratio) ** A.shape[1])))
    
    # Initial solution based on all samples
    best_sol = np.linalg.lstsq(A, b)[0]
    best_val = np.median((b - A.dot(best_sol)) ** 2)
    
    # Try random subsets
    for t in range(trials):
        
        selected = np.zeros(A.shape[0], dtype = bool)
        while selected.sum() < A.shape[1]:
            selected[np.random.randint(A.shape[0])] = True
        
        sol = np.linalg.lstsq(A[selected, :], b[selected])[0]
        val = np.median((b - A.dot(sol)) ** 2)
        if val < best_val:
            best_sol = sol
            best_val = val
    
    return best_sol


def emd(func, S = range(4,14)):
    
    if isinstance(S, int):
        return pyeemd.emd(func, S_number = S)
    else:
    
        imfs = []
        for s in S:
            cur_imfs = pyeemd.emd(func, S_number = s)
            for i in range(cur_imfs.shape[0]):
                if i >= len(imfs):
                    imfs.append([cur_imfs[i,:]])
                else:
                    imfs[i].append(cur_imfs[i,:])
        
        np_imfs = np.ndarray(cur_imfs.shape)
        for i in range(cur_imfs.shape[0]):
            np_imfs[i,:] = np.asarray(imfs[i]).mean(axis = 0)
        return np_imfs


def average_period(func):
    
    distances = []
    lastPeak = None
    for i in range(1, func.size - 1):
        l, m, r = func.flat[i-1:i+2]
        if (l < m) and (r < m):
            if lastPeak is not None:
                distances.append(i - lastPeak)
            lastPeak = i
    return np.mean(distances) if len(distances) > 0 else 0.0


if __name__ == '__main__':

    # Load some time series with obvious trends
    ids = ['real_6', 'real_13', 'real_15', 'real_28', 'real_43', 'real_44', 'real_55', 'real_56', 'real_65']
    data = { func['id']: func for func in datasets.loadYahooDataset(subset = 'real')['A1Benchmark'] if func['id'] in ids }

    # Detrend
    for id in ids:
        func = data[id]['ts'].ravel()
        
        # Fit a robust linear regression line to the data and take the residuals as detrended time series
        A = np.hstack((np.arange(0.0, float(len(func))).reshape((len(func), 1)), np.ones((len(func), 1))))
        #line_params = m_estimation(A, func.reshape(len(func), 1))
        line_params = lmeds(A, func.reshape(len(func), 1))
        linear_trend = A.dot(line_params)
        detrended_linear = func - linear_trend.ravel()
        
        # Estimate seasonality and linear trend together:
        # x_t = a_0 + b_0 * t + a_j + b_j * t/period + e_t
        gt_period = 24
        A = np.zeros((len(func), 2 * gt_period + 2))
        A[:,0] = 1                          # intercept term a_0
        A[:,1] = np.arange(0.0, len(func))  # linear term b*t
        for t in range(len(func)):          # seasonal terms
            A[t, 2 + (t % gt_period)] = 1
            A[t, 2 + gt_period + (t % gt_period)] = float(t) / gt_period
        ols_seasonality = np.linalg.lstsq(A, func)[0]
        ols_seasonal_ts = A.dot(ols_seasonality)
        norm_func_ols = func - ols_seasonal_ts
        
        # Empirical Mode Decomposition (EMD)
        imfs = emd(func)
        trend_imfs = (np.array([average_period(imfs[i,:]) for i in range(imfs.shape[0])]) == 0)
        imf_trend = np.sum(imfs[trend_imfs,:], axis = 0)
        norm_func_imf = func - imf_trend
        
        # Plot
        fig = plt.figure()
        fig.canvas.set_window_title('Detrended {}'.format(id))
        ax = fig.add_subplot(411, title = 'Original Time Series with Fitted Trends')
        ax.plot(func, 'b-', linear_trend, 'r:', ols_seasonal_ts, 'r--', imf_trend, 'g-')
        ax = fig.add_subplot(412, title = 'Linear Trend Removed')
        ax.plot(detrended_linear, 'b-')
        ax = fig.add_subplot(413, title = 'Seasonality and Linear Trend Removed')
        ax.plot(norm_func_ols, 'b-')
        ax = fig.add_subplot(414, title = 'Trend Removed by EMD')
        ax.plot(norm_func_imf, 'b-')
        plt.show()
