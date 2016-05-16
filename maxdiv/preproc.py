""" preprocessing routines for time-series """

import numpy as np
from scipy.linalg import lstsq, sqrtm

from . import maxdiv_util


def get_available_methods():
    return ['local_linear', 'td']


def local_linear_regression(X, window_size=5):
    """ Local linear regression also known as linear predictive coding (LPC) """
    dimension = X.shape[0]
    n = X.shape[1]
    
    # TODO: should we integrate also the model error
    params = np.zeros([dimension*2, n])
    for i in range(n):
        C = np.zeros(dimension*dimension)
        b = np.zeros(dimension)

        start = max(0, i-window_size)
        end = min(n-1, i+window_size)
        P = np.linspace(0.0, 1.0, end-start)
        P = np.vstack( [P, np.ones(P.shape)] )
        b = X[:, start:end]
        # can be sign. speeded-up
        param, _, _, _ = lstsq(P.T, b.T)
        params[:, i] = param.ravel()
    return params


def td(X, m=3):
    """ time-delay transformation """
    newX = np.copy(X)
    for i in range(1,m):
        shift = np.arange(-i, X.shape[1]-i, 1)
        shift[shift<0] = 0
        newX = np.vstack([newX, X[:, shift]])
    return newX


def normalize_time_series(ts):
    """ Normalizes each dimension of a time series by subtracting the mean and dividing by the maximum. """
    
    ts = (ts.T - ts.mean(axis = 1)).T
    ts = (ts.T / np.abs(ts).max(axis = 1)).T
    return ts


def detect_periods(ts):
    """ Detects the length of periods after which some patterns in the given time series are repeating.
    
    The given time series will be transformed into 1D Fourier space, where a threshold operation based
    on the standard deviation of the frequencies will be applied to find unusually high frequencies.
    
    The return value of this function is a tuple containing two lists: the first one gives the length of
    the periods as floats and the second one the corresponding frequencies as integers. The lists will be
    sorted in descending order by the coefficient of the frequencies.
    
    If the given time series is multivariate, each dimension will "vote" for some frequencies by
    accumulating the corresponding values from the power spectrum.
    """
    
    ts = maxdiv_util.enforce_multivariate_timeseries(ts)
    
    freq = np.fft.fft(ts)           # Transform to Fourier domain
    ps = (freq * freq.conj()).real  # Compute Power Spectrum (Autocorrelation)
    ps[:,0] = 0                     # Ignore the zero-frequency component, which always is the highest peak
    
    periods = {}
    for d in range(ts.shape[0]):
        # Threshold
        th = np.mean(ps[d,:]) + 3 * np.std(ps[d,:])
        period = (ps[d, :(ps.shape[1]//2)+1].ravel() > th)
        # Ignore implausibly long periods
        period[0:7] = False
        # Accumulate scores
        period_ind = np.where(period)[0]
        for p in period_ind:
            if p not in periods:
                periods[p] = ps[d, p]
            else:
                periods[p] += ps[d, p]
    
    # Sort by scores in descending order
    periods = np.array(sorted(periods.keys(), key = lambda p: periods[p], reverse = True))
    return float(ts.shape[1]) / periods, periods


def deseasonalize_ft(ts):
    """ Deseasonalizes a given time series by eliminating extreme frequencies in the Fourier spectrum.
    
    Multivariate time series will be deseasonalized separately dimension by dimension.
    """
    
    nts = maxdiv_util.enforce_multivariate_timeseries(ts)
    freq = np.fft.fft(nts)
    for d in range(nts.shape[0]):
        _, periods = detect_periods(nts[d,:])
        if len(periods) > 0:
            freq[d, np.concatenate((periods, -periods))] = 0
    norm_ts = np.fft.ifft(freq).real
    return norm_ts if ts.ndim > 1 else norm_ts.ravel()


def deseasonalize_zscore(ts, period_len):
    """ Deseasonalizes a given time series using the Z Score method.
    
    The time series will be divided into groups of samples with a distance of `period_len`, which correspond to
    a specific time within the period. The mean of each group will then by subtracted from the samples in the group
    and the result will be divided by the standard deviation of the group.
    This functions returns a deseasonalized copy of `ts`.
    """
    
    norm_func = ts.copy()
    for h in range(period_len):
        if (ts.ndim == 1) or (ts.shape[0] == 1):
            values = ts.flat[h::period_len]
            norm_func.flat[h::period_len] -= np.mean(values)
            norm_func.flat[h::period_len] /= np.std(values)
        else:
            values = ts[:, h::period_len]
            mu = np.mean(values, axis = 1)
            cov = np.cov(values)
            zeromean_X = (values.T - mu).T
            norm_func[:,h::period_len] = np.dot(sqrtm(np.linalg.inv(cov)), zeromean_X)
    return norm_func


def detrend_linear(ts):
    """ Removes a linear trend from a given time series by subtracting a robustly fitted line.
    
    Multivariate time series will be detrended separately dimension by dimension.
    """
    
    # First column of A: 0 to N-1. Second column of A: constantly 1.
    N = len(ts) if ts.ndim == 1 else ts.shape[1]
    A = np.hstack((np.arange(0.0, float(N)).reshape((N, 1)), np.ones((N, 1))))
    
    # Fit robust lines
    line_params = np.ndarray((2, 1 if ts.ndim == 1 else ts.shape[0]))
    for d in range(line_params.shape[1]):
        line_params[:,d] = maxdiv_util.m_estimation(A, ts.reshape((N, 1)) if ts.ndim == 1 else ts[d,:].T).ravel()
    
    # Subtract linear trend
    linear_trend = A.dot(line_params)
    detrended_linear = ts - linear_trend.ravel() if ts.ndim == 1 else ts - linear_trend.T
    return detrended_linear


def detrend_ols(ts, periods = None, linear_trend = True, linear_season_trend = True, return_model_params = False):
    """ Deseasonalizes and detrends a given time series by ordinary least squares.
    
    Each sample `y_t` in the given time series `ts` will be modelled according to
    `y_t = a_0 + b_0 * t + a_j + b_j * t/period_len + e_t`,
    where `j` is the season of the sample.
    The residuals `e_t` will be returned as deseasonalized and detrended time series.
    
    Multivariate time series will be detrended separately dimension by dimension.
    
    `periods` is a list of tupels which specify the number of seasonal units and the length
    of each unit. For example, for hourly sampled data `[(365, 24), (24, 1)]` would mean that
    there are seasonal effects across the day as well as across the year. This would assume, that
    the diurnal effects are independent from the seasonal effects across the year. Thus, `[(365*24, 1)]`
    whould be an alternative, but could mean too many degrees of freedom for a robust estimation from
    the available data.
    For simplicity, if an integer `x` is given, it will be equivalent to `[(x, 1)]`.
    If `periods` is set to `None`, this method will try to detect the seasonal patterns in the data based
    on the autocorrelation function / power spectrum.
    
    `linear_trend` specifies if the term `b_0 * t` should be included in the model, which corresponds
    to a linear trend in the data.
    
    `linear_season_trend` specifies if the terms `b_j * t/period` should be included in the model, which
    correspond to a linear change of the seasonal components. This should be used carefully, since it
    would interpolate a sudden change in the seasonal pattern as a smooth transition in the model, which
    will almost never fit the actual data.
    
    Returns: the deseasonalized and detrended time series. If `return_model_param` is set to true, a
    tupel consisting of the resulting time series and the estimated model parameters will be returned instead.
    """
    
    # Convert parameters to canonical format
    func = maxdiv_util.enforce_multivariate_timeseries(ts)
    if isinstance(periods, int):
        periods = [(periods, 1)]
    elif periods is None:
        # Detect seasonality automatically
        p, _ = detect_periods(func[0,:])
        periods = { int(round(period_len)) : 1 for period_len in p}
        # Each dimension in the time series may vote for a period
        for i in range(1, func.shape[0]):
            p, _ = detect_periods(func[i,:])
            for period_len in p:
                period_len_int = int(round(period_len))
                if period_len_int not in periods:
                    periods[period_len_int] = 1
                else:
                    periods[period_len_int] += 1
        # Select the period with the most votes
        if len(periods) == 0:
            return ts
        periods = sorted(periods.keys(), key = lambda p: (periods[p], p), reverse = True)
        periods = [(periods[0], 1)]
    
    # Construct model matrix
    num_season_coeffs = sum(season_num for season_num, season_len in periods)
    num_params = 1 + num_season_coeffs
    if linear_trend:
        num_params += 1
    if linear_season_trend:
        num_params += num_season_coeffs
    A = np.zeros((func.shape[1], num_params))
    # intercept term a_0
    A[:,0] = 1
    # linear term b_0 * t
    if linear_trend:
        A[:,1] = np.arange(0.0, func.shape[1])
    # seasonal terms
    for t in range(func.shape[1]):
        offs = 2 if linear_trend else 1
        for season_num, season_len in periods:
            ind = ((t // season_len) % season_num)
            A[t, offs + ind] = 1
            if linear_season_trend:
                A[t, offs + num_season_coeffs + ind] = float(t) / season_num
            offs += season_num
    
    # Estimate parameters and detrend time series
    ols_seasonality = np.linalg.lstsq(A, func.T)[0]
    ols_seasonal_ts = A.dot(ols_seasonality).T
    norm_func_ols = func - ols_seasonal_ts
    if ts.ndim == 1:
        norm_func_ols = norm_func_ols.ravel()
    return (norm_func_ols, ols_seasonality) if return_model_params else norm_func_ols
