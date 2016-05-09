import numpy as np
from .eval import pointwiseLabelsToIntervals
from .maxdiv_util import calc_gaussian_kernel, IoU


def hotellings_t(X):
    """ Global version of hotellings_t """
    
    mu = np.mean(X, axis = 1)
    cov = np.cov(X)
    zeromean_X = (X.T - mu).T
    if X.shape[0]==1:
        norm_X = zeromean_X/cov
        scores = zeromean_X*norm_X
        scores = np.ravel(scores)
    else:
        invcov = np.linalg.inv(cov)
        norm_X = np.dot(invcov, zeromean_X)
        scores = np.sum(norm_X * zeromean_X, axis=0)
    return scores


def pointwiseKDE(X, kernel_sigma_sq = 1.0):
    """ Scores every point in the time series by Kernel Density Estimation.
    
    Note: The values of X should be in range [-1,1] to avoid numeric problems.
    """
    
    # Compute kernel matrix
    K = calc_gaussian_kernel(X, kernel_sigma_sq, False)
    # Score points by their unlikelihood
    return (1.0 - K.mean(axis = 0))


def pointwiseScoresToIntervals(scores, min_length = 0):
    
    sorted_scores = sorted(scores)
    first_th = sorted_scores[int(len(scores) * 0.7)]
    max_score = sorted_scores[-1]
    
    thresholds = np.linspace(first_th, max_score, 10, endpoint = False)
    scores = np.array(scores)
    regions = []
    for th in thresholds[::-1]:
        regions += [(a, b, scores[a:b].min()) for a, b in pointwiseLabelsToIntervals(scores >= th) if b - a >= min_length]
    
    # Non-maxima suppression
    include = np.ones(len(regions), dtype = bool) # suppressed intervals will be set to False
    for i in range(len(regions)):
        if include[i]:
            a, b, score = regions[i]
            # Exclude intervals with a lower score overlapping this one
            for j in range(i + 1, len(regions)):
                if include[j] and (IoU(a, b - a, regions[j][0], regions[j][1] - regions[j][0]) > 0.5):
                    include[j] = False
    
    return [r for i, r in enumerate(regions) if include[i]]


def pointwiseRegionProposals(func, extint_min_len = 20, extint_max_len = 150,
                             method = 'hotellings_t', filter = [-1, 0, 1], useMAD = False, sd_th = 1.5, **kwargs):
    """ A generator yielding proposals for possibly anomalous regions.
    
    `func` is the time series to propose possibly anomalous regions for as d-by-n matrix, where
    `d` is the number of attributes and `n` is the number of data points.
    Note that time-delay pre-processing should have been applied to `func` for good results in
    advance and will *not* be done by this function.
    
    `extint_min_len` and `extint_max_len` specify the minimum and the maximum length of the
    proposed regions, respectively.
    
    `method` specifies the point-wise anomaly detection method to derive region proposals from.
    Possible values are 'hotellings_t' and 'kde'.
    
    `filter` is a 1-d filter mask which will be convolved with the series of point-wise scores.
    The result of this convolution will be used to derive region proposals.
    If `filter` is set to None, the actual scores will be used directly.
    
    For generating region proposals, all points above a specific threshold will be considered.
    This threshold is `m + sd_th * sd`, where `sd` is the standard deviation of the scores and
    `m` is the mean.
    If `useMAD` is set to `True`, the median will be used as a robust estimate of the mean and
    the *Median Absolute Deviation* (MAD) as a robust estimate for the standard deviation.
    
    Yields: (a, b, score) tuples, where `a` is the beginning (inclusively) and `b` is the end
            (exclusively) of the proposed region. `score` is a confidence value for the proposal
            in the range [0,1].
    """
    
    METHODS = { 'hotellings_t' : hotellings_t, 'kde' : pointwiseKDE }
    if method not in METHODS:
        raise NameError('Invalid point-wise scoring method: {}'.format(method))
    
    # Compute scores
    scores = METHODS[method](func)
    
    # Filter scores
    if filter is not None:
        pad = (len(filter) - 1) // 2
        padded_scores = np.concatenate((scores[:pad], scores, scores[-pad:]))
        scores = np.abs(np.convolve(padded_scores, filter, 'valid'))
    
    # Determine threshold
    if useMAD:
        score_mean = np.median(scores)
        score_sd = 1.4826 * np.median(np.abs(scores - score_mean)) # robust MAD estimation for the standard deviation
    else:
        score_mean = np.mean(scores)
        score_sd = np.std(scores)
    score_max = np.max(scores)
    if score_max <= 1e-16:
        return
    th = score_mean + sd_th * score_sd
    while not np.any(scores >= th):
        sd_th *= 0.8
        th = score_mean + sd_th * score_sd
    
    # Generate inter-peak proposals
    n = func.shape[1]
    visited = np.zeros(n, dtype = int)
    for i in range(n - extint_min_len + 1):
        if scores[i] >= th:
            for j in range(i + extint_min_len, min(i + extint_max_len, n) + 1):
                if scores[j-1] >= th:
                    yield (i, j, (scores[i] + scores[j-1]) / (2 * score_max))
                    visited[i] += 1
                    visited[j-1] += 1
    
    # Search for isolated peaks and generate proposals with lower threshold
    isolated = np.where((scores >= th) & (visited < 1))[0]
    for i in isolated:
    
        # Search after
        for j in range(i + extint_min_len, min(i + extint_max_len, n) + 1):
            if scores[j-1] >= score_mean:
                yield (i, j, (scores[i] + scores[j-1]) / (2 * score_max))
        
        # Search before
        for j in range(max(i+1 - extint_max_len, 0), i+1 - extint_min_len + 1):
            if scores[j] >= score_mean:
                yield (j, i+1, (scores[i] + scores[j]) / (2 * score_max))
