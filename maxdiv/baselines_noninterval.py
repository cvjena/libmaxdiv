import numpy as np
from sklearn.mixture import GMM
from .eval import pointwiseLabelsToIntervals
from .maxdiv_util import calc_gaussian_kernel, IoU


def hotellings_t(X):
    """ Global version of hotellings_t """
    
    mu = X.mean(axis = 1)
    cov = np.ma.cov(X).filled(0)
    zeromean_X = (X.T - mu).T
    if X.shape[0]==1:
        norm_X = zeromean_X/cov
        scores = zeromean_X*norm_X
        scores = scores.ravel()
    else:
        invcov = np.linalg.inv(cov)
        norm_X = np.dot(invcov, zeromean_X) if not np.ma.isMaskedArray(zeromean_X) else np.ma.dot(invcov, zeromean_X)
        scores = (norm_X * zeromean_X).sum(axis=0)
    if np.ma.isMaskedArray(X):
        scores.mask = X.mask[0,:]
    return scores


def pointwiseKDE(X, kernel_sigma_sq = 1.0):
    """ Scores every point in the time series by Kernel Density Estimation.
    
    Note: The values of X should be in range [-1,1] to avoid numerical problems.
    """
    
    # Compute kernel matrix
    K = calc_gaussian_kernel(X, kernel_sigma_sq, False)
    # Score points by their unlikelihood
    prob = K.mean(axis = 0)
    if np.ma.isMaskedArray(X):
        prob.mask = X.mask[0,:] if X.ndim > 1 else X.mask
    return (1.0 - prob)


def gmm_scores(X, n_components = 2):
    """ Fits a Gaussian Mixture Model to the data and detects anomalies based on that model.
    
    The component with the highest weight will be considered the model for the nominal part of
    the time-series. If the a-priori score for a point under any other component is higher, it
    will be considered anomalous.
    """
    
    gmm = GMM(n_components, 'full', n_init = 10)
    gmm.fit(X.T)
    nominal_component = gmm.weights_.argmax()
    if n_components == 2:
        scores = gmm.score_samples(X.T)[1][:, 1 - nominal_component]
    else:
        scores = (1.0 - gmm.score_samples(X.T)[1][:, nominal_component])
    if np.ma.isMaskedArray(X):
        scores = np.ma.MaskedArray(scores, X.mask[0,:] if X.ndim > 1 else X.mask)
    return scores


def rkde(X, kernel_sigma_sq = 1.0, type = 'hampel'):
    """Perform Robust Kernel Density Estimation (Kim & Scott, 2012) for point-wise anomaly detection.
    
    `type` can either be 'huber' or 'hampel'.
    """
    
    def rho(x, type, params):
        if type == 'huber':
            a = params[0]
            J = np.sum((x[x <= a] ** 2) / 2) + np.sum(a * (x[x > a] - a) + (a*a)/2)
        elif type == 'hampel':
            a, b, c = params
            i1 = (x <= a)
            i2 = ((a < x) & (x <= b))
            i3 = ((b < x) & (x <= c))
            i4 = (c < x)
            p = -a/(c-b)
            q = a*c/(c-b)
            r = a*b - (a*a)/2 - (p*b*b)/2 - q*b
            J = np.sum((x[i1] ** 2) / 2)
            J += np.sum(a * (x[i2] - a) + (a*a)/2)
            J += np.sum(p * (x[i3] ** 2) / 2 + q * x[i3] + r)
            J += np.sum((p * c * c) / 2 + q * c + r)
        return J/len(x)
    
    def psi(x, type, params):
        if type == 'huber':
            return np.minimum(x, params[0])
        elif type == 'hampel':
            a, b, c = params
            i1 = (x < a)
            i2 = ((x >= a) & (x < b))
            i3 = ((x >= b) & (x < c))
            i4 = (x >=c)
            out = np.ndarray(x.shape)
            out[i1] = x[i1]
            out[i2] = a
            out[i3] = a * (c - x[i3])/(c - b)
            out[i4] = 0
            return out
    
    def parameter_select(K, type):
        n = K.shape[0]
        w = np.ones(n) / n
        tol = 1e-8
        norm2mu = np.dot(w.T, np.dot(K, w))
        normdiff = np.sqrt(K.diagonal()) + norm2mu - 2 * K.dot(w)
        J = normdiff.sum() / n
        while True:
            J_old = J
            w = 1.0 / normdiff
            w /= w.sum()
            norm2mu = np.dot(w.T, np.dot(K, w))
            normdiff = np.sqrt(K.diagonal()) + norm2mu - 2 * K.dot(w)
            J = normdiff.sum() / n
            if abs(J_old - J) < J_old * tol:
                break
        if type == 'huber':
            return (np.median(normdiff),)
        elif type == 'hampel':
            return (np.median(normdiff), np.percentile(normdiff, 95), np.max(normdiff))
    
    if type not in ('huber', 'hampel'):
        raise ValueError('Unknown loss function: {}'.format(type))
    
    d, n = X.shape
    K = calc_gaussian_kernel(X, kernel_sigma_sq)
    
    # Find median absolute deviation
    params = parameter_select(K, type)
    
    # Initialize weights
    w = np.ones(n) / n
    tol = 1e-8
    
    # Compute loss of the initial solution in kernel space
    norm2mu = np.dot(w.T, np.dot(K, w)) # norm of the kernel space solution
    normdiff = np.sqrt(K.diagonal()) + norm2mu - 2 * K.dot(w) # norm of the difference between target and solution
    J = rho(normdiff, type, params) # compute loss
    
    # Kernelized Iteratively Re-weighted Least Squares (KIRWLS)
    while True:
        J_old = J
        
        # Obtain new weights
        w = psi(normdiff, type, params) / normdiff
        w /= w.sum()
        
        # Compute loss of the new kernel space solution
        norm2mu = np.dot(w.T, np.dot(K, w)) # norm of the kernel space solution
        normdiff = np.sqrt(K.diagonal()) + norm2mu - 2 * K.dot(w) # norm of the difference between target and solution
        J = rho(normdiff, type, params) # compute loss
        
        # Check termination criterion
        if abs(J_old - J) < J_old * tol:
            break
    
    # Score points by their unlikelihood
    return (1.0 - np.sum(w * K, axis = 1))


def pointwiseScoresToIntervals(scores, min_length = 0):
    
    first_th = np.percentile(scores, 0.7)
    max_score = np.max(scores)
    
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
        padded_scores = np.ma.concatenate((scores[:pad], scores, scores[-pad:]))
        conv_scores = np.abs(np.convolve(padded_scores, filter, 'valid'))
        if np.ma.isMaskedArray(scores) and (np.asarray(filter) == [-1, 0, 1]).all():
            scores = np.ma.MaskedArray(conv_scores, ((scores.mask[max(0, i - 1)] or scores.mask[min(len(scores) - 1, i + 1)]) for i in range(len(scores))))
        else:
            scores = conv_scores
    
    # Determine threshold
    if useMAD:
        score_mean = np.ma.median(scores).filled(0)
        score_sd = 1.4826 * np.ma.median(np.abs(scores - score_mean)).filled(0) # robust MAD estimation for the standard deviation
    else:
        score_mean = scores.mean()
        score_sd = scores.std()
    score_max = scores.max()
    if score_max <= 1e-16:
        return
    th = score_mean + sd_th * score_sd
    while not (scores >= th).any():
        sd_th *= 0.8
        th = score_mean + sd_th * score_sd
    
    # Generate inter-peak proposals
    if np.ma.isMaskedArray(scores):
        scores = scores.filled(min(0, scores.min()))
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
