import numpy as np
from eval import pointwiseLabelsToIntervals
from maxdiv import calc_gaussian_kernel, IoU


def hotellings_t(X):
    """ Global version of hotellings_t """
    
    mu = np.mean(X)
    cov = np.cov(X)
    zeromean_X = X - mu
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
    """ Scores every point in the time series by Kernel Density Estimation. """
    
    # Compute kernel matrix
    K = calc_gaussian_kernel(X, kernel_sigma_sq)
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