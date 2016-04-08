import numpy as np


def hotellings_t(X):
    # global version of hotellings_t 
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

