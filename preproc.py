""" preprocessing routines for time-series """

import numpy as np
from scipy.linalg import solve, lstsq

def get_available_methods():
    return ['local_linear', 'td']

def local_linear_regression(X, window_size=5):
    dimension = X.shape[0]
    n = X.shape[1]
    
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
