import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
import datasets


def mutual_information(func, k, T = 1):
    
    if (k < 2) or (T < 1):
        return 0.0
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    d, n = func.shape
    embed_func = np.vstack([func[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the marginal distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cov_indep = cov.copy()
    cov_indep[:d, d:] = 0
    cov_indep[d:, :d] = 0
    
    # Compute KL divergence between p(x_t, x_(t-T), ..., x_(t - (k-1)*T)) and p(x_t)*p(x_(t-L), ..., x_(t - (k-1)*T))
    return (np.linalg.inv(cov_indep).dot(cov).trace() + np.linalg.slogdet(cov_indep)[1] - np.linalg.slogdet(cov)[1] - embed_func.shape[0]) / 2


if __name__ == '__main__':

    # Parse arguments
    if (len(sys.argv) > 1) and (sys.argv[1] == 'help'):
        print('Usage: {} [<subset>]'.format(sys.argv[0]))
        print()
        print('Plots the average mutual information for different time lags on a given dataset.')
        exit()
    subset = sys.argv[1] if len(sys.argv) > 1 else None

    # Load datasets
    if subset is not None:
        data = datasets.loadDatasets()
        if subset not in data:
            print('Dataset not found: {}'.format(subset))
            exit()
        data = data[subset]
    else:
        data = sum(datasets.loadSyntheticTestbench().values(), [])

    # Compute mutual information for various time lags and all functions in the data set
    lags = np.arange(1, 61)
    mi = np.array([[mutual_information(func['ts'], 2, lag) for func in data] for lag in lags])
    
    # Plot average mutual information for each time lag
    mi_mean = np.mean(mi, axis = 1)
    mi_sd = np.std(mi, axis = 1)
    min_lag = np.argmin(mi_mean)
    mi_th = mi_mean[min_lag] + mi_sd[min_lag]
    print('Time-Lag with minimal Mutual Information: {}'.format(lags[min_lag]))
    print('First lag below min_lag + sd: {}'.format(lags[(mi_mean <= mi_th).nonzero()[0][0]]))
    plt.errorbar(lags, mi_mean, yerr = mi_sd, fmt = '-')
    plt.plot(lags, [mi_th] * len(lags), '--', color = 'gray')
    plt.show()
    
    # Compute mutual information for various embedding dimensions and all functions in the data set
    dims = np.arange(2, 21)
    mi = np.array([[mutual_information(func['ts'], dim) for func in data] for dim in dims])
    
    # Plot average mutual information for each embedding dimension
    mi_mean = np.mean(mi, axis = 1)
    mi_sd = np.std(mi, axis = 1)
    max_dim = np.argmax(mi_mean)
    mi_th = mi_mean[max_dim] - mi_sd[max_dim]
    print('Embedding dimension with maximal Mutual Information: {}'.format(dims[max_dim]))
    print('First dimension above max_dim - sd: {}'.format(dims[(mi_mean >= mi_th).nonzero()[0][0]]))
    plt.errorbar(dims, mi_mean, yerr = mi_sd, fmt = '-')
    plt.plot(dims, [mi_th] * len(dims), '--', color = 'gray')
    plt.show()