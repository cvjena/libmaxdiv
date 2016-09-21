import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
import datasets


def mutual_information(func, k, T = 1):
    
    d, n = func.shape
    
    if (k < 2) or (T < 1):
        # Entropy as a special case of MI
        cov = np.cov(func)
        if d > 1:
            return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (d * (np.log(2 * np.pi) + 1) + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([func[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the marginal distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cov_indep = cov.copy()
    cov_indep[:d, d:] = 0
    cov_indep[d:, :d] = 0
    
    # Compute KL divergence between p(x_t, x_(t-T), ..., x_(t - (k-1)*T)) and p(x_t)*p(x_(t-L), ..., x_(t - (k-1)*T))
    return (np.linalg.inv(cov_indep).dot(cov).trace() + np.linalg.slogdet(cov_indep)[1] - np.linalg.slogdet(cov)[1] - embed_func.shape[0]) / 2


def conditional_entropy(func, k, T = 1):
    
    d, n = func.shape
    
    if (k < 2) or (T < 1):
        # Entropy as a special case
        cov = np.cov(func)
        if d > 1:
            return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (d * (np.log(2 * np.pi) + 1) + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([func[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the conditioned distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cond_cov = cov[:d, :d] - cov[:d, d:].dot(np.linalg.inv(cov[d:, d:]).dot(cov[d:, :d]))
    
    # Compute the conditional entropy H(x_t | x_(t-T), ..., x_(t - (k-1)*T))
    return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cond_cov)[1]) / 2


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
    
    # Plot average mutual information and its gradient for each time lag
    mi_mean = np.mean(mi, axis = 1)
    mi_sd = np.std(mi, axis = 1)
    min_lag = np.argmin(mi_mean)
    mi_th = mi_mean[min_lag] + mi_sd[min_lag]
    print('Time-Lag with minimal Mutual Information: {}'.format(lags[min_lag]))
    print('First lag below min_lag + sd: {}'.format(lags[(mi_mean <= mi_th).nonzero()[0][0]]))
    plt.figure()
    plt.errorbar(lags, mi_mean, yerr = mi_sd, fmt = '-')
    plt.plot(lags, [mi_th] * len(lags), '--', color = 'gray')
    plt.figure()
    plt.plot(lags[1:-1], np.convolve(mi_mean / mi_mean[0], [1, 0, -1], 'valid'), '-r')
    plt.plot(lags[1:-1], [-0.05] * (len(lags) - 2), '--k')
    plt.show()
    
    # Compute conditional entropy for various embedding dimensions and all functions in the data set
    dims = np.arange(2, 61)
    ce = np.array([[conditional_entropy(func['ts'], dim) for func in data] for dim in dims])
    
    # Plot average conditional entropy and its gradient for each embedding dimension
    ce_mean = np.mean(ce, axis = 1)
    ce_sd = np.std(ce, axis = 1)
    min_dim = np.argmin(ce_mean)
    ce_th = ce_mean[min_dim] + ce_sd[min_dim]
    print('Embedding dimension with minimal conditional entropy: {}'.format(dims[min_dim]))
    print('First dimension below min_dim + sd: {}'.format(dims[(ce_mean <= ce_th).nonzero()[0][0]]))
    plt.errorbar(dims, ce_mean, yerr = ce_sd, fmt = '-')
    plt.plot(dims, [ce_th] * len(dims), '--', color = 'gray')
    plt.figure()
    plt.plot(dims[1:-1], np.convolve(ce_mean, [1, 0, -1], 'valid'), '-r')
    plt.plot(dims[1:-1], [-0.01] * (len(dims) - 2), '--k')
    plt.figure()
    for i in range(4):
        plt.plot(dims, ce[:, i])
    plt.show()