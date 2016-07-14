import numpy as np
import scipy.spatial.distance


def enforce_multivariate_timeseries(X):
    if X.ndim==1:
        X = X.reshape((1, len(X)))
    return X


def calc_distance_matrix(X, metric='sqeuclidean'):
    """ Compute pairwise distances between columns in X """
    # results from pdist are usually not stored as a symmetric matrix,
    # therefore, we use squareform to convert it
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.T, metric))
    return D


def calc_gaussian_kernel(X, kernel_sigma_sq = 1.0, normalized=True):
    """ Calculate a normalized Gaussian kernel using the columns of X """
    # Let's first compute the kernel matrix from our squared Euclidean distances in $D$.
    D = calc_distance_matrix(X)
    # compute proper normalized Gaussian kernel values
    K = np.exp(-D/(2.0*kernel_sigma_sq))
    if normalized:
        K = K / ((2*np.pi*kernel_sigma_sq) ** (X.shape[0] / 2))
    return K


def calc_nonstationary_gaussian_kernel(X, kernel_sigma_sq_vec):
    """ Calculate a normalized Gaussian kernel using the columns of X """
    # Let's first compute the kernel matrix from our squared Euclidean distances in $D$.
    dimension = X.shape[0]
    n = X.shape[1]
    D = calc_distance_matrix(X)
    S = np.tile(kernel_sigma_sq_vec, [n,1])
    S_sum = S + S.T
    S_prod = S * S.T
    
    # compute Gaussian kernel values
    K = np.exp(-D/(0.5*S_sum))*(np.power(S_prod,0.25)/np.sqrt(0.5*S_sum))
    return K


def IoU(start1, len1, start2, len2):
    """ Computes the intersection over union of two intervals starting at start1 and start2 with lengths len1 and len2. """
    intersection = max(0, min(start1 + len1, start2 + len2) - max(start1, start2))
    return float(intersection) / (len1 + len2 - intersection)


def td_mutual_information(ts, k, T = 1):
    """ Computes mutual information between each time-step and some number of previous time steps.
    
    ts - The time-series.
    k - Total number of time steps to be taken into account. This may be set to 1 to compute entropy.
    T - Distance between two consecutive time steps.
    
    A multivariate normal distribution is assumed to mode the distribution of the data.
    """
    
    d, n = ts.shape
    
    if (k < 2) or (T < 1):
        # Entropy as a special case of MI
        cov = np.cov(ts)
        if d > 1:
            return (n * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (n * (np.log(2 * np.pi) + 1) + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([ts[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the marginal distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cov_indep = cov.copy()
    cov_indep[:d, d:] = 0
    cov_indep[d:, :d] = 0
    
    # Compute KL divergence between p(x_t, x_(t-T), ..., x_(t - (k-1)*T)) and p(x_t)*p(x_(t-L), ..., x_(t - (k-1)*T))
    return (np.linalg.inv(cov_indep).dot(cov).trace() + np.linalg.slogdet(cov_indep)[1] - np.linalg.slogdet(cov)[1] - embed_func.shape[0]) / 2


def m_estimation(A, b, k = 1.345):
    """ Robust M-Estimation with Huber's function.
    
    This function solves `A * x = b`, but is less sensitive to outliers than ordinary least squares.
    """
    
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


def plot_matrix_with_interval(D, a, b):
    """ Show a given kernel or distance matrix with a highlighted interval """
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(range(D.shape[0]), a*np.ones([D.shape[0],1]), 'r-')
    plt.plot(range(D.shape[0]), b*np.ones([D.shape[0],1]), 'r-')
    plt.imshow(D)
    plt.show()