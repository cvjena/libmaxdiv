import numpy as np
import scipy.spatial.distance


def enforce_multivariate_timeseries(X):
    if X.ndim==1:
        X = np.reshape(X, 1, len(X))


def calc_distance_matrix(X, metric='sqeuclidean'):
    """ Compute pairwise distances between columns in X """
    # results from pdist are usually not stored as a symmetric matrix,
    # therefore, we use squareform to convert it
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.T, 'sqeuclidean'))
    return D

def calc_gaussian_kernel(X, kernel_sigma_sq = 1.0, normalized=True):
    """ Calculate a normalized Gaussian kernel using the columns of X """
    # Let's first compute the kernel matrix from our squared Euclidean distances in $D$.
    dimension = X.shape[0]
    D = calc_distance_matrix(X)
    # compute proper normalized Gaussian kernel values
    K = np.exp(-D/(2.0*kernel_sigma_sq))
    if normalized:
        K = K / ((2*np.pi*kernel_sigma_sq)**(dimension/2.0))
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


def plot_matrix_with_interval(D, a, b):
    """ Show a given kernel or distance matrix with a highlighted interval """
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(range(D.shape[0]), a*np.ones([D.shape[0],1]), 'r-')
    plt.plot(range(D.shape[0]), b*np.ones([D.shape[0],1]), 'r-')
    plt.imshow(D)
    plt.show()