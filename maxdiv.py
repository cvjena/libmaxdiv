# coding: utf-8
# Detection of extreme intervals in multivariate time-series
# Author: Erik Rodner together with Milan Flach (2015)

# Novelty detection by finding by minimizing the KL divergence
# In the following, we will derive a similar algorithm based on Kullback-Leibler (KL) divergence
# between the distribution $p_I$ of data points in the extreme interval $I = [a,b)$
# and the distribution $p_{\Omega}$ of non-extreme data points. We approximate both distributions with a simple kernel density estimate:
#
# $p_I(\mathbf{x}) = \frac{1}{|I|} \sum\limits_{i \in I} K(\mathbf{x}, \mathbf{x}_i)$
#
# with $K$ being a normalized kernel, such that $p_I$ is a proper densitity.
# Similarly, we define $p_{\Omega}$ with $\Omega = \{1, \ldots, n\} \setminus I$.

import numpy as np
import matplotlib.pylab as plt

def calc_euclidean_distances(X):
    """ Compute pairwise distances between columns in X """
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X.T, 'sqeuclidean'))
    return D

def calc_normalized_gaussian_kernel(X, kernel_sigma_sq = 1.0):
    """ Calculate a normalized Gaussian kernel using the columns of X """
    # Let's first compute the kernel matrix from our squared Euclidean distances in $D$.
    dimension = X.shape[0]
    D = calc_euclidean_distances(X)
    # compute proper normalized Gaussian kernel values
    K = np.exp(-D/2.0)/((2*np.pi*kernel_sigma_sq)**(dimension/2))
    return K


# Let's derive the algorithm where we try to maximize the KL divergence between the two distributions:
#
# $\text{KL}^{\alpha}(p_{\Omega}, p_I)
# = \frac{1}{n} \sum\limits_{i=1}^n p_{\Omega}(\mathbf{x}_i) \log \frac{ p_{I}^{\alpha}(\mathbf{x}_i) }{ p_{\Omega}(\mathbf{x}_i) }
# = \frac{1}{n} \sum\limits_{i=1}^n p_{\Omega}(\mathbf{x}_i) \log p_{I}^{\alpha}(\mathbf{x}_i) - \frac{1}{n} \sum\limits_{i=1}^n p_{\Omega}(\mathbf{x}_i) \log ( p_{\Omega}(\mathbf{x}_i) ) $
#
# The above formulation uses a parameterized version of the KL divergence (which will be important to get the right results).
# TODO: However, one should use something like the
# power divergence (http://link.springer.com/article/10.1007/s13571-012-0050-3) or the
# density power divergence (http://biomet.oxfordjournals.org/content/85/3/549.full.pdf).
# Plugging everything together we derive at the following algorithm:

def maxdiv_parzen(K, mode="OMEGA_I", alpha=1.0, extint_min_len = 20, extint_max_len = 150):
    """ Finding an extreme interval """
    interval_scores = score_intervals_parzen(K, mode, alpha, extint_min_len, extint_max_len)

    a, boffset = np.unravel_index(np.argmax(interval_scores), interval_scores.shape)
    b = a + boffset

    return a, b, interval_scores[a, boffset]


def score_intervals_parzen(K, mode="OMEGA_I", alpha=1.0, extint_min_len = 20, extint_max_len = 150):
    """ Evaluating all possible intervals and returning the score matrix """
    # the minimal and maximal length of an extreme interval (extint_*_len)
    # this avoids trivial solutions of just one data point in the interval
    # and saves computation time

    endelement = -1
    K_integral = np.cumsum(K, axis=0)
    sums_all = K_integral[endelement,:]
    n = K_integral.shape[0]

    interval_scores = np.zeros([n, extint_max_len])

    # small constant to avoid problems with log(0)
    eps = 1e-7

    # loop through all possible intervals
    for i in range(n-extint_min_len):
        for j in range(i+extint_min_len, min(i+extint_max_len,n)):
            extreme_interval_length = j-i
            non_extreme_points = n - extreme_interval_length
            # sum up kernel values to get non-normalized
            # kernel density estimates at single points for p_I and p_Omega
            sums_extreme = K_integral[j, :] - K_integral[i, :]
            sums_non_extreme = sums_all - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points

            score = 0.0
            # the mode parameter determines which KL divergence to use
            # mode == SYM does not make much sense right now for alpha != 1.0
            if mode == "OMEGA_I" or mode == "SYM":
                # version for maximizing KL(p_Omega, p_I)
                kl_integrand1 = np.mean(np.log(sums_extreme + eps) *
                                               sums_non_extreme)
                kl_integrand2 = np.mean(np.log(sums_non_extreme + eps) *
                                               sums_non_extreme)
                negative_kl_Omega_I = alpha * kl_integrand1 - kl_integrand2
                score += - negative_kl_Omega_I

            # version for maximizing KL(p_I, p_Omega)
            if mode == "I_OMEGA" or mode == "SYM":
                kl_integrand1 = np.mean(np.log(sums_non_extreme + eps) *
                                        sums_extreme)
                kl_integrand2 = np.mean(np.log(sums_extreme + eps) *
                                        sums_extreme)
                negative_kl_I_Omega = alpha * kl_integrand1 - kl_integrand2
                score += - negative_kl_I_Omega

            if mode == "LAMBDA":
                sums_mixed = alpha * sums_extreme + (1-alpha) * sums_non_extreme
                kl_integrand1 = np.mean(np.log(sums_mixed + eps) * sums_extreme)
                kl_integrand2 = np.mean(np.log(sums_extreme + eps) * sums_extreme)
                negative_kl_I_Mixed = kl_integrand1 - kl_integrand2
                kl_integrand1 = np.mean(np.log(sums_mixed + eps) * sums_non_extreme)
                kl_integrand2 = np.mean(np.log(sums_non_extreme + eps) * sums_non_extreme)
                negative_kl_Omega_Mixed = kl_integrand1 - kl_integrand2

                score += - (alpha * negative_kl_I_Mixed + (1-alpha) * negative_kl_Omega_Mixed)



            interval_scores[i,j-i] = score

    return interval_scores

def maxdiv_gaussian(X, mode="OMEGA_I", alpha=1.0, extint_min_len = 20, extint_max_len = 150):
    """ Finding an extreme interval """
    interval_scores = score_intervals_gaussian(X, mode, alpha, extint_min_len, extint_max_len)

    a, boffset = np.unravel_index(np.argmax(interval_scores), interval_scores.shape)
    b = a + boffset

    return a, b, interval_scores[a, boffset]



def score_intervals_gaussian(X, mode, alpha, extint_min_len, extint_max_len):
    X_integral = np.cumsum(X, axis=1)
    n = X.shape[1]

    interval_scores = np.zeros([n, extint_max_len])
    sums_all = X_integral[:, -1]



    for i in range(n-extint_min_len):
        for j in range(i+extint_min_len, min(i+extint_max_len,n)):
            extreme_interval_length = j-i
            non_extreme_points = n - extreme_interval_length
            sums_extreme = X_integral[:, j] - X_integral[:, i]
            sums_non_extreme = sums_all - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points

            score = 0.0
            # the mode parameter determines which KL divergence to use
            # mode == SYM does not make much sense right now for alpha != 1.0
            if mode == "OMEGA_I" or mode == "SYM":
                kl_Omega_I = np.sum((sums_extreme - sums_non_extreme)**2)
                score += kl_Omega_I

            # version for maximizing KL(p_I, p_Omega)
            if mode == "I_OMEGA" or mode == "SYM":
                kl_I_Omega = np.sum((sums_extreme - sums_non_extreme)**2)
                score += kl_I_Omega

            interval_scores[i,j-i] = score

    return interval_scores





def plot_matrix_with_interval(D, a, b):
    """ Show a given kernel or distance matrix with a highlighted interval """
    plt.figure()
    plt.plot(range(D.shape[0]), a*np.ones([D.shape[0],1]), 'r-')
    plt.plot(range(D.shape[0]), b*np.ones([D.shape[0],1]), 'r-')
    plt.imshow(D)
    plt.show()


