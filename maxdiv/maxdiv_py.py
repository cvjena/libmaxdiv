# coding: utf-8
#
# Detection of extreme intervals in multivariate time-series
# Author: Erik Rodner (2015-) & Björn Barz (2016)

# Python implementation of several variants of the "Maximally Divergent Intervals (MDI)" algorithm.

# In the following, we will derive an algorithm based on Kullback-Leibler (KL) divergence
# between the distribution $p_I$ of data points in the extreme interval $I = [a,b)$
# and the distribution $p_{\Omega}$ of non-extreme data points. We approximate both distributions with a simple kernel density estimate:
#
# $p_I(\mathbf{x}) = \frac{1}{|I|} \sum\limits_{i \in I} K(\mathbf{x}, \mathbf{x}_i)$
#
# with $K$ being a normalized kernel, such that $p_I$ is a proper densitity.
# Similarly, we define $p_{\Omega}$ with $\Omega = \{1, \ldots, n\} \setminus I$.

import numpy as np
from numpy.linalg import slogdet, inv, solve
from scipy.linalg import solve_triangular, cholesky, cho_factor, cho_solve
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process.gaussian_process import l1_cross_distances
import math, time, types, warnings


#
# Maximally divergent regions using Kernel Density Estimation
#
def maxdiv_parzen(K, intervals, mode = 'I_OMEGA', alpha = 1.0, **kwargs):
    """ Scores given intervals by using Kernel Density Estimation.
    
    `K` is a symmetric kernel matrix whose components are K(|i - j|) for a given kernel K.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    # compute integral sums for each column within the kernel matrix 
    K_integral = np.cumsum(K if not np.ma.isMaskedArray(K) else K.filled(0), axis = 0)
    # the sum of all kernel values for each column
    # is now given in the last row
    sums_all = K_integral[-1,:]
    # n is the number of data points considered
    n = K_integral.shape[0]
    if np.ma.isMaskedArray(K):
        i = 0
        while (K.mask[i,:].all()):
            i += 1
        mask = K.mask[i,:]
        numValidSamples = n - mask.sum()
    else:
        numValidSamples = n

    # list of results
    scores = []

    # small constant to avoid problems with log(0)
    eps = 1e-7

    # indicators for points inside and outside of the anomalous region
    extreme = np.zeros(n, dtype=bool)
    non_extreme = np.ones(n, dtype=bool)
    # loop through all intervals
    for a, b, base_score in intervals:
    
        score = 0.0
        
        extreme[:] = False
        extreme[a:b] = True
        non_extreme = np.logical_not(extreme)
        if np.ma.isMaskedArray(K):
            extreme[mask] = False
            non_extreme[mask] = False

        # number of data points in the current interval
        extreme_interval_length = b - a if not np.ma.isMaskedArray(K) else b - a - mask[a:b].sum()
        # number of data points outside of the current interval
        non_extreme_points = numValidSamples - extreme_interval_length
        
        # compute the KL divergence
        # the mode parameter determines which KL divergence to use
        # mode == SYM does not make much sense right now for alpha != 1.0
        if mode == "IS_I_OMEGA":
            # for comments see OMEGA_I
            # this is a very experimental mode that exploits importance sampling
            sums_extreme = K_integral[b-1, :] - (K_integral[a-1, :] if a > 0 else 0)
            sums_non_extreme = sums_all - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            weights = sums_extreme / (sums_non_extreme + eps)
            weights[extreme] = 1.0
            weights /= np.sum(weights)
            kl_integrand1 = np.sum(weights * np.log(sums_non_extreme + eps))
            kl_integrand2 = np.sum(weights * np.log(sums_extreme + eps))
            negative_kl_I_Omega = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_I_Omega


        if mode == "OMEGA_I" or mode == "SYM":
            # sum up kernel values to get non-normalized
            # kernel density estimates at single points for p_I and p_Omega
            # we use the integral sums in K_integral
            # sums_extreme and sums_non_extreme are vectors of size n
            sums_extreme = K_integral[b-1, non_extreme] - (K_integral[a-1, non_extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[non_extreme] - sums_extreme
            # divide by the number of data points to get the final
            # parzen scores for each data point
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points

            # version for maximizing KL(p_Omega, p_I)
            # in this case we have p_Omega 
            kl_integrand1 = np.mean(np.log(sums_extreme + eps))
            kl_integrand2 = np.mean(np.log(sums_non_extreme + eps))
            negative_kl_Omega_I = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_Omega_I

        # version for maximizing KL(p_I, p_Omega)
        if mode == "I_OMEGA" or mode == "SYM":
            # for comments see OMEGA_I
            sums_extreme = K_integral[b-1, extreme] - (K_integral[a-1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            kl_integrand1 = np.mean(np.log(sums_non_extreme + eps))
            kl_integrand2 = np.mean(np.log(sums_extreme + eps))
            negative_kl_I_Omega = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_I_Omega
        
        # Cross Entropy
        if mode == "CROSSENT" or mode == "CROSSENT_TS":
            sums_extreme = K_integral[b-1, extreme] - (K_integral[a-1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            score -= np.sum(np.log(sums_non_extreme + eps)) if mode == "CROSSENT_TS" else np.mean(np.log(sums_non_extreme + eps))
        
        # Jensen-Shannon Divergence
        if mode == 'JSD':
            jsd = 0.0
            
            # Compute p_I and p_Omega for extremal points
            sums_extreme = K_integral[b-1, extreme] - (K_integral[a-1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            # Compute (p_I + p_Omega)/2 for extremal points
            sums_combined = (sums_extreme + sums_non_extreme) / 2
            # Compute sum over extremal region
            jsd += np.mean(np.log2(sums_extreme + eps) - np.log2(sums_combined + eps))
            
            # Compute p_I and p_Omega for non-extremal points
            sums_extreme = K_integral[b-1, non_extreme] - (K_integral[a-1, non_extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[non_extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            # Compute (p_I + p_Omega)/2 for non-extremal points
            sums_combined = (sums_extreme + sums_non_extreme) / 2
            # Compute sum over non-extremal region
            jsd += np.mean(np.log2(sums_non_extreme + eps) - np.log2(sums_combined + eps))
            
            score += jsd / 2.0

        # store the score
        scores.append((a, b, score))
    
    return scores

#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian_globalcov(X, intervals, mode = 'I_OMEGA', gaussian_mode = 'GLOBAL_COV', **kwargs):
    """ Scores given intervals by assuming gaussian distributions with equal covariance.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0,:].count()

    X_integral = np.cumsum(X if not np.ma.isMaskedArray(X) else X.filled(0), axis=1)
    sums_all = X_integral[:, -1]
    if (gaussian_mode == 'GLOBAL_COV') and (dimension > 1):
        cov = np.ma.cov(X).filled(0)
        cov_chol = cho_factor(cov)
        logdet = slogdet(cov)[1]

    scores = []

    eps = 1e-7
    for a, b, base_score in intervals:
        
        extreme_interval_length = b - a if not np.ma.isMaskedArray(X) else X[0,a:b].count()
        non_extreme_points = numValidSamples - extreme_interval_length
        
        sums_extreme = X_integral[:, b-1] - (X_integral[:, a-1] if a > 0 else 0)
        sums_non_extreme = sums_all - sums_extreme
        sums_extreme /= extreme_interval_length
        sums_non_extreme /= non_extreme_points

        diff = sums_extreme - sums_non_extreme
        if (gaussian_mode == 'GLOBAL_COV') and (dimension > 1):
            score = diff.T.dot(cho_solve(cov_chol, diff))
            if (mode == 'CROSSENT') or (mode == 'CROSSENT_TS'):
                score += slogdet
        else:
            score = np.sum(diff * diff)
        if (mode == 'CROSSENT') or (mode == 'CROSSENT_TS'):
            score += dimension * (1 + np.log(2 * np.pi))
        scores.append((a, b, score))

    return scores


#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian(X, intervals, mode = 'I_OMEGA', gaussian_mode = 'COV', **kwargs):
    """ Scores given intervals by assuming gaussian distributions.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    if gaussian_mode in ('COV_TS', 'TS'):
        gaussian_mode = 'COV'
        mode = 'TS'
    
    if gaussian_mode!='COV':
        return maxdiv_gaussian_globalcov(X, intervals, mode, gaussian_mode)

    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0,:].count()
    X_integral = np.cumsum(X if not np.ma.isMaskedArray(X) else X.filled(0), axis=1)
    sums_all = X_integral[:, -1]
    scores = []

    # compute integral series of the outer products
    # we will use this to compute covariance matrices
    outer_X = np.apply_along_axis(lambda x: np.ravel(np.outer(x,x)), 0, X)
    if np.ma.isMaskedArray(X):
        outer_X[:,X.mask[0,:]] = 0
    outer_X_integral = np.cumsum(outer_X, axis=1)
    outer_sums_all = outer_X_integral[:, -1]
    
    if mode == 'TS':
        ts_mean = X.shape[0] + (X.shape[0] * (X.shape[0] + 1)) / 2
        ts_sd = np.sqrt(2 * ts_mean)

    eps = 1e-7
    for a, b, base_score in intervals:
        
        score = 0.0
        
        extreme_interval_length = b - a if not np.ma.isMaskedArray(X) else X[0,a:b].count()
        non_extreme_points = numValidSamples - extreme_interval_length
        
        sums_extreme = X_integral[:, b-1] - (X_integral[:, a-1] if a > 0 else 0)
        sums_non_extreme = sums_all - sums_extreme
        sums_extreme /= extreme_interval_length
        sums_non_extreme /= non_extreme_points

        outer_sums_extreme = outer_X_integral[:, b-1] - (outer_X_integral[:, a-1] if a > 0 else 0)
        outer_sums_non_extreme = outer_sums_all - outer_sums_extreme
        outer_sums_extreme /= extreme_interval_length
        outer_sums_non_extreme /= non_extreme_points

        cov_extreme = np.reshape(outer_sums_extreme, [dimension, dimension]) - \
                np.outer(sums_extreme, sums_extreme) + eps * np.eye(dimension)
        cov_non_extreme = np.reshape(outer_sums_non_extreme, [dimension, dimension]) - \
                np.outer(sums_non_extreme, sums_non_extreme) + eps * np.eye(dimension)

        if mode != 'JSD':
            logdet_extreme = slogdet(cov_extreme)[1]
            logdet_non_extreme = slogdet(cov_non_extreme)[1]
            diff = sums_extreme - sums_non_extreme

        # the mode parameter determines which KL divergence to use
        # mode == SYM does not make much sense right now for alpha != 1.0
        if mode == "OMEGA_I" or mode == "SYM":
            # alternative version using implicit inversion
            #kl_Omega_I = np.dot(diff, solve(cov_extreme, diff.T) )
            #kl_Omega_I += np.sum(np.diag(solve(cov_extreme, cov_non_extreme)))
            inv_cov_extreme = inv(cov_extreme)
            # term for the mahalanobis distance
            kl_Omega_I = np.dot(diff, np.dot(inv_cov_extreme, diff.T))
            # trace term
            kl_Omega_I += np.trace(np.dot(inv_cov_extreme, cov_non_extreme))
            # logdet terms
            kl_Omega_I += logdet_extreme - logdet_non_extreme - dimension
            score += kl_Omega_I

        # version for maximizing KL(p_I, p_Omega)
        if mode in ("I_OMEGA", "SYM", "TS"):
            inv_cov_non_extreme = inv(cov_non_extreme)
            # term for the mahalanobis distance
            kl_I_Omega = np.dot(diff, np.dot(inv_cov_non_extreme, diff.T))
            # trace term
            kl_I_Omega += np.trace(np.dot(inv_cov_non_extreme, cov_extreme))
            # logdet terms
            kl_I_Omega += logdet_non_extreme - logdet_extreme - dimension
            if mode == 'TS':
                score += (extreme_interval_length * kl_I_Omega - ts_mean) / ts_sd
            else:
                score += kl_I_Omega
        
        # version for maximizing Cross Entropy
        if mode in ("CROSSENT", "CROSSENT_TS"):
            inv_cov_non_extreme = inv(cov_non_extreme)
            # term for the mahalanobis distance
            ce_I_Omega = np.dot(diff, np.dot(inv_cov_non_extreme, diff.T))
            # trace term
            ce_I_Omega += np.trace(np.dot(inv_cov_non_extreme, cov_extreme))
            # logdet term
            ce_I_Omega += logdet_non_extreme + dimension * np.log(2 * np.pi)
            if mode == 'CROSSENT_TS':
                score += (extreme_interval_length * ce_I_Omega - ts_mean) / ts_sd
            else:
                score += ce_I_Omega
        
        # Jensen-Shannon Divergence
        if mode == 'JSD':
            # Compute probability densities
            pdf_extreme     = multivariate_normal.pdf(X.T, sums_extreme, cov_extreme)
            pdf_non_extreme = multivariate_normal.pdf(X.T, sums_non_extreme, cov_non_extreme)
            pdf_combined    = (pdf_extreme + pdf_non_extreme) / 2
            if np.ma.isMaskedArray(X):
                pdf_extreme = np.ma.MaskedArray(pdf_extreme, X.mask[0,:])
                pdf_non_extreme = np.ma.MaskedArray(pdf_non_extreme, X.mask[0,:])
                pdf_combined = np.ma.MaskedArray(pdf_combined, X.mask[0,:])
            # Compute JSD
            jsd_extreme     = (np.log2(pdf_extreme[a:b] + eps) - np.log2(pdf_combined[a:b] + eps)).mean()
            jsd_non_extreme = (np.log2(np.concatenate((pdf_non_extreme[:a], pdf_non_extreme[b:])) + eps)
                                      - np.log2(np.concatenate((pdf_combined[:a], pdf_combined[b:])) + eps)).mean()
            score += (jsd_extreme + jsd_non_extreme) / 2.0
        
        #print score, cov_extreme, cov_non_extreme, diff

        scores.append((a, b, score))

    return scores


#
# Maximally divergent regions using an Ensemble of Random Projection Histograms
#
def maxdiv_erph(X, intervals, mode = 'I_OMEGA', num_hist = 100, num_bins = None, discount = 1, **kwargs):
    """ Scores given intervals by estimating the joint likelihood over all attributes using an ensemble
    of histograms over random 1d projections.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    
    `num_hist` specifies the number of histograms / random projections to be used. Should be
    much greater than `d`.
    
    `num_bins` specifies the number of bins to be used per histogram. If set to `None` an
    individual number of bins will be determined automatically for each histogram.
    
    `discount` is a constant value added to the count of each bin in order to make unseen values
    not completely unlikely.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0,:].count()
    scores = []
    
    if (num_bins is not None) and (num_bins < 1):
        num_bins = None
    
    if (discount < 1e-7):
        discount = 1e-7
    
    # Generate random projections
    proj_dims = int(round(np.sqrt(dimension)))
    dim_range = np.arange(dimension)
    proj = np.zeros((num_hist, dimension))
    for i in range(num_hist):
        np.random.shuffle(dim_range)
        proj[i, dim_range[:proj_dims]] = np.random.randn(proj_dims)
    
    # Project data and initialize histograms
    Xp = proj.dot(X) if not np.ma.isMaskedArray(X) else np.ma.dot(proj, X)
    hist = [Histogram1D(Xp[i,:], num_bins, store_data = False) for i in range(num_hist)]
    ind = np.array([hist[i].indices(Xp[i,:]) for i in range(num_hist)])
    counts = [np.array([[1.0 if ind[i, j] == b else 0.0 for j in range(n)] for b in range(hist[i].num_bins)]).cumsum(axis = 1) for i in range(num_hist)]

    # Score intervals
    extreme = np.zeros(n, dtype=bool)
    non_extreme = np.ones(n, dtype=bool)
    counts_inner, counts_outer = [None] * num_hist, [None] * num_hist
    prob_inner, prob_outer = [None] * num_hist, [None] * num_hist
    logprob_inner, logprob_outer = [None] * num_hist, [None] * num_hist
    for a, b, base_score in intervals:
        
        score = 0.0
        
        extreme[:] = False
        extreme[a:b] = True
        non_extreme = np.logical_not(extreme)

        # number of data points in the current interval
        extreme_interval_length = b - a  if not np.ma.isMaskedArray(X) else X[0,a:b].count()
        # number of data points outside of the current interval
        non_extreme_points = numValidSamples - extreme_interval_length
        
        # Compute histograms and probability density estimates
        for i in range(num_hist):
            counts_inner[i] = counts[i][:, b-1] - (counts[i][:, a-1] if a > 0 else 0.0)
            counts_outer[i] = counts[i][:, -1] - counts_inner[i]
            prob_inner[i] = hist[i].num_bins * (counts_inner[i] + discount) / (extreme_interval_length + hist[i].num_bins * discount)
            prob_outer[i] = hist[i].num_bins * (counts_outer[i] + discount) / (non_extreme_points + hist[i].num_bins * discount)
            logprob_inner[i] = np.log(prob_inner[i])
            logprob_outer[i] = np.log(prob_outer[i])

        # Compute divergence
        if mode == "OMEGA_I" or mode == "SYM":
            for i in range(num_hist):
                score += np.sum(counts_outer[i] * (logprob_outer[i] - logprob_inner[i])) / (non_extreme_points * num_hist)
        
        if mode == "I_OMEGA" or mode == "SYM":
            for i in range(num_hist):
                score += np.sum(counts_inner[i] * (logprob_inner[i] - logprob_outer[i])) / (extreme_interval_length * num_hist)
        
        if mode == "CROSSENT" or mode == "CROSSENT_TS":
            quotient = num_hist if mode == "CROSSENT_TS" else (extreme_interval_length * num_hist)
            for i in range(num_hist):
                score -= np.sum(counts_inner[i] * logprob_outer[i]) / quotient
        
        if mode == 'JSD':
            jsd = 0.0
            
            for i in range(n):
                if not np.ma.is_masked(X[0,i]):
                    p_I = sum(logprob_inner[h][ind[h,i]] for h in range(num_hist)) / num_hist
                    p_Omega = sum(logprob_outer[h][ind[h,i]] for h in range(num_hist)) / num_hist
                    logprob_combined = np.log((np.exp(p_I) + np.exp(p_Omega)) / 2)
                    if extreme[i]:
                        jsd += (p_I - logprob_combined) / extreme_interval_length
                    else:
                        jsd += (p_Omega - logprob_combined) / non_extreme_points
            
            score += jsd / (np.log(2.0) * 2.0)

        # store the score
        scores.append((a, b, score))

    return scores


class Histogram1D(object):
    
    def __init__(self, X, num_bins = None, store_data = True):
        """ Initializes a 1d histogram for the data in the vector X.
        
        If `num_bins` is set to `None`, the number of bins will be determined from the data.
        
        If `store_data` is `False`, the histogram will be initialized, but empty.
        """

        object.__init__(self)
        self.vmin, self.vmax = X.min(), X.max()
        
        if num_bins is None:
            
            max_pml = 0.0
            argmax_pml = None
            last_ll = np.ndarray((20,))
            last_pen = np.ndarray((20,))
            for b in range(2, len(X) // 2 + 1):
                self.num_bins = b
                self.fit(X)
                pml, ll, penalty = self._penalizedML()
                last_ll[b % 20] = ll
                last_pen[b % 20] = penalty
                if (argmax_pml is None) or (pml > max_pml):
                    max_pml, argmax_pml = pml, b
                elif (b >= 22) and (b - argmax_pml > 20) and (np.mean(last_ll[1:] - last_ll[:-1]) < np.mean(last_pen[1:] - last_pen[:-1])):
                    break
            self.num_bins = argmax_pml
            
        else:
            self.num_bins = num_bins
        
        if store_data:
            self.fit(X)
        else:
            self.counts = np.zeros(self.num_bins, dtype = int)
            self.N = 0
    
    def fit(self, X, ind = False):
        """ Reset the histogram and add the samples in X. """
        
        self.counts = np.zeros(self.num_bins, dtype = int)
        self.N = len(X) if not np.ma.isMaskedArray(X) else X.count()
        if not ind:
            X = self.indices(X)
        self.counts = np.array([(X == i).sum() for i in range(self.num_bins)])
    
    def pdf(self, X, ind = False):
        """ Retrieve the frequencies of the samples in X. """
        
        if not ind:
            X = self.indices(X)
        prob = self.num_bins * self.counts[X].astype(float) / self.N
        if np.ma.isMaskedArray(X):
            prob = np.ma.MaskedArray(X, X.mask)
        return prob
    
    def indices(self, X):
        """ Retrieve the indices of the bins for the samples in X. """
        
        ind = ((X - self.vmin) * self.num_bins / (self.vmax - self.vmin)).astype(int)
        ind[ind < 0] = 0
        ind[ind >= self.num_bins] = self.num_bins - 1
        return ind
    
    def _penalizedML(self):
        """ The penalized maximum likelihood of the histogram. """
        
        ll = np.sum(self.counts.astype(float) * np.log(self.num_bins * self.counts.astype(float) / self.N + 1e-12))
        penalization = self.num_bins - 1 + np.log(self.num_bins) ** 2.5
        return ll - penalization, ll, penalization


#
# Maximally divergent regions using Gaussian Process Regression
#
def maxdiv_gp(X, intervals, mode = 'I_OMEGA', theta = 30, train_step = 5, **kwargs):
    """ Scores given intervals by fitting a Gaussian Process to them.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """
    
    dimension, n = X.shape
    scores = []
    
    # Fit gaussian process parameters to time-series
    if theta is None:
        gp = GaussianProcess(thetaL = 0.1, thetaU = 1000, nugget = 1e-8, normalize = False)
    else:
        gp = GaussianProcess(theta0 = theta, nugget = 1e-8, normalize = False)
    gp.fit(np.linspace(0, 1, n, endpoint = True).reshape(n, 1), X.T)
    
    # Compute characteristic length scale
    ls = int(np.sqrt(0.5 / gp.theta_).flat[0] * n + 0.5)
    
    # Compute regression function
    f = gp.regr(gp.X)
    
    # Compute correlation matrix
    D, ij = l1_cross_distances(gp.X)
    r = gp.corr(gp.theta_, D)
    corr = np.eye(n) * (1. + gp.nugget)
    corr[ij[:, 0], ij[:, 1]] = r
    corr[ij[:, 1], ij[:, 0]] = r

    # Search for maximally divergent intervals
    eps = 1e-7
    for a, b, base_score in intervals:
        
        score = 0.0
        
        extreme_interval_length = b - a
        non_extreme_points = n - extreme_interval_length
        
        timesteps_extreme = np.arange(a, b)
        #timesteps_non_extreme = np.setdiff1d(np.arange(n), timesteps_extreme)
        timesteps_non_extreme = np.concatenate((
            np.arange(max(0, a - ls), b, train_step),
            np.arange(b, min(n, b + ls), train_step)
        ))

        mu, sigma = condition_gp(gp, f, corr, timesteps_non_extreme, timesteps_extreme)
        ll = multivariate_normal.logpdf(gp.y[timesteps_extreme, :].T - mu.T, None, sigma + np.eye(sigma.shape[0]) * eps).sum()
        
        if mode != 'TS':
            score -= ll / extreme_interval_length
        else:
            score -= ll
        
        scores.append((a, b, score))

    return scores


def condition_gp(gp, f, corr, train_x, test_x):
    
    ## Checks and pre-processing ##
    
    # Get shapes
    n = gp.X.shape[0]
    n_test = len(test_x)
    n_train = len(train_x)
    n_targets = gp.y.shape[1]
    
    ## Compute regression and correlation functions for training and test set ##
    
    # Split up regression function into pieces mu and m'
    mu_train = f[train_x, :]    # mu
    mu_test = f[test_x, :]      # m'
    
    # Extract correlations S, S' and S'' from overall correlation matrix
    i, j = np.meshgrid(train_x, train_x, indexing = 'ij')
    S_train = corr[i, j]        # S
    i, j = np.meshgrid(test_x, test_x, indexing = 'ij')
    S_test = corr[i, j]         # S''
    i, j = np.meshgrid(test_x, train_x, indexing = 'ij')
    S_test_train = corr[i, j]   # S'^T
    
    # Cholesky decomposition of S = C * C^T
    C = cholesky(S_train, lower = True)
    
    ## Compute conditioned mean ##
    # m' + S'^T * S^-1 * (y - mu)
    mu = mu_test + np.dot(S_test_train, cho_solve((C, True), gp.y[train_x, :] - mu_train))
    
    ## Compute conditioned covariance matrix ##
    # S'' - S'^T * S^-1 * S'
    
    # Compute C^-1 * S' (C is the Cholesky decomposition of S)
    rt = solve_triangular(C, S_test_train.T, lower = True)
    # Now: S'^T * S^-1 * S' = S'^T * C^-T * C^-1 * S' = (C^-1 * S')^T * (C^-1 * S') = rt^T * rt
    S_sub = np.dot(rt.T, rt)
    
    # Conditioned covariance
    sigma = S_test - S_sub
    
    return mu, sigma
