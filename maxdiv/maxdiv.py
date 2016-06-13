# coding: utf-8
#
# Detection of extreme intervals in multivariate time-series
# Author: Erik Rodner (2015-) & Björn Barz (2016)

# Novelty detection by minimizing the KL divergence
# In the following, we will derive a similar algorithm based on Kullback-Leibler (KL) divergence
# between the distribution $p_I$ of data points in the extreme interval $I = [a,b)$
# and the distribution $p_{\Omega}$ of non-extreme data points. We approximate both distributions with a simple kernel density estimate:
#
# $p_I(\mathbf{x}) = \frac{1}{|I|} \sum\limits_{i \in I} K(\mathbf{x}, \mathbf{x}_i)$
#
# with $K$ being a normalized kernel, such that $p_I$ is a proper densitity.
# Similarly, we define $p_{\Omega}$ with $\Omega = \{1, \ldots, n\} \setminus I$.

import numpy as np
from numpy.linalg import slogdet, inv, solve
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal
import math, time, types
from . import maxdiv_util, preproc
from .baselines_noninterval import pointwiseRegionProposals

def get_available_methods():
    return ['parzen', 'gaussian_cov', 'gaussian_id_cov', 'gaussian_global_cov']


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


#
# Maximally divergent regions using Kernel Density Estimation
#
def maxdiv_parzen(K, intervals, mode = 'I_OMEGA', alpha = 1.0, score_merge_coeff = None, **kwargs):
    """ Scores given intervals by using Kernel Density Estimation.
    
    `K` is a symmetric kernel matrix whose components are K(|i - j|) for a given kernel K.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    The scores should be in the range [0,1] and will be integrated into the final interval
    score if `score_merge_coeff` is not `None`. The proposed score and the divergence-based
    score will be combined according to the following equation:
    
    `score = score_merge_coeff * divergence_score + (1.0 - score_merge_coeff) * proposed_score`
    
    The divergence-based scores will be scaled to be in range [0,1].
    This scaling won't be performed if score merging is disabled by setting `score_merge_coeff`
    to `None`.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    # compute integral sums for each column within the kernel matrix 
    K_integral = np.cumsum(K, axis=0)
    # the sum of all kernel values for each column
    # is now given in the last row
    sums_all = K_integral[-1,:]
    # n is the number of data points considered
    n = K_integral.shape[0]

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

        # number of data points in the current interval
        extreme_interval_length = b - a
        # number of data points outside of the current interval
        non_extreme_points = n - extreme_interval_length
        
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
        scores.append((a, b, score, base_score) if score_merge_coeff is not None else (a, b, score))
    
    # Merge divergence and proposal scores
    if score_merge_coeff is None:
        return scores
    else:
        # Apply sigmoid function to scale scores to [0,1)
        if mode != 'JSD':
            for i, (a, b, score, base_score) in enumerate(scores):
                scores[i] = (a, b, 2.0 / (1.0 + math.exp(-0.1 * scores[i][2])) - 1.0, base_score)
        # Combine divergence-based scores with proposed scores linearly
        return [(a, b, score_merge_coeff * score + (1.0 - score_merge_coeff) * base_score) for a, b, score, base_score in scores]

#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian_globalcov(X, intervals, mode = 'I_OMEGA', gaussian_mode = 'GLOBAL_COV', score_merge_coeff = None, **kwargs):
    """ Scores given intervals by assuming gaussian distributions with equal covariance.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    The scores should be in the range [0,1] and will be integrated into the final interval
    score if `score_merge_coeff` is not `None`. The proposed score and the divergence-based
    score will be combined according to the following equation:
    
    `score = score_merge_coeff * divergence_score + (1.0 - score_merge_coeff) * proposed_score`
    
    The divergence-based scores will be scaled to be in range [0,1].
    This scaling won't be performed if score merging is disabled by setting `score_merge_coeff`
    to `None`.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    dimension, n = X.shape

    X_integral = np.cumsum(X, axis=1)
    sums_all = X_integral[:, -1]
    if (gaussian_mode == 'GLOBAL_COV') and (dimension > 1):
        cov = np.cov(X)
        cov_chol = cho_factor(cov)

    scores = []

    eps = 1e-7
    for a, b, base_score in intervals:
        
        extreme_interval_length = b - a
        non_extreme_points = n - extreme_interval_length
        
        sums_extreme = X_integral[:, b-1] - (X_integral[:, a-1] if a > 0 else 0)
        sums_non_extreme = sums_all - sums_extreme
        sums_extreme /= extreme_interval_length
        sums_non_extreme /= non_extreme_points

        diff = sums_extreme - sums_non_extreme
        if (gaussian_mode == 'GLOBAL_COV') and (dimension > 1):
            score = diff.T.dot(cho_solve(cov_chol, diff))
        else:
            score = np.sum(diff * diff)
        scores.append((a, b, score, base_score) if score_merge_coeff is not None else (a, b, score))

    if score_merge_coeff is None:
        return scores
    else:
        # Apply sigmoid function to scale scores to [0,1)
        for i, (a, b, score, base_score) in enumerate(scores):
            scores[i] = (a, b, 2.0 / (1.0 + math.exp(-0.02 * scores[i][2])) - 1.0, base_score)
        # Combine divergence-based scores with proposed scores linearly
        return [(a, b, score_merge_coeff * score + (1.0 - score_merge_coeff) * base_score) for a, b, score, base_score in scores]


#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian(X, intervals, mode = 'I_OMEGA', gaussian_mode = 'COV', score_merge_coeff = None, **kwargs):
    """ Scores given intervals by assuming gaussian distributions.
    
    `X` is a d-by-n matrix with `n` data points, each with `d` attributes.
    
    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.
    The scores should be in the range [0,1] and will be integrated into the final interval
    score if `score_merge_coeff` is not `None`. The proposed score and the divergence-based
    score will be combined according to the following equation:
    
    `score = score_merge_coeff * divergence_score + (1.0 - score_merge_coeff) * proposed_score`
    
    The divergence-based scores will be scaled to be in range [0,1].
    This scaling won't be performed if score merging is disabled by setting `score_merge_coeff`
    to `None`.
    
    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    if gaussian_mode!='COV':
        return maxdiv_gaussian_globalcov(X, intervals, mode, gaussian_mode)

    dimension, n = X.shape
    X_integral = np.cumsum(X, axis=1)
    sums_all = X_integral[:, -1]
    scores = []

    # compute integral series of the outer products
    # we will use this to compute covariance matrices
    outer_X = np.apply_along_axis(lambda x: np.ravel(np.outer(x,x)), 0, X)
    outer_X_integral = np.cumsum(outer_X, axis=1)
    outer_sums_all = outer_X_integral[:, -1]

    eps = 1e-7
    for a, b, base_score in intervals:
        
        score = 0.0
        
        extreme_interval_length = b - a
        non_extreme_points = n - extreme_interval_length
        
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
            _, logdet_extreme = slogdet(cov_extreme)
            _, logdet_non_extreme = slogdet(cov_non_extreme)
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
            kl_Omega_I += logdet_extreme - logdet_non_extreme
            score += kl_Omega_I

        # version for maximizing KL(p_I, p_Omega)
        if mode == "I_OMEGA" or mode == "SYM":
            inv_cov_non_extreme = inv(cov_non_extreme)
            # term for the mahalanobis distance
            kl_I_Omega = np.dot(diff, np.dot(inv_cov_non_extreme, diff.T))
            # trace term
            kl_I_Omega += np.trace(np.dot(inv_cov_non_extreme, cov_extreme))
            # logdet terms
            kl_I_Omega += logdet_non_extreme - logdet_extreme
            score += kl_I_Omega
        
        # Jensen-Shannon Divergence
        if mode == 'JSD':
            # Compute probability densities
            pdf_extreme     = multivariate_normal.pdf(X.T, sums_extreme, cov_extreme)
            pdf_non_extreme = multivariate_normal.pdf(X.T, sums_non_extreme, cov_non_extreme)
            pdf_combined    = (pdf_extreme + pdf_non_extreme) / 2
            # Compute JSD
            jsd_extreme     = np.mean(np.log2(pdf_extreme[a:b] + eps) - np.log2(pdf_combined[a:b] + eps))
            jsd_non_extreme = np.mean(np.log2(np.concatenate((pdf_non_extreme[:a], pdf_non_extreme[b:])) + eps)
                                      - np.log2(np.concatenate((pdf_combined[:a], pdf_combined[b:])) + eps))
            score += (jsd_extreme + jsd_non_extreme) / 2.0
        
        #print score, cov_extreme, cov_non_extreme, diff

        scores.append((a, b, score, base_score) if score_merge_coeff is not None else (a, b, score))

    # Merge divergence and proposal scores
    if score_merge_coeff is None:
        return scores
    else:
        # Apply sigmoid function to scale scores to [0,1)
        if mode != 'JSD':
            for i, (a, b, score, base_score) in enumerate(scores):
                scores[i] = (a, b, 2.0 / (1.0 + math.exp(-0.02 * scores[i][2])) - 1.0, base_score)
        # Combine divergence-based scores with proposed scores linearly
        return [(a, b, score_merge_coeff * score + (1.0 - score_merge_coeff) * base_score) for a, b, score, base_score in scores]

#
# Search non-overlapping regions
#
def find_max_regions(intervals, num_intervals = None, overlap_th = 0.0):
    """ Given a list of scored intervals, we select the `num_intervals` intervals
        which are non-overlapping and have the highest score.
        
        `intervals` must be a list of (a, b, score) tuples which define intervals [a,b) with
        a corresponding anomaly score.
        
        `overlap_th` specifies a threshold for non-maxima suppression: Intervals with an Intersection
        over Union (IoU) greater than this threshold will be considered overlapping.
        
        `num_intervals` may be set to None to retrieve all non-overlapping regions.
        
        Returns: List of 3-tuples (a, b, score), specifying the score for an interval [a,b).
                 This list will be ordered decreasingly by the score.
    """
    
    # Shortcut if only the maximum is of interest
    if num_intervals == 1:
        return [max(intervals, key = lambda x: x[2])]
    
    # Sort intervals by scores in descending order
    intervals.sort(key = lambda x: x[2], reverse = True)
    
    # Non-maximum suppression
    n = len(intervals) # total number of intervals
    include = np.ones(n, dtype = bool) # suppressed intervals will be set to False
    found_intervals = 0
    for i in range(n):
        if include[i]:
            
            # Terminate non-maxima suppression if we already have found enough intervals
            found_intervals += 1
            if (num_intervals is not None) and (found_intervals >= num_intervals):
                include[i+1:] = False
                break
            
            # Exclude intervals with a lower score overlapping this one
            a, b = intervals[i][:2]
            for j in range(i + 1, n):
                if include[j] and ((intervals[j][2] == 0) or (maxdiv_util.IoU(a, b - a, intervals[j][0], intervals[j][1] - intervals[j][0]) > overlap_th)):
                    include[j] = False
    
    # Return list of remaining intervals
    return [intvl for intvl, incl in zip(intervals, include) if incl]


#
# Wrapper and utility functions
#
def maxdiv(X, method = 'gaussian_cov', num_intervals = 1, proposals = 'dense', useLibMaxDiv = None, **kwargs):
    """ Wrapper function for calling maximum divergent regions """
    
    if useLibMaxDiv != False:
        try:
            from . import libmaxdiv_wrapper
            return libmaxdiv_wrapper.maxdiv(X, method, num_intervals, proposals, **kwargs)
        except:
            if useLibMaxDiv == True:
                raise
    
    if 'preproc' in kwargs:
        preprocs = kwargs['preproc'] if isinstance(kwargs['preproc'], list) or isinstance(kwargs['preproc'], tuple) else [kwargs['preproc']]
        preprocMethods = {
            'normalize'         : preproc.normalize_time_series,
            'local_linear'      : preproc.local_linear_regression,
            'td'                : preproc.td,
            'deseasonalize'     : preproc.detrend_ols,
            'deseasonalize_ft'  : preproc.deseasonalize_ft,
            'detrend_linear'    : preproc.detrend_linear
        }
        for prep in preprocs:
            if (prep is not None) and (prep in preprocMethods):
                X = preprocMethods[prep](X)
            elif prep is not None:
                raise Exception("Unknown preprocessing method {}".format(prep))
        del kwargs['preproc']
    
    if 'proposalparameters' in kwargs:
        proposalParameters = kwargs['proposalparameters']
        del kwargs['proposalparameters']
    else:
        proposalParameters = {}
    if ('extint_min_len' in kwargs) and ('extint_min_len' not in proposalParameters):
        proposalParameters['extint_min_len'] = kwargs['extint_min_len']
    if ('extint_max_len' in kwargs) and ('extint_max_len' not in proposalParameters):
        proposalParameters['extint_max_len'] = kwargs['extint_max_len']
    
    if proposals in ['hotellings_t', 'kde']:
        proposalParameters['method'] = proposals
        proposals = 'pointwise'
    
    if isinstance(proposals, types.GeneratorType) or isinstance(proposals, list):
        intervals = proposals
    if proposals == 'dense':
        intervals = denseRegionProposals(X, **proposalParameters)
        kwargs['score_merge_coeff'] = None
    elif proposals == 'pointwise':
        intervals = pointwiseRegionProposals(X, **proposalParameters)
    elif isinstance(proposals, types.FunctionType):
        intervals = proposals(X, **proposalParameters)
    else:
        raise Exception('Unknown proposal generator: {}'.format(proposals))

    if 'kernelparameters' in kwargs:
        kernelparameters = kwargs['kernelparameters']
        del kwargs['kernelparameters']
    else:
        kernelparameters = {'kernel_sigma_sq': 1.0}

    if method == 'parzen':
        # compute kernel matrix first (Gaussian kernel)
        K = maxdiv_util.calc_gaussian_kernel(X, **kernelparameters)
        # obtain the interval [a,b] of the extreme event with score score
        interval_scores = maxdiv_parzen(K, intervals, **kwargs)

    elif method.startswith('gaussian'):
        if 'alpha' in kwargs:
            del kwargs['alpha']
        kwargs['gaussian_mode'] = method[9:].upper()
        interval_scores = maxdiv_gaussian(X, intervals, **kwargs)
        
    else:
        raise Exception("Unknown method {}".format(method))

    if any(math.isnan(score) for a, b, score in interval_scores):
        raise Exception("NaNs found in interval_scores!")

    if 'extint_min_len' in kwargs:
        interval_min_length = kwargs['extint_min_len']
    else:
        interval_min_length = 20
    
    # get the K best non-overlapping regions
    regions = find_max_regions(interval_scores, num_intervals)

    return regions


def denseRegionProposals(func, extint_min_len = 20, extint_max_len = 150, **kwargs):
    """ A generator that yields all possible regions with size between `extint_min_len` and `extint_max_len`. """
    
    n = func.shape[1]
    for i in range(n - extint_min_len + 1):
        for j in range(i + extint_min_len, min(i + extint_max_len, n) + 1):
            yield (i, j, 0.0)
