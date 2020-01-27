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
import math, types, warnings
from . import maxdiv_util, preproc
from .baselines_noninterval import pointwiseRegionProposals

def get_available_methods():
    return ['parzen', 'gaussian_cov', 'gaussian_id_cov', 'gaussian_global_cov', 'erph']


#
# Wrapper delegating calls either to the C++ or the Python implementation
#
def maxdiv(X, method = 'gaussian_cov', num_intervals = 1, proposals = 'dense', useLibMaxDiv = None, **kwargs):
    """ Universal wrapper function for running the MDI algorithm.
    
    Arguments:
    
    - `X`: The time-series as d-by-n numpy array, where `d` is the number of attributes and `n` is the number of
           time-steps. If the data contains missing values, the array must have been masked.
    
    - `method`: Method for probability density estimation. One of: 'gaussian_cov', 'gaussian_global_cov',
                'gaussian_id_cov', 'parzen', 'erph'
    
    - `num_intervals`: Number of detections to be returned. If set to `None`, all detections will be returned
                       (after applying non-maximum suppression).
    
    - `proposals`: Interval proposal method to be used. The following options are available:
                    - 'dense': full scan over all possible intervals
                    - 'hotellings_t': generate proposals based on Hotelling's T^2 scores
                    - 'kde': generate proposals based on point-wise KDE scores
    
    - `useLibMaxDiv`: If set to `None`, this function tries to delegate the call to `libmaxdiv`. If the library is not
                      available or the call failed for some reason, the Python implementation of the MDI algorithm
                      will be used as a fallback automatically, but a warning will be printed to stderr.
                      If this parameter is set to `True`, `libmaxdiv` *must* be used and an exception will be raised
                      if it is not available. If set to `False`, the Python implementation will be used.
    
    - `mode`: Divergence to be used. Possible options:
                - Variants of the KL divergence: 'I_OMEGA', 'OMEGA_I', 'SYM', 'TS' (unbiased KL divergence),
                - 'CROSSENT' (cross entropy)
                - 'JSD' (Jensen-Shannon divergence)
    
    - `extint_min_len`: Minimum length of the anomalous intervals
    
    - `extint_max_len`: Maximum length of the anomalous intervals
    
    - `overlap_th`: Overlap threshold for non-maximum suppression: Intervals with a greater IoU will be considered overlapping. Default: 0.5
    
    - `td_dim`: Time-Delay Embedding Dimension (may be set to 0 for automatic determination)
    
    - `td_lag`: Time-Lag for Time-Delay Embedding (may be set to 0 for automatic determination)
    
    - `preproc`: List of pre-processing methods to be applied. Possible methods are:
                 - 'td': Time-Delay Embedding with automatic determination of the embedding dimension
                 - 'normalize': Normalize the values in the time-series
                 - 'deseasonalize': Deseasonalization based on ordinary least squares
                 - 'deseasonalize_ft': Deseasonalization based on the Fourier Transform
                 - 'detrend_linear': Linear detrending
    
    - `kernel_sigma_sq`: Kernel variance for 'parzen' method
    
    - `num_hist`: The number of histograms used by the ERPH estimator.
    
    - `num_bins`: The number of bins in the histograms used by the ERPH estimator (0 = auto).
    
    - `discount`: Discount added to all bins of the histograms of the ERPH estimator in order to make unseen values not completely unlikely.
    
    - `pca_dim`: Reduce data to the given number of dimensions using PCA.
    
    - `random_projection_dim`: Project data onto the given number of random projection vectors.
    
    - `prop_th`: Threshold for pointwise interval proposals (default: 1.5)
    
    - `prop_mad`: Use MAD to determine the threshold for interval proposals.
    
    - `prop_unfiltered`: If set to true, pointwise scores will be used directly for proposals instead of their gradient.
    
    Returns: List of detections as (a, b, score) tuples, where `a` is the index of the first time-step inside
             of the detected interval and `b` is the first time-step just outside of the interval. The detections
             are sorted by their score in descending order.
    """
    
    if useLibMaxDiv != False:
        try:
            from . import libmaxdiv_wrapper
            return libmaxdiv_wrapper.maxdiv(X, method, num_intervals, proposals, **kwargs)
        except:
            if useLibMaxDiv == True:
                raise
            else:
                warnings.warn('libmaxdiv could not be loaded. Falling back to the Python implementation, but this will be much slower and results may be different.', RuntimeWarning, stacklevel = 2)
    
    from .maxdiv_py import maxdiv_parzen, maxdiv_gaussian, maxdiv_erph, maxdiv_gp

    if (not np.ma.isMaskedArray(X)) and np.isnan(X).any():
        X = np.ma.mask_cols(np.ma.masked_invalid(X))
    
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
                if (prep != 'td') or ('td_dim' not in kwargs) or (kwargs['td_dim'] == 1):
                    X = preprocMethods[prep](X)
            elif prep is not None:
                raise Exception("Unknown preprocessing method {}".format(prep))
        del kwargs['preproc']
    
    if ('td_dim' in kwargs) and (kwargs['td_dim'] != 1):
        td_dim = kwargs['td_dim'] if (kwargs['td_dim'] is not None) and (kwargs['td_dim'] > 0) else None
        if 'td_lag' in kwargs:
            td_lag = kwargs['td_lag'] if (kwargs['td_lag'] is not None) and (kwargs['td_lag'] > 0) else None
        else:
            td_lag = 1
        X = preproc.td(X, td_dim, td_lag)
    
    if ('pca_dim' in kwargs) and (kwargs['pca_dim'] > 0):
        X = preproc.pca_projection(X, kwargs['pca_dim'])
    if ('random_projection_dim' in kwargs) and (kwargs['random_projection_dim'] > 0):
        X = preproc.sparse_random_projection(X, kwargs['random_projection_dim'])
    
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
    elif proposals == 'dense':
        intervals = denseRegionProposals(X, **proposalParameters)
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
        from .maxdiv_py import maxdiv_parzen
        # compute kernel matrix first (Gaussian kernel)
        K = maxdiv_util.calc_gaussian_kernel(X, normalized = False, **kernelparameters)
        # obtain the interval [a,b] of the extreme event with score score
        interval_scores = maxdiv_parzen(K, intervals, **kwargs)
    
    elif method == 'gaussian_process':
        from .maxdiv_py import maxdiv_gp
        interval_scores = maxdiv_gp(X, intervals, **kwargs)

    elif method.startswith('gaussian'):
        from .maxdiv_py import maxdiv_gaussian
        if 'alpha' in kwargs:
            del kwargs['alpha']
        kwargs['gaussian_mode'] = method[9:].upper()
        interval_scores = maxdiv_gaussian(X, intervals, **kwargs)
    
    elif method == 'erph':
        from .maxdiv_py import maxdiv_erph
        interval_scores = maxdiv_erph(X, intervals, **kwargs)
        
    else:
        raise Exception("Unknown method {}".format(method))

    if any(math.isnan(score) for a, b, score in interval_scores):
        raise Exception("NaNs found in interval_scores!")
    
    # get the K best non-overlapping regions
    regions = find_max_regions(interval_scores, num_intervals, kwargs['overlap_th'] if 'overlap_th' in kwargs else 0.0)

    return regions


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


def denseRegionProposals(func, extint_min_len = 20, extint_max_len = 150, **kwargs):
    """ A generator that yields all possible regions with size between `extint_min_len` and `extint_max_len`. """
    
    n = func.shape[1]
    for i in range(n - extint_min_len + 1):
        if not np.ma.is_masked(func[0,i]):
            for j in range(i + extint_min_len, min(i + extint_max_len, n) + 1):
                if not np.ma.is_masked(func[0, j - 1]):
                    yield (i, j, 0.0)
