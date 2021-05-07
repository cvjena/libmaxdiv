"""Wrapper for libmaxdiv.

This module provides a low-level interface to libmaxdiv as an instance of a wrapper class
called _LibMaxDiv. That instance is stored in this module's dictionary's field 'libmaxdiv', which may
be None if the library could not be found or loaded.
The _LibMaxDiv class provides exactly the same functions as libmaxdiv.

In addition, this module provides a 'maxdiv' function which mimics the interface of maxdiv.maxdiv,
but delegates the call to libmaxdiv instead of performing the computations in Python code.

This wrapper assumes that libmaxdiv has been compiled using double floating point precision.
If single precision is being used instead, the value of `maxdiv_scalar` has to be adjusted accordingly.
"""

from ctypes import *
from ctypes import util
import numpy as np
import os.path


# Scalar floating point type used by libmaxdiv
maxdiv_scalar = c_double


# enumeration constants according to  libmaxdiv.h
enums = {
    'MAXDIV_PROPOSAL_SEARCH'    : 0,
    
    'MAXDIV_KL_DIVERGENCE'      : 0,
    'MAXDIV_JS_DIVERGENCE'      : 1,
    'MAXDIV_CROSS_ENTROPY'      : 2,
    
    'MAXDIV_KDE'        : 0,
    'MAXDIV_GAUSSIAN'   : 1,
    'MAXDIV_ERPH'       : 2,
    
    'MAXDIV_DENSE_PROPOSALS'                    : 0,
    'MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST'    : 1,
    'MAXDIV_POINTWISE_PROPOSALS_KDE'            : 2,
    
    'MAXDIV_KL_I_OMEGA' : 0,
    'MAXDIV_KL_OMEGA_I' : 1,
    'MAXDIV_KL_SYM'     : 2,
    'MAXDIV_KL_UNBIASED': 3,
    
    'MAXDIV_GAUSSIAN_COV_FULL'      : 0,
    'MAXDIV_GAUSSIAN_COV_SHARED'    : 1,
    'MAXDIV_GAUSSIAN_COV_ID'        : 2,
    
    'MAXDIV_BORDER_POLICY_AUTO'     : 0,
    'MAXDIV_BORDER_POLICY_CONSTANT' : 1,
    'MAXDIV_BORDER_POLICY_MIRROR'   : 2,
    'MAXDIV_BORDER_POLICY_VALID'    : 3,
    
    'MAXDIV_NORMALIZE_NONE' : 0,
    'MAXDIV_NORMALIZE_MAX'  : 1,
    'MAXDIV_NORMALIZE_SD'   : 2,
    
    'MAXDIV_DETREND_NONE'   : 0,
    'MAXDIV_DETREND_LINEAR' : 1,
    'MAXDIV_DETREND_OLS'    : 2,
    'MAXDIV_DETREND_ZSCORE' : 3,
    
    'MAXDIV_PROJECT_NONE'   : 0,
    'MAXDIV_PROJECT_PCA'    : 1,
    'MAXDIV_PROJECT_RANDOM' : 2
}


# index vector with 5 elements (1 temporal, 3 spatial and 1 attribute dimension)
index_vector_t = c_uint * 5

# index vector with 4 elements (1 temporal and 3 spatial dimensions)
point_t = c_uint * 4

# detection_t structure definition according to libmaxdiv.h
class detection_t(Structure):
    _fields_ = [('range_start', point_t),
                ('range_end', point_t),
                ('score', maxdiv_scalar)]

# anonymous structures nested in maxdiv_params_t
class pointwise_proposal_params_t(Structure):
    _fields_ = [('gradient_filter', c_bool),
                ('mad', c_bool),
                ('sd_th', maxdiv_scalar),
                ('kernel_sigma_sq', maxdiv_scalar)]

class erph_params_t(Structure):
    _fields_ = [('num_hist', c_uint),
                ('num_bins', c_uint),
                ('discount', maxdiv_scalar)]

class embedding_params_t(Structure):
    _fields_ = [('kt', c_uint),
                ('kx', c_uint),
                ('ky', c_uint),
                ('kz', c_uint),
                ('dt', c_uint),
                ('dx', c_uint),
                ('dy', c_uint),
                ('dz', c_uint),
                ('temporal_borders', c_int),
                ('spatial_borders', c_int)]

class detrending_params_t(Structure):
    _fields_ = [('method', c_int),
                ('linear_degree', c_uint),
                ('ols_period_num', c_uint),
                ('ols_period_len', c_uint),
                ('ols_linear_trend', c_bool),
                ('ols_linear_season_trend', c_bool),
                ('z_period_len', c_uint)]

class projection_params_t(Structure):
    _fields_ = [('method', c_int),
                ('ndims', c_uint)]

class preproc_params_t(Structure):
    _fields_ = [('normalization', c_int),
                ('embedding', embedding_params_t),
                ('detrending', detrending_params_t),
                ('dimensionality_reduction', projection_params_t)]

# maxdiv_params_t structure definition according to libmaxdiv.h
class maxdiv_params_t(Structure):
    _fields_ = [('strategy', c_int),
                ('divergence', c_int),
                ('estimator', c_int),
                ('min_size', point_t),
                ('max_size', point_t),
                ('stride', point_t),
                ('overlap_th', maxdiv_scalar),
                ('proposal_generator', c_int),
                ('pointwise_proposals', pointwise_proposal_params_t),
                ('kl_mode', c_int),
                ('kernel_sigma_sq', maxdiv_scalar),
                ('gaussian_cov_mode', c_int),
                ('erph', erph_params_t),
                ('preproc', preproc_params_t)]



# Pointer types
c_uint_p = POINTER(c_uint)
maxdiv_scalar_p = POINTER(maxdiv_scalar)
detection_p = POINTER(detection_t)
maxdiv_params_p = POINTER(maxdiv_params_t)



class _LibMaxDiv(object):

    def __init__(self, library):
        """Sets up the function prototypes of the library."""
        
        object.__init__(self)
        self._lib = library
        
        # maxdiv_init_params function
        self._register_func('maxdiv_init_params',
            (c_void_p, maxdiv_params_p),
            ((1, 'params'),)
        )
        
        # maxdiv_compile_pipeline function
        self._register_func('maxdiv_compile_pipeline',
            (c_uint, maxdiv_params_p),
            ((1, 'params'),),
            self._errcheck_compile_pipeline
        )
        
        # maxdiv_free_pipeline function
        self._register_func('maxdiv_free_pipeline',
            (c_void_p, c_uint),
            ((1, 'handle'),)
        )
        
        # maxdiv_exec function
        self._register_func('maxdiv_exec',
            (c_void_p, c_uint, maxdiv_scalar_p, index_vector_t, detection_p, c_uint_p, c_bool, c_bool, maxdiv_scalar),
            ((1, 'pipeline'), (1, 'data'), (1, 'shape'), (1, 'detection_buf'), (1, 'detection_buf_size'),
             (1, 'const_data', True), (1, 'custom_missing_value', False), (1, 'missing_value', 0))
        )

        # maxdiv_score_intervals function
        self._register_func('maxdiv_score_intervals',
            (c_void_p, c_uint, maxdiv_scalar_p, index_vector_t, detection_p, c_uint, c_bool, c_bool, maxdiv_scalar),
            ((1, 'pipeline'), (1, 'data'), (1, 'shape'), (1, 'intervals'), (1, 'num_intervals'),
             (1, 'const_data', True), (1, 'custom_missing_value', False), (1, 'missing_value', 0))
        )
        
        # maxdiv function
        self._register_func('maxdiv',
            (c_void_p, maxdiv_params_p, maxdiv_scalar_p, index_vector_t, detection_p, c_uint_p, c_bool, c_bool, maxdiv_scalar),
            ((1, 'params'), (1, 'data'), (1, 'shape'), (1, 'detection_buf'), (1, 'detection_buf_size'),
             (1, 'const_data', True), (1, 'custom_missing_value', False), (1, 'missing_value', 0))
        )
    
    
    def _register_func(self, funcName, paramtypes, paramflags, errcheck = None):
        
        prototype = CFUNCTYPE(*paramtypes)
        self.__dict__[funcName] = prototype((funcName, self._lib), paramflags)
        if errcheck is not None:
            self.__dict__[funcName].errcheck = errcheck


    @staticmethod
    def _errcheck_compile_pipeline(result, func, args):
        if (result == 0):
            raise RuntimeError('[libmaxdiv] Invalid parameters')
        return args



def _search_libmaxdiv():
    basedir = os.path.dirname(__file__)
    search_names = ('maxdiv', 'libmaxdiv', 'libmaxdiv.so', os.path.join('.', 'libmaxdiv.so'), \
                    os.path.join('bin', 'maxdiv'), os.path.join('bin', 'libmaxdiv'), os.path.join('bin', 'libmaxdiv.so'), \
                    os.path.join('build', 'maxdiv'), os.path.join('build', 'libmaxdiv'), os.path.join('build', 'libmaxdiv.so'), \
                    os.path.join('libmaxdiv', 'maxdiv'), os.path.join('libmaxdiv', 'libmaxdiv'), os.path.join('libmaxdiv', 'libmaxdiv.so'), \
                    os.path.join('libmaxdiv', 'bin', 'maxdiv'), os.path.join('libmaxdiv', 'bin', 'libmaxdiv'), os.path.join('libmaxdiv', 'bin', 'libmaxdiv.so'), \
                    os.path.join('libmaxdiv', 'build', 'maxdiv'), os.path.join('libmaxdiv', 'build', 'libmaxdiv'), os.path.join('libmaxdiv', 'build', 'libmaxdiv.so'), \
                    os.path.join(basedir, 'maxdiv'), os.path.join(basedir, 'libmaxdiv'), os.path.join(basedir, 'libmaxdiv.so'), \
                    os.path.join(basedir, 'bin', 'maxdiv'), os.path.join(basedir, 'bin', 'libmaxdiv'), os.path.join(basedir, 'bin', 'libmaxdiv.so'), \
                    os.path.join(basedir, 'build', 'maxdiv'), os.path.join(basedir, 'build', 'libmaxdiv'), os.path.join(basedir, 'build', 'libmaxdiv.so'), \
                    os.path.join(basedir, 'libmaxdiv', 'maxdiv'), os.path.join(basedir, 'libmaxdiv', 'libmaxdiv'), os.path.join(basedir, 'libmaxdiv.so'), \
                    os.path.join(basedir, 'libmaxdiv', 'bin', 'maxdiv'), os.path.join(basedir, 'libmaxdiv', 'bin', 'libmaxdiv'), os.path.join(basedir, 'libmaxdiv', 'bin', 'libmaxdiv.so'), \
                    os.path.join(basedir, 'libmaxdiv', 'build', 'maxdiv'), os.path.join(basedir, 'libmaxdiv', 'build', 'libmaxdiv'), os.path.join(basedir, 'libmaxdiv', 'build', 'libmaxdiv.so'), \
                    util.find_library('maxdiv'))
    for n in search_names:
        if not n is None:
            try:
                lib = CDLL(n)
                if os.path.exists(n + '.dll'):
                    fn = n + '.dll'
                elif os.path.exists(n + '.so'):
                    fn = n + '.so'
                else:
                    fn = n
                return _LibMaxDiv(lib), fn
            except (OSError, TypeError):
                pass
    return None, None



def maxdiv(X, method = 'gaussian_cov', num_intervals = 1, proposals = 'dense', **kwargs):
    """ Semantically equivalent to `maxdiv.maxdiv`, but delegates the computation to libmaxdiv. """
    
    if libmaxdiv is None:
        raise RuntimeError('libmaxdiv could not be found or loaded.')
    
    if X.ndim == 1:
        X = X.reshape((1, len(X)))
    isSpatioTemporal = (X.ndim == 5)
    
    # Set parameters
    params = maxdiv_params_t()
    libmaxdiv.maxdiv_init_params(params)
    
    # Length
    params.min_size[:] = [kwargs['extint_min_len'] if 'extint_min_len' in kwargs else 20] * len(params.min_size)
    if 'extint_max_len' in kwargs:
        params.max_size[:] = [kwargs['extint_max_len']] * len(params.max_size)
    if 'stride' in kwargs:
        params.stride[:] = [kwargs['stride']] * len(params.stride)
    
    # Overlap Threshold
    if 'overlap_th' in kwargs:
        params.overlap_th = kwargs['overlap_th']
    
    # Method
    method = method.lower()
    if method in ('gaussian_cov', 'gaussian_cov_ts', 'gaussian_ts'):
        params.estimator = enums['MAXDIV_GAUSSIAN']
        params.gaussian_cov_mode = enums['MAXDIV_GAUSSIAN_COV_FULL']
        if method in ('gaussian_cov_ts', 'gaussian_ts'):
            kwargs['mode'] = 'TS'
    elif method == 'gaussian_global_cov':
        params.estimator = enums['MAXDIV_GAUSSIAN']
        params.gaussian_cov_mode = enums['MAXDIV_GAUSSIAN_COV_SHARED']
    elif method == 'gaussian_id_cov':
        params.estimator = enums['MAXDIV_GAUSSIAN']
        params.gaussian_cov_mode = enums['MAXDIV_GAUSSIAN_COV_ID']
    elif method == 'parzen':
        params.estimator = enums['MAXDIV_KDE']
        if ('kernelparameters' in kwargs) and ('kernel_sigma_sq' in kwargs['kernelparameters']):
            params.kernel_sigma_sq = kwargs['kernelparameters']['kernel_sigma_sq']
        elif 'kernel_sigma_sq' in kwargs:
            params.kernel_sigma_sq = kwargs['kernel_sigma_sq']
    elif method == 'erph':
        params.estimator = enums['MAXDIV_ERPH']
        if 'num_hist' in kwargs:
            params.erph.num_hist = kwargs['num_hist']
        if 'num_bins' in kwargs:
            params.erph.num_bins = kwargs['num_bins'] if (kwargs['num_bins'] is not None) and (kwargs['num_bins'] > 0) else 0
        if 'discount' in kwargs:
            params.erph.discount = kwargs['discount']
    else:
        raise ValueError('Unknown method: {}'.format(method))
    
    # Divergence
    if 'mode' in kwargs:
        mode = kwargs['mode'].upper()
        if mode == 'I_OMEGA':
            params.divergence = enums['MAXDIV_KL_DIVERGENCE']
            params.kl_mode = enums['MAXDIV_KL_I_OMEGA']
        elif mode == 'OMEGA_I':
            params.divergence = enums['MAXDIV_KL_DIVERGENCE']
            params.kl_mode = enums['MAXDIV_KL_OMEGA_I']
        elif mode == 'SYM':
            params.divergence = enums['MAXDIV_KL_DIVERGENCE']
            params.kl_mode = enums['MAXDIV_KL_SYM']
        elif mode == 'TS':
            params.divergence = enums['MAXDIV_KL_DIVERGENCE']
            params.kl_mode = enums['MAXDIV_KL_UNBIASED']
        elif mode == 'JSD':
            params.divergence = enums['MAXDIV_JS_DIVERGENCE']
        elif mode == 'CROSSENT':
            params.divergence = enums['MAXDIV_CROSS_ENTROPY']
            params.kl_mode = enums['MAXDIV_KL_I_OMEGA']
        elif mode == 'CROSSENT_TS':
            params.divergence = enums['MAXDIV_CROSS_ENTROPY']
            params.kl_mode = enums['MAXDIV_KL_UNBIASED']
        else:
            raise ValueError('Unknown divergence mode: {}'.format(mode))
    
    # Proposal generator
    if (proposals != 'dense') and ('proposalparameters' in kwargs) and ('method' in kwargs['proposalparameters']):
        proposals = kwargs['proposalparameters']['method']
    proposals = proposals.lower()
    if proposals == 'dense':
        params.proposal_generator = enums['MAXDIV_DENSE_PROPOSALS']
    elif proposals == 'hotellings_t':
        params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST']
    elif proposals == 'kde':
        params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_KDE']
    else:
        raise ValueError('Unknown proposal generator: {}'.format(proposals))
    if 'proposalparameters' in kwargs:
        pp = kwargs['proposalparameters']
        if ('filter' in pp) and (pp['filter'] is None):
            params.pointwise_proposals.gradient_filter = False
        if 'useMAD' in pp:
            params.pointwise_proposals.mad = pp['useMAD']
        if 'sd_th' in pp:
            params.pointwise_proposals.sd_th = pp['sd_th']
    
    # Pre-processing
    params.preproc.embedding.kt = 1
    params.preproc.embedding.temporal_borders = enums['MAXDIV_BORDER_POLICY_CONSTANT']
    if 'preproc' in kwargs:
        preprocs = kwargs['preproc'] if isinstance(kwargs['preproc'], list) or isinstance(kwargs['preproc'], tuple) else [kwargs['preproc']]
        from . import preproc
        for prep in preprocs:
            if prep is not None:
                if prep == 'normalize':
                    params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']
                elif prep == 'td':
                    params.preproc.embedding.kt = 0
                elif prep == 'local_linear':
                    if isSpatioTemporal:
                        raise RuntimeError('Local linear regression is not available for spatio-temporal data.')
                    X = preproc.local_linear_regression(X)
                elif prep == 'deseasonalize':
                    if isSpatioTemporal:
                        raise RuntimeError('Automatic seasonality detection is not available for spatio-temporal data.')
                    periods, _ = preproc.detect_periods(X)
                    if len(periods) > 0:
                        params.preproc.detrending.method = enums['MAXDIV_DETREND_OLS']
                        params.preproc.detrending.ols_period_num = periods[0]
                elif prep == 'deseasonalize_ft':
                    if isSpatioTemporal:
                        raise RuntimeError('FT deseasonalization is not available for spatio-temporal data.')
                    X = preproc.deseasonalize_ft(X)
                elif prep == 'detrend_linear':
                    if params.preproc.detrending.method == enums['MAXDIV_DETREND_NONE']:
                        params.preproc.detrending.method = enums['MAXDIV_DETREND_LINEAR']
                else:
                    raise ValueError("Unknown preprocessing method {}".format(prep))
    if ('td_dim' in kwargs) and (kwargs['td_dim'] != 1):
        params.preproc.embedding.kt = kwargs['td_dim'] if (kwargs['td_dim'] is not None) and (kwargs['td_dim'] > 0) else 0
    if 'td_lag' in kwargs:
        params.preproc.embedding.dt = kwargs['td_lag'] if (kwargs['td_lag'] is not None) and (kwargs['td_lag'] > 0) else 0
    if ('pca_dim' in kwargs) and (kwargs['pca_dim'] > 0):
        params.preproc.dimensionality_reduction.method = enums['MAXDIV_PROJECT_PCA']
        params.preproc.dimensionality_reduction.ndims = kwargs['pca_dim']
    elif ('random_projection_dim' in kwargs) and (kwargs['random_projection_dim'] > 0):
        params.preproc.dimensionality_reduction.method = enums['MAXDIV_PROJECT_RANDOM']
        params.preproc.dimensionality_reduction.ndims = kwargs['random_projection_dim']
    
    return maxdiv_exec(X, params, num_intervals)


def maxdiv_exec(X, params, num_intervals = 1):
    """ Runs the MaxDiv algorithm using libmaxdiv with a given set of parameters.
    
    X - np.ndarray with d rows and n columns, where d is the number of attributes and n is
        the number of time steps. Alternatively, spatio-temporal data may be given as
        5-d array with 1 temporal, 3 spatial and 1 attribute dimension.
    params - Either a maxdiv_params_t object or a handle to a compiled pipeline obtained from `libmaxdiv.maxdiv_compile_pipeline()`
    num_intervals - Number of detections to be returned. Can be set to None to return as many
                    detections as possible.
    
    Returns: a list of `(a, b, score)` tuples, where `a` is the first point within a detected interval,
             `b` is the first point right after the interval and `score` is the detection score.
             For non-spatial data, `a` and `b` will be scalars, while they will be lists of indices for
             spatio-temporal data.
    """
    
    if libmaxdiv is None:
        raise RuntimeError('libmaxdiv could not be found or loaded.')
    
    isSpatioTemporal = False
    if X.ndim == 1:
        X = X.reshape((1, len(X)))
    elif X.ndim == 5:
        isSpatioTemporal = True
    elif X.ndim != 2:
        raise ValueError('Unsupported number of data dimensions: {}'.format(X.ndim))
    
    if not (isinstance(params, maxdiv_params_t) or isinstance(params, int)):
        raise ValueError('Parameters must be given as maxdiv_params_t structure or integral handle.')
    
    # Create buffer for detections
    if (num_intervals is None) or (num_intervals < 1):
        num_intervals = 100000
    det_buf_size = c_uint(num_intervals)
    det_buf = (detection_t * num_intervals)()
    
    # Prepare data
    if np.ma.isMaskedArray(X):
        X = X.filled(np.nan)
    X = np.require(X if isSpatioTemporal else X.T, np.float32 if maxdiv_scalar == c_float else np.float64, ['C_CONTIGUOUS'])
    if isSpatioTemporal:
        shape = index_vector_t()
        shape[:] = X.shape
    else:
        shape = index_vector_t(X.shape[0], 1, 1, 1, X.shape[1])
    
    # Run algorithm
    if isinstance(params, maxdiv_params_t):
        libmaxdiv.maxdiv(params, X.ctypes.data_as(maxdiv_scalar_p), shape, det_buf, pointer(det_buf_size), True)
    else:
        libmaxdiv.maxdiv_exec(params, X.ctypes.data_as(maxdiv_scalar_p), shape, det_buf, pointer(det_buf_size), True)
    
    # Convert detections to tuples
    if isSpatioTemporal:
        return [(det_buf[i].range_start[:4], det_buf[i].range_end[:4], det_buf[i].score) for i in range(det_buf_size.value)]
    else:
        return [(det_buf[i].range_start[0], det_buf[i].range_end[0], det_buf[i].score) for i in range(det_buf_size.value)]


def maxdiv_score_intervals(X, params, intervals):
    """ Computes anomaly scores for a given set of individual intervals.
    
    X - np.ndarray with d rows and n columns, where d is the number of attributes and n is
        the number of time steps. Alternatively, spatio-temporal data may be given as
        5-d array with 1 temporal, 3 spatial and 1 attribute dimension.
    params - Either a maxdiv_params_t object or a handle to a compiled pipeline obtained from `libmaxdiv.maxdiv_compile_pipeline()`
    intervals - List of intervals to be scored. Each interval is given as a tuple of first (inclusive) and last (exclusive) index.
                For purely temporal data (X is 2-dimensional), the indices are simply integers specifying the timesteps.
                For spatio-temporal data, each index is a 4-dimensional tuple specifying the point in time and space.
    
    Returns: a list of `(a, b, score)` tuples, where `a` is the first point within a detected interval,
             `b` is the first point right after the interval and `score` is the detection score.
             For non-spatial data, `a` and `b` will be scalars, while they will be lists of indices for
             spatio-temporal data.
             Out-of-bounds intervals and intervals within the padding area removed during pre-processing will receive a NaN score.
    """
    
    if libmaxdiv is None:
        raise RuntimeError('libmaxdiv could not be found or loaded.')
    
    isSpatioTemporal = False
    if X.ndim == 1:
        X = X.reshape((1, len(X)))
    elif X.ndim == 5:
        isSpatioTemporal = True
    elif X.ndim != 2:
        raise ValueError('Unsupported number of data dimensions: {}'.format(X.ndim))
    
    if not (isinstance(params, maxdiv_params_t) or isinstance(params, int)):
        raise ValueError('Parameters must be given as maxdiv_params_t structure or integral handle.')
    
    # Load intervals into buffer
    num_intervals = c_uint(len(intervals))
    interval_buf = (detection_t * len(intervals))()
    for i, (start, end) in enumerate(intervals):
        interval_buf[i].score = 0
        if isSpatioTemporal:
            interval_buf[i].range_start[:] = list(start) + [0]
            interval_buf[i].range_end[:] = list(end) + [0]
        else:
            interval_buf[i].range_start[:] = [start] + [0] * (len(interval_buf[i].range_start) - 1)
            interval_buf[i].range_end[:] = [end] + [1] * (len(interval_buf[i].range_end) - 1)
    
    # Prepare data
    if np.ma.isMaskedArray(X):
        X = X.filled(np.nan)
    X = np.require(X if isSpatioTemporal else X.T, np.float32 if maxdiv_scalar == c_float else np.float64, ['C_CONTIGUOUS'])
    if isSpatioTemporal:
        shape = index_vector_t()
        shape[:] = X.shape
    else:
        shape = index_vector_t(X.shape[0], 1, 1, 1, X.shape[1])
    
    # Run algorithm
    pipeline = libmaxdiv.maxdiv_compile_pipeline(params) if isinstance(params, maxdiv_params_t) else params
    libmaxdiv.maxdiv_score_intervals(pipeline, X.ctypes.data_as(maxdiv_scalar_p), shape, interval_buf, num_intervals, True)
    if isinstance(params, maxdiv_params_t):
        libmaxdiv.maxdiv_free_pipeline(pipeline)
    
    # Convert detections to tuples
    if isSpatioTemporal:
        return [(interval_buf[i].range_start[:4], interval_buf[i].range_end[:4], interval_buf[i].score) for i in range(len(intervals))]
    else:
        return [(interval_buf[i].range_start[0], interval_buf[i].range_end[0], interval_buf[i].score) for i in range(len(intervals))]



# Search library
libmaxdiv, libmaxdiv_path = _search_libmaxdiv()
