import numpy as np
import sys
from collections import OrderedDict

from maxdiv import preproc, eval, libmaxdiv_wrapper
import datasets


# Settings
METHODS = [('KDE', 'I_OMEGA'), ('KDE', 'JSD'), ('GAUSSIAN', 'I_OMEGA'), ('GAUSSIAN', 'OMEGA_I'), ('GAUSSIAN', 'UNBIASED'), ('GAUSSIAN_GLOBAL_COV', 'I_OMEGA')]
PERIOD_LEN = 24
PROPOSALS = sys.argv[1] if len(sys.argv) > 1 else 'dense'
TD_DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TD_LAG = int(sys.argv[3]) if len(sys.argv) > 3 else 2

if PROPOSALS == 'help':
    print('Usage: {} [<proposals = dense>] [<td-dim = 6>] [<td-lag = 2>]'.format(sys.argv[0]))
    exit()

# Load test data
data = datasets.loadDatasets('yahoo_real')['A1Benchmark']

# Check libmaxdiv
if libmaxdiv_wrapper.libmaxdiv is None:
    raise RuntimeError('libmaxdiv could not be found and loaded.')

# Compile pipelines
pipelines = OrderedDict()
pipelines['none'] = []
pipelines['OLS'] = []
pipelines['Z-Score'] = []
params = libmaxdiv_wrapper.maxdiv_params_t()
libmaxdiv_wrapper.libmaxdiv.maxdiv_init_params(params)
params.min_size[0] = 10
params.max_size[0] = 120
params.proposal_generator = libmaxdiv_wrapper.enums['MAXDIV_DENSE_PROPOSALS'] if PROPOSALS == 'dense' else libmaxdiv_wrapper.enums['MAXDIV_POINTWISE_PROPOSALS_' + PROPOSALS.upper()]
params.preproc.normalization = libmaxdiv_wrapper.enums['MAXDIV_NORMALIZE_MAX']
params.preproc.embedding.kt = TD_DIM
params.preproc.embedding.dt = TD_LAG
params.preproc.detrending.ols_period_num = PERIOD_LEN
params.preproc.detrending.z_period_len = PERIOD_LEN

for method, mode in METHODS:

    if method == 'GAUSSIAN_GLOBAL_COV':
        params.estimator = libmaxdiv_wrapper.enums['MAXDIV_GAUSSIAN']
        params.gaussian_cov_mode = libmaxdiv_wrapper.enums['MAXDIV_GAUSSIAN_COV_SHARED']
    else:
        params.estimator = libmaxdiv_wrapper.enums['MAXDIV_' + method]
        params.gaussian_cov_mode = libmaxdiv_wrapper.enums['MAXDIV_GAUSSIAN_COV_FULL']
    if mode == 'JSD':
        params.divergence = libmaxdiv_wrapper.enums['MAXDIV_JS_DIVERGENCE']
    else:
        params.divergence = libmaxdiv_wrapper.enums['MAXDIV_KL_DIVERGENCE']
        params.kl_mode = libmaxdiv_wrapper.enums['MAXDIV_KL_' + mode]
    
    params.preproc.detrending.method = libmaxdiv_wrapper.enums['MAXDIV_DETREND_NONE']
    pipelines['none'].append(libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params))
    params.preproc.detrending.method = libmaxdiv_wrapper.enums['MAXDIV_DETREND_OLS']
    pipelines['OLS'].append(libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params))
    params.preproc.detrending.method = libmaxdiv_wrapper.enums['MAXDIV_DETREND_ZSCORE']
    pipelines['Z-Score'].append(libmaxdiv_wrapper.libmaxdiv.maxdiv_compile_pipeline(params))

# Run detectors
aps = OrderedDict()
aps['none'] = []
aps['FT'] = []
aps['OLS'] = []
aps['Z-Score'] = []
for detrending in aps.keys():
    pipeline_key = 'none' if detrending == 'FT' else detrending
    for i, (method, mode) in enumerate(METHODS):
        sys.stderr.write('{}, {}, {}\n'.format(detrending, method, mode))
        ygts = []
        regions = []
        
        for func in data:
            ygts.append(func['gt'])
            ts = preproc.deseasonalize_ft(func['ts']) if detrending == 'FT' else func['ts']
            regions.append(libmaxdiv_wrapper.maxdiv_exec(ts, pipelines[pipeline_key][i], None))
        
        aps[detrending].append(eval.average_precision(ygts, regions))

# Clean up
for p in pipelines.values():
    for pipeline in p:
        libmaxdiv_wrapper.libmaxdiv.maxdiv_free_pipeline(pipeline)

# Print results
header = 'Deseasonalization'
for method, mode in METHODS:
    header += ';{} ({})'.format(method, mode)
print(header)
for detrending in aps.keys():
    print('{};{}'.format(detrending, ';'.join('{}'.format(ap) for ap in aps[detrending])))