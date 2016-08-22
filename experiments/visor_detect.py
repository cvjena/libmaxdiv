import sys
sys.path.append('..')

import numpy as np
from maxdiv.libmaxdiv_wrapper import *


#data = np.load('../../visor_cnn_16.npy')
data = np.load('../../visor_avg_8.npy')

params = maxdiv_params_t()
libmaxdiv.maxdiv_init_params(params)
params.kl_mode = enums['MAXDIV_KL_UNBIASED']
params.min_size[0] = 72
params.min_size[1] = 10
params.min_size[2] = 10
params.max_size[0] = 288
params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']
params.preproc.embedding.kt = 3
params.preproc.embedding.dt = 4
#params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST']

detections = maxdiv_exec(data, params, 5)

for a, b, score in detections:
    print('{:.1f} - {:.1f} s, {}x{} - {}x{} (Score: {})'.format(a[0] / 25.0, (b[0] - 1) / 25.0, a[1], a[2], b[1] - 1, b[2] - 1, score))
