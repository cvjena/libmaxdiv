""" Runs the MDI algorithm on the Sea Level Pressure dataset. """

import os.path
from utils import *
from maxdiv.libmaxdiv_wrapper import *

params = maxdiv_params_t()
libmaxdiv.maxdiv_init_params(params)
params.min_size[0] = 3
params.min_size[1] = params.min_size[2] = 3
params.max_size[0] = 10
params.kl_mode = enums['MAXDIV_KL_UNBIASED']
#params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST']
params.preproc.embedding.kt = 3
params.preproc.embedding.kx = params.preproc.embedding.ky = 1
params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']

detections = maxdiv_exec(loadSLP(), params, 20)

printDetections(detections)
