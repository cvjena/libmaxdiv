import os.path
from maxdiv_coastdat_utils import *
from maxdiv.libmaxdiv_wrapper import *

dump_file = '../../coastDat_ff-hs-mp_aggregated_ols.dat'
#dump_file = ''

data_params = coastdat_params_t()
coastdat_default_params(data_params)
#data_params.firstLat = data_params.lastLat = 80
#data_params.firstLon = data_params.lastLon = 70
#data_params.spatialPoolingSize = 1
data_params.spatialPoolingSize = 100
data_params.deseasonalization = COASTDAT_DESEAS_OLS_YEAR

if (dump_file != '') and (not os.path.exists(dump_file)):
    coastdat_dump(data_params, dump_file.encode())

params = maxdiv_params_t()
libmaxdiv.maxdiv_init_params(params)
params.min_size[0] = 12
params.max_size[0] = 72
params.kl_mode = enums['MAXDIV_KL_UNBIASED']
params.preproc.embedding.kt = 3
params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']
#params.preproc.detrending.method = enums['MAXDIV_DETREND_ZSCORE']
#params.preproc.detrending.z_period_len = 24*365

num_det = c_uint(20)
detections = (detection_t * num_det.value)()
if dump_file != '':
    coastdat_maxdiv_dump(params, dump_file.encode(), detections, num_det)
else:
    coastdat_maxdiv(params, data_params, detections, num_det)

for i in range(num_det.value):
    printCoastDatDetection(detections[i], data_params)
    print()
