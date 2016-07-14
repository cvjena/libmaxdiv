import sys
sys.path.append('..')

from ctypes import *
from maxdiv.libmaxdiv_wrapper import maxdiv_params_t, maxdiv_params_p, detection_t, detection_p

class coastdat_params_t(Structure):
    _fields_ = [('variables', c_char_p),
                ('firstYear', c_uint),
                ('lastYear', c_uint),
                ('firstLat', c_uint),
                ('lastLat', c_uint),
                ('firstLon', c_uint),
                ('lastLon', c_uint),
                ('spatialPoolingSize', c_uint)]

coastdat_params_p = POINTER(coastdat_params_t)
c_uint_p = POINTER(c_uint)


libcoastdat = CDLL('maxdiv_coastdat.so')

maxdiv_coastdat = CFUNCTYPE(c_int, maxdiv_params_p, coastdat_params_p, detection_p, c_uint_p)(
                            ('maxdiv_coastdat', libcoastdat), ((1, 'params'), (1, 'data_params'), (1, 'detection_buf'), (1, 'detection_buf_size')))

maxdiv_coastdat_context_window_size = CFUNCTYPE(c_int, maxdiv_params_p)(('maxdiv_coastdat_context_window_size', libcoastdat), ((1, 'data_params'),))

maxdiv_coastdat_default_params = CFUNCTYPE(c_void_p, maxdiv_params_p)(('maxdiv_coastdat_default_params', libcoastdat), ((1, 'data_params'),))