import sys
sys.path.append('..')

from ctypes import *
import datetime
from maxdiv.libmaxdiv_wrapper import maxdiv_params_t, maxdiv_params_p, detection_t, detection_p


LAT_OFFS = 51.0
LAT_STEP = 0.05
LON_OFFS = -3.0
LON_STEP = 0.1
YEAR_OFFS = 1958


COASTDAT_DESEAS_NONE        = 0
COASTDAT_DESEAS_OLS_DAY     = 1
COASTDAT_DESEAS_OLS_YEAR    = 2    
COASTDAT_DESEAS_ZSCORE_DAY  = 3
COASTDAT_DESEAS_ZSCORE_YEAR = 4


class coastdat_params_t(Structure):
    _fields_ = [('variables', c_char_p),
                ('firstYear', c_uint),
                ('lastYear', c_uint),
                ('firstLat', c_uint),
                ('lastLat', c_uint),
                ('firstLon', c_uint),
                ('lastLon', c_uint),
                ('spatialPoolingSize', c_uint),
                ('deseasonalization', c_int)]

coastdat_params_p = POINTER(coastdat_params_t)
c_uint_p = POINTER(c_uint)


libcoastdat = CDLL('./maxdiv_coastdat.so')

coastdat_default_params = CFUNCTYPE(c_void_p, coastdat_params_p)(('coastdat_default_params', libcoastdat), ((1, 'data_params'),))

coastdat_dump = CFUNCTYPE(c_int, coastdat_params_p, c_char_p)(('coastdat_dump', libcoastdat), ((1, 'data_params'), (1, 'dump_file')))

coastdat_maxdiv = CFUNCTYPE(c_int, maxdiv_params_p, coastdat_params_p, detection_p, c_uint_p)(
                            ('coastdat_maxdiv', libcoastdat), ((1, 'params'), (1, 'data_params'), (1, 'detection_buf'), (1, 'detection_buf_size')))

coastdat_maxdiv_dump = CFUNCTYPE(c_int, maxdiv_params_p, c_char_p, detection_p, c_uint_p)(
                                ('coastdat_maxdiv_dump', libcoastdat), ((1, 'params'), (1, 'dump_file'), (1, 'detection_buf'), (1, 'detection_buf_size')))

coastdat_context_window_size = CFUNCTYPE(c_int, coastdat_params_p)(('coastdat_context_window_size', libcoastdat), ((1, 'data_params'),))

coastdat_context_window_size_dump = CFUNCTYPE(c_int, c_char_p, c_int)(('coastdat_context_window_size_dump', libcoastdat), ((1, 'dump_file'), (1, 'deseasonalization', COASTDAT_DESEAS_NONE)))


def ind2latlon(ind_y, ind_x, data_params = None):
    if data_params is not None:
        if data_params.spatialPoolingSize > 1:
            ind_y *= data_params.spatialPoolingSize + 0.5
            ind_x *= data_params.spatialPoolingSize + 0.5
        ind_y += data_params.firstLat
        ind_x += data_params.firstLon
    return (LAT_OFFS + ind_y * LAT_STEP, LON_OFFS + ind_x * LON_STEP)

def latlon2ind(lat, lon, data_params = None):
    ind = (int(round((lat - LAT_OFFS) / LAT_STEP)), int(round((lon - LON_OFFS) / LON_STEP)))
    if data_params is not None:
        ind[0] -= data_params.firstLat
        ind[1] -= data_params.firstLon
        if data_params.spatialPoolingSize > 1:
            ind_y //= data_params.spatialPoolingSize
            ind_x //= data_params.spatialPoolingSize
    return ind

def ind2time(t, data_params = None):
    start_year = data_params.firstYear if data_params is not None else YEAR_OFFS
    if start_year < YEAR_OFFS:
        start_year += YEAR_OFFS - 1
    return datetime.datetime(start_year, 1, 1) + datetime.timedelta(seconds = t * 3600)

def time2ind(time, data_params = None):
    start_year = data_params.firstYear if data_params is not None else YEAR_OFFS
    if start_year < YEAR_OFFS:
        start_year += YEAR_OFFS - 1
    return round((time - datetime.datetime(start_year, 1, 1)).total_seconds() / 3600)

def printCoastDatDetection(detection, data_params):
    print('TIMEFRAME: {} - {}'.format(ind2time(detection.range_start[0], data_params), ind2time(detection.range_end[0] - 1, data_params)))
    print('LOCATION:  {start.lat:.2f} N, {start.lon:.2f} E - {end.lat:.2f} N, {end.lon:.2f} E'.format(
        start = ind2latlon(detection.range_start[2], detection.range_start[1], data_params),
        end   = ind2latlon(detection.range_end[2] - 1, detection.range_end[1] - 1, data_params)
    ))
    print('SCORE:     {}'.format(detection.score))
