import sys, os.path
sys.path.append(os.path.join('..', '..'))

from ctypes import *
import datetime, csv
from maxdiv.libmaxdiv_wrapper import maxdiv_params_t, maxdiv_params_p, detection_t, detection_p
from maxdiv.maxdiv_util import IoU


# Wrapper around coastdat C++ library

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


# Historic storm data

def loadHistoricStorms(filename = 'historic_storms.csv'):
    with open(filename) as stormfile:
        storms = list(csv.DictReader(stormfile, delimiter = '\t'))
    for storm in storms:
        storm['START_DATE'] = datetime.datetime.strptime(storm['START_DATE'], '%Y-%m-%d')
        storm['END_DATE'] = datetime.datetime.strptime(storm['END_DATE'], '%Y-%m-%d')
    return storms

historic_storms = loadHistoricStorms()

def matchDetectionWithStorm(detection, data_params):
    detStart = ind2time(detection.range_start[0], data_params)
    detEnd = ind2time(detection.range_end[0], data_params)
    maxOverlapStorm = max((storm, IoU(
                            int(storm['START_DATE'].timestamp()),
                            int((storm['END_DATE'] - storm['START_DATE']).total_seconds()),
                            int(detStart.timestamp()),
                            int((detEnd - detStart).total_seconds())
                        )) for storm in historic_storms, key = lambda x: x[1])
    return maxOverlapStorm[0] if maxOverlapStorm[1] > 0.0 else None

def matchDetectionsWithStorms(detections, data_params):
    matchedDetections = [(detection, matchDetectionWithStorm(detection, data_params)) for detection in detections]
    
    matched = { storm['NAME'] : False for storm in historic_storms }
    totalMatches = 0
    for _, storm in matchedDetections:
        if storm is not None:
            matched[storm['NAME']] = True
            totalMatches += 1

    return (matchedDetections, totalMatches, sum(int(m) for m in matched.values()))


# Helper functions for indexing and formatting

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
    print('LOCATION:  {start[0]:.2f} N, {start[1]:.2f} E - {end[0]:.2f} N, {end[1]:.2f} E'.format(
        start = ind2latlon(detection.range_start[2], detection.range_start[1], data_params),
        end   = ind2latlon(detection.range_end[2] - 1, detection.range_end[1] - 1, data_params)
    ))
    print('SCORE:     {}'.format(detection.score))

def printCoastDatDetections(detections, data_params):
    matchedDetections, totalMatches, uniqueMatches = matchDetectionsWithStorms(detections, data_params)
    for detection, storm in matchedDetections:
        printCoastDatDetection(detection, data_params)
        if storm is not None:
            print('IDENT:     {} ({} - {})'.format(storm['NAME'], storm['START_DATE'].date(), strom['END_DATE'].date()))
        print()
    print('MATCHED DETECTIONS: {:3d}/{}'.format(totalMatches, len(detections)))
    print('UNIQUE MATCHES:     {:3d}/{}'.format(uniqueMatches, len(detections)))
