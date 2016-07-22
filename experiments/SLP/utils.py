import sys, os.path
sys.path.append(os.path.join('..', '..'))

import datetime
import scipy.io
from maxdiv.maxdiv_util import IoU


# Constants

LAT_OFFS = 25.0
LAT_STEP = 2.5
LON_OFFS = -52.5
LON_STEP = 2.5
YEAR_OFFS = 1957


def loadSLP():
    slp = scipy.io.loadmat('SLP_ATL.mat')['pres']
    return slp.reshape(slp.shape + (1, 1))


# Helper functions for indexing and formatting

def ind2latlon(ind1, ind2):
    return (LAT_OFFS + ind1 * LAT_STEP, LON_OFFS + ind2 * LON_STEP)

def latlon2ind(lat, lon):
    return (int(round((lat - LAT_OFFS) / LAT_STEP)), int(round((lon - LON_OFFS) / LON_STEP)))

def ind2date(t):
    return datetime.date(YEAR_OFFS, 1, 1) + datetime.timedelta(days = t)

def date2ind(date):
    return (date - datetime.date(YEAR_OFFS, 1, 1)).days

def printDetection(detection):
    range_start, range_end, score = detection
    print('TIMEFRAME: {} - {}'.format(ind2date(range_start[0]), ind2date(range_end[0] - 1)))
    print('LOCATION:  {start[0]:.2f} N, {start[1]:.2f} E - {end[0]:.2f} N, {end[1]:.2f} E'.format(
        start = ind2latlon(range_start[1], range_start[2]),
        end   = ind2latlon(range_end[1] - 1, range_end[2] - 1)
    ))
    print('SCORE:     {}'.format(score))

def printDetections(detections):
    for detection in detections:
        printDetection(detection)
        print()
