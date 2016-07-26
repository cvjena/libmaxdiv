import sys, os.path
sys.path.append(os.path.join('..', '..'))

import datetime, csv
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


# Historic storm data

def loadHistoricStorms(filename = '../coastdat/historic_storms.csv'):
    with open(filename) as stormfile:
        storms = list(csv.DictReader(stormfile, delimiter = '\t'))
    for storm in storms:
        storm['START_DATE'] = datetime.datetime.strptime(storm['START_DATE'], '%Y-%m-%d').date()
        storm['END_DATE'] = datetime.datetime.strptime(storm['END_DATE'], '%Y-%m-%d').date()
    return storms

historic_storms = loadHistoricStorms()

def matchDetectionWithStorm(detection):
    base_date = datetime.date(YEAR_OFFS, 1, 1)
    maxOverlapStorm = max(((storm, IoU(
                            (storm['START_DATE'] - base_date).days,
                            (storm['END_DATE'] - storm['START_DATE']).days + 1,
                            detection.range_start[0],
                            detection.range_end[0] - detection.range_start[0]
                        )) for storm in historic_storms), key = lambda x: x[1])
    return maxOverlapStorm[0] if maxOverlapStorm[1] > 0.0 else None

def matchDetectionsWithStorms(detections):
    matchedDetections = [(detection, matchDetectionWithStorm(detection, data_params)) for detection in detections]
    
    matched = { storm['NAME'] : False for storm in historic_storms }
    totalMatches = 0
    for _, storm in matchedDetections:
        if storm is not None:
            matched[storm['NAME']] = True
            totalMatches += 1

    return (matchedDetections, totalMatches, sum(int(m) for m in matched.values()))


# Helper functions for indexing and formatting

def ind2latlon(ind1, ind2):
    return (LAT_OFFS + ind1 * LAT_STEP, LON_OFFS + ind2 * LON_STEP)

def latlon2ind(lat, lon):
    return (int(round((lat - LAT_OFFS) / LAT_STEP)), int(round((lon - LON_OFFS) / LON_STEP)))

def ind2date(t):
    return datetime.date(YEAR_OFFS, 1, 1) + datetime.timedelta(days = t)

def date2ind(date):
    return (date - datetime.date(YEAR_OFFS, 1, 1)).days

def storm2str(storm):
    startDate, endDate = storm['START_DATE'].date(), storm['END_DATE'].date()
    if startDate == endDate:
        datestr = startDate.strftime('%b %d')
    elif startDate.month == endDate.month:
        datestr = '{}-{}'.format(startDate.strftime('%b %d'), endDate.strftime('%d'))
    else:
        datestr = '{} - {}'.format(startDate.strftime('%b %d'), endDate.strftime('%b %d'))
    return '{} ({})'.format(storm['NAME'], datestr)

def printDetection(detection):
    range_start, range_end, score = detection
    print('TIMEFRAME: {} - {}'.format(ind2date(range_start[0]), ind2date(range_end[0] - 1)))
    print('LOCATION:  {start[0]:.2f} N, {start[1]:.2f} E - {end[0]:.2f} N, {end[1]:.2f} E'.format(
        start = ind2latlon(range_start[1], range_start[2]),
        end   = ind2latlon(range_end[1] - 1, range_end[2] - 1)
    ))
    print('SCORE:     {}'.format(score))

def printDetections(detections):
    matchedDetections, totalMatches, uniqueMatches = matchDetectionsWithStorms(detections)
    for i, (detection, storm) in enumerate(matchedDetections):
        print('#{}'.format(i))
        printDetection(detection)
        if storm is not None:
            print('IDENT:     {}'.format(storm2str(storm)))
        print()
    print('MATCHED DETECTIONS: {:3d}/{}'.format(totalMatches, len(detections)))
    print('UNIQUE MATCHES:     {:3d}/{}'.format(uniqueMatches, len(detections)))
    print('TOP-10 DETECTIONS:  {:3d}'.format(sum(1 if storm is not None else 0 for _, storm in matchedDetections[:10])))
