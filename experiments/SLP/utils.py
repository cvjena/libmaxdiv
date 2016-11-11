import sys, os.path
sys.path.append(os.path.join('..', '..'))

import datetime, csv
import scipy.io
from maxdiv.maxdiv_util import IoU


# Constants

def getSLPGridSpec():
    """ old specifications of the SLP data """
    return {
        'lat_offs': 25.0,
        'lat_step': 2.5,
        'lon_offs': -52.5,
        'lon_step': 2.5,
        'year_offs': 1957
    }

def loadTensor(matfile, tensorvar):
    """ load a simple tensor from a MATLAB file """
    tensor_data = scipy.io.loadmat(matfile)
    if tensorvar in tensor_data:
        tensor = tensor_data[tensorvar]
    else:
        raise Exception('Variable {} not present in {}\nAvailable variables: {}\n'.format(tensorvar,
	    matfile, tensor_data.keys()))
    print ("Tensor shape: {}".format(tensor.shape))
    return tensor.reshape(tensor.shape + (1, 1))


# Historic event data

def loadHistoricEvents(filename = '../coastdat/historic_storms.csv'):
    """ load historic events from a CSV file """
    with open(filename) as eventfile:
        events = list(csv.DictReader(eventfile, delimiter = '\t'))
    for event in events:
        # historic events are only temporal right now
        event['START_DATE'] = datetime.datetime.strptime(event['START_DATE'], '%Y-%m-%d').date()
        event['END_DATE'] = datetime.datetime.strptime(event['END_DATE'], '%Y-%m-%d').date()
    return events

def matchDetectionWithEvent(detection, historic_events, year_offs):
    base_date = datetime.date(year_offs, 1, 1)
    maxOverlap = max(((event, IoU(
                            (event['START_DATE'] - base_date).days,
                            (event['END_DATE'] - event['START_DATE']).days + 1,
                            detection[0][0],
                            detection[1][0] - detection[0][0]
                        )) for event in historic_events), key = lambda x: x[1])
    return maxOverlap[0] if maxOverlap[1] > 0.0 else None

def matchDetectionsWithEvents(detections, historic_events, year_offs):
    matchedDetections = [(detection, matchDetectionWithEvent(detection, historic_events, year_offs)) for detection in detections]
    
    matched = { event['NAME'] : False for event in historic_events }
    totalMatches = 0
    for _, event in matchedDetections:
        if event is not None:
            matched[event['NAME']] = True
            totalMatches += 1

    return (matchedDetections, totalMatches, sum(int(m) for m in matched.values()))


# Helper functions for indexing and formatting

def ind2latlon(ind1, ind2, lat_offs, lat_step, lon_offs, lon_step):
    return (lat_offs + ind1 * lat_step, lon_offs + ind2 * lon_step)

def latlon2ind(lat, lon, lat_offs, lat_step, lon_offs, lon_step):
    return (int(round((lat - lat_offs) / lat_step)), int(round((lon - lon_offs) / lon_step)))

def ind2date(t, year_offs):
    return datetime.date(year_offs, 1, 1) + datetime.timedelta(days = t)

def date2ind(date):
    return (date - datetime.date(year_offs, 1, 1)).days

def event2str(event):
    startDate, endDate = event['START_DATE'], event['END_DATE']
    if startDate == endDate:
        datestr = startDate.strftime('%b %d')
    elif startDate.month == endDate.month:
        datestr = '{}-{}'.format(startDate.strftime('%b %d'), endDate.strftime('%d'))
    else:
        datestr = '{} - {}'.format(startDate.strftime('%b %d'), endDate.strftime('%b %d'))
    return '{} ({})'.format(event['NAME'], datestr)

def printDetection(detection, gridspec):
    range_start, range_end, score = detection
    y_o = gridspec['year_offs']
    grid_coords = [ gridspec[k] for k in ['lat_offs', 'lat_step', 'lon_offs', 'lon_step' ]]
    print('TIMEFRAME: {} - {}'.format(ind2date(range_start[0], y_o), ind2date(range_end[0] - 1, y_o)))
    print('LOCATION:  {start[0]:.2f} N, {start[1]:.2f} E - {end[0]:.2f} N, {end[1]:.2f} E'.format(
        start = ind2latlon(range_start[1], range_start[2], *grid_coords),
        end   = ind2latlon(range_end[1] - 1, range_end[2] - 1, *grid_coords)
    ))
    print('SCORE:     {}'.format(score))

def printDetections(detections, gridspec, historic_events):
    matchedDetections, totalMatches, uniqueMatches = matchDetectionsWithEvents(detections, historic_events, gridspec['year_offs'])
    for i, (detection, event) in enumerate(matchedDetections):
        print('#{}'.format(i))
        printDetection(detection, gridspec)
        if event is not None:
            print('IDENT:     {}'.format(event2str(event)))
        print()
    print('MATCHED DETECTIONS: {:3d}/{}'.format(totalMatches, len(detections)))
    print('UNIQUE MATCHES:     {:3d}/{}'.format(uniqueMatches, len(detections)))
    print('TOP-10 DETECTIONS:  {:3d}'.format(sum(1 if event is not None else 0 for _, event in matchedDetections[:10])))
