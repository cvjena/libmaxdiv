import numpy as np
import matplotlib.pylab as plt
import maxdiv, eval
import csv, datetime
from collections import OrderedDict


HURRICANE_GT = { \
    'Sandy'     : (datetime.date(2012,10,22), datetime.date(2012,10,29)),
    'Rafael'    : (datetime.date(2012,10,12), datetime.date(2012,10,18)),
    'Isaac'     : (datetime.date(2012, 8,22), datetime.date(2012, 8,25))
}


def read_hpw_csv(csvFile):
    """ Reads HPW data from a CSV file.
    
    The CSV file must contain 4 fields per line:
    date as 'yyyy-m-d-h', wind speed, air pressure and wave height
    The first line is assumed to be field headings and will be ignored.
    """
    
    # Read data from CSV file into ordered dict
    data = OrderedDict()
    with open(csvFile) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            date = line[0][:line[0].rfind('-')]
            fields = [float(x) for x in line[1:]]
            if not np.any(np.isnan(fields)):
                if date not in data:
                    data[date] = []
                data[date].append(fields)
    
    # Take day-wise means and store them in a numpy array
    ts = np.ndarray((3, len(data)))
    for i, (date, values) in enumerate(data.items()):
        ts[:,i] = np.array(values).mean(axis = 0).T
    dates = [datetime.date(*(int(x) for x in date.split('-'))) for date in data.keys()]
    
    return ts, dates


def normalize_time_series(ts):
    """ Normalizes each dimension of a time series by subtracting the mean and dividing by the maximum. """
    
    ts = (ts.T - ts.mean(axis = 1)).T
    ts = (ts.T / ts.max(axis = 1)).T
    return ts


if __name__ == '__main__':

    # Load data
    data, dates = read_hpw_csv('HPW_2012_41046.csv')
    data = normalize_time_series(data)
    
    # Detect
    regions = maxdiv.maxdiv(data, 'parzen_proper', preproc = 'td', extint_min_len = 3, extint_max_len = 30, num_intervals = 5)
    
    # Console output
    print('-- Ground Truth --')
    for name, (a, b) in HURRICANE_GT.items():
        print('{:{}s}: {!s} - {!s}'.format(name, max(len(n) for n in HURRICANE_GT.keys()), a, b - datetime.timedelta(days = 1)))
    print('\n-- Detected Intervals --')
    for a, b, score in regions:
        print('{!s} - {!s} (Score: {})'.format(dates[a], dates[b-1], score))
    
    # Plot
    ygt = [((a - dates[0]).days, (b - dates[0]).days) for a, b in HURRICANE_GT.values()]
    eval.plotDetections(data, regions, ygt,
                        ticks = { (d-dates[0]).days : d.strftime('%b %Y') for d in (datetime.date(2012,mon,1) for mon in range(6, 12)) })