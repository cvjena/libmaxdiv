import csv, json, datetime, os.path
import numpy as np
from maxdiv.eval import pointwiseLabelsToIntervals
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle


DATASETS = ['synthetic', 'nab_real', 'nab_artifical', 'yahoo_real']
TYPES = ['interval', 'point', 'change']

BASEPATH = os.path.dirname(os.path.realpath(__file__))


def loadDatasets(datasets = None, types = None):
    """ Common interface for loading several datasets.
    
    `datasets` - A list of datasets to be loaded. Possible values are listed in `DATASETS`.
                 If set to `None`, all available datasets will be loaded.
    
    `types` - Types of anomalies to be included. Possible values are listed in `TYPES`.
              If set to `None`, types of anomalies won't be restricted.
    
    Returns: A dictionary whose keys are names of datasets and whose values are lists with
             one dictionary for each time series in that dataset. Those dictionaries will
             contain the following keys:
                - `ts` - The actual time series as d-by-n matrix with `d` dimensions and `n`samples.
                - `ticks` - List with values on the time axis corresponding to the elements of `ts`.
                            These values will always be spaced uniformly and either be numbers or
                            instances of `datetime.datetime`, `datetime.date` or `datetime.time`.
                - `gt` - Ground-truth regions as `(a, b)` tuples specifying an interval `[a,b)` where
                         `a` and `b` are indices for the `ts` or the `ticks` array..
                - `type` - The type of the anomalies contained in the time series.
                - `id` - A dataset-specific identifier of the time series within the data set.
    """
    
    if (datasets is None) or (len(datasets) == 0):
        datasets = DATASETS
    elif isinstance(datasets, str):
        datasets = [datasets]
    if (types is None) or (len(types) == 0):
        types = TYPES
    elif isinstance(types, str):
        types = [types]
    types = set(types)
    
    data = {}
    for ds in datasets:
        if (ds == 'synthetic') and ('interval' in types):
            data.update(loadSyntheticTestbench())
        elif (ds == 'synthetic_small') and ('interval' in types):
            data.update(loadSyntheticTestbench(BASEPATH + '/testcube_small.pickle'))
        elif ds == 'synthetic_normal':
            data.update(loadSyntheticNormalTestbench())
        elif ds.startswith('nab_'):
            data.update(loadNabDataset(subset = ds[4:], types = types))
        elif ds.startswith('yahoo_'):
            data.update(loadYahooDataset(subset = ds[6:], types = types, minAnomalyLength = 10))
        else:
            raise ValueError('Unknown datasat: {}'.format(ds))
    
    return data


def loadSyntheticTestbench(filename = BASEPATH + '/testcube.pickle'):
    """ Loads a synthetic test bench with different types of anomalous intervals.
    
    The testbench can be generated using `testbench.py`.
    This function will load it from the Pickle file created by that script and returns
    a dictionary in the format described in `loadDatasets()`.
    """
    
    with open(filename, 'rb') as fin:
        cube = pickle.load(fin)
        f = cube['f']
        y = cube['y']
    
    return { ftype : [{
        'ts'    : func,
        'ticks' : list(range(func.shape[1])),
        'gt'    : pointwiseLabelsToIntervals(ygt),
        'type'  : ['interval'] * len(ygt),
        'id'    : i
    } for i, (func, ygt) in enumerate(zip(f[ftype], y[ftype]))] for ftype in f }


def loadSyntheticNormalTestbench(filename = BASEPATH + '/testcube_normal.pickle'):
    """ Loads a synthetic test bench with time-series without anomalies.
    
    The testbench can be generated using `testbench.py`.
    This function will load it from the Pickle file created by that script and returns
    a dictionary in the format described in `loadDatasets()`.
    """
    
    with open(filename, 'rb') as fin:
        f = pickle.load(fin)
    
    return { ftype : [{
        'ts'    : func,
        'ticks' : list(range(func.shape[1])),
        'gt'    : [],
        'type'  : [],
        'id'    : i
    } for i, func in enumerate(f[ftype])] for ftype in f }


def loadNabDataset(annotations = BASEPATH + '/../../datasets/NAB-1.0/labels/combined_windows.json',
                   datadir = BASEPATH + '/../../datasets/NAB-1.0/data/',
                   subset = None, types = None):
    """ Loads data from the Numenta Anomaly Benchmark (NAB).
    
    NAB GitHub repository: https://github.com/numenta/NAB
    
    `annotations` - path of the `combined_windows.json` file from NAB.
    
    `datadir` - path of the `data` directory from NAB.
    
    `subset` - If different from `None`, only a subset of the NAB data will be loaded.
               Possible values are 'real' and 'artificial'.
    
    `types` - Types of anomalies to be included. Possible values are listed in `TYPES`.
              If set to `None`, types of anomalies won't be restricted.
    
    Returns: a dictionary in the format described in `loadDatasets()`.
    """
    
    if (types is None) or (len(types) == 0):
        types = TYPES
    types = set(types)
    
    with open(annotations) as af:
        metadata = json.load(af)
    
    data = {}
    for filename, regions in metadata.items():
        setName, tsName = filename.split('/')
        if (len(regions) > 0) and ((subset is None) or (setName.startswith(subset))):
            
            # Load data
            dates, values = [], []
            delta = None
            
            with open(os.path.join(datadir, filename)) as df:
                for i, d in enumerate(csv.DictReader(df)):
                    
                    date = datetime.datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
                    val = float(d['value'])
                    
                    # Fill gaps
                    #if i > 1:
                    #    while date - dates[-1] > delta:
                    #        dates.append(dates[-1] + delta)
                    #        values.append(values[-1])
                    #elif i == 1:
                    #    delta = date - dates[-1]
                    
                    dates.append(date)
                    values.append(val)
            
            # Convert [a,b] datetime regions to [a,b) index intervals
            intervals, intvl_types = [], []
            for region in regions:
                
                if isinstance(region, list):
                    region = { 'start' : region[0], 'end' : region[1], 'type' : 'interval' }
                
                regionStart = datetime.datetime.strptime(region['start'], '%Y-%m-%d %H:%M:%S.%f')
                regionEnd = datetime.datetime.strptime(region['end'], '%Y-%m-%d %H:%M:%S.%f')
                a = dates.index(regionStart)
                b = dates.index(regionEnd) + 1
                if a >= b:
                    raise ValueError('Invalid ground-truth region: ({}, {}) [{!s} - {!s}]'.format(a, b, regionStart, regionEnd))
                
                if region['type'] in types:
                    intervals.append((a, b))
                    intvl_types.append(region['type'])
            
            # Append to set
            if setName not in data:
                data[setName] = []
            if len(intervals) > 0:
                data[setName].append({
                    'ts'    : np.array(values).reshape(1, len(values)),
                    'ticks' : dates,
                    'gt'    : intervals,
                    'type'  : intvl_types,
                    'id'    : os.path.splitext(tsName)[0]
                })
    
    return data


def loadYahooDataset(datadir = BASEPATH + '/../../datasets/yahoo-s5-v1_0/', minAnomalyLength = 0, subset = None, types = None):
    """ Loads data from the Numenta Anomaly Benchmark (NAB).
    
    Download of the Yahoo! Webscope S5 Dataset: http://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70
    (Note: Interval annotations are not part of that data set. They have been created manually by bjoern.barz@uni-jena.de)
    
    `datadir` - Path of the directory containing the dataset.
    
    `minAnomalyLength` - Only anomalies with at least this length will be added to the ground-truth.
                         If a time series does not have any long enough anomaly, it will be excluded from the dataset.
    
    `subset` - If different from `None`, only a subset of the data will be loaded.
               Possible values are 'real' (a.k.a. 'A1Benchmark') and 'synthetic' (subsumes 'A2Benchmark', 'A3Benchmark' and 'A4Benchmark').
    
    `types` - Types of anomalies to be included. Possible values are listed in `TYPES`.
              If set to `None`, types of anomalies won't be restricted.
    
    Returns: a dictionary in the format described in `loadDatasets()`.
    """
    
    DIRS = {
        'all'       : ['A{}Benchmark'.format(i) for i in range(1, 5)],
        'real'      : ['A1Benchmark'],
        'synthetic' : ['A{}Benchmark'.format(i) for i in range(2, 5)]
    }
    for i in range(1, 5):
        dir = 'A{}Benchmark'.format(i)
        DIRS[dir] = [dir]
    
    if subset is None:
        subset = 'all'
    if (types is None) or (len(types) == 0):
        types = TYPES
    types = set(types)
    
    data = {}
    for dir in DIRS[subset]:
    
        data[dir] = []
        
        with open(os.path.join(datadir, dir, 'anom_intervals.json')) as af:
            metadata = json.load(af)
        
        for filename, regions in metadata.items():
            
            # Load data
            timestamps, values = [], []
            with open(os.path.join(datadir, dir, filename)) as df:
                for d in csv.DictReader(df):
                    timestamps.append(int(d['timestamps' if 'timestamps' in d else 'timestamp']))
                    values.append(float(d['value']))
            
            # Convert regions to canonical format and prune intervals which are too small or of the wrong type
            intervals, intvl_types = [], []
            for region in regions:
                
                if isinstance(region, list):
                    region = { 'start' : region[0], 'end' : region[1], 'type' : 'interval' }
                
                # Convert [a,b] timestamp regions to [a,b) index intervals
                a = timestamps.index(region['start'])
                b = timestamps.index(region['end']) + 1
                if a >= b:
                    raise ValueError('Invalid ground-truth region: ({}, {}) [{!s} - {!s}]'.format(a, b, region['start'], region['end']))
                
                if (b - a >= minAnomalyLength) and (region['type'] in types):
                    intervals.append((a, b))
                    intvl_types.append(region['type'])
            
            # Append to set
            if len(intervals) > 0:
                data[dir].append({
                    'ts'    : np.array(values).reshape(1, len(values)),
                    'ticks' : timestamps,
                    'gt'    : intervals,
                    'types' : intvl_types,
                    'id'    : os.path.splitext(filename)[0]
                })
    
    return data
