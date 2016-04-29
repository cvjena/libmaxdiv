""" Tries different parameter combinations for point-wise interval proposing. """

import sys
sys.path.append('..')

import numpy as np
import maxdiv, eval
from collections import OrderedDict
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle

# Constants
PROPMETHODS = ['hotellings_t', 'kde']
METHOD = 'gaussian_cov'
MODE = 'I_OMEGA'

MEDIAN = [True, False]
FILTERED = [True, False]
THS = np.linspace(0, 2, 21, endpoint = True)

propmeth = sys.argv[1] if (len(sys.argv) > 1) and (sys.argv[1] in PROPMETHODS) else PROPMETHODS[0]

# Load synthetic test data
with open('../testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']


# Try different parameter combinations for interval proposing
ap = OrderedDict()      # Average Precision
mean_ap = OrderedDict() # Mean Average Precision
labels = OrderedDict()  # Labels for the different combinations
for useMedian in MEDIAN:
    for filtered in FILTERED:
        id = (useMedian, filtered)
        ap[id] = {}
        mean_ap[id] = {}
        labels[id] = '{}, {}'.format('median' if useMedian else 'mean', 'gradients' if filtered else 'scores')
        print('Testing {}'.format(labels[id]))
        
        for sd_th in THS:
            ygts = []
            regions = []
            aps = []
            
            propparams = { 'useMedian' : useMedian, 'sd_th' : sd_th }
            if not filtered:
                propparams['filter'] = None
            
            for ftype in f:
                ygts += y[ftype]
                cur_regions = []
                for func in f[ftype]:
                    cur_regions.append(maxdiv.maxdiv(func, method = METHOD, mode = MODE, preproc = 'td',
                                                     num_intervals = None, extint_min_len = 10, extint_max_len = 50,
                                                     proposals = propmeth, proposalparameters = propparams))
                aps.append(eval.average_precision(y[ftype], cur_regions))
                regions += cur_regions
                
            ap[id][sd_th] = eval.average_precision(ygts, regions)
            mean_ap[id][sd_th] = np.mean(aps)


# Print results as table
maxLabelLen = max(len(lbl) for lbl in labels.values())
hdiv_len = 5 + sum(len(lbl) + 3 for lbl in labels.values()) # length of horizontal divider

print('\n-- Overall Average Precision --\n')

print('     |' + '|'.join(' {} '.format(lbl) for lbl in labels.values()))
print('{:-<{}s}'.format('', hdiv_len))
for sd_th in THS:
    row = '{:4.2f} '.format(sd_th)
    for id, aps in ap.items():
        row += '| {:>{}.4f} '.format(aps[sd_th], len(labels[id]))
    print(row)

print('\n-- Mean Average Precision --\n')

print('     |' + '|'.join(' {} '.format(lbl) for lbl in labels.values()))
print('{:-<{}s}'.format('', hdiv_len))
for sd_th in THS:
    row = '{:4.2f} '.format(sd_th)
    for id, aps in mean_ap.items():
        row += '| {:>{}.4f} '.format(aps[sd_th], len(labels[id]))
    print(row)