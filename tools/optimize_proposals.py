""" Tries different parameter combinations for point-wise interval proposing. """

import sys
sys.path.append('..')

import numpy as np
from collections import OrderedDict

from maxdiv import maxdiv, eval
import datasets

# Constants
PROPMETHODS = ['hotellings_t', 'kde']
METHOD = 'gaussian_cov'
MODE = 'I_OMEGA'

MAD = [True, False]
FILTERED = [True, False]
THS = np.concatenate((np.linspace(0, 2, 20, endpoint = False), np.linspace(2, 4, 9, endpoint = True)))

propmeth = sys.argv[1] if (len(sys.argv) > 1) and (sys.argv[1] in PROPMETHODS) else PROPMETHODS[0]
dataset = sys.argv[2] if len(sys.argv) > 2 else 'synthetic'

# Load test data
data = datasets.loadDatasets(dataset)

# Try different parameter combinations for interval proposing
ap = OrderedDict()      # Average Precision
mean_ap = OrderedDict() # Mean Average Precision
labels = OrderedDict()  # Labels for the different combinations
for filtered in FILTERED:
    for useMAD in MAD:
        id = (filtered, useMAD)
        ap[id] = {}
        mean_ap[id] = {}
        labels[id] = '{}, {}'.format('median' if useMAD else 'mean', 'gradients' if filtered else 'scores')
        print('Testing {}'.format(labels[id]))
        sys.stdout.flush()
        
        for sd_th in THS:
            ygts = []
            regions = []
            aps = []
            
            propparams = { 'useMAD' : useMAD, 'sd_th' : sd_th }
            if not filtered:
                propparams['filter'] = None
            
            for ftype in data:
                gts = []
                cur_regions = []
                for func in data[ftype]:
                    gts.append(func['gt'])
                    cur_regions.append(maxdiv.maxdiv(func['ts'], method = METHOD, mode = MODE, preproc = 'normalize',
                                                     td_dim = 6, td_lag = 2,
                                                     num_intervals = None, extint_min_len = 20, extint_max_len = 100,
                                                     proposals = propmeth, proposalparameters = propparams))
                aps.append(eval.average_precision(gts, cur_regions))
                ygts += gts
                regions += cur_regions
                
            ap[id][sd_th] = eval.average_precision(ygts, regions)
            mean_ap[id][sd_th] = np.mean(aps)


# Print results as table
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
