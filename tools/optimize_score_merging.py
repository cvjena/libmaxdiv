""" Tries different coefficients for a linear combination of divergence and proposals scores (`score_merge_coeff` parameter). """

import sys
sys.path.append('..')

import numpy as np
import maxdiv, datasets, eval
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle

# Constants
PROPMETHODS = ['hotellings_t', 'kde']
METHODS = ['parzen','gaussian_cov']
MODES = ['I_OMEGA', 'JSD'] # KL divergence modes
COEFFS = np.linspace(0, 1, 21, endpoint = True)

propmeth = sys.argv[1] if (len(sys.argv) > 1) and (sys.argv[1] in PROPMETHODS) else PROPMETHODS[0]
dataset = sys.argv[2] if len(sys.argv) > 2 else 'synthetic'

# Load test data
data = datasets.loadDatasets(dataset)


# Try all combinations of methods and divergence modes with different combination coefficients
ap = {}         # Average Precision
mean_ap = {}    # Mean Average Precision
labels = {}     # Labels for the different combinations
for method in METHODS:
    for mode in MODES:
        id = (method, mode)
        ap[id] = {}
        mean_ap[id] = {}
        labels[id] = '{} ({})'.format(method, mode)
        print('Testing {}'.format(labels[id]))
        
        for coeff in COEFFS:
            ygts = []
            regions = []
            aps = []
            
            for ftype in data:
                gts = []
                cur_regions = []
                for func in data[ftype]:
                    gts.append(func['gt'])
                    cur_regions.append(maxdiv.maxdiv(func['ts'], method = method, mode = mode, preproc = 'td',
                                                     num_intervals = None, extint_min_len = 10, extint_max_len = 50,
                                                     proposals = propmeth, score_merge_coeff = coeff))
                aps.append(eval.average_precision(gts, cur_regions))
                ygts += gts
                regions += cur_regions
                
            ap[id][coeff] = eval.average_precision(ygts, regions)
            mean_ap[id][coeff] = np.mean(aps)


# Print results as table
maxLabelLen = max(len(lbl) for lbl in labels.values())
hdiv_len = 5 + sum(len(lbl) + 3 for lbl in labels.values()) # length of horizontal divider

print('\n-- Overall Average Precision --\n')

print('     |' + '|'.join(' {} '.format(lbl) for lbl in labels.values()))
print('{:-<{}s}'.format('', hdiv_len))
for coeff in COEFFS:
    row = '{:4.2f} '.format(coeff)
    for id, aps in ap.items():
        row += '| {:>{}.4f} '.format(aps[coeff], len(labels[id]))
    print(row)

print('\n-- Mean Average Precision --\n')

print('     |' + '|'.join(' {} '.format(lbl) for lbl in labels.values()))
print('{:-<{}s}'.format('', hdiv_len))
for coeff in COEFFS:
    row = '{:4.2f} '.format(coeff)
    for id, aps in mean_ap.items():
        row += '| {:>{}.4f} '.format(aps[coeff], len(labels[id]))
    print(row)