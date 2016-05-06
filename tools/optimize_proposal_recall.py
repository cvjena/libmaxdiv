""" Tries different thresholds for point-wise interval proposing and evaluates precison and recall of the proposed intervals. """

import sys
sys.path.append('..')

import numpy as np
from collections import OrderedDict

from maxdiv import preproc, eval
from maxdiv.baselines_noninterval import pointwiseRegionProposals
import datasets

# Constants
PROPMETHODS = ['hotellings_t', 'kde']
THS = np.concatenate((np.linspace(0, 2, 20, endpoint = False), np.linspace(2, 4, 9, endpoint = True)))

# Parse parameters
dataset = sys.argv[1] if len(sys.argv) > 1 else 'synthetic'
extint_max_len = max(10, int(sys.argv[2])) if len(sys.argv) > 2 else 50

# Load test data
data = datasets.loadDatasets(dataset)

# Try different thresholds for interval proposing
results = OrderedDict()
for propmeth in PROPMETHODS:
    results[propmeth] = OrderedDict()
    for sd_th in THS:
        ygts = []
        regions = []
        
        for ftype in data:
            for func in data[ftype]:
                ygts.append(func['gt'])
                regions.append(list(pointwiseRegionProposals(preproc.td(func['ts']), method = propmeth, sd_th = sd_th,
                                                                        extint_min_len = 10, extint_max_len = extint_max_len)))
            
        results[propmeth][sd_th] = eval.recall_precision(ygts, regions, multiAsFP = False)


# Print results as table
labels = ('Recall', 'Precision', 'F1-Score')
hdiv_len = 5 + sum(len(lbl) + 3 for lbl in labels) # length of horizontal divider

for propmeth, res in results.items():
    print('\n-- {} --\n'.format(propmeth))

    print('     |' + '|'.join(' {} '.format(lbl) for lbl in labels))
    print('{:-<{}s}'.format('', hdiv_len))
    for sd_th, (recall, precision) in res.items():
        row = '{:4.2f} '.format(sd_th)
        for lbl, val in zip(labels, (recall, precision, (2 * precision * recall) / (precision + recall))):
            row += '| {:>{}.4f} '.format(val, len(lbl))
        print(row)
