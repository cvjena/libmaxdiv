import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
from collections import Counter

from maxdiv import maxdiv, preproc, eval
import datasets


# Parameters
method = sys.argv[1] if len(sys.argv) > 1 else 'gaussian_cov'
extremetype = sys.argv[2] if len(sys.argv) > 2 else None
td_lag = int(sys.argv[3]) if len(sys.argv) > 3 else 1

if method == 'help':
    print('{} <method = gaussian_cov> [ <extremetype> [ <td-lag = 1> ] ]'.format(sys.argv[0]))
    exit()

# Load data
data = datasets.loadDatasets('synthetic')
ftypes = [extremetype] if extremetype is not None else data.keys()

# Find the best embedding dimension for every single time series
aucs = {}
aps = {}
all_gt = []
all_regions = []
best_k = {}
for ftype in ftypes:
    print('-- {} --'.format(ftype))

    ygts = []
    regions = []
    aucs[ftype] = []
    best_k[ftype] = Counter()
    for i, func in enumerate(data[ftype]):
        ygts.append(func['gt'])
        k_best, ap_best, auc_best = 0, 0.0, 0.0
        regions_best = []
        for k in range(3, 13):
            detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                                       extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                                       td_dim = k, td_lag = td_lag)
            cur_ap = eval.average_precision([func['gt']], [detections])
            cur_auc = eval.auc(func['gt'], detections, func['ts'].shape[1])
            if (k_best == 0) or (cur_ap > ap_best) or ((cur_ap == ap_best) and (cur_auc > auc_best)):
                k_best, ap_best, auc_best, regions_best = k, cur_ap, cur_auc, detections
        for r in range(len(regions_best) - 1, -1, -1):
            regions_best[r] = (regions_best[r][0], regions_best[r][1], regions_best[r][2] / regions_best[0][2])
        regions.append(regions_best)
        aucs[ftype].append(auc_best)
        best_k[ftype][k_best] += 1
        print ("Best k: {}".format(k_best))
    
    aps[ftype] = eval.average_precision(ygts, regions)
    print ("AP: {}".format(aps[ftype]))
    
    plt.bar(np.asarray(best_k[ftype].keys()) - 0.5, best_k[ftype].values(), 1)
    plt.title(ftype)
    plt.show()
    
    all_regions += regions
    all_gt += ygts

print('-- Best k --')
for ftype, counts in best_k.items():
    print('{}: {} ({} - {})'.format(ftype, max(counts.keys(), key = lambda k: counts[k]), min(counts.keys()), max(counts.keys())))

print('-- Aggregated AUC --')
for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in aps:
    print ("{}: {}".format(ftype, aps[ftype]))
print ("OVERALL AP: {}".format(eval.average_precision(all_gt, all_regions)))