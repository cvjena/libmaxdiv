import sys
sys.path.append('..')

import numpy as np

from maxdiv import preproc, eval
from maxdiv.baselines_noninterval import pointwiseRegionProposals
import datasets


# Parse parameters
propmeth = sys.argv[1] if len(sys.argv) > 1 else 'hotellings_t'
dataset = sys.argv[2] if len(sys.argv) > 2 else 'synthetic'
extint_max_len = max(10, int(sys.argv[3])) if len(sys.argv) > 3 else 100
td_dim = max(1, int(sys.argv[4])) if len(sys.argv) > 4 else 1
td_lag = max(1, int(sys.argv[5])) if len(sys.argv) > 5 else 1

# Load test data
data = datasets.loadDatasets(dataset, 'interval')

# Try different thresholds for interval proposing
regions = []
gts = []
for ftype in data:
    for func in data[ftype]:
        gts.append(func['gt'])
        ts = preproc.normalize_time_series(func['ts'])
        if td_dim > 1:
            ts = preproc.td(ts, td_dim, td_lag)
        regions.append(list(pointwiseRegionProposals(ts, method = propmeth, sd_th = -3.0, extint_min_len = 10, extint_max_len = extint_max_len)))

print('Overall AP: {}'.format(eval.average_precision(gts, regions)))
