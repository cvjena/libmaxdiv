import maxdiv
import csv
import numpy as np

m = []
with open('/Users/erik/Documents/kernel_matrix.csv', 'r') as f:
    csvf = csv.DictReader(f)
    for line in csvf:
        v = [ float(line['x{}'.format(i)]) for i in range(1,301) ]
        m.append(v)

K = np.vstack(m)
intervals = maxdiv.maxdiv_parzen_proper_sampling(K, mode="OMEGA_I", alpha=1.0, extint_min_len = 2, extint_max_len = 100)
regions = maxdiv.calc_max_nonoverlapping_regions(intervals, 10, 2)

print regions
