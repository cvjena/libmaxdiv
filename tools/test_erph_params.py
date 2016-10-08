""" Compares the performance of the MDI algorithm with ERPH models for different numbers of histograms and bins. """

import sys, os.path
import subprocess
import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else 'I_OMEGA'

NUM_HIST = (5, 25, 100, 250)
NUM_BINS = (5, 10, 25, 50, 0)
NUM_IT = 5

overall_ap = np.ndarray((len(NUM_HIST), len(NUM_BINS), NUM_IT))
mean_ap = np.ndarray((len(NUM_HIST), len(NUM_BINS), NUM_IT))

for i, num_hist in enumerate(NUM_HIST):
    for j, num_bins in enumerate(NUM_BINS):
        print('Running with {} histograms and {} bins...'.format(num_hist, num_bins))
        sys.stdout.flush()
        
        for k in range(NUM_IT):
            
            output = subprocess.run([
                'python', os.path.join('..', 'experiments', 'synthetic', 'run_tests.py'),
                '--novis', '--datasets', 'synthetic_hd', '--td_dim', '6', '--td_lag', '2',
                '--method', 'erph', '--mode', mode, '--num_hist', str(num_hist), '--num_bins', str(num_bins)],
                stdout = subprocess.PIPE, universal_newlines = True, check = True).stdout
        
            lines = output.split('\n')
            overall_ap[i, j, k] = float(lines[-2].strip().split()[-1])
            mean_ap[i, j, k] = float(lines[-3].strip().split()[-1])

overall_ap = overall_ap.mean(axis = 2)
mean_ap = mean_ap.mean(axis = 2)

print('\n-- OVERALL AP --\n')
for i in range(overall_ap.shape[0]):
    print(','.join(str(overall_ap[i, j]) for j in range(overall_ap.shape[1])))

print('\n-- MEAN AP --\n')
for i in range(mean_ap.shape[0]):
    print(','.join(str(mean_ap[i, j]) for j in range(mean_ap.shape[1])))
