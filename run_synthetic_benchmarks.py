import numpy as np
from maxdiv import maxdiv, preproc, eval
import maxdiv_tools, datasets
import sys, argparse, time

# Constants
METHODS = maxdiv.get_available_methods() # available probability density estimators
MODES = ['I_OMEGA', 'OMEGA_I', 'SYM', 'TS', 'JSD'] # Divergence modes
PREPROC = [None, 'td'] # preprocessing methods
canonical_order = ['meanshift', 'meanshift_hard', 'meanshift5', 'meanshift5_hard', 'amplitude_change', 'frequency_change', 'mixed',
                   'meanshift_multvar', 'amplitude_change_multvar', 'frequency_change_multvar', 'mixed_multvar']

# Set up CLI argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])
parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=20, type=int)
parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=100, type=int)
parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)
parser.add_argument('--num_intervals', help='number of intervals to be retrieved (0 = all)', default=0, type=int)
parser.add_argument('--csv', help='Format output as CSV instead of tables.', action='store_true')

args = parser.parse_args()

# Prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters() if parameter_name in args_dict}
if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
    parameters['num_intervals'] = None

# Load synthetic test data
data = datasets.loadDatasets('synthetic')

# Determine set of extreme types to run tests for
extremetypes = set(args.extremetypes) & set(data.keys())
if len(extremetypes) == 0:
    extremetypes = data.keys()


# Try all combinations of preprocessing methods, density estimators and divergence modes
# for all types of extremes and store the results in dictionaries
auc = {}    # Area under ROC curve
auc_sd = {} # Standard deviation of AUC scores
aps = {}    # Average Precision
times = { method : [] for method in METHODS } # Lists of runtimes for each method
labels = {} # Labels for the different combinations
all_gt = {}
all_regions = {}
for fi, ftype in enumerate(extremetypes):
    sys.stderr.write('-- Testing on {} ({}/{}) --\n'.format(ftype, fi+1, len(extremetypes)))
    
    if ftype not in canonical_order:
        canonical_order.append(ftype)
    
    for preproc in PREPROC:
        for method in METHODS:
            for mode in MODES:
                
                if (mode == 'TS') and (method != 'gaussian_cov'):
                    continue
                
                id = (preproc, method, mode)
                if id not in auc:
                    auc[id] = {}
                    auc_sd[id] = {}
                    aps[id] = {}
                    all_gt[id] = []
                    all_regions[id] = []
                    labels[id] = '{}, {}, td = {}'.format(method, mode, 3 if preproc is not None else 1)
                
                sys.stderr.write('- {} -\n'.format(labels[id]))
            
                aucs = []
                regions = []
                ygts = []
                
                for i, func in enumerate(data[ftype]):
                    time_start = time.time()
                    regions.append(maxdiv.maxdiv(func['ts'], useLibMaxDiv = True,
                                                 method = method, preproc = preproc, mode = mode,
                                                 kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters))
                    time_stop = time.time()
                    
                    ygts.append(func['gt'])
                    aucs.append(eval.auc(func['gt'], regions[-1], func['ts'].shape[1]))
                    if (preproc is None) and (func['ts'].shape[1] == 250):
                        times[method].append(time_stop - time_start)
                
                auc[id][ftype] = np.mean(aucs)
                auc_sd[id][ftype] = np.std(aucs)
                aps[id][ftype] = eval.average_precision(ygts, regions)
                
                all_gt[id] += ygts
                all_regions[id] += regions


# Store test results on disk
#with open('benchmark_results.pickle', 'wb') as fout:
#    pickle.dump({ 'auc' : auc, 'auc_sd' : auc_sd, 'aps' : aps, 'times' : times }, fout)


if args.csv:
    
    # Print results as CSV
    print('--- AP ---\n')
    header = 'method'
    for ftype in canonical_order:
        if ftype in extremetypes:
            header += ';{}'.format(ftype)
    header += ';OVERALL'
    print(header)
    for preproc in PREPROC:
        for method in METHODS:
            for mode in MODES:
                id = (preproc, method, mode)
                if id in labels:
                    row = labels[id]
                    for ftype in canonical_order:
                        if ftype in extremetypes:
                            row += ';{:.3f}'.format(aps[id][ftype])
                    row += ';{:.3f}'.format(eval.average_precision(all_gt[id], all_regions[id]))
                    print(row)
    
    print('\n')
    
    print('--- AUC ---\n')
    header = 'method'
    for ftype in canonical_order:
        if ftype in extremetypes:
            header += ';{}'.format(ftype)
    print(header)
    for preproc in PREPROC:
        for method in METHODS:
            for mode in MODES:
                id = (preproc, method, mode)
                if id in labels:
                    row = labels[id]
                    for ftype in canonical_order:
                        if ftype in extremetypes:
                            row += ';{:.3f}'.format(auc[id][ftype])
                    print(row)
    
    print('\n')
    
    print('--- AUC SD ---\n')
    header = 'method'
    for ftype in canonical_order:
        if ftype in extremetypes:
            header += ';{}'.format(ftype)
    print(header)
    for preproc in PREPROC:
        for method in METHODS:
            for mode in MODES:
                id = (preproc, method, mode)
                if id in labels:
                    row = labels[id]
                    for ftype in canonical_order:
                        if ftype in extremetypes:
                            row += ';{:.3f}'.format(auc_sd[id][ftype])
                    print(row)

else:

    # Print results as tables
    maxLabelLen = max(len(lbl) for lbl in labels.values())
    hdiv_len = maxLabelLen + 1 + sum(max(len(ftype), 18) + 3 for ftype in extremetypes) # length of horizontal divider
    fmtHeader = '{:^' + str(maxLabelLen+1) + 's}|' + '|'.join(' {:^18s} '.format(ftype) for ftype in extremetypes)

    print('\n')

    print(fmtHeader.format('AUC'))
    print('{:=<{}s}'.format('', hdiv_len))
    num = 0
    for preproc in PREPROC:
        for method in METHODS:
            if num > 0:
                print('{:-<{}s}'.format('', hdiv_len))
            for mode in MODES:
                id = (preproc, method, mode)
                if id in labels:
                    row = '{:{}s} '.format(labels[id], maxLabelLen)
                    for ftype in extremetypes:
                        row += '|{:>{}s} '.format('{:.3f} (+/- {:.3f})'.format(auc[id][ftype], auc_sd[id][ftype]), max(len(ftype), 18) + 1)
                    print(row)
            num += 1

    print('\n')

    print(fmtHeader.format('Average Precision'))
    print('{:=<{}s}'.format('', hdiv_len))
    num = 0
    for preproc in PREPROC:
        for method in METHODS:
            if num > 0:
                print('{:-<{}s}'.format('', hdiv_len))
            for mode in MODES:
                id = (preproc, method, mode)
                if id in labels:
                    row = '{:{}s} '.format(labels[id], maxLabelLen)
                    for ftype in extremetypes:
                        row += '| {:{}.4f} '.format(aps[id][ftype], max(len(ftype), 18))
                    print(row)
            num += 1

    print('\n')

    maxMethodLen = max(len(m) for m in METHODS)
    print('Mean runtime for series of length 250')
    print('-------------------------------------')
    for method in METHODS:
        print('{:{}s} | {:{}.3f} s'.format(method, maxMethodLen, np.mean(times[method]), 37 - 5 - maxMethodLen))