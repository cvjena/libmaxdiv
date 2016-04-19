import numpy as np
import maxdiv, maxdiv_tools, preproc, eval
import argparse, time
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle

# Constants
METHODS = maxdiv.get_available_methods() # available probability density estimators
METHODS.remove('parzen') # 'parzen' was buggy and has been fixed in 'parzen_proper'
MODES = ['OMEGA_I', 'SYM', 'I_OMEGA', 'LAMBDA', 'IS_I_OMEGA'] # KL divergence modes
PREPROC = [None, 'td'] # preprocessing methods

# Set up CLI argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])
parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=10, type=int)
parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=50, type=int)
parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)
parser.add_argument('--num_intervals', help='number of intervals to be retrieved (0 = all)', default=5, type=int)

args = parser.parse_args()

# Prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters() if parameter_name in args_dict}
if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
    parameters['num_intervals'] = None

# Load synthetic test data
with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

# Determine set of extreme types to run tests for
extremetypes = set(args.extremetypes) & set(f.keys())
if len(extremetypes) == 0:
    extremetypes = f.keys()


# Try all combinations of preprocessing methods, density estimators and divergence modes
# for all types of extremes and store the results in dictionaries
auc = {}    # Area under ROC curve
auc_sd = {} # Standard deviation of AUC scores
aps = {}    # Average Precision
times = { method : [] for method in METHODS } # Lists of runtimes for each method
labels = {} # Labels for the different combinations
for fi, ftype in enumerate(extremetypes):

    print('-- Testing on {} ({}/{}) --'.format(ftype, fi+1, len(extremetypes)))
    funcs = f[ftype]
    ygts = y[ftype]
    
    for preproc in PREPROC:
        for method in METHODS:
            for mode in MODES:
                
                id = (preproc, method, mode)
                if id not in auc:
                    auc[id] = {}
                    auc_sd[id] = {}
                    aps[id] = {}
                    labels[id] = '{}, {}, {} preproc.'.format(method, mode, preproc if preproc is not None else 'no')
                
                print('- {} -'.format(labels[id]))
            
                aucs = []
                regions = []
                
                for i in range(len(funcs)):
                    func = funcs[i]
                    ygt = ygts[i]
                    time_start = time.time()
                    regions.append(maxdiv.maxdiv(func,
                                                 method = method, preproc = preproc, mode = mode,
                                                 kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters))
                    time_stop = time.time()
                    
                    aucs.append(eval.auc(ygt, regions[-1]))
                    if (preproc is None) and (func.shape[1] == 250):
                        times[method].append(time_stop - time_start)
                
                auc[id][ftype] = np.mean(aucs)
                auc_sd[id][ftype] = np.std(aucs)
                aps[id][ftype] = eval.average_precision(ygts, regions)


# Store test results on disk
with open('benchmark_results.pickle', 'wb') as fout:
    pickle.dump({ 'auc' : auc, 'auc_sd' : auc_sd, 'aps' : aps, 'times' : times }, fout)


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