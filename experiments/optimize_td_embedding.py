import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import argparse

from sklearn.gaussian_process import GaussianProcess

from maxdiv import maxdiv, preproc, eval
import datasets


def find_best_k(func, method, td_lag):
    
    # Find embedding dimension which maximizes AP
    k_best, ap_best, auc_best = 0, 0.0, 0.0
    regions_best = []
    for k in range(3, 21):
        detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                                   extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                                   td_dim = k, td_lag = td_lag)
        cur_ap = eval.average_precision([func['gt']], [detections])
        cur_auc = eval.auc(func['gt'], detections, func['ts'].shape[1])
        if (k_best == 0) or (cur_ap > ap_best) or ((cur_ap == ap_best) and (cur_auc > auc_best)):
            k_best, ap_best, auc_best, regions_best = k, cur_ap, cur_auc, detections
    
    return regions_best, k_best


def td_from_mi(func, method, td_lag):
    
    # Determine Time Lag with minimum Mutual Information
    k = min(range(2, int(0.05 * func['ts'].shape[1])), key = lambda k: mutual_information(func['ts'], 2, k - 1)) // td_lag
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_relative_mi(func, method, td_lag, th = 0.0002):
    
    th *= func['ts'].shape[0]
    # Determine Time Lag based on the ratio of Mutual Information and Entropy
    entropy = mutual_information(func['ts'], 1)
    rmi = np.array([mutual_information(func['ts'], 2, d) / entropy for d in range(1, int(0.05 * func['ts'].shape[1]))])
    drmi = np.convolve(rmi, [-1, 0, 1], 'valid')
    if np.any(drmi <= th):
        k = (np.where(drmi <= th)[0][0] + 3) // td_lag
    else:
        k = (drmi.argmin() + 3) // td_lag
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_mi_gradient(func, method, td_lag, th = 0.15):
    
    th *= func['ts'].shape[0]
    # Determine Time Lag based on the steepness of decrease of mutual information
    mi = np.array([mutual_information(func['ts'], 2, d) for d in range(1, int(0.05 * func['ts'].shape[1]))])
    dmi = np.convolve(mi, [-1, 0, 1], 'valid')
    if np.any(dmi <= th):
        k = (np.where(dmi <= th)[0][0] + 3) // td_lag
    else:
        k = (dmi.argmin() + 3) // td_lag
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_length_scale(func, method, td_lag):
    
    # Determine Length Scale of Gaussian Process
    ls = length_scale(func['ts'])
    # Set Embedding Dimension
    k = int(max(1, min(0.05 * func['ts'].shape[1], round(ls / td_lag))))
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def mutual_information(ts, k, T = 1):
    
    d, n = ts.shape
    
    if (k < 2) or (T < 1):
        # Entropy as a special case of MI
        cov = np.cov(ts)
        if d > 1:
            return (n * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (n * (np.log(2 * np.pi) + 1) + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([ts[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the marginal distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cov_indep = cov.copy()
    cov_indep[:d, d:] = 0
    cov_indep[d:, :d] = 0
    
    # Compute KL divergence between p(x_t, x_(t-T), ..., x_(t - (k-1)*T)) and p(x_t)*p(x_(t-L), ..., x_(t - (k-1)*T))
    return (np.linalg.inv(cov_indep).dot(cov).trace() + np.linalg.slogdet(cov_indep)[1] - np.linalg.slogdet(cov)[1] - embed_func.shape[0]) / 2


def length_scale(ts):
    
    scales = []
    
    X = np.arange(0, ts.shape[1]).reshape((ts.shape[1], 1))
    GP = GaussianProcess(thetaL = 0.1, thetaU = 100, nugget = 1e-8)
    for d in range(ts.shape[0]):
        GP.fit(X, ts[d, :].T)
        scales.append((GP.theta_, np.exp(GP.reduced_likelihood_function_value_)))
    
    norm = np.sum([w for s, w in scales])
    return np.sum([s * w / norm for s, w in scales])


# Constants
optimizers = { 'best_k' : find_best_k, 'mi' : td_from_mi, 'rmi' : td_from_relative_mi, 'dmi' : td_from_mi_gradient, 'gp_ls' : td_from_length_scale }

# Parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', help='MaxDiv method', choices = maxdiv.get_available_methods() + ['gaussian_ts'], default = 'gaussian_cov')
parser.add_argument('--optimizer', help='Optimization method for Time-Delay Embedding', choices = optimizers.keys(), default = 'best_k')
parser.add_argument('--plot', action='store_true', help='Plot histograms of embedding dimensions for each extreme type')
parser.add_argument('--datasets', help='datasets to be loaded', nargs='+', default=['synthetic'])
parser.add_argument('--subsets', help='subsets of the datasets to be tested', nargs='+',default=[])
parser.add_argument('--td_lag', help='Time-Lag for Time-Delay Embedding', default=1, type=int)
args = parser.parse_args()

# Load data
data = datasets.loadDatasets(args.datasets)
ftypes = args.subsets if len(args.subsets) > 0 else data.keys()

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
        
        det, k_best = optimizers[args.optimizer](func, args.method, args.td_lag)
        
        # Divide scores by maximum score since their range differs widely depending on the dimensionality
        if args.method not in ('gaussian_cov_ts', 'gaussian_ts'):
            for r in range(len(det) - 1, -1, -1):
                det[r] = (det[r][0], det[r][1], det[r][2] / det[0][2])
        
        regions.append(det)
        aucs[ftype].append(eval.auc(func['gt'], det, func['ts'].shape[1]))
        best_k[ftype][k_best] += 1
        print ("Best k: {}".format(k_best))
    
    aps[ftype] = eval.average_precision(ygts, regions)
    print ("AP: {}".format(aps[ftype]))
    
    if args.plot:
        plt.bar(np.array(list(best_k[ftype].keys())) - 0.5, list(best_k[ftype].values()), 1)
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