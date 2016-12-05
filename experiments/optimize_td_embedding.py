""" Evaluates various methods for automatic optimization of time-delay embedding parameters. """


import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import argparse

from sklearn.gaussian_process import GaussianProcess

from maxdiv import maxdiv, maxdiv_util, preproc, eval
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


def rank_aggregation(func, method, td_lag):
    
    # Collect scores for all intervals with various embedding dimensions
    regions = {}
    for k in range(3, 21):
        detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                                   extint_min_len = 20, extint_max_len = 100, num_intervals = None, overlap_th = 1.0,
                                   td_dim = k, td_lag = td_lag)
        for a, b, score in detections:
            if (a, b) not in regions:
                regions[(a, b)] = np.zeros(18)
            regions[(a, b)][k - 3] = score
    
    # Sort detections by Approximate Kemeny Rank Aggregation
    # (an interval is preferred over another one if the majority of rankers does so)
    detections = sorted(regions.keys(), key = lambda intvl: KemenyCompare(regions, intvl), reverse = True)
    
    # Assign inverse rank as detection score
    for i, (a, b) in enumerate(detections):
        detections[i] = (a, b, len(detections) - i)
    
    return maxdiv.find_max_regions(detections), 0

class KemenyCompare:
    def __init__(self, regions, intvl):
        self.regions = regions
        self.intvl = intvl
    def cmp(self, other):
        return (self.regions[self.intvl] > self.regions[other.intvl]).sum() - (self.regions[self.intvl] < self.regions[other.intvl]).sum()
    def __lt__(self, other):
        return self.cmp(other) < 0
    def __gt__(self, other):
        return self.cmp(other) > 0
    def __eq__(self, other):
        return self.cmp(other) == 0
    def __le__(self, other):
        return self.cmp(other) <= 0
    def __ge__(self, other):
        return self.cmp(other) >= 0
    def __ne__(self, other):
        return self.cmp(other) != 0


def td_from_mi(func, method, td_lag):
    
    # Determine Time Lag with minimum Mutual Information
    k = min(range(2, int(0.05 * func['ts'].shape[1])), key = lambda k: mutual_information(func['ts'], 2, k - 1)) // td_lag
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_relative_mi(func, method, td_lag, th = 0.05):
    
    # Determine Time Lag based on "normalized" Mutual Information
    rmi = np.array([mutual_information(func['ts'], 2, d) for d in range(1, int(0.05 * func['ts'].shape[1]))])
    rmi /= rmi[0]
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


def td_from_relative_ce(func, method, td_lag, th = 0.005):
    
    # Determine Time Lag based on "normalized" Mutual Information
    rce = np.array([conditional_entropy(func['ts'], d, td_lag) for d in range(1, int(0.05 * func['ts'].shape[1] / td_lag))])
    rce /= rce[0]
    drce = np.convolve(rce, [-1, 0, 1], 'valid')
    if np.any(drce <= th):
        k = (np.where(drce <= th)[0][0] + 2)
    else:
        k = (drce.argmin() + 2)
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_ce_gradient(func, method, td_lag, th = 0.001):
    
    # Determine Time Lag based on the steepness of decrease of conditional entropy
    ce = np.array([conditional_entropy(func['ts'], d, td_lag) for d in range(1, int(0.05 * func['ts'].shape[1] / td_lag))])
    dce = np.convolve(ce, [-1, 0, 1], 'valid')
    if np.any(dce <= th):
        k = (np.where(dce <= th)[0][0] + 2)
    else:
        k = (dce.argmin() + 2)
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_length_scale(func, method, td_lag, factor = 0.3):
    
    # Determine Length Scale of Gaussian Process
    ls = length_scale(func['ts'])
    # Set Embedding Dimension
    k = int(max(1, min(0.05 * func['ts'].shape[1], round(factor * ls / td_lag))))
    # Detect regions
    detections = maxdiv.maxdiv(func['ts'], method = method, mode = 'I_OMEGA',
                               extint_min_len = 20, extint_max_len = 100, num_intervals = None,
                               td_dim = k, td_lag = td_lag)
    return detections, k


def td_from_false_neighbors(func, method, td_lag, Rtol = 1.0, Ntol = 0.001):
    
    d, n = func['ts'].shape
    Rtol2 = Rtol * Rtol
    
    # Determine embedding dimension based on false nearest neighbors
    dist = maxdiv_util.calc_distance_matrix(func['ts'])
    cumdist = dist.copy()
    fnn = []
    max_k = int(0.05 * func['ts'].shape[1])
    for k in range(1, max_k + 1):
        
        cur_fnn = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                id = max(0, i - k * td_lag)
                jd = max(0, j - k * td_lag)
                if dist[id, jd] / cumdist[i, j] > Rtol2:
                    cur_fnn += 1
                cumdist[i, j] += dist[id, jd]
        fnn.append(cur_fnn)
        
        if (len(fnn) >= 3) and (abs(fnn[-3] - fnn[-1]) <= Ntol * abs(fnn[0] - fnn[2])):
            k -= 2
            break
    
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
            return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (np.log(2 * np.pi) + 1 + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([ts[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the marginal distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cov_indep = cov.copy()
    cov_indep[:d, d:] = 0
    cov_indep[d:, :d] = 0
    
    # Compute KL divergence between p(x_t, x_(t-T), ..., x_(t - (k-1)*T)) and p(x_t)*p(x_(t-L), ..., x_(t - (k-1)*T))
    return (np.linalg.inv(cov_indep).dot(cov).trace() + np.linalg.slogdet(cov_indep)[1] - np.linalg.slogdet(cov)[1] - embed_func.shape[0]) / 2


def conditional_entropy(ts, k, T = 1):
    
    d, n = ts.shape
    
    if (k < 2) or (T < 1):
        # Entropy as a special case
        cov = np.cov(ts)
        if d > 1:
            return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cov)[1]) / 2
        else:
            return (d * (np.log(2 * np.pi) + 1) + np.log(cov)) / 2
    
    # Time-Delay Embedding with the given embedding dimension and time lag
    embed_func = np.vstack([ts[:, ((k - i - 1) * T):(n - i * T)] for i in range(k)])
    
    # Compute parameters of the joint and the conditioned distributions assuming a normal distribution
    cov = np.cov(embed_func)
    cond_cov = cov[:d, :d] - cov[:d, d:].dot(np.linalg.inv(cov[d:, d:]).dot(cov[d:, :d]))
    
    # Compute the conditional entropy H(x_t | x_(t-T), ..., x_(t - (k-1)*T))
    return (d * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(cond_cov)[1]) / 2


def length_scale(ts):
    
    X = np.linspace(0, 1, ts.shape[1], endpoint = True).reshape(ts.shape[1], 1)
    GP = GaussianProcess(thetaL = 0.1, thetaU = 1000, nugget = 1e-8, normalize = False)
    GP.fit(X, ts.T)
    return np.sqrt(0.5 / GP.theta_.flat[0]) * ts.shape[1]


# Constants
optimizers = {
    'best_k'            : find_best_k,
    'rank_aggregation'  : rank_aggregation,
    'mi'                : td_from_mi,
    'rmi'               : td_from_relative_mi,
    'dmi'               : td_from_mi_gradient,
    'ce'                : td_from_ce_gradient,
    'rce'               : td_from_relative_ce,
    'gp_ls'             : td_from_length_scale,
    'fnn'               : td_from_false_neighbors
}

# Parameters
parser = argparse.ArgumentParser(description = 'Evaluate various methods for automatic optimization of time-delay embedding parameters.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', help='MaxDiv method', choices = maxdiv.get_available_methods() + ['gaussian_ts'], default = 'gaussian_ts')
parser.add_argument('--optimizer', help='Optimization method for Time-Delay Embedding', choices = optimizers.keys(), default = 'best_k')
parser.add_argument('--plot', action='store_true', help='Plot histograms of embedding dimensions for each extreme type')
parser.add_argument('--datasets', help='datasets to be loaded', nargs='+', default=['synthetic'])
parser.add_argument('--subsets', help='subsets of the datasets to be tested', nargs='+',default=[])
parser.add_argument('--td_lag', help='Time-Lag for Time-Delay Embedding', default=1, type=int)
parser.add_argument('--dump', help='Dump detections for each time-series to the specified CSV file', default='')
args = parser.parse_args()

# Load data
data = datasets.loadDatasets(args.datasets)
ftypes = args.subsets if len(args.subsets) > 0 else data.keys()

# Find the best embedding dimension for every single time series
aucs = {}
aps = {}
all_ids = []
all_gt = []
all_regions = []
best_k = {}
for ftype in ftypes:
    print('-- {} --'.format(ftype))

    func_ids = []
    ygts = []
    regions = []
    aucs[ftype] = []
    best_k[ftype] = Counter()
    for i, func in enumerate(data[ftype]):
        func_ids.append('{}_{:03d}'.format(ftype, i))
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
    
    all_ids += func_ids
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

# Dump detections
if args.dump:
    with open(args.dump, 'w') as dumpFile:
        dumpFile.write('Func,Start,End,Score\n')
        for id, regions in zip(all_ids, all_regions):
            for a, b, score in regions:
                dumpFile.write('{},{},{},{}\n'.format(id, a, b, score))
