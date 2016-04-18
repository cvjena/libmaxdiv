import numpy as np
import maxdiv, preproc, eval
import matplotlib.pylab as plt
import argparse, time

# ensure reproducable results
np.random.seed(0)

def sample_gp(length, sigma, n=1, noise=0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    X = np.linspace(0, 1, length, False).reshape([1, length])
    zeroY = np.zeros(length)
    K = maxdiv.calc_gaussian_kernel(X, sigma / length) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(zeroY, K, n)

def sample_gp_with_meanshift(length, anomaly_length, sigma = 5.0, n=1, noise=0.001):
    
    gp = sample_gp(length, sigma, n, noise)
    anom_start = int(np.random.randint(0, length - anomaly_length))
    gp[0,anom_start:anom_start+anomaly_length] -= np.random.rand()*1.0 + 3.0
    return gp, [(anom_start, anom_start + anomaly_length)]


# Set up CLI argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--methods', help='maxdiv method', choices=maxdiv.get_available_methods(), nargs='+', default=[])
parser.add_argument('--preproc', help='use a pre-processing method', default='td', choices=preproc.get_available_methods())
parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'LAMBDA', 'IS_I_OMEGA'], default='I_OMEGA')
parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)

args = parser.parse_args()
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in ('preproc', 'mode', 'alpha')}

methods = set(args.methods) & set(maxdiv.get_available_methods())
if len(methods) == 0:
    methods = maxdiv.get_available_methods()

n = 250 # length of time series
m = 20 # number of time series per anomaly length

# Measure performance with various lengths of anomalies
aucs = { method : [] for method in methods }
aps = { method : [] for method in methods }
ratios = []
for l in range(15, 121, 15):
    
    funcs = [sample_gp_with_meanshift(n, l) for i in range(m)]
    
    for method in methods:
        auc = []
        regions = []
        
        for i in range(m):
            gp, ygt = funcs[i]
            regions.append(maxdiv.maxdiv(gp, method = method, num_intervals = 5, extint_min_len = 10, extint_max_len = 120,
                                         kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters))
            auc.append(eval.auc(ygt, regions[-1], n))
        
        aucs[method].append(np.mean(auc))
        aps[method].append(eval.average_precision([ygt for gp, ygt in funcs], regions))
    
    ratios.append(float(l) / n)

# Plot results
fig_auc = plt.figure()
sp_auc = fig_auc.add_subplot(111, xlabel = 'Length of anomaly / Length of time series', ylabel = 'AUC')
fig_ap = plt.figure()
sp_ap = fig_ap.add_subplot(111, xlabel = 'Length of anomaly / Length of time series', ylabel = 'Average Precision')
for method in methods:
    sp_auc.plot(ratios, aucs[method], marker = 'x', label = method)
    sp_ap.plot(ratios, aps[method], marker = 'x', label = method)
sp_auc.legend(loc = 'lower right')
sp_ap.legend(loc = 'lower right')
fig_auc.savefig('anomaly_ratio_auc.svg')
fig_ap.savefig('anomaly_ratio_ap.svg')
plt.show()