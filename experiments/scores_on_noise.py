import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
import scipy.stats

from maxdiv import maxdiv, maxdiv_util, preproc, eval


# Parameters
method = sys.argv[1] if len(sys.argv) > 1 else 'gaussian_cov'
try:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000 # number of samples
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 1  # dimensionality of samples
    td_embed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dataset = 'noise'
except ValueError:
    dataset = sys.argv[2]
    td_embed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
extint_min_len = 5   # minimum interval length
extint_max_len = 100 # maximum interval length

if method == 'help':
    print('Testing on noise (default): {} <method = gaussian_cov> <n = 1000> <dim = 1> <td-embed = 1>'.format(sys.argv[0]))
    print('Testing on synthetic data:  {} <method = gaussian_cov> <dataset> <td-embed = 1>'.format(sys.argv[0]))
    print('Methods: gaussian_cov, gaussian_cov_ts, gaussian_global_cov, gaussian_id_cov, gaussian_id_cov_normalized, parzen, compare')
    exit()

methods = ['parzen', 'gaussian_id_cov', 'gaussian_cov', 'gaussian_cov_ts'] if method == 'compare' else [method]

if dataset == 'noise':
    # Sample a time series consisting of pure white noise
    np.random.seed(0)
    ts = np.random.randn(dim, n)
else:
    import datasets
    ts = datasets.loadSyntheticTestbench()[dataset][0]['ts']

# Retrieve scores for all intervals
pts = preproc.td(ts, td_embed) if td_embed > 1 else ts
scores = dict()
for meth in methods:
    proposals = maxdiv.denseRegionProposals(pts, extint_min_len, extint_max_len)
    if meth == 'gaussian_id_cov_normalized':
        norm_scores = maxdiv.maxdiv_gaussian(pts, proposals, mode = 'I_OMEGA', gaussian_mode = 'ID_COV')
        # Compute theoretical means and standard deviations of the chi^2 distributions
        X = np.arange(extint_min_len, extint_max_len + 1)
        scales = 1.0 / X - 1.0 / (pts.shape[1] - X)
        chi_mean = pts.shape[0] * scales
        chi_sd = np.sqrt(2 * pts.shape[0] * (scales ** 2))
        # Normalize scores
        for i, (a, b, score) in enumerate(norm_scores):
            ind = b - a - extint_min_len
            norm_scores[i] = (a, b, (score - chi_mean[ind]) / chi_sd[ind])
        # Add a constant offset to avoid negative scores
        base_score = min(score for _, _, score in norm_scores) - 0.01
        for i, (a, b, score) in enumerate(norm_scores):
            norm_scores[i] = (a, b, score - base_score)
        scores[meth] = norm_scores
    elif meth == 'gaussian_cov_ts':
        scores[meth] = maxdiv.maxdiv_gaussian(pts, proposals, mode = 'TS', gaussian_mode = 'COV')
        # de-normalize
        chi_mean = (pts.shape[0] * (pts.shape[0] + 3)) / 2
        chi_sd = np.sqrt(2 * chi_mean)
        for i, (a, b, score) in enumerate(scores[meth]):
            scores[meth][i] = (a, b, score * chi_sd + chi_mean)
    elif meth.startswith('gaussian'):
        scores[meth] = maxdiv.maxdiv_gaussian(pts, proposals, mode = 'I_OMEGA', gaussian_mode = meth[9:].upper())
    elif meth == 'parzen':
        K = maxdiv_util.calc_gaussian_kernel(pts)
        scores[meth] = maxdiv.maxdiv_parzen(K, proposals, mode = 'I_OMEGA')
    else:
        print('Unknown method: {}'.format(meth))
        exit()

if method != 'compare':

    # Plot top 5 detections
    detections = maxdiv.find_max_regions(scores[method])
    eval.plotDetections(ts, detections[:5])

    # Plot histogram of the length of detected intervals
    plt.title('Histogram of the Length of Detected Intervals for Method "{}"'.format(method))
    plt.hist([b - a for a, b, score in detections])
    plt.show()

# Plot average score against interval length
if method == 'compare':
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Average Score depending on Interval Length')
    ax[1].set_title('Variance depending on Interval Length')
else:
    plt.title('Average Score depending on Interval Length for Method "{}"'.format(method))

for i, meth in enumerate(methods):
    avg_scores = np.zeros(extint_max_len - extint_min_len + 1)
    var_scores = np.zeros(len(avg_scores))
    num_scores = np.zeros(len(avg_scores))
    for a, b, score in scores[meth]:
        ind = b - a - extint_min_len
        avg_scores[ind] += score
        var_scores[ind] += score * score
        num_scores[ind] += 1
    avg_scores /= num_scores
    var_scores = var_scores / num_scores - avg_scores * avg_scores
    X = np.arange(extint_min_len, extint_max_len + 1)
    if method != 'compare':
        plt.errorbar(X, avg_scores, fmt = '-', yerr = var_scores, ecolor = '#A0A0A0', errorevery = 5, label = 'Empirical mean and variance')
    else:
        ax[0].plot(X, avg_scores, label = meth)
        ax[1].plot(X, var_scores, label = meth)

if method in ('gaussian_id_cov', 'gaussian_global_cov'):
    # Plot theoretical means and variances of the chi^2 distributions
    scales = 1.0 / X - 1.0 / (pts.shape[1] - X)
    plt.plot(X, pts.shape[0] * scales, '--r', label = 'Theoretical chi^2 mean')
    if td_embed <= 1:
        plt.plot(X, pts.shape[0] * scales + 2 * pts.shape[0] * (scales ** 2), '--', color = '#A0A0A0', label = 'Theoretical chi^2 variance')
        plt.plot(X, pts.shape[0] * scales - 2 * pts.shape[0] * (scales ** 2), '--', color = '#A0A0A0')
elif method == 'gaussian_cov_ts':
    # Plot theoretical mean and variance of the chi^2 distribution
    mean = pts.shape[0] + (pts.shape[0] * (pts.shape[0] + 1)) / 2
    variance = 2 * mean
    plt.plot(X, [mean] * len(X), '--r', label = 'Theoretical chi^2 mean')
    plt.plot(X, [mean + variance] * len(X), '--', color = '#A0A0A0', label = 'Theoretical chi^2 variance')
    plt.plot(X, [mean - variance] * len(X), '--', color = '#A0A0A0')
elif method not in ('compare', 'gaussian_id_cov_normalized'):
    # Try to estimate the functional form of the mean and variance based on the scores
    mean = 1.0 / X - 1.0 / (pts.shape[1] - X)
    scale, offs = np.linalg.lstsq(np.vstack((mean, np.ones(len(mean)))).T, avg_scores)[0]
    vscale, voffs = np.linalg.lstsq(np.vstack((mean ** 2, np.ones(len(mean)))).T, var_scores)[0]
    plt.plot(X, mean * scale + offs, '--r', label = 'Guessed mean')
    plt.plot(X, mean * scale + offs + (mean ** 2) * vscale + voffs, '--', color = '#A0A0A0', label = 'Guessed variance')
    plt.plot(X, mean * scale + offs - (mean ** 2) * vscale + voffs, '--', color = '#A0A0A0')
    #gamma_params = [scipy.stats.gamma.fit([score for a, b, score in scores if b - a == x]) for x in X]
    #mean = np.array([scipy.stats.gamma.mean(*params) for params in gamma_params])
    #variance = np.array([scipy.stats.gamma.var(*params) for params in gamma_params])
    #plt.plot(X, mean, '--r', label = 'Guessed mean')
    #plt.plot(X, mean + variance, '--', color = '#A0A0A0', label = 'Guessed variance')
    #plt.plot(X, mean - variance, '--', color = '#A0A0A0')

if method == 'compare':
    ax[0].set_ylabel('Average Score')
    ax[1].set_ylabel('Variance')
    for axis in ax:
        axis.set_xlabel('Interval Length')
        axis.legend()
else:
    plt.xlabel('Interval Length')
    plt.ylabel('Average Score')
    plt.legend()
plt.show()

if method != 'compare':
    scores = scores[method]

    # Analyze the distribution of scores
    plt.title('Distribution of Scores for Method "{}"'.format(method))
    plt.hist([score for a, b, score in scores], 100, normed = True)
    if method == 'gaussian_cov_ts':
        X = np.linspace(plt.xlim()[0], plt.xlim()[1], 250)
        plt.plot(X, scipy.stats.chi2.pdf(X, (pts.shape[0] * (pts.shape[0] + 3)) / 2), '-r', lw = 2)
    plt.show()

    # Compare distribution of scores with chi-squared or gamma distribution
    fig, ax = plt.subplots(2, 2)
    for i, l in enumerate([10, 20, 40, 80]):
        axes = ax.flat[i]
        axes.set_title('Distribution of Scores for Intervals of Length {}'.format(l))
        axes.hist([score for a, b, score in scores if b - a == l], 100, normed = True)
        X = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 250)
        if method in ['gaussian_id_cov', 'gaussian_global_cov']:
            # scores are chi2 distributed
            axes.plot(X, scipy.stats.chi2.pdf(X, pts.shape[0], scale = (1.0/l + 1.0/(l - pts.shape[1]))), '-r', lw = 2)
        elif method == 'gaussian_cov_ts':
            # scores are chi2 distributed independent of the length of the intervals
            axes.plot(X, scipy.stats.chi2.pdf(X, (pts.shape[0] * (pts.shape[0] + 3)) / 2), '-r', lw = 2)
        else:
            # scores are gamma distributed
            # [memo: just a wild guess for the scale parameter: (td_embed + 1) * (1/l + 1/(l - n))]
            gamma_params = scipy.stats.gamma.fit([score for a, b, score in scores if b - a == l], floc = pts.shape[0])
            print(gamma_params)
            axes.plot(X, scipy.stats.gamma.pdf(X, *gamma_params), '-r', lw = 2)
    plt.show()