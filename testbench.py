""" Synthetic Test Bench for MaxDiv """

import maxdiv
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle

# ensure reproducable results
np.random.seed(0)

def sample_gp(X, meany, sigma, n=1, noise=0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    K = maxdiv.calc_gaussian_kernel(X, sigma) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)

def sample_gp_nonstat(X, meany, sigmas, n=1, noise=0.001):
    """ sample a function from a non-stationary Gaussian process """
    # http://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf
    K = maxdiv.calc_nonstationary_gaussian_kernel(X, sigmas) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)


def sample_interval(n, minlen, maxlen):
    """ sample the bounds of an interval """
    defect_start = int(np.random.randint(0,n-minlen))
    defect_end = int(np.random.randint(defect_start+minlen,min(defect_start+maxlen,n)))
    defect = np.zeros(n, dtype=bool)
    defect[defect_start:defect_end] = True
    return defect, defect_start, defect_end


X = np.arange(0,1,0.004)
X = np.reshape(X, [1, len(X)])
n = X.shape[1]

y = {}
f = {}

# simple mean shift
numf = 10
zeroy = np.zeros(X.shape[1])

print ("Generating time series of length {}".format(n))
defect_maxlen = int(0.2*n)
defect_minlen = int(0.05*n)
print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

sigma = 0.02
y['meanshift'] = []
f['meanshift'] = sample_gp(X, zeroy, sigma, numf)
for i in range(numf):
    defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
    y['meanshift'].append(defect)
    f['meanshift'][i,defect] -= np.random.rand()*0.5 + 0.5
#    plt.plot(X.T, f['meanshift'][i])
#plt.show()

y['contmeanshift'] = []
plt.figure()
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['contmeanshift'].append(defect)
    contmean = np.copy(zeroy)
    halflen = (b-a)/2
    contmean[a:(a+halflen)] = np.linspace(0,1,halflen)
    contmean[(a+halflen):b] = np.linspace(1,0,b-a-halflen)
    func = sample_gp(X, contmean, sigma, 1)
    if 'contmeanshift' in f:
        f['contmeanshift'] = np.vstack([f['contmeanshift'], func])
    else:
        f['contmeanshift'] = func
#    plt.plot(X.T, func[0])
#plt.show()
 
y['frequency_change'] = []
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['frequency_change'].append(defect)
    func = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
    if 'frequency_change' in f:
        f['frequency_change'] = np.vstack([f['frequency_change'], func])
    else:
        f['frequency_change'] = func
    plt.plot(X.T, func[0])
plt.show()

 
with open('testcube.pickle', 'wb') as fout:
    pickle.dump({'f': f, 'y': y}, fout)
