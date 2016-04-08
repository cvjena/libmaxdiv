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
gps = sample_gp(X, zeroy, sigma, numf)
f['meanshift'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
for i in range(numf):
    defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
    y['meanshift'].append(defect)
    f['meanshift'][i,0,defect] -= np.random.rand()*1.0 + 3.0 # easy
    #f['meanshift'][i,0,defect] -= np.random.rand()*0.5 + 0.5 # hard
#    plt.plot(X.T, f['meanshift'][i])
#plt.show()

sigma = 0.02
y['meanshift_multvar'] = []
gps = sample_gp(X, zeroy, sigma, numf*4)
f['meanshift_multvar'] = np.reshape(gps, [numf, 4, gps.shape[1]])
for i in range(numf):
    defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
    y['meanshift_multvar'].append(defect)
    f['meanshift_multvar'][i,0,defect] -= np.random.rand()*1.0 + 3.0 # easy
    #f['meanshift'][i,0,defect] -= np.random.rand()*0.5 + 0.5 # hard
#    plt.plot(X.T, f['meanshift'][i])
#plt.show()

y['amplitude_change'] = []
f['amplitude_change'] = np.zeros([numf, 1, n])
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['amplitude_change'].append(defect)
    func = sample_gp(X, zeroy, sigma, 1)
    sigmaw = (b-a)/4.0
    mu = (a+b)/2.0
    gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
    gauss[gauss>0.2] = 0.2
    func = func * (2.0*gauss/np.max(gauss)+1)
    f['amplitude_change'][i, 0] = func
#    plt.plot(X.T, func[0])
#plt.show()
 
y['amplitude_change_multvar'] = []
f['amplitude_change_multvar'] = np.zeros([numf, 4, n])
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['amplitude_change_multvar'].append(defect)
    func = sample_gp(X, zeroy, sigma, 4)
    sigmaw = (b-a)/4.0
    mu = (a+b)/2.0
    gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
    gauss[gauss>0.2] = 0.2
    func[0,:] = func[0,:] * (2.0*gauss/np.max(gauss)+1)
    f['amplitude_change_multvar'][i, :] = func
#    plt.plot(X.T, func[0])
#plt.show()

y['frequency_change'] = []
f['frequency_change'] = np.zeros([numf, 1, n])
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['frequency_change'].append(defect)
    func = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
    f['frequency_change'][i, 0] = func
#    plt.plot(X.T, func[0])
#plt.show()

y['frequency_change_multvar'] = []
f['frequency_change_multvar'] = np.zeros([numf, 5, n])
for i in range(numf):
    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
    y['frequency_change_multvar'].append(defect)
    func_defect = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
    func_ok = sample_gp(X, zeroy, sigma, 4)
    f['frequency_change_multvar'][i] = np.vstack([func_defect, func_ok])

 
with open('testcube.pickle', 'wb') as fout:
    pickle.dump({'f': f, 'y': y}, fout)
