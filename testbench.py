""" Synthetic Test Bench for MaxDiv """

import maxdiv
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle

def sample_gp(X, meany, sigma, n=1, noise=0.001):
    K = maxdiv.calc_normalized_gaussian_kernel(X, sigma) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)

X = np.arange(0,1,0.004)
X = np.reshape(X, [1, len(X)])
n = X.shape[1]

y = {}
f = {}

# simple mean shift
numf = 10
zeroy = np.zeros(X.shape[1])
sigma = 1.0
f['meanshift'] = sample_gp(X, zeroy, sigma, numf)

print ("Generating time series of length {}".format(n))
defect_maxlen = int(0.2*n)
defect_minlen = int(0.05*n)
print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

#plt.figure()
y['meanshift'] = []
for i in range(numf):
    defect_start = int(np.random.randint(0,n-defect_minlen))
    defect_end = int(np.random.randint(defect_start+defect_minlen,min(defect_start+defect_maxlen,n)))
    defect = np.zeros(n, dtype=bool)
    defect[defect_start:defect_end] = True
    y['meanshift'].append(defect)
    f['meanshift'][i][defect] -= np.random.rand()*2.0 + 0.5 
    #plt.plot(X.T, f1[i])
#plt.show()

with open('testcube.pickle', 'wb') as fout:
    pickle.dump({'f': f, 'y': y}, fout)
