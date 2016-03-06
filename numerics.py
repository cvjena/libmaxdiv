import numpy as np
from numpy.linalg import slogdet, solve, inv, det, cholesky
import time

def tic():
    return time.time()

def toc(t, desc='something'):
    print ("Time used for {}: {}".format(desc, time.time()-t))

def timeit(func, num_iterations, desc):
    times = []
    for i in range(num_iterations):
        t = time.time()
        func()
        td = time.time() - t
        times.append(td)
    print ("Time used for {}: {} (+/- {})".format(desc, \
            np.mean(times), np.std(times)))

def cholall(A):
    G = cholesky(A)
    return np.prod(np.diag(G))

    return deta, loga

n = 100
d = 10
X = np.random.randn(n, d)
O = np.dot(X.T, X)

X2 = np.random.randn(n, d)
O2 = np.dot(X.T, X)

numit = 10000
timeit(lambda: slogdet(O), numit, 'slogdet')
timeit(lambda: det(O), numit, 'det')

v = np.random.randn(d, 1)
timeit(lambda: np.dot(v.T, np.dot(inv(O), v)), numit, 'inv+bilinear')
timeit(lambda: np.dot(v.T, solve(O, v)), numit, 'solve+bilinear')

timeit(lambda: np.sum(np.diag(np.dot(inv(O),O2))), numit, 'inv+trace')
timeit(lambda: np.sum(np.diag(solve(O, O2))), numit, 'solve+trace')

indices = range(d)
timeit(lambda: np.add(O, np.eye(d)), numit, 'add-eye')
timeit(lambda: np.add(O[indices,indices], 1.0), numit, 'add-diag-using-indices')

