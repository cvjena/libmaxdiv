import numpy as np
import matplotlib.pylab as plt
import preproc
from baselines_noninterval import *
import sys
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle
    

METHODS = { 'hotellings_t' : hotellings_t, 'kde' : pointwiseKDE }
GRAD_FILTER = [-1, 0, 1]


if len(sys.argv) != 3:
    print('Usage: {} <method> <extremetype>'.format(sys.argv[0]))
else:
    
    # Load test data
    with open('testcube.pickle', 'rb') as fin:
        cube = pickle.load(fin)
        f = cube['f']
        y = cube['y']

    # Parse arguments
    method, ftype = sys.argv[1:]
    if method not in METHODS:
        print('Unknown method: {}\nPossible values: {}'.format(method, ', '.join(METHODS)))
        exit()
    if ftype not in f:
        print('Unknown extreme type: {}'.format(ftype))
        exit()
    
    # Detect and plot anomaly boundaries
    for func in f[ftype]:
        
        # Compute scores for each point
        scores = METHODS[method](preproc.td(func))

        # Score statistics
        score_mean = np.mean(scores)
        score_sd = np.std(scores)
        
        # Compute gradient of scores
        pad = (len(GRAD_FILTER) - 1) // 2
        padded_scores = np.concatenate((scores[:pad], scores, scores[-pad:]))
        score_gradient = np.abs(np.convolve(padded_scores, GRAD_FILTER, 'valid'))
        score_gradient_mean = np.median(score_gradient)
        score_gradient_sd = np.std(score_gradient)
        
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(311, ylabel = 'Function')
        ax.plot(func.T, color = 'b')
        ax = fig.add_subplot(312, ylabel = 'Scores')
        ax.plot(scores, color = 'r')
        ax.plot([0, len(scores)-1], [score_mean] * 2, '--k')
        ax.plot([0, len(scores)-1], [score_mean + 1.5 * score_sd] * 2, '--b')
        ax = fig.add_subplot(313, ylabel = 'Gradient of Scores')
        ax.plot(score_gradient, color = 'r')
        ax.plot([0, len(scores)-1], [score_gradient_mean] * 2, '--k')
        ax.plot([0, len(scores)-1], [score_gradient_mean + 1.5 * score_gradient_sd] * 2, '--b')
        plt.show()