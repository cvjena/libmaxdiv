import numpy as np
import matplotlib.pylab as plt
import datasets
import sys

from maxdiv import preproc
from maxdiv.baselines_noninterval import *
    

METHODS = { 'hotellings_t' : hotellings_t, 'kde' : pointwiseKDE }
GRAD_FILTER = [-1, 0, 1]


if len(sys.argv) < 3:
    print('Usage: {} <method> [<dataset>/]<extremetype> [<embedding-dimension = 3>]'.format(sys.argv[0]))
else:

    # Parse arguments
    method, ftype = sys.argv[1:3]
    embed_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if method not in METHODS:
        print('Unknown method: {}\nPossible values: {}'.format(method, ', '.join(METHODS)))
        exit()
    
    # Load test data
    if '/' in ftype:
        dataset, ftype = ftype.split('/', 1)
    else:
        dataset = 'synthetic'
    try:
        data = datasets.loadDatasets(dataset)
    except:
        print('Unknown dataset: {}'.format(dataset))
        exit()
    if ftype not in data:
        print('Unknown extreme type: {}'.format(ftype))
        exit()
    
    # Detect and plot anomaly boundaries
    for func in data[ftype]:
        
        # Compute scores for each point
        scores = METHODS[method](preproc.td(func['ts'], embed_dim))

        # Score statistics
        score_mean = np.mean(scores)
        score_sd = np.std(scores)
        score_median = np.median(scores)
        score_mad = 1.4826 * np.median(np.abs(scores - score_median))
        
        # Compute gradient of scores
        pad = (len(GRAD_FILTER) - 1) // 2
        padded_scores = np.concatenate((scores[:pad], scores, scores[-pad:]))
        score_gradient = np.abs(np.convolve(padded_scores, GRAD_FILTER, 'valid'))
        score_gradient_mean = np.mean(score_gradient)
        score_gradient_sd = np.std(score_gradient)
        score_gradient_median = np.median(score_gradient)
        score_gradient_mad = 1.4826 * np.median(np.abs(score_gradient - score_gradient_mean))
        
        # Plot
        fig = plt.figure()
        fig.canvas.set_window_title('{} / {}'.format(ftype, func['id']))
        ax = fig.add_subplot(311, ylabel = 'Function')
        ax.plot(func['ts'].T, color = 'b')
        ax = fig.add_subplot(312, ylabel = 'Scores')
        ax.plot(scores, color = 'r')
        ax.plot([0, len(scores)-1], [score_mean] * 2, '--k')
        ax.plot([0, len(scores)-1], [score_mean + 1.5 * score_sd] * 2, '--b')
        #ax.plot([0, len(scores)-1], [score_median] * 2, ':k')
        #ax.plot([0, len(scores)-1], [score_median + 1.5 * score_mad] * 2, ':b')
        ax = fig.add_subplot(313, ylabel = 'Gradient of Scores')
        ax.plot(score_gradient, color = 'r')
        ax.plot([0, len(scores)-1], [score_gradient_mean] * 2, '--k')
        ax.plot([0, len(scores)-1], [score_gradient_mean + 1.5 * score_gradient_sd] * 2, '--b')
        #ax.plot([0, len(scores)-1], [score_gradient_median] * 2, ':k')
        #ax.plot([0, len(scores)-1], [score_gradient_median + 1.5 * score_gradient_mad] * 2, ':b')
        plt.show()