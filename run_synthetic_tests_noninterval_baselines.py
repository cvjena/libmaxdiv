import numpy as np
import matplotlib.pylab as plt
import maxdiv, maxdiv_tools, preproc, eval
import argparse
import sklearn
import sklearn.metrics
import time
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from baselines_noninterval import *
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle


def pointwiseScoresToIntervals(scores, min_length = 0):
    
    sorted_scores = sorted(scores)
    first_th = sorted_scores[int(len(scores) * 0.7)]
    max_score = sorted_scores[-1]
    
    thresholds = np.linspace(first_th, max_score, 10, endpoint = False)
    scores = np.array(scores)
    regions = []
    for th in thresholds[::-1]:
        regions += [(a, b, scores[a:b].min()) for a, b in eval.pointwiseLabelsToIntervals(scores >= th) if b - a >= min_length]
    
    # Non-maxima suppression
    include = np.ones(len(regions), dtype = bool) # suppressed intervals will be set to False
    for i in range(len(regions)):
        if include[i]:
            a, b, score = regions[i]
            # Exclude intervals with a lower score overlapping this one
            for j in range(i + 1, len(regions)):
                if include[j] and (maxdiv.IoU(a, b - a, regions[j][0], regions[j][1] - regions[j][0]) > 0.5):
                    include[j] = False
    
    return [r for i, r in enumerate(regions) if include[i]]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])
parser.add_argument('--preproc', help='preprocessing method', choices=[None,'td'],default=None)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=10, type=int)

args = parser.parse_args()

with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

extremetypes = set(args.extremetypes)

aucs = {}
aps = {}
for ftype in f:
    print ('-- {} --'.format(ftype))
    if len(extremetypes)>0 and not ftype in extremetypes:
        continue

    funcs = f[ftype]
    ygts = y[ftype]
    regions = []
    aucs[ftype] = []
    for i in range(len(funcs)):
        if args.preproc == 'td':
            func = preproc.td(funcs[i])
        else:
            func = funcs[i]
        ygt = ygts[i]
        
        scores = hotellings_t(func)
        regions.append(pointwiseScoresToIntervals(scores, args.extint_min_len))
        if not args.novis:
            eval.plotDetections(funcs[i], regions[-1], ygt, silent = False)
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ygt, scores, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))
    
    aps[ftype] = eval.average_precision(ygts, regions)

print('-- Aggregated AUC --')
for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in aps:
    print ("{}: {}".format(ftype, aps[ftype]))