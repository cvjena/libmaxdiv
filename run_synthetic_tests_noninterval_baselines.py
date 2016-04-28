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


METHODS = { 'hotellings_t' : hotellings_t, 'kde' : pointwiseKDE }


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', help='scoring method', choices=METHODS, required=True)
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
        
        scores = METHODS[args.method](func)
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