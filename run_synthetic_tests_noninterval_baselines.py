import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
import maxdiv
import maxdiv_tools
import preproc
import argparse
import sklearn
import sklearn.metrics
import time
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from baselines_noninterval import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])

args = parser.parse_args()

with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

extremetypes = set(args.extremetypes)

aucs = {}
num = 0
for ftype in f:
    print ftype, extremetypes
    if len(extremetypes)>0 and not ftype in extremetypes:
        continue

    funcs = f[ftype]
    ygts = y[ftype]
    aucs[ftype] = []
    for i in range(len(funcs)):
        func = funcs[i]
        ygt = ygts[i]
        
        scores = hotellings_t(func)
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ygt, scores, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))

for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))
