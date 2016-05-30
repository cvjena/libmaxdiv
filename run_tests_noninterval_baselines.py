import numpy as np
import matplotlib.pylab as plt
import argparse, time
import sklearn, sklearn.metrics
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from maxdiv import maxdiv, preproc, eval
from maxdiv.baselines_noninterval import *
import maxdiv_tools, datasets


METHODS = { 'hotellings_t' : hotellings_t, 'kde' : pointwiseKDE }


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', help='scoring method', choices=METHODS, required=True)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--datasets', help='datasets to be loaded', nargs='+', default=datasets.DATASETS)
parser.add_argument('--subsets', help='subsets of the datasets to be tested', nargs='+',default=[])
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=datasets.TYPES)
parser.add_argument('--preproc', help='preprocessing method', choices=[None,'td'],default=None)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=10, type=int)

args = parser.parse_args()

data = datasets.loadDatasets(args.datasets, args.extremetypes)
subsets = set(args.subsets)

aucs = {}
aps = {}
all_gt = []
all_regions = []
for ftype in data:
    print ('-- {} --'.format(ftype))
    if len(subsets)>0 and not ftype in subsets:
        continue

    ygts = []
    regions = []
    aucs[ftype] = []
    for func in data[ftype]:
        if args.preproc == 'td':
            pfunc = preproc.td(func['ts'])
        else:
            pfunc = func['ts']
        
        scores = METHODS[args.method](pfunc)
        ygts.append(func['gt'])
        regions.append(pointwiseScoresToIntervals(scores, args.extint_min_len))
        if not args.novis:
            eval.plotDetections(func['ts'], regions[-1], func['gt'], silent = False)
        
        auc = eval.auc(func['gt'], regions[-1], func['ts'].shape[1])
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))
    
    aps[ftype] = eval.average_precision(ygts, regions)
    
    all_regions += regions
    all_gt += ygts

print('-- Aggregated AUC --')
for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in aps:
    print ("{}: {}".format(ftype, aps[ftype]))
print ("OVERALL AP: {}".format(eval.average_precision(all_gt, all_regions)))