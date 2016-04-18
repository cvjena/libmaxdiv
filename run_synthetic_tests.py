import numpy as np
import matplotlib.pylab as plt
import maxdiv, maxdiv_tools, preproc, eval
import argparse, time
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])
parser.add_argument('--demomode', help='show results with a given delay and store images to disk', action='store_true')

maxdiv_tools.add_algorithm_parameters(parser)

args = parser.parse_args()

# prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters()}
if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
    parameters['num_intervals'] = None


with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

extremetypes = set(args.extremetypes)

detailedvis = False

aucs = {}
aps = {}
num = 0
for ftype in f:
    if len(extremetypes)>0 and not ftype in extremetypes:
        continue
    
    print('-- {} --'.format(ftype))

    funcs = f[ftype]
    ygts = y[ftype]
    aucs[ftype] = []
    regions = []
    for i in range(len(funcs)):
        func = funcs[i]
        ygt = ygts[i]
        gt_regions = eval.pointwiseLabelsToIntervals(ygt)
        regions.append(maxdiv.maxdiv(func, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters))

        if not args.novis:
            if args.demomode and (num == 0):
                plt.figure()
                plt.ion()
                plt.show()
            eval.plotDetections(func, regions[-1], gt_regions, export = '{}_{:010}.png'.format(ftype, i) if args.demomode else None, detailedvis = detailedvis)
            if args.demomode:
                plt.draw()
                time.sleep(0.25)
                plt.clf()
            
        auc = eval.auc(ygt, regions[-1])
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))
                
        num += 1
    
    aps[ftype] = eval.average_precision(ygts, regions, plot = detailedvis and not args.novis)
    print ("AP: {}".format(aps[ftype]))

print('-- Aggregated AUC --')
for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in aps:
    print ("{}: {}".format(ftype, aps[ftype]))
