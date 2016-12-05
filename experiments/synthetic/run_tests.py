""" CLI for running the MDI algorithm on given datasets and computing performance metrics. """

import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
import matplotlib.pylab as plt
import argparse, time

from maxdiv import maxdiv, preproc, eval
import cli_tools, datasets

parser = argparse.ArgumentParser(description = 'Run the MDI algorithm on a given dataset and measure performance.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--datasets', help='datasets to be loaded', nargs='+', default=['synthetic'])
parser.add_argument('--subsets', help='subsets of the datasets to be tested', nargs='+',default=[])
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=datasets.TYPES)
parser.add_argument('--demomode', help='show results with a given delay and store images to disk', action='store_true')
parser.add_argument('--dump', help='Dump detections for each time-series to the specified CSV file', default='')

cli_tools.add_algorithm_parameters(parser)

args = parser.parse_args()

canonical_order = ['meanshift', 'meanshift_hard', 'meanshift5', 'meanshift5_hard', 'amplitude_change', 'frequency_change', 'mixed',
                   'meanshift_multvar', 'amplitude_change_multvar', 'frequency_change_multvar', 'mixed_multvar']

# prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in cli_tools.get_algorithm_parameters()}
if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
    parameters['num_intervals'] = None
parameters['kernelparameters'] = { 'kernel_sigma_sq' : args.kernel_sigma_sq }
parameters['proposalparameters'] = { 'useMAD' : args.prop_mad, 'sd_th' : args.prop_th }
if args.prop_unfiltered:
    parameters['proposalparameters']['filter'] = None

# Load datasets
data = datasets.loadDatasets(args.datasets, args.extremetypes)
subsets = set(args.subsets)

detailedvis = False

aucs = {}
aps = {}
all_ids = []
all_gt = []
all_regions = []
num = 0
for ftype in data:
    if len(subsets)>0 and not ftype in subsets:
        continue
    if ftype not in canonical_order:
        canonical_order.append(ftype)
    
    print('-- {} --'.format(ftype))

    func_ids = []
    ygts = []
    regions = []
    aucs[ftype] = []
    for i, func in enumerate(data[ftype]):
        func_ids.append('{}_{:03d}'.format(ftype, i))
        ygts.append(func['gt'])
        regions.append(maxdiv.maxdiv(func['ts'], **parameters))

        if not args.novis:
            if args.demomode and (num == 0):
                plt.figure()
                plt.ion()
                plt.show()
            eval.plotDetections(func['ts'], regions[-1], func['gt'],
                                silent = False,
                                export = '{}_{:010}.png'.format(ftype, i) if args.demomode else None,
                                detailedvis = detailedvis)
            if args.demomode:
                plt.draw()
                time.sleep(0.25)
                plt.clf()
            
        auc = eval.auc(func['gt'], regions[-1], func['ts'].shape[1])
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))
                
        num += 1
    
    aps[ftype] = eval.average_precision(ygts, regions, plot = detailedvis and not args.novis)
    print ("AP: {}".format(aps[ftype]))
    
    all_ids += func_ids
    all_regions += regions
    all_gt += ygts

# Print results
print('-- Aggregated AUC --')
for ftype in canonical_order:
    if ftype in aucs:
        print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in canonical_order:
    if ftype in aps:
        print ("{}: {}".format(ftype, aps[ftype]))
print ("MEAN AP: {}".format(np.mean(list(aps.values()))))
print ("OVERALL AP: {}".format(eval.average_precision(all_gt, all_regions)))

# Dump detections
if args.dump:
    with open(args.dump, 'w') as dumpFile:
        dumpFile.write('Func,Start,End,Score\n')
        for id, regions in zip(all_ids, all_regions):
            for a, b, score in regions:
                dumpFile.write('{},{},{},{}\n'.format(id, a, b, score))