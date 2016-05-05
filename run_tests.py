import numpy as np
import matplotlib.pylab as plt
import maxdiv, maxdiv_tools, preproc, eval, datasets
import argparse, time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--datasets', help='datasets to be loaded', nargs='+', default=datasets.DATASETS)
parser.add_argument('--subsets', help='subsets of the datasets to be tested', nargs='+',default=[])
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=datasets.TYPES)
parser.add_argument('--demomode', help='show results with a given delay and store images to disk', action='store_true')

maxdiv_tools.add_algorithm_parameters(parser)

args = parser.parse_args()

# prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters()}
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
all_gt = []
all_regions = []
num = 0
for ftype in data:
    if len(subsets)>0 and not ftype in subsets:
        continue
    
    print('-- {} --'.format(ftype))

    ygts = []
    regions = []
    aucs[ftype] = []
    for i, func in enumerate(data[ftype]):
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
    
    aps[ftype] = eval.average_precision(ygts, regions, detailedvis and not args.novis)
    print ("AP: {}".format(aps[ftype]))
    
    all_regions += regions
    all_gt += ygts

print('-- Aggregated AUC --')
for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))

print('-- Average Precision --')
for ftype in aps:
    print ("{}: {}".format(ftype, aps[ftype]))
print ("OVERALL AP: {}".format(eval.average_precision(all_gt, all_regions)))
