import sys
sys.path.append('..')

import maxdiv
import maxdiv_tools
import glob
import argparse
import os
import csv
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
maxdiv_tools.add_algorithm_parameters(parser)
parser.add_argument('--nabroot', help='data folder of the NAB dataset', required=True)
parser.add_argument('--methodtag', help='name of the method used to create the results folder', default='maxdiv')
args = parser.parse_args()

args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters()}
 
for fn in glob.iglob(os.path.join(args.nabroot, 'data/*/*.csv')):
    print ("Processing time series {}".format(fn))

    # read the multivariate time series
    X, times = maxdiv_tools.read_csv_timeseries(fn, selected_variables=None, timecol='timestamp', 
            timeformat='%Y-%m-%d %H:%M:%S', maxdatapoints=None)
        
    regions = maxdiv.maxdiv(X, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters)
    
    # store the dates in times, X (one-dimensional), and the scores
    dataset_dirname = os.path.basename(os.path.dirname(fn))
    dataset_name = os.path.basename(fn)
    outfn = os.path.join(args.nabroot, 'results', args.methodtag, dataset_dirname, args.methodtag + '_' + dataset_name)
    print ("Storing results in {}".format(outfn))

    n = X.shape[1]
    scores = np.zeros(n)
    labels = np.zeros(n)
    for a, b, score in regions:
        scores[a:b] = score
        labels[a:b] = 1
    with open(outfn, 'w') as fout:
        csvout = csv.DictWriter(fout, fieldnames=['timestamp', 'value', 
            'anomaly_score', 'label'])
        csvout.writeheader()
        for i in range(n):
            csvout.writerow({'timestamp': times[i],
                          'value': X[0, i], 
                          'anomaly_score': scores[i], 
                          'label': labels[i]})
#                          'reward_low_FP_rate': scores[i],
#                          'reward_high_TP_rate': scores[i]})


