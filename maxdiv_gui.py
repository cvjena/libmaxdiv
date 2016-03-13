#!/usr/bin/env python
# -*- coding: utf8 -*-
""" GUI/Analysis program for finding extreme intervals using maximally divergent regions """

import maxdiv
import preproc
import numpy as np
import matplotlib.pylab as plt
import argparse
import csv
import datetime
import time
import sys
import cProfile as profile
from scipy.io import savemat
#try:
#    raise Exception("Skip Gooey support")
#    from gooey import Gooey, GooeyParser
#    gooey_installed = True
#except:
#    print ("Install Gooey for a fancy GUI")
#    def Gooey(func):
#        return func
#    gooey_installed = False

__author__ = "Erik Rodner"

#@Gooey
def main():
    #if gooey_installed:
    #    parser = GooeyParser()
    #else:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #if gooey_installed:
    #    parser.add_argument('--input', help='time series in CSV format with column names',
    #            required=True, type=argparse.FileType('r'), widget='FileChooser')
    #else:
    parser.add_argument('--input', help='time series in CSV format with column names', required=True)
    parser.add_argument('--method', help='maxdiv method', choices=maxdiv.get_available_methods(), required=True)
    parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
    parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=20, type=int)
    parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=250, type=int)
    parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)
    parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'LAMBDA'], default='I_OMEGA')
    parser.add_argument('--timecol', help='name of the column for the date-time specification in the CSV file', default='DateTime')
    parser.add_argument('--variables', help='names of variables to consider', nargs='+')
    parser.add_argument('--maxdatapoints', help='maximum number of data points (for debugging)', type=int)
    parser.add_argument('--timeformat', help='format used for strptime to convert time specifications, ' +
            'see http://strftime.org for more information', default='%Y-%m-%d %H:%M:%S')
    parser.add_argument('--vissample', help='number of non-extreme points sampled for visualization', default=-1, type=int)
    parser.add_argument('--visborder', help='number of data points additionally displayed after and before extreme', default=10, type=int)
    parser.add_argument('--num_intervals', help='number of intervals to be displayed', default=5, type=int)
    parser.add_argument('--novis', action='store_true', help='skip the visualization')
    parser.add_argument('--profile', action='store_true', help='run the profiler')
    parser.add_argument('--matout', help='.mat file for results', default='results.mat')
    parser.add_argument('--preproc', help='use a pre-processing method', default=None, choices=preproc.get_available_methods())
    parser.add_argument('--outfmt', help='output format used to store results', default='matlab', choices=['matlab', 'nabcsv'] )
    parser.parse_args()
    args = parser.parse_args()

    # prepare parameters for calling maxdiv
    method_parameter_names = ['extint_min_len', 'extint_max_len', 'alpha', 'mode', 'method', 'num_intervals', 'preproc']
    args_dict = vars(args)
    parameters = {parameter_name: args_dict[parameter_name] for parameter_name in method_parameter_names}

    # read the multivariate time series
    print ("Reading the time series")
    X = []
    times = []
    with open(args.input, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if not args.timecol in reader.fieldnames:
            raise Exception("No column with name {} found in the file".format(args.timecol))
        if args.variables is None:
            variables = list(reader.fieldnames)
        else:
            variables = args.variables
        if args.timecol in variables:
            variables.remove(args.timecol)
        print ("Variables used: {}".format(variables))

        for row in reader:
            time_string = row[args.timecol]
            try:
                current_time = datetime.datetime.strptime(time_string, args.timeformat)
            except:
                raise Exception("Unable to convert the time specification {} using the format {}".format(time_string, args.timeformat))
            times.append(current_time)
            vector = [ float(row[v]) for v in variables ]
            X.append(vector)

            if not args.maxdatapoints is None and len(X) >= args.maxdatapoints:
                break

    X = np.vstack(X).T
    print ("Data points in the time series: {}".format(X.shape[1]))
    print ("Dimensions for each data point: {}".format(X.shape[0]))

    if args.profile:
        pr = profile.Profile()
        pr.enable()
    regions = maxdiv.maxdiv(X, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters)
    if args.profile:
        pr.disable()
        pr.dump_stats('profile-stats.pr')
        pr.print_stats(sort=1)
        sys.exit(0)

    for region_index, region in enumerate(regions):
        a, b, score = region
        a_time = times[a]
        b_time = times[b]
        print ("Extreme interval detected between data points {} and {}".format(a, b))
        print ("Extreme interval detected between {} and {}".format(a_time, b_time))
        print ("Score of the interval: {}".format(score))

        # visualization
        if not args.novis:
            plt.figure()
            x, av, bv = maxdiv.show_interval(X, a, b, args.visborder)

            steps = (bv-av)//10
            plt.xticks(x[::steps], times[av:bv:steps], rotation=30)
            plt.title('Detected Extreme in the Time Series')
            plt.savefig('extreme-{:05d}.pdf'.format(region_index)) 

            if X.shape[0]>=2:
                plt.figure()
                non_extreme_sample = range(0,a) + range(b,X.shape[1])
                if args.vissample>0:
                    non_extreme_sample = np.random.choice(non_extreme_sample, args.vissample)
                plt.scatter(X[0,non_extreme_sample], X[1,non_extreme_sample], color='blue')
                plt.scatter(X[0,a:b], X[1,a:b], color='red')
                plt.title('Data Distributions of the Extreme and All Data')
                plt.legend(['sampled from non-extreme', 'extreme'])
                plt.savefig('extreme-distribution-{:05d}.pdf'.format(region_index)) 
            else:
                print ("The time series has only one dimension, therefore the data distribution plot is skipped.")
            plt.show()

    if args.outfmt=='nabcsv':
        # store the dates in times, X (one-dimensional), and the scores
        n = X.shape[1]
        scores = np.zeros(n)
        labels = np.zeros(n)
        for a, b, score in regions:
            scores[a:b] = score
            labels[a:b] = 1
        with open(args.matout, 'w') as fout:
            csvout = csv.DictWriter(fout, fieldnames=['timestamp', 'value', 
                'anomaly_score', 'reward_high_TP_rate', 'reward_low_FP_rate', 'label'])
            csvout.writeheader()
            for i in range(n):
                csvout.writerow({'timestamp': times[i],
                              'value': X[0, i], 
                              'anomaly_score': scores[i], 
                              'label': labels[i],
                              'reward_low_FP_rate': scores[i],
                              'reward_high_TP_rate': scores[i]})

    else:
        # Since savemat cannot handle None values, we skip
        # all values from the settings with None values
        data_to_store = { key:value for value, key in enumerate(args_dict) if not value is None }
        data_to_store['regions'] = regions
        savemat(args.matout, data_to_store)

#
# Main program
#
main()
