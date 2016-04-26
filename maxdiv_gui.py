#!/usr/bin/env python
# -*- coding: utf8 -*-
""" GUI/Analysis program for finding extreme intervals using maximally divergent regions """

import maxdiv
import maxdiv_tools
import preproc
import numpy as np
import matplotlib.pylab as plt
import argparse
import csv
import datetime
import time
import psutil
import os
import sys
import cProfile as profile
from scipy.io import savemat
from eval import show_interval
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
    parser.add_argument('--timecol', help='name of the column for the date-time specification in the CSV file', default='DateTime')
    parser.add_argument('--variables', help='names of variables to consider', nargs='+')
    parser.add_argument('--maxdatapoints', help='maximum number of data points (for debugging)', type=int)
    parser.add_argument('--timeformat', help='format used for strptime to convert time specifications, ' +
            'see http://strftime.org for more information', default='%Y-%m-%d %H:%M:%S')
    parser.add_argument('--vissample', help='number of non-extreme points sampled for visualization', default=-1, type=int)
    parser.add_argument('--visborder', help='number of data points additionally displayed after and before extreme', default=10, type=int)
    parser.add_argument('--novis', action='store_true', help='skip the visualization')
    parser.add_argument('--profile', action='store_true', help='run the profiler')
    parser.add_argument('--matout', help='.mat file for results', default='results.mat')
    parser.add_argument('--outfmt', help='output format used to store results', default='matlab', choices=['matlab', 'nabcsv'] )
    maxdiv_tools.add_algorithm_parameters(parser)
    
    args = parser.parse_args()

    # read the multivariate time series
    X, times = maxdiv_tools.read_csv_timeseries(args.input, args.variables, args.timecol, 
            args.timeformat, args.maxdatapoints)

    # prepare parameters for calling maxdiv
    args_dict = vars(args)
    parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters()}

    #
    # Running our fancy maxdiv method
    #
    if args.profile:
        pr = profile.Profile()
        pr.enable()
    regions = maxdiv.maxdiv(X, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters)
    if args.profile:
        pr.disable()
        pr.dump_stats('profile-stats.pr')
        pr.print_stats(sort=1)
        sys.exit(0)

    process = psutil.Process(os.getpid())
    print ("Memory used so far: {} MB".format(process.memory_info().rss/1024/1024))

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
            x, av, bv = show_interval(X, a, b, args.visborder)

            steps = (bv-av)//10
            plt.xticks(x[::steps], times[av:bv:steps], rotation=30)
            plt.title('Detected Extreme in the Time Series')
            plt.gcf().tight_layout()
            plt.savefig('extreme-{:05d}.pdf'.format(region_index), bbox_inches='tight') 

            if X.shape[0]>=2:
                plt.figure()
                non_extreme_sample = range(0,a) + range(b,X.shape[1])
                if args.vissample>0:
                    non_extreme_sample = np.random.choice(non_extreme_sample, args.vissample)
                plt.scatter(X[0,non_extreme_sample], X[1,non_extreme_sample], color='blue')
                plt.scatter(X[0,a:b], X[1,a:b], color='red')
                plt.title('Data Distributions of the Extreme and All Data')
                plt.legend(['sampled from non-extreme', 'extreme'])
                plt.savefig('extreme-distribution-{:05d}.pdf'.format(region_index), bbox_inches='tight') 
            else:
                print ("The time series has only one dimension, therefore the data distribution plot is skipped.")
            plt.show()

    if args.outfmt=='nabcsv':
        raise Exception("Use the script nab_experiment.py")
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
