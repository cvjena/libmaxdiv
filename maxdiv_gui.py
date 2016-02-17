#!/usr/bin/env python
# -*- coding: utf8 -*-
""" GUI/Analysis program for finding extreme intervals using maximally divergent regions """

import maxdiv
import numpy as np
import matplotlib.pylab as plt
import argparse
import csv
import datetime
import time

__author__ = "Erik Rodner"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', help='time series in CSV format with column names', required=True)
parser.add_argument('--method', help='maxdiv method', choices=['parzen', 'gaussian'])
parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=20, type=int)
parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=250, type=int)
parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', default='1.0')
parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'LAMBDA'], default='I_OMEGA')
parser.add_argument('--timecol', help='name of the column for the date-time specification in the CSV file', default='DateTime')
parser.add_argument('--variables', help='names of variables to consider', nargs='+')
parser.add_argument('--timeformat', help='format used for strptime to convert time specifications, ' +
        'see https://docs.python.org/2/library/time.html#time.strftime for more information', default='%Y-%m-%d %H:%M:%S')
args = parser.parse_args()

# prepare parameters for calling maxdiv
method_parameter_names = ['extint_min_len', 'extint_max_len', 'alpha', 'mode', 'method']
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in method_parameter_names}

# read the multivariate time series
print ("Reading the time series from {}".format(args.input))
X = []
times = []
with open(args.input, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    if not args.timecol in reader.fieldnames:
        raise Exception("No column with name {} found in the file".format(args.timerow))
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
            current_time = time.strptime(time_string, args.timeformat)
        except:
            raise Exception("Unable to convert the time specification {} using the format {}".format(time_string, args.timeformat))
        times.append(current_time)
        vector = [ float(row[v]) for v in variables ]
        X.append(vector)

X = np.vstack(X)

a, b, score = maxdiv.maxdiv(X, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters)

