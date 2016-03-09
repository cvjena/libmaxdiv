import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
import maxdiv
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', help='maxdiv method', choices=maxdiv.get_available_methods(), required=True)
parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=20, type=int)
parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=250, type=int)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--num_intervals', help='number of intervals to be displayed', default=5, type=int)
parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)
parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'LAMBDA'], default='I_OMEGA')

parser.parse_args()
args = parser.parse_args()

# prepare parameters for calling maxdiv
method_parameter_names = ['extint_min_len', 'extint_max_len', 'alpha', 'mode', 'method', 'num_intervals']
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in method_parameter_names}


with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

for ftype in f:
    funcs = f[ftype]
    ygts = y[ftype]
    for i in range(len(funcs)):
        func = np.reshape(funcs[i], [1, len(funcs[i])])
        ygt = ygts[i]
        regions = maxdiv.maxdiv(func, **parameters)

        if not args.novis:
            plt.figure()
            for i in range(len(regions)):
                a, b, score = regions[i]
                maxdiv.show_interval(range(len(funcs[i])), func, a, b, 10000)
            plt.show()

