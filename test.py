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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--novis', action='store_true', help='skip the visualization')
parser.add_argument('--extremetypes', help='types of extremes to be tested', nargs='+',default=[])
parser.add_argument('--demomode', help='show results with a given delay and store images to disk', action='store_true')

maxdiv_tools.add_algorithm_parameters(parser)

args = parser.parse_args()

# prepare parameters for calling maxdiv
args_dict = vars(args)
parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters()}


with open('testcube.pickle', 'rb') as fin:
    cube = pickle.load(fin)
    f = cube['f']
    y = cube['y']

extremetypes = set(args.extremetypes)

detailedvis = False

aucs = {}
num = 0
for ftype in f:
    if len(extremetypes)>0 and not ftype in extremetypes:
        continue

    funcs = f[ftype]
    ygts = y[ftype]
    aucs[ftype] = []
    for i in range(len(funcs)):
        func = funcs[i]
        ygt = ygts[i]
        regions = maxdiv.maxdiv(func, kernelparameters={'kernel_sigma_sq': args.kernel_sigma_sq}, **parameters)

        scores = np.zeros(len(ygt))
        for i in range(len(regions)):
            a, b, score = regions[i]
            print "Region {}/{}: {} - {}".format(i, len(regions), a, b)
            scores[a:b] = score

            if not args.novis:
                if num==0:
                    plt.figure()
                    if args.demomode:
                        plt.ion()
                        plt.show()


                # assuming that there is only one extreme present
                a_gt = np.min(np.nonzero(ygt))
                b_gt = np.max(np.nonzero(ygt))
                maxdiv.show_interval(func, a_gt, b_gt, 10000, 'r', 1.0, plot_function=False, border=True)
                maxdiv.show_interval(func, a, b, 10000)

                patch_detected_extreme = mpatches.Patch(color='blue', alpha=0.3, label='detect. extreme')
                patch_gt_extreme = mlines.Line2D([], [], color='red', label='gt extreme')
                patch_time_series = mlines.Line2D([], [], color='blue', label='time series')

                plt.legend(handles=[patch_time_series, patch_gt_extreme, patch_detected_extreme], loc='center', mode='expand', ncol=3, bbox_to_anchor=(0,1,1,0), shadow=True, fancybox=True)

                if detailedvis:
                    plt.figure()
                    if func.shape[0]==1:
                        h_nonextreme, bin_edges = np.histogram( np.hstack([ func[0,:a], func[0, b:] ]), bins=40 )
                        h_extreme, _ = np.histogram(func[0,a:b], bins=bin_edges)
                        bin_means = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        plt.plot(bin_means, h_extreme)
                        plt.plot(bin_means, h_nonextreme)
                    else:
                        X_nonextreme = np.hstack([ func[:2, :a], func[:2, b:] ])
                        X_extreme = func[:2, a:b]
                        plt.plot( X_nonextreme[0], X_nonextreme[0], 'bo' )
                        plt.plot( X_extreme[0], X_extreme[0], 'r+' )
                if args.demomode:
                    plt.savefig('vis{:010}.png'.format(num))
                    plt.draw()
                    #time.sleep(0.25)
                    plt.clf()
                else:
                    plt.show()
                num += 1
            
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ygt, scores, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs[ftype].append(auc)
        print ("AUC: {}".format(auc))

for ftype in aucs:
    print ("{}: {} (+/- {})".format(ftype, np.mean(aucs[ftype]), np.std(aucs[ftype])))
