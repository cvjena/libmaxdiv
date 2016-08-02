import sys
sys.path.append('..')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv, os.path
from glob import glob
from bisect import bisect

from maxdiv import maxdiv


DATASETS = ['chfdb', 'ltstdb', 'mitdb', 'stdb']
ROOT_DIR = '/home/barz/anomaly-detection/ECG/'


def readECG(id):
    
    with open(ROOT_DIR + id + '.csv') as f:
        data = np.array([[float(x) for x in d] for d in csv.reader(f)])
    
    timesteps = data[:, 0]
    ecg = data[:, 1:].T
    
    anomalies = []
    with open(ROOT_DIR + id + '.ann') as f:
        for l in f:
            if l.strip() != '':
                anomalies.append(float(l.split()[0].strip()))
    
    return ecg, timesteps, anomalies


def classifyDetections(detections, timesteps, anomalies):
    
    det = []
    for a, b, score in detections:
        i, j = bisect(anomalies, timesteps[a]), bisect(anomalies, timesteps[b-1])
        if j > i: # interval contains an anomaly
            isTP = True
        elif (i > 0) and (timesteps[a] - anomalies[i-1] < 1):   # nearby anomaly to the left
            isTP = True
        elif (j < len(anomalies)) and (anomalies[j] - timesteps[b - 1] < 1):   # nearby anomaly to the right
            isTP = True
        else:
            isTP = False
        det.append((a, b, score, isTP))
    return det


def runOnDataset(dataset, params):
    
    detections = []
    numAnomalies = 0
    records = [os.path.splitext(os.path.basename(file))[0] for file in glob(ROOT_DIR + dataset + '/*.csv')]
    for record in records:
        ecg, timesteps, anomalies = readECG('{}/{}'.format(dataset, record))
        if len(anomalies) > 0:
            print('Running detector on {}/{}'.format(dataset, record))
            sys.stdout.flush()
            detections.append(classifyDetections(maxdiv.maxdiv(ecg, **params), timesteps, anomalies))
            numAnomalies += len(anomalies)
    return detections, numAnomalies


def recall_precision(detections, numPositive):
    
    # Flatten detections array and sort them descendingly by their score
    sorted_regions = sorted((region for det in detections for region in det), key = lambda r: r[2], reverse = True)
    
    # Indicators for true and fals positives
    tp = np.array([isTP for a, b, score, isTP in sorted_regions])
    fp = np.logical_not(tp)
    
    # Compute recall and precision
    tp = tp.cumsum()
    fp = fp.cumsum()
    return (tp / numPositive, tp / (tp + fp) if len(detections) > 0 else tp)


def average_precision(detections, numPositive, plotFilename = None, plotTitle = ''):
    
    # Determine recall and precision for all thresholds and compute interpolated AP
    recall, precision = recall_precision(detections, numPositive)
    ap = 0.0
    for i in range(len(recall) - 2, -1, -1):
        
        if recall[i] != recall[i+1]:
            ap += (recall[i+1] - recall[i]) * precision[i+1]
        
        precision[i] = max(precision[i+1], precision[i])

    ap += recall[0] * precision[0]
    
    # Plot recall/precision curve
    if plotFilename is not None:
        fig = plt.figure()
        sp = fig.add_subplot(111, title = '{} (AP: {:.2f})'.format(plotTitle, ap), xlabel = 'Recall', ylabel = 'Precision', ylim = (0.0, 1.05))
        sp.plot(recall, precision)
        fig.savefig(plotFilename)
    
    return ap


if __name__ == '__main__':
    import argparse, maxdiv_tools
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', help='maxdiv method', choices=maxdiv.get_available_methods(), default = 'gaussian_cov')
    parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'TS', 'LAMBDA', 'IS_I_OMEGA', 'JSD'], default='TS')
    parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
    parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=200, type=int)
    parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=400, type=int)
    parser.add_argument('--num_intervals', help='number of intervals', default=1000, type=int)
    parser.add_argument('--deseas', help='apply FT deseasonalization', action = 'store_true')
    parser.add_argument('--td_dim', help='Time-Delay Embedding Dimension', default=3, type=int)
    parser.add_argument('--td_lag', help='Time-Lag for Time-Delay Embedding', default=1, type=int)
    
    args = parser.parse_args()
    args_dict = vars(args)
    parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters() if parameter_name in args_dict}
    if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
        parameters['num_intervals'] = None
    parameters['kernelparameters'] = { 'kernel_sigma_sq' : args.kernel_sigma_sq }
    
    suffix = args.method + '_' + args.mode
   
    parameters['preproc'] = ['normalize'] 
    if args.deseas:
        parameters['preproc'].append('deseasonalize_ft')
        suffix += '_ft'
    
    all_detections = []
    num_anomalies = 0
    for ds in DATASETS:
        detections, numPositive = runOnDataset(ds, parameters)
        ap = average_precision(detections, numPositive, 'precrec_{}_{}.svg'.format(ds, suffix), ds)
        print('Processed {} ECGs from dataset {} (AP: {})'.format(len(detections), ds, ap))
        sys.stdout.flush()
        all_detections += detections
        num_anomalies += numPositive
    
    ap = average_precision(all_detections, num_anomalies, 'precrec_overall_{}.svg'.format(suffix), 'Overall')
    print('Overall AP: {}'.format(ap))
