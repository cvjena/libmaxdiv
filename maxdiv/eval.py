import numpy as np
import sklearn.metrics
import matplotlib.pylab as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from .maxdiv_util import IoU


def auc(ygt, regions, n = 0):
    """ Computes the Area Under the ROC curve for a list of detected region triples (a, b, score). 
        
    The ground-truth annotations are given by `ygt` either as pointwise boolean labels or as list of regions (a, b).
    `n` gives the total length of the time series and is only reguired if `ygt` is given as list of regions.
    """
    
    # Convert ground-truth to pointwise labels if given as regions
    if isinstance(ygt, list) and (isinstance(ygt[0], tuple) or isinstance(ygt[0], list)):
    
        if n <= 0:
            raise ValueError('The length of the time series must be specified if ground-truth is given as list of regions.')
        
        gt = np.zeros(n, dtype = bool)
        for a, b in ygt:
            gt[a:b] = True
        
    else:
        gt = ygt
        n = len(ygt)
    
    # Convert list of detected regions to scored points
    scores = np.zeros(n)
    for a, b, score in regions:
        scores[a:b] = score
    
    # Compute AUC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt, scores, pos_label = 1)
    return sklearn.metrics.auc(fpr, tpr)


def average_precision(ygt, regions, overlap = 0.5, plot = False):
    """ Computes the Average Precision for a list of detected region triples (a, b, score). 
        
    The ground-truth annotations are given by `ygt` as a list of either pointwise boolean labels or lists of regions (a, b).
    Each element of `ygt` corresponds to one time series and contains the ground-truth intervals for that time series.
    The `regions` list must have the same length as `ygt` and contains an arbitrary number of detected intervals for each
    time series.
    
    Alternatively, `ygt` may be the total number of ground-truth intervals and the detections in `regions` must then
    be 4-tuples `(a, b, score, tp)`, where `tp` specifies whether they are a true positive detection or not.
    
    `overlap` is a threshold for the Intersection over Union (IoU) criterion. Detections must have an IoU with
    a ground-truth region greater than this threshold for being considered a true positive.
    
    If `plot` is set to true, a recall/precision curve will be plotted and shown.
    """
    
    # Convert ground-truth to intervals if given as pointwise labels
    if isinstance(ygt, list):
        if len(ygt) != len(regions):
            raise ValueError('Different number of time series for ground-truth and detections.')
        ygt = [gt if (isinstance(gt[0], tuple) or isinstance(gt[0], list)) else pointwiseLabelsToIntervals(gt) for gt in ygt]
    
    # Determine recall and precision for all thresholds and compute interpolated AP
    recall, precision = recall_precision(ygt, regions, overlap, 'all')
    ap = 0.0
    for i in range(len(recall) - 2, -1, -1):
        
        if recall[i] != recall[i+1]:
            ap += (recall[i+1] - recall[i]) * precision[i+1]
        
        precision[i] = max(precision[i+1], precision[i])

    ap += recall[0] * precision[0]
    
    # Plot recall/precision curve
    if plot:
        fig = plt.figure()
        sp = fig.add_subplot(111, xlabel = 'Recall', ylabel = 'Precision', ylim = (0.0, 1.05))
        sp.plot(recall, precision)
        plt.show()
    
    return ap


def recall_precision(ygt, regions, overlap = 0.5, th = None, multiAsFP = True):
    """ Computes recall and precision for a list of detected region triples (a, b, score). 
        
    The ground-truth annotations are given by `ygt` as a list of lists of regions (a, b). Each list
    corresponds to one time series and contains the ground-truth intervals for that time series.
    The `regions` list must have the same length as `ygt` and contains an arbitrary number of detected
    intervals for each time series.
    
    Alternatively, `ygt` may be the total number of ground-truth intervals and the detections in `regions`
    must then be 4-tuples `(a, b, score, tp)`, where `tp` specifies whether they are a true positive detection or not.
    
    `overlap` is a threshold for the Intersection over Union (IoU) criterion. Detections must have an IoU with
    a ground-truth region greater than this threshold for being considered a true positive.
    
    `th` specifies the detection threshold; detections with a lower score will be ignored.
    Set this to None to compute recall and precision over all detections.
    The special value 'all' can be used to compute lists of corresponding recall and precision values for every
    possible threshold (which is more efficient than calling this function once for every threshold).
    
    `multiAsFP` controls whether subsequent detections of the same region will be ignored (`False`) or counted
    as false positives (`True`).
    
    Returns: a `(recall, precision)` tuple, where `recall` and `precision` are floats or arrays of floats if
             `th` is set to `'all'`. In the latter case, the lists will be sorted in ascending order by recall.
    """
    
    # Wrap ygt and regions in a lists if there is only one time series
    if isinstance(ygt, list) and (not (isinstance(ygt[0], list) and ((len(ygt[0]) == 0) or isinstance(ygt[0][0], list) or isinstance(ygt[0][0], tuple)))):
        ygt = [ygt]
    if not (isinstance(regions[0], list) and ((len(regions[0]) == 0) or isinstance(regions[0][0], list) or isinstance(regions[0][0], tuple))):
        regions = [regions]
    if isinstance(ygt, list) and (len(ygt) != len(regions)):
        raise ValueError('Different number of time series for ground-truth and detections.')
    
    if th == 'all':
    
        if not isinstance(ygt, list):
        
            # Flatten regions array and sort detections descendingly by their score
            sorted_regions = sorted((region for detections in regions for region in detections), key = lambda r: r[2], reverse = True)
            
            # Count true positives for all thresholds
            tp = np.array([int(isTP) for a, b, score, isTP in sorted_regions]).cumsum()
            numDet = np.arange(1, len(sorted_regions) + 1)

            # Account for detections with exactly the same score
            mask = np.array([((i >= len(tp) - 1) or (sorted_regions[i][2] != sorted_regions[i+1][2])) for i in range(len(tp))])
            if not mask.all():
                tp = tp[mask]
                numDet = numDet[mask]
            
            # Compute recall and precision
            return (tp / ygt, tp / numDet if len(sorted_regions) > 0 else tp)
        
        else:
        
            # Flatten regions array after associating each region with its time series and sort them descendingly by their score
            sorted_regions = sorted((tuple(region) + (ts,) for ts, detections in enumerate(regions) for region in detections), key = lambda r: r[2], reverse = True)
        
            # Indicators for true and false positives
            tp = np.zeros(len(sorted_regions))
            fp = np.zeros(len(sorted_regions))
            mask = np.ones(len(sorted_regions), dtype = bool)
            
            # Determine for each detection if it is a true or a false positive
            region_detected = [[False] * len(gt) for gt in ygt] # prevent multiple detections of the same interval
            for i, (a, b, score, ts) in enumerate(sorted_regions):
                
                isTP, isSubsequentDetection = False, False
                for i_gt, (a_gt, b_gt) in enumerate(ygt[ts]):
                    if (not (region_detected[ts][i_gt] and multiAsFP)) and (IoU(a, b - a, a_gt, b_gt - a_gt) >= overlap):
                        if region_detected[ts][i_gt]:
                            isSubsequentDetection = True
                        isTP = True
                        region_detected[ts][i_gt] = True
                        break
                
                if not isSubsequentDetection:
                    if isTP:
                        tp[i] = 1
                    else:
                        fp[i] = 1
                
                # Account for detections with exactly the same score
                if (i < len(sorted_regions) - 1) and (score == sorted_regions[i+1][2]):
                    mask[i] = False
        
            # Compute recall and precision
            tp = tp.cumsum()[mask]
            fp = fp.cumsum()[mask]
            return (tp / sum(len(gt) for gt in ygt), tp / (tp + fp) if len(sorted_regions) > 0 else tp)
    
    else:
    
        if not isinstance(ygt, list):
        
            # Count true positives for a specific threshold
            tp = sum(int(isTP) for detections in regions for a, b, score, isTP in detections if score >= th)
            
            # Compute recall and precision
            return (float(tp) / ygt, float(tp) / max(1, sum(len(detections) for detections in regions)))
        
        else:
    
            # Count true and false positives for a specific threshold
            tp, fp = 0, 0
            region_detected = [[False] * len(gt) for gt in ygt] # prevent multiple detections of the same interval
            for i in range(len(regions)):
                for a, b, score in regions[i]:
                    if (th is None) or (score >= th):
                    
                        isTP = False
                        for i_gt, (a_gt, b_gt) in enumerate(ygt[i]):
                            if (not region_detected[i][i_gt]) and (IoU(a, b - a, a_gt, b_gt - a_gt) >= overlap):
                                isTP = True
                                region_detected[i][i_gt] = True
                                break
                        
                        if isTP:
                            tp += 1
                        else:
                            fp += 1
            
            # Calculate recall and precision
            return (float(tp) / sum(len(gt) for gt in ygt), float(tp) / (tp + fp) if tp + fp > 0 else 1.0)


def plotDetections(func, regions, gt = [], ticks = None, export = None, silent = True, detailedvis = False):
    """ Plots a time series and highlights both detected regions and ground-truth regions.
    
    `regions` specifies the detected regions as (a, b, score) tuples.
    
    `gt` specifies the ground-truth regions either as (a, b) tuples or as pointwise boolean labels.
    
    Custom labels for the ticks on the x axis may be specified via the `ticks` parameter, which expects
    a dictionary with tick locations as keys and tick labels as values.
    
    If `export` is set to a string, the plot will be saved to that filename instead of being shown.
    
    Setting `silent` to False will print the detected intervals to the console.
    """
    
    # Convert pointwise ground-truth to list of regions
    if (len(gt) > 0) and (not (isinstance(gt[0], tuple) or isinstance(gt[0], list))):
        gt = pointwiseLabelsToIntervals(gt)
    
    # Plot time series and ground-truth intervals
    plotted_function = False
    for a, b in gt:
        show_interval(func, a, b, 10000, 'r', 1.0, plot_function = not plotted_function, border = True)
        plotted_function = True
    
    # Plot detected intervals with color intensities corresponding to their score
    if len(regions) > 0:
        minScore = min(r[2] for r in regions)
        maxScore = max(r[2] for r in regions)
        for i in range(len(regions)):
            a, b, score = regions[i]
            if not silent:
                print ("Region {}/{}: {} - {} (Score: {})".format(i, len(regions), a, b, score))
            intensity = float(score - minScore) / (maxScore - minScore) if minScore < maxScore else 1.0
            show_interval(func, a, b, 10000, plot_function = not plotted_function,
                          color = (0.8 - intensity * 0.8, 0.8 - intensity * 0.8, 1.0))
            plotted_function = True
        
            # Show supplementary visualization
            if detailedvis:
                mainFigNum = plt.gcf().number
                detailfig = plt.figure()
                if func.shape[0]==1:
                    h_nonextreme, bin_edges = np.histogram( np.hstack([ func[0,:a], func[0, b:] ]), bins=40 )
                    h_extreme, _ = np.histogram(func[0,a:b], bins=bin_edges)
                    bin_means = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    plt.plot(bin_means, h_extreme, figure = detailfig)
                    plt.plot(bin_means, h_nonextreme, figure = detailfig)
                else:
                    X_nonextreme = np.hstack([ func[:2, :a], func[:2, b:] ])
                    X_extreme = func[:2, a:b]
                    plt.plot( X_nonextreme[0], X_nonextreme[0], 'bo', figure = detailfig )
                    plt.plot( X_extreme[0], X_extreme[0], 'r+', figure = detailfig )
                plt.figure(mainFigNum)

    # Draw legend
    patch_detected_extreme = mpatches.Patch(color='blue', alpha=0.3, label='detect. extreme')
    patch_gt_extreme = mlines.Line2D([], [], color='red', label='gt extreme')
    patch_time_series = mlines.Line2D([], [], color='blue', label='time series')

    plt.legend(handles=[patch_time_series, patch_gt_extreme, patch_detected_extreme], loc='center', mode='expand', ncol=3, bbox_to_anchor=(0,1,1,0), shadow=True, fancybox=True)
    
    # Set tick labels
    if ticks is not None:
        ax = plt.gca()
        ax.set_xticks(list(ticks.keys()))
        ax.set_xticklabels(list(ticks.values()))
    
    # Display plot
    if export:
        plt.savefig(export)
    else:
        plt.show()


def show_interval(f, a, b, visborder=100, color='b', alpha=0.3, plot_function=True, border=False):
    """ Plot a timeseries together with a marked interval """
    
    av = max(a - visborder, 0)
    bv = min(b + visborder, f.shape[1])
    x = range(av, bv)
    minv = np.min(f[:, av:bv])
    maxv = np.max(f[:, av:bv])
    if plot_function:
        for i in range(f.shape[0]):
            plt.plot(x, f[i,av:bv], color='blue')

    if border:
        plt.plot([ a, a, b, b, a ], [minv, maxv, maxv, minv, minv], color=color, alpha=alpha, linewidth=3)
    else:
        plt.fill([ a, a, b, b ], [minv, maxv, maxv, minv], color=color, alpha=alpha)


    yborder = abs(maxv-minv)*0.05
    plt.ylim([minv-yborder, maxv+yborder])

    return x, av, bv


def pointwiseLabelsToIntervals(labels):
    """Converts an array of boolean labels for each point in a time series to a list of contiguous intervals."""
    
    regions = []
    current_start = None
    
    for i, lbl in enumerate(labels):
        # Begin of new interval
        if (current_start is None) and lbl:
            current_start = i
        # End of current interval
        elif (current_start is not None) and (not lbl):
            regions.append((current_start, i))
            current_start = None
    
    # Terminate trailing interval
    if current_start is not None:
        regions.append((current_start, len(labels)))
    
    return regions
