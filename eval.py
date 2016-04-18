import maxdiv
import numpy as np
import sklearn.metrics
import matplotlib.pylab as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


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
    
    `overlap` is a threshold for the Intersection over Union (IoU) criterion. Detections must have an IoU with
    a ground-truth region greater than this threshold for being considered a true positive.
    
    If `plot` is set to true, a recall/precision curve will be plotted and shown.
    """
    
    # Check parameters
    if len(ygt) != len(regions):
        raise ValueError('Different number of time series for ground-truth and detections.')
    
    # Convert ground-truth to intervals if given as pointwise labels
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


def recall_precision(ygt, regions, overlap = 0.5, th = None):
    """ Computes recall and precision for a list of detected region triples (a, b, score). 
        
    The ground-truth annotations are given by `ygt` as a list of lists of regions (a, b). Each list
    corresponds to one time series and contains the ground-truth intervals for that time series.
    The `regions` list must have the same length as `ygt` and contains an arbitrary number of detected
    intervals for each time series.
    
    `overlap` is a threshold for the Intersection over Union (IoU) criterion. Detections must have an IoU with
    a ground-truth region greater than this threshold for being considered a true positive.
    
    `th` specifies the detection threshold; detections with a lower score will be ignored.
    Set this to None to compute recall and precision over all detections.
    The special value 'all' can be used to compute lists of corresponding recall and precision values for every
    possible threshold (which is more efficient than calling this function once for every threshold).
    
    Returns: a `(recall, precision)` tuple or, where `recall` and `precision` are floats or arrays of floats
             `th` is set to `'all'`. In the latter case, the lists will be sorted in ascending order by recall.
    """
    
    # Wrap ygt and regions in a lists if there is only one time series
    if not (isinstance(ygt[0], list) and (isinstance(ygt[0][0], list) or isinstance(ygt[0][0], tuple))):
        ygt = [ygt]
    if not (isinstance(regions[0], list) and (isinstance(regions[0][0], list) or isinstance(regions[0][0], tuple))):
        regions = [regions]
    if len(ygt) != len(regions):
        raise ValueError('Different number of time series for ground-truth and detections.')
    
    if th == 'all':
    
        # Flatten regions array after associating each region with its time series and sort them descendingly by their score
        sorted_regions = sorted((tuple(region) + (ts,) for ts, detections in enumerate(regions) for region in detections), key = lambda r: r[2], reverse = True)
        
        # Indicators for true and fals positives
        tp = np.zeros(len(sorted_regions))
        fp = np.zeros(len(sorted_regions))
        
        # Determine for each detection if it is a true or a false positive
        region_detected = [[False] * len(gt) for gt in ygt] # prevent multiple detections of the same interval
        for i, (a, b, score, ts) in enumerate(sorted_regions):
            
            isTP = False
            for i_gt, (a_gt, b_gt) in enumerate(ygt[ts]):
                if (not region_detected[ts][i_gt]) and (maxdiv.IoU(a, b - a, a_gt, b_gt - a_gt) >= overlap):
                    isTP = True
                    region_detected[ts][i_gt] = True
                    break
            
            if isTP:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute recall and precision
        tp = tp.cumsum()
        fp = fp.cumsum()
        return (tp / sum(len(gt) for gt in ygt), tp / (tp + fp))
    
    else:
    
        # Count true and false positives for a specific threshold
        tp, fp = 0, 0
        region_detected = [[False] * len(gt) for gt in ygt] # prevent multiple detections of the same interval
        for i in range(len(regions)):
            for a, b, score in regions[i]:
                if (th is None) or (score >= th):
                
                    isTP = False
                    for i_gt, (a_gt, b_gt) in enumerate(ygt[i]):
                        if (not region_detected[i][i_gt]) and (maxdiv.IoU(a, b - a, a_gt, b_gt - a_gt) >= overlap):
                            isTP = True
                            region_detected[i][i_gt] = True
                            break
                    
                    if isTP:
                        tp += 1
                    else:
                        fp += 1
        
        # Calculate recall and precision
        return (float(tp) / sum(len(gt) for gt in ygt), float(tp) / (tp + fp))


def plotDetections(func, regions, gt = [], export = None, detailedvis = False):
    """ Plots a time series and highlights both detected regions and ground-truth regions.
    
    `regions` specifies the detected regions as (a, b, score) tuples.
    `gt` specifies the ground-truth regions either as (a, b) tuples or as pointwise boolean labels.
    If `export` is set to a string, the plot will be saved to that filename instead of being shown.
    """
    
    # Convert pointwise ground-truth to list of regions
    if not (isinstance(gt[0], tuple) or isinstance(gt[0], list)):
        gt = pointwiseLabelsToIntervals(gt)
    
    # Plot time series and ground-truth intervals
    plotted_function = False
    for a, b in gt:
        maxdiv.show_interval(func, a, b, 10000, 'r', 1.0, plot_function = not plotted_function, border = True)
        plotted_function = True
    
    # Plot detected intervals with color intensities corresponding to their score
    minScore = min(r[2] for r in regions)
    maxScore = max(r[2] for r in regions)
    for i in range(len(regions)):
        a, b, score = regions[i]
        print ("Region {}/{}: {} - {} (Score: {})".format(i, len(regions), a, b, score))
        intensity = float(score - minScore) / (maxScore - minScore) if minScore < maxScore else 1.0
        maxdiv.show_interval(func, a, b, 10000, plot_function = not plotted_function,
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
       
    # Display plot
    if export:
        plt.savefig(export)
    else:
        plt.show()


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