""" Checks the difference between two dumps of detections created by run_tests.py for significance. """

import numpy as np
import scipy.stats
import random, csv, argparse

from maxdiv import eval
from maxdiv.maxdiv_util import IoU
import datasets


def testSignificance(det, numSamples = 1000, printProgress = False):
    """ Performs several significance tests for Overall and Mean Average Precision for a given set of detections.
    
    The parameter `det` has to be a list of detections like the one returned by `alignDetections()`, i.e. the
    dictionaries have to contain the keys 'ftype', 'fnum', 'det1', 'det2' and 'num_gt'.
    
    `numSamples` specifies how many permutations or resamples are to be generated for the permutation or the
    bootstrap test, respectively.
    
    Returns: a dictionary with the following items:
        - `ap1`: Overall Average Precision of the first set of detections
        - `ap2`: Overall Average Precision of the second set of detections
        - `ap_diff`: `ap2 - ap1`
        - `map1`: Mean Average Precision of the first set of detections over all function types
        - `map2`: Mean Average Precision of the second set of detections over all function types
        - `map_diff`: `map2 - map1`
        - `ap_permutation_p`: p-Value for `ap_diff` according to the permutation test
        - `ap_permutation_ci`: 99% Confidence Interval for `ap_permutation_p` according to the bootstrap test
        - `ap_bootstrap_ci90`: 90% Confidence Interval for `ap_diff` according to the bootstrap test
        - `ap_bootstrap_ci95`: 95% Confidence Interval for `ap_diff` according to the bootstrap test
        - `ap_bootstrap_ci99`: 99% Confidence Interval for `ap_diff` according to the bootstrap test
        - `map_permutation_p`: p-Value for `map_diff` according to the permutation test
        - `map_permutation_ci`: 99% Confidence Interval for `map_permutation_p` according to the bootstrap test
        - `map_bootstrap_ci90`: 90% Confidence Interval for `map_diff` according to the bootstrap test
        - `map_bootstrap_ci95`: 95% Confidence Interval for `map_diff` according to the bootstrap test
        - `map_bootstrap_ci99`: 99% Confidence Interval for `map_diff` according to the bootstrap test
        - `map_ttest`: p-Value for `map_diff` according to Student's paired t-test
    
    The `map_*` items will only be present if there is more than one function type.
    
    Note that `map_ttest` takes only the AP for each function type into account, while `map_bootstrap_p` and
    `map_permutation_p` resample from the set of all functions, regardless of their type.
    """
    
    # Compute observed differences of Overall AP and Mean AP
    ap1, ap2, ap_diff = apDiff(det)
    aps1, aps2 = apPerType(det)
    map1, map2 = np.mean(aps1), np.mean(aps2)
    map_diff = map2 - map1
    results = { 'ap1' : ap1, 'ap2' : ap2, 'ap_diff' : ap_diff, 'map1' : map1, 'map2' : map2, 'map_diff' : map_diff }
    
    # Run tests for Overall AP
    if printProgress:
        print('Running Permutation Test for Overall AP...')
    p, conf = permutationTest(det, lambda d: apDiff(d)[-1], ap_diff, numSamples)
    results['ap_permutation_p'] = p
    results['ap_permutation_ci'] = conf
    if printProgress:
        print('Running Bootstrap Test for Overall AP...')
    conf = bootstrapTest(det, lambda d: apDiff(d)[-1], numSamples)
    results['ap_bootstrap_ci90'] = conf[0]
    results['ap_bootstrap_ci95'] = conf[1]
    results['ap_bootstrap_ci99'] = conf[2]
    
    # Run tests for Mean AP
    if len(aps1) > 1:
        if printProgress:
            print('Running Permutation Test for Mean AP...')
        p, conf = permutationTest(det, lambda d: mapDiff(d)[-1], map_diff, numSamples)
        results['map_permutation_p'] = p
        results['map_permutation_ci'] = conf
        if printProgress:
            print('Running Bootstrap Test for Mean AP...')
        conf = bootstrapTest(det, lambda d: mapDiff(d)[-1], numSamples)
        results['map_bootstrap_ci90'] = conf[0]
        results['map_bootstrap_ci95'] = conf[1]
        results['map_bootstrap_ci99'] = conf[2]
        results['map_ttest'] = scipy.stats.ttest_rel(aps1, aps2)[1]
    
    return results


def printSignificanceResults(results):
    """ Pretty-prints the significance test results returned from `testSignificance()`. """
    
    indicators = { True : '+', False : '-' }
    
    print('\n--- Overall AP ---\n')
    print('{:.5f} vs. {:.5f} (Difference: {:.5f})\n'.format(results['ap1'], results['ap2'], results['ap_diff']))
    print('Permutation: p = {:.5f} (99% Confidence Interval: [{:.5f}, {:.5f}])'.format(results['ap_permutation_p'], *results['ap_permutation_ci']))
    print('Bootstrap:')
    for alpha in (90, 95, 99):
        ci = results['ap_bootstrap_ci{}'.format(alpha)]
        print('    {}% Confidence Interval: [{:.5f}, {:8.5f}]{}'.format(alpha, ci[0], ci[1], ' - significant difference' if (ci[0] > 0) or (ci[1] < 0) else ''))
    
    if 'map_permutation_p' in results:
        print('\n--- Mean AP ---\n')
        print('{:.5f} vs. {:.5f} (Difference: {:.5f})\n'.format(results['map1'], results['map2'], results['map_diff']))
        print('Student\'s T:  p = {:.5f}'.format(results['map_ttest']))
        print('Permutation:  p = {:.5f} (99% Confidence Interval: [{:.5f}, {:.5f}])'.format(results['map_permutation_p'], *results['map_permutation_ci']))
        print('Bootstrap:')
        for alpha in (90, 95, 99):
            ci = results['map_bootstrap_ci{}'.format(alpha)]
            print('    {}% Confidence Interval: [{:.5f}, {:8.5f}]{}'.format(alpha, ci[0], ci[1], ' - significant difference' if (ci[0] > 0) or (ci[1] < 0) else ''))
    
    print('\n--- Summary ---\n')
    print(' Significance Level | 10% |  5% |  1%')
    print('--------------------------------------')
    print('AP, Permutation     |{:^5s}|{:^5s}|{:^5s}'.format(*tuple(indicators[results['ap_permutation_ci'][1] < th] for th in (0.1, 0.05, 0.01))))
    print('AP, Bootstrap       |{:^5s}|{:^5s}|{:^5s}'.format(*tuple(indicators[(results['ap_bootstrap_ci{}'.format(alpha)][0] > 0) or (results['ap_bootstrap_ci{}'.format(alpha)][1] < 0)] for alpha in (90, 95, 99))))
    if 'map_permutation_p' in results:
        print('--------------------------------------')
        print('MAP, Permutation    |{:^5s}|{:^5s}|{:^5s}'.format(*tuple(indicators[results['map_permutation_ci'][1] < th] for th in (0.1, 0.05, 0.01))))
        print('MAP, Bootstrap      |{:^5s}|{:^5s}|{:^5s}'.format(*tuple(indicators[(results['map_bootstrap_ci{}'.format(alpha)][0] > 0) or (results['map_bootstrap_ci{}'.format(alpha)][1] < 0)] for alpha in (90, 95, 99))))
        print('MAP, Student\'s T    |{:^5s}|{:^5s}|{:^5s}'.format(*tuple(indicators[results['map_ttest'] < th] for th in (0.1, 0.05, 0.01))))
    
    if ((results['ap_permutation_ci'][0] < 0.05) and (results['ap_permutation_ci'][1] > 0.05)) or (('map_permutation_ci' in results) and (results['map_permutation_ci'][0] < 0.05) and (results['map_permutation_ci'][1] > 0.05)):
        print('\nNOTE: To increase the accuracy of the p-Value computed by the permutation test, increase --num_samples.')


def bootstrapTest(det, stat, numSamples = 1000):
    """ Performs the bootstrap significance test.
    
    The bootstrap test estimates confidence intervals of the test statistic by resampling multiple times with replacement
    from the given sample and computing the test statistic for each re-sample. Confidence intervals can then be established
    from percentiles of the bootstrap replicates of the statistic. For example, for a significant improvement of AP on a
    5% significance level, 0 must not be contained in the 95% bootstrap confidence interval.
    
    Reference: Larry Wasserman: "All of Statistics" (section 9.3)
    
    The parameter `det` has to be a list of detections like the one returned by `alignDetections()`, i.e. the
    dictionaries have to contain the keys 'ftype', 'fnum', 'det1', 'det2' and 'num_gt'.
    
    `stat` is a function that, when called with a list in the format of `det`, returns the value of a test statistic.
    
    `obs` is the value of the test statistic to compute a p-value for.
    
    `numSamples` specifies the number of resamples.
    
    Returns: tuple with the 90%, 95% and 99% confidence intervals for the test statistic
    """
    
    stats = [stat([random.choice(det) for j in range(len(det))]) for i in range(numSamples)]
    return tuple((np.percentile(stats, alpha_half), np.percentile(stats, 100 - alpha_half)) for alpha_half in (5, 2.5, 0.5))


def permutationTest(det, stat, obs, numSamples = 1000):
    """ Performs the permutation (aka Fisher's randomization) significance test.
    
    The permutation test assumes the validity of the null-hypothesis that the distribution of the performance
    of the two algorithms to be compared is identical. Under this hypothesis, the assignment of detections
    to algorithms is interchangeable. The permutation test estimates the distribution of the test statistic under
    the null-hypothesis by computing it for all permutations of labels. In this Monte Carlo version, the exact
    p-Value is approximated by computing the statistic for a limited number of random permutations.
    
    The parameter `det` has to be a list of detections like the one returned by `alignDetections()`, i.e. the
    dictionaries have to contain the keys 'ftype', 'fnum', 'det1', 'det2' and 'num_gt'.
    
    `stat` is a function that, when called with a list in the format of `det`, returns the value of a test statistic.
    
    `obs` is the value of the test statistic to compute a p-value for.
    
    `numSamples` specifies the number of permutations.
    
    Returns: tuple with 2 elements:
        - the p-Value of `obs`,
        - a 2-tuple giving the lower and upper bound of the 95% confidence limit of that p-Value.
    """
    
    stats = []
    for i in range(numSamples):
        permDet = []
        for d in det:
            x = random.randint(0, 1)
            permDet.append({
                'ftype' : d['ftype'],
                'fnum'  : d['fnum'],
                'det1'  : d['det1' if x == 0 else 'det2'],
                'det2'  : d['det2' if x == 0 else 'det1'],
                'num_gt': d['num_gt']
            })
        stats.append(stat(permDet))
    p = (np.abs(stats) >= np.abs(obs)).sum() / numSamples
    conf = (scipy.stats.binom.ppf(0.005, n = numSamples, p = p) / numSamples, scipy.stats.binom.ppf(0.995, n = numSamples, p = p) / numSamples)
    if conf[0] > conf[1]:
        conf = (conf[1], conf[1])
    return p, conf


def apDiff(det):
    """ Computes the difference of Overall Average Precision for two sets of detections.
    
    The detections have to be given as a list of dictionaries with the keys 'det1', 'det2'
    and 'num_gt' (number of ground-truth intervals).
    
    Returns: Tuple with 3 elements: AP of the 1st detections, AP of the 2nd detections, difference of both.
    """
    
    num_gt = sum(d['num_gt'] for d in det)
    ap1 = eval.average_precision(num_gt, [d['det1'] for d in det])
    ap2 = eval.average_precision(num_gt, [d['det2'] for d in det])
    return ap1, ap2, ap2 - ap1


def mapDiff(det):
    """ Computes the difference of Mean Average Precision over all function types for two sets of detections.
    
    The detections have to be given as a list of dictionaries with the keys 'ftype', 'det1', 'det2'
    and 'num_gt' (number of ground-truth intervals).
    
    Returns: Tuple with 3 elements: MAP of the 1st detections, MAP of the 2nd detections, difference of both.
    """
    
    aps1, aps2 = apPerType(det)
    map1 = np.mean(aps1)
    map2 = np.mean(aps2)
    return map1, map2, map2 - map1


def apPerType(det):
    """ Computes the Average Precision of two set of detections separately for each function type.
    
    The detections have to be given as a list of dictionaries with the keys 'ftype', 'det1', 'det2'
    and 'num_gt' (number of ground-truth intervals).
    
    Returns: Tuple with 2 lists containing the AP for each function type.
    """
    
    types = set(d['ftype'] for d in det)
    aps1, aps2 = [], []
    for ftype in types:
        num_gt = sum(d['num_gt'] for d in det if d['ftype'] == ftype)
        aps1.append(eval.average_precision(num_gt, [d['det1'] for d in det if d['ftype'] == ftype]))
        aps2.append(eval.average_precision(num_gt, [d['det2'] for d in det if d['ftype'] == ftype]))
    return aps1, aps2


def loadDetectionDump(dumpFile):
    """ Loads detections for a dataset from a CSV file.
    
    The CSV file is expected to have a header row and four columns containing a unique ID of the function
    the respective detection belongs to, the first time step in the detected interval, the first time step
    just outside of the detected interval and the detection score.
    
    This function returns a dictionary which maps function IDs to a list of detections as (a, b, score) tuples.
    """
    
    det = {}
    with open(dumpFile) as f:
        for l in csv.DictReader(f):
            if l['Func'] not in det:
                det[l['Func']] = []
            det[l['Func']].append((int(l['Start']), int(l['End']), float(l['Score'])))
    return det


def alignDetections(det1, det2, funcs, addMissingFuncs = True, quiet = True):
    """ Associates functions in two sets of detections with each other and with the ground-truth from a given dataset.
    
    `det1` and `det2` must be dictionaries mapping unique function IDs to lists of detections given as (a, b, score) tuples
    and `funcs` must be a dataset as returned from `datasets.loadDatasets()`. Function IDs must consist of two parts, a type
    identifier and an index, separated by an underscore, e.g. 'meanshift_hard_005'.
    
    If `addMissingFuncs` is true, functions that are contained in the dataset `funcs`, but neither in `det1` nor in `det2` will
    be added to the resulting list with their number of ground-truth intervals, but with empty lists of detections.
    
    If `quiet` is set to False, errors about functions missing in the dataset or in the list of detections will be printed to
    stdout.
    
    This function returns a list of dictionaries with the following elements:
    - `ftype`: the type identifier of the function
    - `fnum`: the index of the function
    - `det1` and `det2`: lists of detections for this function as (a, b, score, isTP) tuples
    - `num_gt`: number of ground-truth intervals for that function according to the given dataset.
    """
    
    func_ids = set(det1.keys()) | set(det2.keys())
    hits = { ftype : [False] * len(functions) for ftype, functions in funcs.items() }
    det = []
    for id in func_ids:
        
        tok = id.split('_')
        ftype = '_'.join(tok[:-1])
        fnum = int(tok[-1], 10)
        
        if (ftype in funcs) and (len(funcs[ftype]) > fnum):
            hits[ftype][fnum] = True
            gt = funcs[ftype][fnum]['gt']
        else:
            print('Function not found in dataset: {}'.format(id))
            gt = []
        
        det.append({
            'ftype' : ftype,
            'fnum'  : fnum,
            'det1'  : annotateDetections(det1[id], gt) if id in det1 else [],
            'det2'  : annotateDetections(det2[id], gt) if id in det2 else [],
            'num_gt': len(gt)
        })
    
    if addMissingFuncs:
        for ftype, hit in hits.items():
            for i, h in enumerate(hits):
                if not h:
                    print('Adding missing function: {}_{:03d}'.format(ftype, i))
                    det.append({ 'ftype' : ftype, 'fnum'  : i, 'det1'  : [], 'det2'  : [], 'num_gt': len(funcs[ftype][i]['gt']) })
    
    return det


def annotateDetections(det, gt, overlap = 0.5):
    """ Labels each detection in the given list as true or false positive according to a given list of ground-truth intervals.
    
    `det` is expected to be a list of detections as `(a, b, score)` tuples and `gt` must be a list of ground-truth intervals as
    `(a, b)` tuples.
    
    `overlap` is a threshold for the Intersection over Union (IoU) criterion. Detections must have an IoU with
    a ground-truth region greater than this threshold for being considered a true positive.
    
    This function returns a new list of detections as `(a, b, score, isTP)` tuples, where `isTP` specifies whether the detection
    is a true positive or not.
    """
    
    newDet = []
    region_detected = [False] * len(gt) # prevent multiple detections of the same interval
    for a, b, score in det:
        
        isTP = False
        for i_gt, (a_gt, b_gt) in enumerate(gt):
            if (not region_detected[i_gt]) and (IoU(a, b - a, a_gt, b_gt - a_gt) >= overlap):
                isTP = True
                region_detected[i_gt] = True
                break
        
        newDet.append((a, b, score, isTP))
    
    return newDet



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     description = 'Checks the difference between two dumps of detections created by run_tests.py for significance.')
    parser.add_argument('--dump1', help = 'CSV file with the detections of the first algorithm', required = True)
    parser.add_argument('--dump2', help = 'CSV file with the detections of the second algorithm', required = True)
    parser.add_argument('--datasets', help = 'The dataset which the detection dumps refer to', nargs = '+', default = ['synthetic'])
    parser.add_argument('--extremetypes', help = 'Types of extremes which have been tested', nargs = '+', default = datasets.TYPES)
    parser.add_argument('--ignore_missing', help = 'Ignore functions from the dataset which no detections are present for', action = 'store_true')
    parser.add_argument('--num_samples', help = 'Number of samples for permutation and bootstrap test', type = int, default = 1000)

    args = parser.parse_args()
    
    # Load dataset
    data = datasets.loadDatasets(args.datasets, args.extremetypes)
    
    # Load detections
    det1 = loadDetectionDump(args.dump1)
    det2 = loadDetectionDump(args.dump2)
    
    # Associate detections with each other and with ground-truth
    det = alignDetections(det1, det2, data, not args.ignore_missing, quiet = False)
    
    # Run significance tests and print results
    results = testSignificance(det, args.num_samples, printProgress = True)
    printSignificanceResults(results)