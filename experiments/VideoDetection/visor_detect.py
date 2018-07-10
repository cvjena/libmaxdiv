""" Runs the MDI algorithm on CNN features extracted from a video an reduced to 16 dimensions using PCA. """

import sys
sys.path.append('../..')

import argparse
import numpy as np
from maxdiv.libmaxdiv_wrapper import maxdiv_exec, libmaxdiv, maxdiv_params_t, enums


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Runs the MDI algorithm on pre-computed CNN features from a video and returns a list of detections.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features', type = str, help = 'Path to a numpy file containing the pre-computed features as 5-dimensional array (time, x, y, 1, channels).')
    parser.add_argument('--proposals', action = 'store_true', default = False, help = 'Use interval proposals to speed up computations.')
    parser.add_argument('--divergence', type = str, choices = ['KL_UNBIASED', 'KL_I_OMEGA', 'CROSS_ENTROPY'], default = 'KL_UNBIASED', help = 'Divergence measure to be used.')
    parser.add_argument('--num', type = int, default = 5, help = 'Number of detections to be returned.')
    args = parser.parse_args()
    
    data = np.load(args.features)

    params = maxdiv_params_t()
    libmaxdiv.maxdiv_init_params(params)
    if args.divergence.startswith('KL_'):
        params.divergence = enums['MAXDIV_KL_DIVERGENCE']
        params.kl_mode = enums['MAXDIV_{}'.format(args.divergence)]
    else:
        params.divergence = enums['MAXDIV_{}'.format(args.divergence)]
    params.min_size[0] = 75
    params.min_size[1] = 10
    params.min_size[2] = 5
    params.max_size[0] = 300
    params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']
    params.preproc.embedding.kt = 3
    params.preproc.embedding.dt = 4
    if args.proposals:
        params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST']

    detections = maxdiv_exec(data, params, args.num)

    for a, b, score in detections:
        print('{:.1f} - {:.1f} s, {}x{} - {}x{} (Score: {})'.format(a[0] / 25.0, (b[0] - 1) / 25.0, a[1], a[2], b[1] - 1, b[2] - 1, score))
