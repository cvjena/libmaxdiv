//  Copyright (C) 2016 Bjoern Barz (University of Jena)
//
//  This file is part of libmaxdiv.
//
//  libmaxdiv is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  libmaxdiv is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with libmaxdiv. If not, see <http://www.gnu.org/licenses/>.

#include "libmaxdiv.h"
#include "search_strategies.h"
#include "pointwise_detectors.h"
#include <vector>
#include <algorithm>
using namespace MaxDiv;


std::vector< std::shared_ptr<SearchStrategy> > maxdiv_pipelines;


void maxdiv_init_params(maxdiv_params_t * params)
{
    if (params == NULL)
        return;
    
    // General Parameters
    params->strategy = MAXDIV_PROPOSAL_SEARCH;
    params->divergence = MAXDIV_KL_DIVERGENCE;
    params->estimator = MAXDIV_GAUSSIAN;
    std::fill(params->min_size, params->min_size + MAXDIV_INDEX_DIMENSION - 1, 0);
    std::fill(params->max_size, params->max_size + MAXDIV_INDEX_DIMENSION - 1, 0);
    params->overlap_th = 0.0;
    
    // Proposal Search Parameters
    params->proposal_generator = MAXDIV_DENSE_PROPOSALS;
    params->pointwise_proposals.gradient_filter = PointwiseProposalGenerator::defaultParams.gradientFilter;
    params->pointwise_proposals.mad = PointwiseProposalGenerator::defaultParams.mad;
    params->pointwise_proposals.sd_th = PointwiseProposalGenerator::defaultParams.sd_th;
    params->pointwise_proposals.kernel_sigma_sq = 1.0;
    
    // Divergence Parameters
    params->kl_mode = MAXDIV_KL_I_OMEGA;
    
    // Estimator Parameters
    params->kernel_sigma_sq = 1.0;
    params->gaussian_cov_mode = MAXDIV_GAUSSIAN_COV_FULL;
    params->erph.num_hist = 100;
    params->erph.num_bins = 0;
    params->erph.discount = 1.0;
    
    // Preprocessing Parameters
    params->preproc.normalization = MAXDIV_NORMALIZE_NONE;
    params->preproc.embedding.kt = 0;
    params->preproc.embedding.kx = 1;
    params->preproc.embedding.ky = 1;
    params->preproc.embedding.kz = 1;
    params->preproc.embedding.dt = 1;
    params->preproc.embedding.dx = 1;
    params->preproc.embedding.dy = 1;
    params->preproc.embedding.dz = 1;
    params->preproc.embedding.temporal_borders = MAXDIV_BORDER_POLICY_AUTO;
    params->preproc.embedding.spatial_borders = MAXDIV_BORDER_POLICY_AUTO;
    params->preproc.detrending.method = MAXDIV_DETREND_NONE;
    params->preproc.detrending.linear_degree = 1;
    params->preproc.detrending.ols_period_num = 0;
    params->preproc.detrending.ols_period_len = 1;
    params->preproc.detrending.ols_linear_trend = true;
    params->preproc.detrending.ols_linear_season_trend = false;
    params->preproc.detrending.z_period_len = 0;
    params->preproc.dimensionality_reduction.method = MAXDIV_PROJECT_NONE;
    params->preproc.dimensionality_reduction.ndims = 0;
}


unsigned int maxdiv_compile_pipeline(const maxdiv_params_t * params)
{
    if (params == NULL)
        return 0;
    
    if (params->strategy != MAXDIV_PROPOSAL_SEARCH)
        return 0;
    
    // Create density estimator
    std::shared_ptr<DensityEstimator> densityEstimator;
    GaussianDensityEstimator::CovMode gaussian_cov_mode = static_cast<GaussianDensityEstimator::CovMode>(params->gaussian_cov_mode);
    switch (params->estimator)
        {
            case MAXDIV_KDE:
                densityEstimator = std::make_shared<KernelDensityEstimator>(params->kernel_sigma_sq);
                break;
            case MAXDIV_GAUSSIAN:
                densityEstimator = std::make_shared<GaussianDensityEstimator>(gaussian_cov_mode);
                break;
            case MAXDIV_ERPH:
                densityEstimator = std::make_shared<EnsembleOfRandomProjectionHistograms>(params->erph.num_hist, params->erph.num_bins, params->erph.discount);
                break;
            default:
                return 0;
        }
    
    // Create divergence
    std::shared_ptr<Divergence> divergence;
    KLDivergence::KLMode kl_mode = static_cast<KLDivergence::KLMode>(params->kl_mode);
    switch (params->divergence)
    {
        case MAXDIV_KL_DIVERGENCE:
            divergence = std::make_shared<KLDivergence>(densityEstimator, kl_mode);
            break;
        case MAXDIV_JS_DIVERGENCE:
            divergence = std::make_shared<JSDivergence>(densityEstimator);
            break;
        case MAXDIV_CROSS_ENTROPY:
            divergence = std::make_shared<CrossEntropy>(densityEstimator, kl_mode);
            break;
        default:
            return 0;
    }
    
    // Create proposal generator
    std::shared_ptr<ProposalGenerator> proposals;
    
    IndexRange lengthRange;
    std::copy(params->min_size, params->min_size + MAXDIV_INDEX_DIMENSION - 1, lengthRange.a.ind);
    std::copy(params->max_size, params->max_size + MAXDIV_INDEX_DIMENSION - 1, lengthRange.b.ind);
    
    PointwiseProposalGenerator::Params ppParams;
    ppParams.gradientFilter = params->pointwise_proposals.gradient_filter;
    ppParams.mad = params->pointwise_proposals.mad;
    ppParams.sd_th = params->pointwise_proposals.sd_th;
    MaxDivScalar ppKernelSigmaSq = params->pointwise_proposals.kernel_sigma_sq;
    
    switch (params->proposal_generator)
    {
        case MAXDIV_DENSE_PROPOSALS:
            proposals = std::make_shared<DenseProposalGenerator>(lengthRange);
            break;
        
        case MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST:
            ppParams.scorer = &hotellings_t;
            proposals = std::make_shared<PointwiseProposalGenerator>(lengthRange, ppParams);
            break;
        
        case MAXDIV_POINTWISE_PROPOSALS_KDE:
            ppParams.scorer = [ppKernelSigmaSq](const DataTensor & data) { return pointwise_kde(data, ppKernelSigmaSq); };
            proposals = std::make_shared<PointwiseProposalGenerator>(lengthRange, ppParams);
            break;
        
        default:
            return 0;
    }
    
    // Create pre-processing pipeline
    std::shared_ptr<PreprocessingPipeline> preproc = std::make_shared<PreprocessingPipeline>();
    
    switch (params->preproc.normalization)
    {
        case MAXDIV_NORMALIZE_NONE:
            break;
        case MAXDIV_NORMALIZE_MAX:
            preproc->push_back(std::make_shared<Normalizer>(false));
            break;
        case MAXDIV_NORMALIZE_SD:
            preproc->push_back(std::make_shared<Normalizer>(true));
            break;
        default:
            return 0;
    }
    
    switch (params->preproc.detrending.method)
    {
        case MAXDIV_DETREND_NONE:
            break;
        case MAXDIV_DETREND_LINEAR:
            preproc->push_back(std::make_shared<LinearDetrending>(params->preproc.detrending.linear_degree));
            break;
        case MAXDIV_DETREND_OLS:
            if (params->preproc.detrending.ols_period_num > 1 && params->preproc.detrending.ols_period_len > 0)
            {
                preproc->push_back(std::make_shared<OLSDetrending>(
                    OLSDetrending::Period{params->preproc.detrending.ols_period_num, params->preproc.detrending.ols_period_len},
                    params->preproc.detrending.ols_linear_trend, params->preproc.detrending.ols_linear_season_trend
                ));
            }
            break;
        case MAXDIV_DETREND_ZSCORE:
            if (params->preproc.detrending.z_period_len > 1)
                preproc->push_back(std::make_shared<ZScoreDeseasonalization>(params->preproc.detrending.z_period_len));
            break;
        default:
            return 0;
    }
    
    if (params->preproc.embedding.kt != 1)
    {
        preproc->push_back(std::make_shared<TimeDelayEmbedding>(
            params->preproc.embedding.kt, params->preproc.embedding.dt, static_cast<BorderPolicy>(params->preproc.embedding.temporal_borders)
        ));
    }
    
    if (params->preproc.embedding.kx > 0 && params->preproc.embedding.ky > 0 && params->preproc.embedding.kz > 0
            && params->preproc.embedding.dx > 0 && params->preproc.embedding.dy > 0 && params->preproc.embedding.dz > 0
            && (params->preproc.embedding.kx > 1 || params->preproc.embedding.ky > 1 || params->preproc.embedding.kz > 1))
    {
        preproc->push_back(std::make_shared<SpatialNeighbourEmbedding>(
            params->preproc.embedding.kx, params->preproc.embedding.ky, params->preproc.embedding.kz,
            params->preproc.embedding.dx, params->preproc.embedding.dy, params->preproc.embedding.dz,
            static_cast<BorderPolicy>(params->preproc.embedding.spatial_borders)
        ));
    }
    
    if (params->preproc.dimensionality_reduction.ndims > 0)
        switch (params->preproc.dimensionality_reduction.method)
        {
            case MAXDIV_PROJECT_NONE:
                break;
            case MAXDIV_PROJECT_PCA:
                preproc->push_back(std::make_shared<PCAProjection>(params->preproc.dimensionality_reduction.ndims));
                break;
            case MAXDIV_PROJECT_RANDOM:
                preproc->push_back(std::make_shared<SparseRandomProjection>(params->preproc.dimensionality_reduction.ndims));
                break;
            default:
                return 0;
        }
    
    // Put everything together and construct the SearchStrategy
    std::shared_ptr<ProposalSearch> detector = std::make_shared<ProposalSearch>(divergence, proposals, preproc);
    detector->setOverlapTh(params->overlap_th);
    maxdiv_pipelines.push_back(detector);
    return maxdiv_pipelines.size();
}


void maxdiv_free_pipeline(unsigned int handle)
{
    if (handle <= maxdiv_pipelines.size())
        maxdiv_pipelines[handle - 1].reset();
}


void maxdiv_exec(unsigned int pipeline, MaxDivScalar * data, const unsigned int * shape,
                 detection_t * detection_buf, unsigned int * detection_buf_size,
                 bool const_data, bool custom_missing_value, MaxDivScalar missing_value)
{
    if (detection_buf_size == NULL || *detection_buf_size == 0)
        return;
    
    // Determine data shape
    ReflessIndexVector dataShape;
    if (shape != NULL)
        std::copy(shape, shape + MAXDIV_INDEX_DIMENSION, dataShape.ind);
    
    // Check parameters
    if (pipeline == 0 || pipeline > maxdiv_pipelines.size() || !maxdiv_pipelines[pipeline - 1] || data == NULL || shape == NULL || dataShape.prod() == 0)
    {
        *detection_buf_size = 0;
        return;
    }
    
    // Run detection pipeline
    DetectionList detections;
    if (const_data)
    {
        if (custom_missing_value)
        {
            std::shared_ptr<DataTensor> data_tensor(new DataTensor(DataTensor(data, dataShape)));
            data_tensor->mask(missing_value);
            detections = (*(maxdiv_pipelines[pipeline - 1]))(data_tensor, *detection_buf_size);
        }
        else
            detections = (*(maxdiv_pipelines[pipeline - 1]))(std::make_shared<const DataTensor>(data, dataShape), *detection_buf_size);
    }
    else
    {
        std::shared_ptr<DataTensor> data_tensor(new DataTensor(data, dataShape));
        if (custom_missing_value)
            data_tensor->mask(missing_value);
        detections = (*(maxdiv_pipelines[pipeline - 1]))(data_tensor, *detection_buf_size);
    }
    
    // Copy detections to the buffer
    detection_t * raw_det = detection_buf;
    for (DetectionList::const_iterator det = detections.begin(); det != detections.end(); ++det, ++raw_det)
    {
        std::copy(det->a.ind, det->a.ind + MAXDIV_INDEX_DIMENSION - 1, raw_det->range_start);
        std::copy(det->b.ind, det->b.ind + MAXDIV_INDEX_DIMENSION - 1, raw_det->range_end);
        raw_det->score = det->score;
    }
    *detection_buf_size = detections.size();
}


void maxdiv_score_intervals(unsigned int pipeline, MaxDivScalar * data, const unsigned int * shape,
                            detection_t * intervals, unsigned int num_intervals,
                            bool const_data, bool custom_missing_value, MaxDivScalar missing_value)
{
    if (intervals == NULL || num_intervals == 0)
        return;
    
    // Determine data shape
    ReflessIndexVector dataShape;
    if (shape != NULL)
        std::copy(shape, shape + MAXDIV_INDEX_DIMENSION, dataShape.ind);
    
    // Check parameters
    if (pipeline == 0 || pipeline > maxdiv_pipelines.size() || !maxdiv_pipelines[pipeline - 1] || data == NULL || shape == NULL || dataShape.prod() == 0)
        return;
    
    // Convert detection_t array to DetectionList
    DetectionList detections;
    IndexVector rangeStart(dataShape, ReflessIndexVector()), rangeEnd(dataShape, ReflessIndexVector());
    for (unsigned int i = 0; i < num_intervals; ++i)
    {
        std::copy(intervals[i].range_start, intervals[i].range_start + MAXDIV_INDEX_DIMENSION - 1, rangeStart.ind);
        std::copy(intervals[i].range_end, intervals[i].range_end + MAXDIV_INDEX_DIMENSION - 1, rangeEnd.ind);
        detections.push_back(Detection(rangeStart, rangeEnd));
    }
    
    // Score intervals
    if (const_data)
    {
        if (custom_missing_value)
        {
            std::shared_ptr<DataTensor> data_tensor(new DataTensor(DataTensor(data, dataShape)));
            data_tensor->mask(missing_value);
            maxdiv_pipelines[pipeline - 1]->scoreIntervals(data_tensor, detections);
        }
        else
            maxdiv_pipelines[pipeline - 1]->scoreIntervals(std::make_shared<const DataTensor>(data, dataShape), detections);
    }
    else
    {
        std::shared_ptr<DataTensor> data_tensor(new DataTensor(data, dataShape));
        if (custom_missing_value)
            data_tensor->mask(missing_value);
        maxdiv_pipelines[pipeline - 1]->scoreIntervals(data_tensor, detections);
    }
    
    // Write computed scores back to intervals buffer
    detection_t * intvl = intervals;
    for (DetectionList::const_iterator det = detections.begin(); det != detections.end(); ++det, ++intvl)
        intvl->score = det->score;
}


void maxdiv(const maxdiv_params_t * params, MaxDivScalar * data, const unsigned int * shape,
            detection_t * detection_buf, unsigned int * detection_buf_size,
            bool const_data, bool custom_missing_value, MaxDivScalar missing_value)
{
    if (detection_buf_size == NULL || *detection_buf_size == 0)
        return;
    
    unsigned int pipeline;
    if (params == NULL || data == NULL || shape == NULL || (pipeline = maxdiv_compile_pipeline(params)) == 0)
    {
        *detection_buf_size = 0;
        return;
    }
    maxdiv_exec(pipeline, data, shape, detection_buf, detection_buf_size, const_data, custom_missing_value, missing_value);
    maxdiv_free_pipeline(pipeline);
}
