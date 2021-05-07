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

/**
* @file
*
* Procedural C-style interface to libmaxdiv, the Maximally Divergent Intervals anomaly detector.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#ifndef LIBMAXDIV
#define LIBMAXDIV

#include "config.h"

#ifdef __cplusplus
extern "C"
{
#endif


/**
* @brief Indices specifying a detected anomalous sub-block of a multi-dimensional tensor, along with a detection score
*/
typedef struct {
    unsigned int range_start[MAXDIV_INDEX_DIMENSION - 1]; /**< Indices of the first point in the sub-block along the first 4 dimensions. */
    unsigned int range_end[MAXDIV_INDEX_DIMENSION - 1]; /**< Indices of the last point in the sub-block along the first 4 dimensions. */
    MaxDivScalar score; /**< The score of the detection (higher is better). */
} detection_t;


enum maxdiv_search_strategy_t { MAXDIV_PROPOSAL_SEARCH };

enum maxdiv_divergence_t
{
    MAXDIV_KL_DIVERGENCE, /**< Kullback-Leibler Divergence */
    MAXDIV_JS_DIVERGENCE, /**< Jensen-Shannon Divergence */
    MAXDIV_CROSS_ENTROPY  /**< Cross-Entropy */
};

enum maxdiv_estimator_t
{
    MAXDIV_KDE,         /**< Kernel Density Estimation */
    MAXDIV_GAUSSIAN,    /**< Normal Distribution */
    MAXDIV_ERPH         /**< Ensemble of Random Projection Histograms */
};

enum maxdiv_proposal_generator_t
{
    MAXDIV_DENSE_PROPOSALS, /**< Dense Proposals (proposes every possible range) */
    MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST, /**< Proposals based an point-wise Hotelling's T^2 scores */
    MAXDIV_POINTWISE_PROPOSALS_KDE /**< Proposals based an point-wise KDE scores */
};

enum maxdiv_kl_mode_t
{
    MAXDIV_KL_I_OMEGA, /**< Integrate over the inner (extremal) range. */
    MAXDIV_KL_OMEGA_I, /**< Integrate over the outer (nominal) range. */
    MAXDIV_KL_SYM,     /**< Symmetric version of the KL divergence: I_OMEGA + OMEGA_I */
    MAXDIV_KL_UNBIASED /**< A variant of I_OMEGA which is normalized regarding the length of the interval and the dimensionality of the time-series. */
};

enum maxdiv_gaussian_cov_mode_t
{
    MAXDIV_GAUSSIAN_COV_FULL,   /**< Use separate covariance matrices for the inner and the outer distribution. */
    MAXDIV_GAUSSIAN_COV_SHARED, /**< Assume a global covariance matrix. */
    MAXDIV_GAUSSIAN_COV_ID      /**< Assume the identity matrix as covariance matrix. */
};

enum maxdiv_border_policy_t
{
    MAXDIV_BORDER_POLICY_AUTO,       /**< Choose `VALID` if the invalid border would not be larger than 5% of the data, otherwise `MIRROR`. */
    MAXDIV_BORDER_POLICY_CONSTANT,   /**< Constant padding with the value which is nearest to the border. */
    MAXDIV_BORDER_POLICY_MIRROR,     /**< Mirror data at the borders. */
    MAXDIV_BORDER_POLICY_VALID       /**< Crop result to the valid region. */
};

enum maxdiv_normalization_t
{
    MAXDIV_NORMALIZE_NONE,  /**< No normalization */
    MAXDIV_NORMALIZE_MAX,   /**< Scale time-series by its maximum. */
    MAXDIV_NORMALIZE_SD     /**< Scale time-series by its standard deviation. */
};

enum maxdiv_detrending_method_t
{
    MAXDIV_DETREND_NONE,    /**< No detrending */
    MAXDIV_DETREND_LINEAR,  /**< Remove linear trends by fitting a polynomial to the data */
    MAXDIV_DETREND_OLS,     /**< Deseasonalization and detrending by Ordinary Least Squares estimation */
    MAXDIV_DETREND_ZSCORE   /**< Z Score deseasonalization */
};

enum maxdiv_projection_method_t
{
    MAXDIV_PROJECT_NONE,    /**< Don't perform dimensionality reduction */
    MAXDIV_PROJECT_PCA,     /**< Use Principal Components Analysis (PCA) for dimensionality reduction */
    MAXDIV_PROJECT_RANDOM   /**< Project data onto sparse random projection vectors */
};


typedef struct {
    
    /* General Parameters */
    maxdiv_search_strategy_t strategy; /**< The strategy used to search for possibly anomalous intervals. */
    maxdiv_divergence_t divergence; /**< The divergence used to measure the deviation of an interval from the rest of the time-series. */
    maxdiv_estimator_t estimator; /**< The distribution model to be fit to the data. */
    unsigned int min_size[MAXDIV_INDEX_DIMENSION - 1]; /**< Minimum size size of the detected ranges along each of the 4 dimensions. */
    unsigned int max_size[MAXDIV_INDEX_DIMENSION - 1]; /**< Maximum size size of the detected ranges along each of the 4 dimensions. `0` means no limit. */
    unsigned int stride[MAXDIV_INDEX_DIMENSION - 1]; /**< Step size between two consecutive ranges to be checked along each of the 4 dimensions. Only applies to dense proposals. */
    MaxDivScalar overlap_th; /**< Overlap threshold for non-maximum suppression: Intervals with a greater IoU will be considered overlapping. */
    
    /* ProposalSearch Parameters */
    maxdiv_proposal_generator_t proposal_generator; /**< The proposal generator to be used. */
    struct
    {
        bool gradient_filter; /**< Specifies whether to apply a gradient filter to the obtained scores. */
        bool mad; /**< Specifies whether to use *Median Absolute Deviation (MAD)* for a robust computation of mean and standard deviation of the scores. */
        MaxDivScalar sd_th; /**< Thresholds for scores will be `mean + sd_th * standard_deviation`. */
        MaxDivScalar kernel_sigma_sq; /**< The variance of the Gauss kernel used by KDE. */
    } pointwise_proposals; /**< Parameters for the proposal generator if `strategy` is `MAXDIV_PROPOSAL_SEARCH` and `proposal_generator` is `MAXDIV_POINTWISE_PROPOSALS_*`. */
    
    /* Divergence Parameters */
    maxdiv_kl_mode_t kl_mode; /**< Variant of the KL divergence. */
    
    /* Estimator Parameters */
    MaxDivScalar kernel_sigma_sq; /**< The variance of the Gauss kernel used by `MAXDIV_KDE`. */
    maxdiv_gaussian_cov_mode_t gaussian_cov_mode; /**< Specifies how the covariance matrix is estimated by `MAXDIV_GAUSSIAN`. */
    struct
    {
        unsigned int num_hist; /**< The number of histograms in the ensemble. */
        unsigned int num_bins; /**< The number of bins per histogram (0 = determine automatically for each histogram). */
        MaxDivScalar discount; /**< Discount to be added to all histogram bins in order to make unseen values not completely unlikely. */
    } erph; /**< Parameters regarding the ERPH estimator. */
    
    /* Preprocessing Parameters */
    struct
    {
        maxdiv_normalization_t normalization; /**< Normalization mode */
        
        struct
        {
            unsigned int kt; /**< Time-Delay Embedding Dimension (1 = no TD embedding, 0 = automatic determination) */
            unsigned int kx; /**< Spatial Embedding Dimension along the first spatial axis */
            unsigned int ky; /**< Spatial Embedding Dimension along the second spatial axis */
            unsigned int kz; /**< Spatial Embedding Dimension along the third spatial axis */
            unsigned int dt; /**< Time-Lag (0 = automatic determination) */
            unsigned int dx; /**< Spacing between neighbours along the first spatial axis */
            unsigned int dy; /**< Spacing between neighbours along the second spatial axis */
            unsigned int dz; /**< Spacing between neighbours along the third spatial axis */
            maxdiv_border_policy_t temporal_borders; /**< Policy to be applied at the beginning of the time series for embedding. */
            maxdiv_border_policy_t spatial_borders; /**< Policy to be applied at the borders of the time series for embedding. */
        } embedding; /**< Parameters for time-delay and spatial-neighbour embedding. */
        
        struct
        {
            maxdiv_detrending_method_t method; /**< Detrending method */
            unsigned int linear_degree; /**< Degree of the polynomial for linear detrending. */
            unsigned int ols_period_num; /**< Number of seasonal units for OLS deseasonalization. */
            unsigned int ols_period_len; /**< Length of each seasonal units for OLS deseasonalization. */
            bool ols_linear_trend; /**< Specifies whether to include a global linear trend in the model. */
            bool ols_linear_season_trend; /**< Specifies whether to include a linear trend of each seasonal unit in the model. */
            unsigned int z_period_len; /**< Number of seasonal groups for Z Score deseasonalization. */
        } detrending; /**< Detrending parameters */
        
        struct
        {
            maxdiv_projection_method_t method; /**< Dimensionality reduction method */
            unsigned int ndims; /**< New number of dimensions. */
        } dimensionality_reduction; /**< Parameters for dimensionality reduction. */
    } preproc; /**< Preprocessing parameters */
    
} maxdiv_params_t;


/**
* Initializes a structure with the parameters for the MaxDiv algorithm with the default values.
*
* @param[in,out] params Pointer to the parameter structure to be initialized.
*/
void maxdiv_init_params(maxdiv_params_t * params);

/**
* Builds a processing pipeline from a set of parameters used by `maxdiv_exec()`.
*
* @param[in] params Pointer to a structure with the parameters for the MDI algorithm. It must have been initialized
* by calling `maxdiv_init_params()`.
*
* @return Returns an internal handle to the pipeline which can be passed to `maxdiv_exec()`. `0` will be returned
* in the case of failure.
*
* @note You have to free the pipeline using `maxdiv_free_pipeline()` when you're done with it to avoid
* memory leaks.
*/
unsigned int maxdiv_compile_pipeline(const maxdiv_params_t * params);

/**
* Frees a processing pipeline built by `maxdiv_compile_pipeline()`.
*
* @param[in] handle The internal handle to the pipeline returned by `maxdiv_compile_pipeline()`.
*/
void maxdiv_free_pipeline(unsigned int handle);


/**
* Uses a processing pipeline built in advance to search for maximally divergent intervals in spatio-temporal data.
*
* @param[in] pipeline The internal handle to the processing pipeline obtained by `maxdiv_compile_pipeline()`.
*
* @param[in] data Pointer to the raw data array. The array must have as many elements as the product of all elements
* of the `shape` array and its memory must be layed out in such a way that the last dimension is changing fastest.
*
* @param[in] shape Pointer to an array with 5 elements which specify the size of each dimension of the given data.  
* Those five dimensions are: time, x, y, z, attribute.  
* The first dimension is the temporal dimension, i.e. the time axis.  
* The second, third and fourth dimensions are spatial dimensions. The distance between each
* pair of consecutive spatial indices is assumed to be constant.  
* The last dimension is the feature or attribute dimension for multivariate time series.
*
* @param[out] detection_buf Pointer to a buffer where the detected intervals will be stored.
*
* @param[in,out] detection_buf_size Pointer to the number of elements allocated for `detection_buf`. The integer
* pointed to will be set to the actual number of elements written to the buffer.
*
* @param[in] const_data If `false`, the given data may be modified. Otherwise, a copy will be made.
*
* @param[in] custom_missing_value If missing values in the given data aren't encoded as `NaN`, but another special
* floating point value, set this to `true` and specify the missing value in `missing_value`.
*
* @param[in] missing_value A special floating point value which is used to encode missing values in the data.
* This parameter has no effect if `custom_missing_value` is set to `false`.
*/
void maxdiv_exec(unsigned int pipeline, MaxDivScalar * data, const unsigned int * shape,
                 detection_t * detection_buf, unsigned int * detection_buf_size,
                 bool const_data = true, bool custom_missing_value = false, MaxDivScalar missing_value = 0);


/**
* Uses a processing pipeline built in advance to compute the anomaly score of individual user-defined intervals in spatio-temporal data.
*
* @param[in] pipeline The internal handle to the processing pipeline obtained by `maxdiv_compile_pipeline()`.
*
* @param[in] data Pointer to the raw data array. The array must have as many elements as the product of all elements
* of the `shape` array and its memory must be layed out in such a way that the last dimension is changing fastest.
*
* @param[in] shape Pointer to an array with 5 elements which specify the size of each dimension of the given data.  
* Those five dimensions are: time, x, y, z, attribute.  
* The first dimension is the temporal dimension, i.e. the time axis.  
* The second, third and fourth dimensions are spatial dimensions. The distance between each
* pair of consecutive spatial indices is assumed to be constant.  
* The last dimension is the feature or attribute dimension for multivariate time series.
*
* @param[in,out] intervals Pointer to an array of `detection_t` objects specifying the intervals to be scored.
* The computed scores will be written to the `score` attribute of the provided objects.
*
* @param[in] num_intervals Number of elements in `intervals`.
*
* @param[in] const_data If `false`, the given data may be modified. Otherwise, a copy will be made.
*
* @param[in] custom_missing_value If missing values in the given data aren't encoded as `NaN`, but another special
* floating point value, set this to `true` and specify the missing value in `missing_value`.
*
* @param[in] missing_value A special floating point value which is used to encode missing values in the data.
* This parameter has no effect if `custom_missing_value` is set to `false`.
*
* @note Out-of-bounds intervals and intervals within the padding area removed during pre-processing will receive a NaN score.
*/
void maxdiv_score_intervals(unsigned int pipeline, MaxDivScalar * data, const unsigned int * shape,
                            detection_t * intervals, unsigned int num_intervals,
                            bool const_data = true, bool custom_missing_value = false, MaxDivScalar missing_value = 0);


/**
* Searches for maximally divergent intervals in spatio-temporal data.
*
* @note This is basically a shortcut for the following sequence of function invocations:
*     1. `maxdiv_compile_pipeline`
*     2. `maxdiv_exec`
*     3. `maxdiv_free_pipeline`
* Thus, if you're intending to call this function multiple times with the same set of parameters, it would be more
* efficient to compile and free the pipeline by yourself and use `maxdiv_exec` instead.
*
* @param[in] params Pointer to a structure with the parameters for the algorithm. It must have been initialized
* by calling `maxdiv_init_params()`.
*
* @param[in] data Pointer to the raw data array. The array must have as many elements as the product of all elements
* of the `shape` array and its memory must be layed out in such a way that the last dimension is changing fastest.
*
* @param[in] shape Pointer to an array with 5 elements which specify the size of each dimension of the given data.  
* Those five dimensions are: time, x, y, z, attribute.  
* The first dimension is the temporal dimension, i.e. the time axis.  
* The second, third and fourth dimensions are spatial dimensions. The distance between each
* pair of consecutive spatial indices is assumed to be constant.  
* The last dimension is the feature or attribute dimension for multivariate time series.
*
* @param[out] detection_buf Pointer to a buffer where the detected intervals will be stored.
*
* @param[in,out] detection_buf_size Pointer to the number of elements allocated for `detection_buf`. The integer
* pointed to will be set to the actual number of elements written to the buffer.
*
* @param[in] const_data If `false`, the given data may be modified. Otherwise, a copy will be made.
*
* @param[in] custom_missing_value If missing values in the given data aren't encoded as `NaN`, but another special
* floating point value, set this to `true` and specify the missing value in `missing_value`.
*
* @param[in] missing_value A special floating point value which is used to encode missing values in the data.
* This parameter has no effect if `custom_missing_value` is set to `false`.
*/
void maxdiv(const maxdiv_params_t * params, MaxDivScalar * data, const unsigned int * shape,
            detection_t * detection_buf, unsigned int * detection_buf_size,
            bool const_data = true, bool custom_missing_value = false, MaxDivScalar missing_value = 0);


#ifdef __cplusplus
}
#endif

#endif
