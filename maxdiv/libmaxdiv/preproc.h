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

#ifndef MAXDIV_PREPROC_H
#define MAXDIV_PREPROC_H

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include "DataTensor.h"

namespace MaxDiv
{


/**
* @brief Specifies behaviour to be applied when data beyond the end of a DataTensor would have to be accessed.
*/
enum class BorderPolicy
{
    AUTO, /**< Choose `VALID` if the invalid border would not be larger than 5% of the data, otherwise `MIRROR`. */
    CONSTANT, /**< Constant padding with the value which is nearest to the border. */
    MIRROR, /**< Mirror data at the borders. */
    VALID /**< Crop result to the valid region. */
};


/**
* @brief Time Delay Embedding
*
* Given a positive **time delay** @p T and a positive **embedding dimension** @p k, this
* function transforms each sample `x(t, x, y, z)` in the data tensor @p data at time step t to
* `x' = [x(t, x, y, z), x(t - T, x, y, z), ..., x(t - (k - 1) * T, x, y, z)]`.
*
* Put simply, the attributes of the samples are extended by the attributes of `k-1` previous
* samples, each one with a distance of `T` time steps.
*
* Since there aren't any previous samples at the beginning of the time series, the @p borders
* parameter specifies whether to mirror the time series there, expand it constantly or crop the
* result to the valid region.
*
* @return Returns the extended data tensor.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
template<typename Scalar>
DataTensor_<Scalar> time_delay_embedding(const DataTensor_<Scalar> & data, int k = 3, int T = 1, BorderPolicy borders = BorderPolicy::MIRROR)
{
    assert(k > 0 && T > 0);
    if (k == 1 || data.length() <= 1)
        return data;
    
    // Determine size of invalid border if any
    typename DataTensor_<Scalar>::Index borderSize = 0;
    if (borders == BorderPolicy::AUTO || borders == BorderPolicy::VALID)
    {
        borderSize = (k - 1) * T;
        if (borderSize >= data.length() || (borders == BorderPolicy::AUTO && borderSize * 20 > data.length()))
        {
            borders = BorderPolicy::MIRROR;
            borderSize = 0;
        }
    }
    
    // Create new tensor
    ReflessIndexVector newShape = data.shape();
    newShape.d *= k;
    newShape.t -= borderSize;
    DataTensor_<Scalar> xdata(newShape);
    
    // Set up some variables
    auto newTM = xdata.asTemporalMatrix();
    const auto tm = data.asTemporalMatrix();
    typename DataTensor_<Scalar>::Index samplesPerTime = data.shape().prod(1, 3),
                                        nAttr = data.shape().d,
                                        nAttrEx = newShape.d;
    IndexVector missingInd(data.shape(), 0);
    missingInd.shape.d = 1;
    
    // Copy data
    int t, pt, d;
    typename DataTensor_<Scalar>::Index s, newCol, oldCol;
    for (t = 0; t < newTM.rows(); ++t)
        for (d = 0; d < k; ++d)
        {
            // Determine index of previous time step
            if (borders == BorderPolicy::MIRROR)
            {
                pt = std::abs(t + static_cast<int>(borderSize) - (d * T) % (2 * tm.rows() - 2));
                if (pt >= tm.rows())
                    pt = std::abs(2 * (tm.rows() - 1) - pt);
            }
            else
                pt = std::max(t + static_cast<int>(borderSize) - (d * T), 0);
            
            // Copy sample by sample
            for (s = 0, oldCol = 0, newCol = d * nAttr; s < samplesPerTime; s++, oldCol += nAttr, newCol += nAttrEx)
                newTM.block(t, newCol, 1, nAttr) = tm.block(pt, oldCol, 1, nAttr);
            
            // Propagate missing values
            if (data.hasMissingSamples())
            {
                missingInd.t = pt;
                missingInd.x = missingInd.y = missingInd.z = 0;
                for (; missingInd.t == pt; ++missingInd)
                    if (data.isMissingSample(missingInd))
                        xdata.setMissingSample(t, missingInd.x, missingInd.y, missingInd.z);
            }
        }
    
    return xdata;
}


/**
* @brief Spatial Neighbour Embedding
*
* Given positive **neighbour distances** @p Dx, @p Dy and @p Dz and positive **embedding dimensions**
* @p kx, @p ky and @p kz, this function transforms each sample `x(t, x, y, z)` in the data tensor
* @p data at spatial location `(x, y, z)` to
* `x' = [x(t, x, y, z), x(t, x +/- Dx, y +/- Dy, z +/- Dz), ..., x(t, x +/- (kx-1) * Dx, y +/- (ky-1) * Dy, z +/- (kz-1) * Dz)]`.
*
* Put simply, the attributes of the samples are extended by the attributes of `2 * k - 1` spatial
* neighbours along each dimension, each one with a distance of `D` locations.
* 
* `kx`, `ky` or `kz` may be set to 1 to disable spatial embedding along the respective axis.
*
* The @p borders parameter controls whether the data will be mirrored or constantly expanded on the
* borders or just cropped to the valid region.
*
* @return Returns the extended data tensor.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
template<typename Scalar>
DataTensor_<Scalar> spatial_neighbour_embedding(const DataTensor_<Scalar> & data,
                                                int kx = 2, int ky = 2, int kz = 2,
                                                int Dx = 1, int Dy = 1, int Dz = 1,
                                                BorderPolicy borders = BorderPolicy::MIRROR)
{
    assert(kx > 0 && ky > 0 && kz > 0 && Dx > 0 && Dy > 0 && Dz > 0);
    if (kx == 1 && ky == 1 && kz == 1)
        return data;
    
    // Determine size of invalid border if any
    ReflessIndexVector borderSize;
    if (borders == BorderPolicy::AUTO || borders == BorderPolicy::VALID)
    {
        borderSize.x = (data.width() > 1) ? (kx - 1) * Dx : 0;
        borderSize.y = (data.height() > 1) ? (ky - 1) * Dy : 0;
        borderSize.z = (data.depth() > 1) ? (kz - 1) * Dz : 0;
        if ((borderSize.vec() >= data.shape().vec()).any() || (borders == BorderPolicy::AUTO && (borderSize.vec() * 20 > data.shape().vec()).any()))
        {
            borders = BorderPolicy::MIRROR;
            borderSize.vec().setZero();
        }
    }
    
    // Create new tensor
    ReflessIndexVector newShape = data.shape();
    if (newShape.x <= 1)
        kx = 1;
    if (newShape.y <= 1)
        ky = 1;
    if (newShape.z <= 1)
        kz = 1;
    newShape.d *= (2 * kx - 1) * (2 * ky - 1) * (2 * kz - 1);
    newShape.vec() -= borderSize.vec() * 2;
    DataTensor_<Scalar> xdata(newShape);
    
    // Set up some variables
    typename DataTensor_<Scalar>::Index nLocations = data.shape().prod(1, 3),
                                        nAttr = data.shape().d,
                                        nAttrEx = newShape.d,
                                        missingTime;
    
    // Copy data
    IndexVector loc = xdata.makeIndexVector(); // current location
    loc.shape.t = loc.shape.d = 1;
    IndexVector neighbour = loc;
    int w = data.width(), h = data.height(), d = data.depth();
    int dx, dy, dz, attr;
    for (; loc.t < loc.shape.t; ++loc)
    {
        auto locMat = xdata.location(loc.x, loc.y, loc.z);
        
        for (dx = -1 * kx + 1, attr = 0; dx < kx; dx++)
        {
            // Determine index of x-neighbour
            if (w > 1)
            {
                if (borders == BorderPolicy::MIRROR)
                {
                    neighbour.x = std::abs(static_cast<int>(loc.x + borderSize.x) + (dx * Dx) % (2 * w - 2));
                    if (neighbour.x >= static_cast<IndexVector::Index>(w))
                        neighbour.x = std::abs(2 * (w - 1) - static_cast<int>(neighbour.x));
                }
                else
                    neighbour.x = std::max(std::min(static_cast<int>(loc.x + borderSize.x) + (dx * Dx), w - 1), 0);
            }
            
            for (dy = -1 * ky + 1; dy < ky; dy++)
            {
                // Determine index of y-neighbour
                if (h > 1)
                {
                    if (borders == BorderPolicy::MIRROR)
                    {
                        neighbour.y = std::abs(static_cast<int>(loc.y + borderSize.y) + (dy * Dy) % (2 * h - 2));
                        if (neighbour.y >= static_cast<IndexVector::Index>(h))
                            neighbour.y = std::abs(2 * (h - 1) - static_cast<int>(neighbour.y));
                    }
                    else
                        neighbour.y = std::max(std::min(static_cast<int>(loc.y + borderSize.y) + (dy * Dy), h - 1), 0);
                }
            
                for (dz = -1 * kz + 1; dz < kz; dz++, attr += nAttr)
                {
                    // Determine index of z-neighbour
                    if (d > 1)
                    {
                        if (borders == BorderPolicy::MIRROR)
                        {
                            neighbour.z = std::abs(static_cast<int>(loc.z + borderSize.z) + (dz * Dz) % (2 * d - 2));
                            if (neighbour.z >= static_cast<IndexVector::Index>(d))
                                neighbour.z = std::abs(2 * (d - 1) - static_cast<int>(neighbour.z));
                        }
                        else
                            neighbour.z = std::max(std::min(static_cast<int>(loc.z + borderSize.z) + (dz * Dz), d - 1), 0);
                    }
                    
                    // Copy attributes for all time steps
                    locMat.block(0, attr, locMat.rows(), nAttr) = data.location(neighbour.x, neighbour.y, neighbour.z);
                    
                    // Propagate missing values
                    if (data.hasMissingSamples())
                        for (missingTime = 0; missingTime < newShape.t; ++missingTime)
                            if (data.isMissingSample(missingTime, neighbour.x, neighbour.y, neighbour.z))
                                xdata.setMissingSample(missingTime, loc.x, loc.y, loc.z);
                }
            }
        }
    }
    
    return xdata;
}


/**
* @brief Abstract base class for components of the pre-processing pipeline
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class Preprocessor
{
public:

    /**
    * Applies some pre-processing to @p dataIn and stores the result in @p dataOut.
    * The input and output tensor are allowed to be identical; a temporary copy will be
    * made in this case if necessary.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `dataOut` in order to allow for chaining.
    */
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const =0;
    
    /**
    * Some pre-processors may crop the data to a smaller sub-block. In this case, this method
    * specifies the size of the border that would be cut off at the beginning of the given DataTensor
    * @p data, so that this offset can be added later to detected ranges.
    *
    * @return Returns the size of the border that is cut off at the beginning of each dimension.
    * A vector of zeroes will be returned if no cropping is performed by this pre-processor.
    */
    virtual ReflessIndexVector borderSize(const DataTensor & data) const { return ReflessIndexVector(); };

};


/**
* @brief Abstract base class for pre-processers that can work in-place
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class InplacePreprocessor : public Preprocessor
{
public:

    /**
    * Applies some pre-processing to @p data in-place.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `data` in order to allow for chaining.
    */
    virtual DataTensor & operator()(DataTensor & data) const =0;
    
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override
    {
        return (*this)(dataOut = dataIn);
    };

};


/**
* @brief Pipeline of sequentially executed pre-processing components.
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class PreprocessingPipeline : public std::vector< std::shared_ptr<Preprocessor> >
{
public:

    typedef std::vector< std::shared_ptr<Preprocessor> > Base;
    
    using Base::size_type;
    using Base::iterator;
    using Base::const_iterator;
    
    PreprocessingPipeline() : Base(), m_timing() {};
    explicit PreprocessingPipeline(size_type count) : Base(count), m_timing() {};
    PreprocessingPipeline(const PreprocessingPipeline & other) : Base(other), m_timing() {};
    PreprocessingPipeline(PreprocessingPipeline && other) : Base(other), m_timing() {};
    PreprocessingPipeline(std::initializer_list< std::shared_ptr<Preprocessor> > init) : Base(init), m_timing() {};
    
    /**
    * Applies each pre-processor in the pipeline sequentially to @p data.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `data`.
    */
    virtual DataTensor & operator()(DataTensor & data) const;
    
    /**
    * Applies each pre-processor in the pipeline sequentially to @p dataIn and stores
    * the result in @p dataOut.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `dataOut`.
    */
    inline DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const
    {
        return (*this)(dataOut = dataIn);
    };
    
    /**
    * Some pre-processors may crop the data to a smaller sub-block. In this case, this method
    * specifies the accumulated size of the border that would be cut off at the beginning of the
    * given DataTensor @p data, so that this offset can be added later to detected ranges.
    *
    * @return Returns the size of the border that is cut off at the beginning of each dimension.
    * A vector of zeroes will be returned if no cropping is performed.
    */
    virtual ReflessIndexVector borderSize(const DataTensor & data) const;
    
    virtual void enableProfiling(bool enabled = true);
    
    const std::vector<float> & getTiming() const { return this->m_timing; };

protected:
    
    mutable std::vector<float> m_timing;
    
};


/**
* @brief Time Delay Embedding pre-processor
*
* This is mainly a wrapper around time_delay_embedding().
* Refer to that function's documentation for details.
*
* In addition, this class can automatically determine appropriate parameters for Time-Delay embedding.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class TimeDelayEmbedding : public Preprocessor
{
public:

    int k; /**< Embedding Dimension (may be 0 for automatic determination) */
    int T; /**< Time Delay (may be 0 for automatic determination) */
    BorderPolicy borderPolicy; /**< How to handle the start of the time series */
    Scalar opt_th; /**< Threshold for automatic parameter determination. See `getContextWindowSize()`. */
    DataTensor::Index maxContextWindowSize; /**< Maximum size of context windows for automatic parameter determination. */

    TimeDelayEmbedding() : k(0), T(1), borderPolicy(BorderPolicy::AUTO), opt_th(0.05), maxContextWindowSize(200) {};
    TimeDelayEmbedding(int k, int T = 1, BorderPolicy borders = BorderPolicy::AUTO)
    : k(k), T(T), borderPolicy(borders), opt_th(0.05), maxContextWindowSize(200) {};
    
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override;
    
    /**
    * This method specifies the size of the border that would be cut off at the beginning of the given time series
    * @p data if `borderPolicy` is `VALID` or `AUTO`.
    *
    * @return Returns the size of the border that is cut off at the beginning of each dimension.
    * A vector of zeroes will be returned if no cropping is performed by this pre-processor.
    */
    virtual ReflessIndexVector borderSize(const DataTensor & data) const;
    
    /**
    * Determines the parameters to be used for time-delay embedding.
    *
    * If both the `k` and the `T` attribute of this object are set to a value greater than 0,
    * this method will just return those values.
    *
    * Otherwise, it will determine appropriate values automatically based on Mutual Information.
    *
    * @param[in] data The data tensor to perform time-delay embedding on. If the data contain
    * missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Returns a pair with the values of the parameters `k` and `T` to be used for
    * the time-delay embedding of the given data.
    */
    virtual std::pair<int, int> getEmbeddingParams(const DataTensor & data) const;
    
    /**
    * Determines the size of the relevant context window for a given time-series @p data.
    *
    * The size of this window is determined based on the mutual information between two points
    * in the time-series with varying distance.
    * The `opt_th` attribute sets a threshold on the gradient of MI. If MI drops slowlier than
    * this threshold, the respective context window size will be chosen.
    *
    * For spatio-temporal data, this will be done for each location separately and the median context
    * window size will be returned.
    *
    * @return Returns the number of timesteps contained in the context for a sample in the time-series
    * (including that sample itself).
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual int determineContextWindowSize(const DataTensor & data) const;

};


/**
* @brief Spatial Neighbour Embedding pre-processor
*
* This is just a wrapper around spatial_neighbour_embedding().
* Refer to that function's documentation for details.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class SpatialNeighbourEmbedding : public Preprocessor
{
public:

    int kx; /**< Embedding Dimension along the x-axis */
    int ky; /**< Embedding Dimension along the y-axis */
    int kz; /**< Embedding Dimension along the z-axis */
    int Dx; /**< Spacing between neighbours along the x-axis */
    int Dy; /**< Spacing between neighbours along the y-axis */
    int Dz; /**< Spacing between neighbours along the z-axis */
    BorderPolicy borderPolicy; /**< How to handle the borders of the data */

    SpatialNeighbourEmbedding() : kx(0), ky(0), kz(0), Dx(1), Dy(1), Dz(1), borderPolicy(BorderPolicy::AUTO) {};
    SpatialNeighbourEmbedding(int kx, int ky, int kz, int Dx = 1, int Dy = 1, int Dz = 1, BorderPolicy borders = BorderPolicy::AUTO)
    : kx(kx), ky(ky), kz(kz), Dx(Dx), Dy(Dy), Dz(Dz), borderPolicy(borders) {};
    
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override
    {
        if (kx == 0 && ky == 0 && kz == 0)
            dataOut = spatial_neighbour_embedding(dataIn); // use default parameters
        else
            dataOut = spatial_neighbour_embedding(dataIn, kx, ky, kz, Dx, Dy, Dz, borderPolicy);
        return dataOut;
    };
    
    /**
    * This method specifies the size of the border that would be cut off at the top-left corner of the given
    * DataTensor @p data if `borderPolicy` is `VALID` or `AUTO`.
    *
    * @return Returns the size of the border that is cut off at the beginning of each dimension.
    * A vector of zeroes will be returned if no cropping is performed by this pre-processor.
    */
    virtual ReflessIndexVector borderSize(const DataTensor & data) const
    {
        ReflessIndexVector bs;
        if (this->borderPolicy == BorderPolicy::AUTO || this->borderPolicy == BorderPolicy::VALID)
        {
            bs.x = (data.width() > 1) ? (kx - 1) * Dx : 0;
            bs.y = (data.height() > 1) ? (ky - 1) * Dy : 0;
            bs.z = (data.depth() > 1) ? (kz - 1) * Dz : 0;
            if ((bs.vec() >= data.shape().vec()).any() || (this->borderPolicy == BorderPolicy::AUTO && (bs.vec() * 20 > data.shape().vec()).any()))
                bs.vec().setZero();
        }
        return bs;
    };

};


/**
* @brief Normalizes the values in a tensor
*
* Subtracts the mean values of each attribute from the samples and divides them either by the maximum
* or by the standard deviation of the attributes.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class Normalizer : public InplacePreprocessor
{
public:

    using InplacePreprocessor::operator();

    bool scaleBySD; /**< Specifies if the data should be scaled by their standard deviation or by their maximum. */
    
    Normalizer();
    
    /**
    * @param[in] scaleBySD Specifies if the data should be scaled by their standard deviation (`true`)
    * or by their maximum (`false`).
    */
    Normalizer(bool scaleBySD);
    
    virtual DataTensor & operator()(DataTensor & data) const override;

};


/**
* @brief Removes polynomial (usually linear) trends from a time series.
*
* Fits a polynomial of a given degree to the time series for each spatial location
* and each attribute separately and returns the residuals as de-trended time series.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class LinearDetrending : public Preprocessor
{
public:

    unsigned int degree; /**< Degree of the polynomial to fit to the data. */
    
    LinearDetrending();
    
    /**
    * @param[in] deg Degree of the polynomial to fit to the data.
    * @param[in] storeParams If set to true, the estimated coefficients of the fitted
    * polynomial will be stored, so that they can be retrieved by calling `getParams()`
    * after each call to `operator()`.
    */
    LinearDetrending(unsigned int deg, bool storeParams = false);
    
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override;
    
    /**
    * Returns the parameters of the fitted polynomials computed during the last call
    * to `operator()`.
    *
    * @return Returns a DataTensor with similar shape to the last input data, but with
    * a time dimension of length `degree + 1`. An empty tensor is returned, if this object
    * has been constructed with `storeParams = false`.
    */
    const DataTensor & getParams() const { return this->m_params; };

protected:

    bool m_storeParams;
    mutable DataTensor m_params;

};


/**
* @brief Deseasonalizes and detrends a time series by ordinary least squares.
*
* Each sample `y_t` in the a time series for a fixed spatial location and attribute
* will be modelled according to `y_t = a_0 + b_0 * t + a_j + b_j * t/period_len + e_t`,
* where `j` is the season of the sample.
* The residuals `e_t` will be returned as deseasonalized and detrended time series.
*
* Multivariate time series will be detrended separately dimension by dimension,
* each spatial location will be handled separately as well.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class OLSDetrending : public Preprocessor
{
public:

    struct Period
    {
        unsigned int num;
        unsigned int len;
    };
    
    typedef std::vector<Period> PeriodVector;

    PeriodVector periods; /**< Vector of tupels specifying the number of seasonal units and the length of each unit. */
    bool linear_trend; /**< Specifies whether to include the linear term `b_0 * t` in the model. */
    bool linear_season_trend; /**< Specifies whether to include the terms `b_j * t/period` in the model. */
    
    OLSDetrending() = delete;
    
    /**
    * @param[in] periods A vector of tupels which specify the number of seasonal units and the length
    * of each unit. For example, for hourly sampled data `[(365, 24), (24, 1)]` would mean that
    * there are seasonal effects across the day as well as across the year. This would assume, that
    * the diurnal effects are independent from the seasonal effects across the year. Thus, `[(365*24, 1)]`
    * whould be an alternative, but could mean too many degrees of freedom for a robust estimation from
    * the available data.
    * 
    * @param[in] linear_trend Specifies if the term `b_0 * t` should be included in the model, which corresponds
    * to a linear trend in the data.
    *
    * @param[in] linear_season_trend Specifies if the terms `b_j * t/period` should be included in the model, which
    * correspond to a linear change of the seasonal components. This should be used carefully, since it
    * would interpolate a sudden change in the seasonal pattern as a smooth transition in the model, which
    * will almost never fit the actual data.
    *
    * @param[in] storeParams If set to true, the estimated coefficients of the fitted seasonal component
    * will be stored, so that they can be retrieved by calling `getParams()` after each call to `operator()`.
    */
    OLSDetrending(PeriodVector periods, bool linear_trend = true, bool linear_season_trend = false, bool storeParams = false);
    
    /**
    * @param[in] period A tupel which specifies the number of seasonal units and the length
    * of each unit. For example, for hourly sampled data `(24, 1)` would mean that tere are
    * seasonal effects across the day.
    * 
    * @param[in] linear_trend Specifies if the term `b_0 * t` should be included in the model, which corresponds
    * to a linear trend in the data.
    *
    * @param[in] linear_season_trend Specifies if the terms `b_j * t/period` should be included in the model, which
    * correspond to a linear change of the seasonal components. This should be used carefully, since it
    * would interpolate a sudden change in the seasonal pattern as a smooth transition in the model, which
    * will almost never fit the actual data.
    *
    * @param[in] storeParams If set to true, the estimated coefficients of the fitted seasonal component
    * will be stored, so that they can be retrieved by calling `getParams()` after each call to `operator()`.
    */
    OLSDetrending(Period period, bool linear_trend = true, bool linear_season_trend = false, bool storeParams = false);
    
    /**
    * @param[in] period A tupel which specifies the number of seasonal units. For example, for hourly sampled
    * data `24` would mean that tere are seasonal effects across the day.
    * 
    * @param[in] linear_trend Specifies if the term `b_0 * t` should be included in the model, which corresponds
    * to a linear trend in the data.
    *
    * @param[in] linear_season_trend Specifies if the terms `b_j * t/period` should be included in the model, which
    * correspond to a linear change of the seasonal components. This should be used carefully, since it
    * would interpolate a sudden change in the seasonal pattern as a smooth transition in the model, which
    * will almost never fit the actual data.
    *
    * @param[in] storeParams If set to true, the estimated coefficients of the fitted seasonal component
    * will be stored, so that they can be retrieved by calling `getParams()` after each call to `operator()`.
    */
    explicit OLSDetrending(unsigned int period, bool linear_trend = true, bool linear_season_trend = false, bool storeParams = false);
    
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override;
    
    /**
    * @return Returns the number of parameters of the model.
    */
    virtual unsigned int getNumParams() const;
    
    /**
    * @return Returns the total number of seasonal units in the model, i.e. the sum of the `num` attribute
    * of all periods.
    */
    unsigned int totalSeasonNum() const;
    
    /**
    * Returns the parameters of the fitted seasonal component computed during the last call
    * to `operator()`.
    *
    * @return Returns a DataTensor with similar shape to the last input data, but with
    * a time dimension with the same length as the number of parameters. An empty tensor is returned,
    * if this object has been constructed with `storeParams = false`.
    */
    const DataTensor & getParams() const { return this->m_params; };

protected:

    bool m_storeParams;
    mutable DataTensor m_params;

};


/**
* @brief Deseasonalizes a given time series using the Z Score method.
*
* The time series at each spatial location will be divided into groups of samples with a distance of `period_len`,
* which correspond to a specific time within the period. The mean of each group will then by subtracted from the
* samples in the group and the result will be divided by the standard deviation of the group.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ZScoreDeseasonalization : public InplacePreprocessor
{
public:

    using InplacePreprocessor::operator();

    unsigned int period_len; /**< Number of seasonal groups. */
    
    ZScoreDeseasonalization() = delete;
    
    /**
    * @param[in] period_len The number of seasonal groups (e.g. `24` for hourly sampled data and diurnal seasonality).
    */
    ZScoreDeseasonalization(unsigned int period_len);
    
    virtual DataTensor & operator()(DataTensor & data) const override;

};


/**
* @brief Performs dimensionality reduction using Principal Components Analysis (PCA).
*
* This class computes a mean feature vector \f$m \in \mathbb{R}^d\f$ for the data
* \f$x \in \mathbb{R}^{n \times d}\f$ and a matrix \f$A \in \mathbb{R}^{d \times k}\f$
* that can be used to reduce the dimensionality of the data by computing: \f$x' = (x - m) \cdot A\f$.
*
* PCA is used to find a matrix \f$A\f$ that maximizes the amount of variance in the reduced
* feature space.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class PCAProjection : public Preprocessor
{
public:

    DataTensor::Index k; /**< Maximum number of principal components to keep. */
    Scalar variability; /**< Fraction of the data's variability to capture. */

    PCAProjection() = delete;
    
    /**
    * @param[in] k Number of principal components to keep.
    */
    PCAProjection(DataTensor::Index k) : k(k), variability(1.0) {};
    
    /**
    * @param[in] k Maximum number of principal components to keep.
    *
    * @param[in] v Fraction of the data's variability to capture (value between 0 and 1).
    */
    PCAProjection(DataTensor::Index k, Scalar v) : k(k), variability(v) {};

    /**
    * Computes the first `k` principal components of the samples in @p dataIn and stores them in
    * @p dataOut.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `dataOut` in order to allow for chaining.
    */
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override;

};


/**
* @brief Performs dimensionality reduction using sparse random projection vectors.
*
* A given data set \f$x \in \mathbb{R}^{n \times d}\f$ is projected onto a space with a different dimensionality by
* \f$x' = x \cdot W^T\f$, where \f$W \in \mathbb{R}^{k \times d}\f$ is a collection of \f$n\f$ sparse random projection
* vectors, each of those having only \f$\sqrt{d}\f$ non-zero entries which are drawn from the standard normal distribution.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class SparseRandomProjection : public Preprocessor
{
public:

    DataTensor::Index k; /**< Number of random projections. */

    SparseRandomProjection() = delete;
    
    /**
    * @param[in] k Number of random projections.
    */
    SparseRandomProjection(DataTensor::Index k) : k(k) {};

    /**
    * Applies the random projections to @p dataIn and stores the result in @p dataOut.
    *
    * @note If the data contain missing values, they must have been masked by calling `DataTensor::mask()`.
    *
    * @return Reference to `dataOut` in order to allow for chaining.
    */
    virtual DataTensor & operator()(const DataTensor & dataIn, DataTensor & dataOut) const override;

};

}

#endif
