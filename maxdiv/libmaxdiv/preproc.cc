#include "preproc.h"
#include <chrono>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
using namespace MaxDiv;

DataTensor & PreprocessingPipeline::operator()(DataTensor & data) const
{
    if (this->m_timing.empty())
    {
        for (const_iterator prep = this->begin(); prep != this->end(); ++prep)
            (**prep)(data, data);
    }
    else
    {
        std::size_t i = 0;
        for (const_iterator prep = this->begin(); prep != this->end(); ++prep, ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            (**prep)(data, data);
            auto stop = std::chrono::high_resolution_clock::now();
            this->m_timing[i] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.f;
        }
    }
    return data;
}

void PreprocessingPipeline::enableProfiling(bool enabled)
{
    if (enabled)
        this->m_timing.assign(this->size(), 0);
    else
        this->m_timing.clear();
}


Normalizer::Normalizer() : scaleBySD(false) {};
Normalizer::Normalizer(bool scaleBySD) : scaleBySD(scaleBySD) {};

DataTensor & Normalizer::operator()(DataTensor & data) const
{
    if (!data.empty())
    {
        data -= data.data().colwise().mean();
        if (this->scaleBySD)
            data /= data.data().cwiseAbs2().colwise().mean().cwiseSqrt();
        else
            data /= data.data().cwiseAbs().colwise().maxCoeff();
    }
    return data;
}


LinearDetrending::LinearDetrending() : degree(1), m_storeParams(false), m_params() {};
LinearDetrending::LinearDetrending(unsigned int deg, bool storeParams) : degree(deg), m_storeParams(storeParams), m_params() {};

DataTensor & LinearDetrending::operator()(const DataTensor & dataIn, DataTensor & dataOut) const
{
    if (dataIn.empty())
        return (dataOut = dataIn);
    
    dataOut = dataIn;
    if (this->m_storeParams)
    {
        ReflessIndexVector paramsShape = dataIn.shape();
        paramsShape.t = this->degree + 1;
        this->m_params.resize(paramsShape);
    }
    
    // Construct the design matrix
    ScalarMatrix A(dataIn.length(), this->degree + 1);
    A.col(0).setConstant(1); // intercept term
    for (unsigned int d = 1; d <= this->degree; d++)
    {
        A.col(d).setLinSpaced(static_cast<Scalar>(0), static_cast<Scalar>(dataIn.length() - 1));
        A.col(d) = A.col(d).cwiseProduct(A.col(d - 1));
    }
    Eigen::ColPivHouseholderQR<ScalarMatrix> qr(A);
    
    // Perform least squares estimation for each spatial location and attribute
    auto tsOut = dataOut.asTemporalMatrix();
    Sample params(this->degree + 1);
    ScalarMatrix::Index ts;
    
    #pragma omp parallel for private(params)
    for (ts = 0; ts < tsOut.cols(); ++ts)
    {
        // Fit polynomial
        params = qr.solve(tsOut.col(ts));
        // Subtract trend
        tsOut.col(ts).noalias() -= A * params;
        // Store parameters
        if (this->m_storeParams)
            this->m_params.asTemporalMatrix().col(ts) = params;
    }
    
    return dataOut;
}


OLSDetrending::OLSDetrending(PeriodVector periods, bool linear_trend, bool linear_season_trend, bool store_params)
: periods(periods), linear_trend(linear_trend), linear_season_trend(linear_season_trend), m_storeParams(store_params), m_params()
{}

OLSDetrending::OLSDetrending(Period period, bool linear_trend, bool linear_season_trend, bool store_params)
: periods({period}), linear_trend(linear_trend), linear_season_trend(linear_season_trend), m_storeParams(store_params), m_params()
{}

OLSDetrending::OLSDetrending(unsigned int period, bool linear_trend, bool linear_season_trend, bool store_params)
: periods({ {period, 1} }), linear_trend(linear_trend), linear_season_trend(linear_season_trend), m_storeParams(store_params), m_params()
{}

unsigned int OLSDetrending::getNumParams() const
{
    unsigned int numSeasonCoeffs = this->totalSeasonNum();
    unsigned int numParams = 1 + numSeasonCoeffs;
    if (this->linear_trend)
        numParams += 1;
    if (this->linear_season_trend)
        numParams += numSeasonCoeffs;
    return numParams;
}

unsigned int OLSDetrending::totalSeasonNum() const
{
    unsigned int numSeasonCoeffs = 0;
    for (PeriodVector::const_iterator period = this->periods.begin(); period != this->periods.end(); ++period)
        numSeasonCoeffs += period->num;
    return numSeasonCoeffs;
}

DataTensor & OLSDetrending::operator()(const DataTensor & dataIn, DataTensor & dataOut) const
{
    if (dataIn.empty())
        return (dataOut = dataIn);
    
    dataOut = dataIn;
    unsigned int numParams = this->getNumParams();
    unsigned int numSeasonCoeffs = this->totalSeasonNum();
    
    if (this->m_storeParams)
    {
        ReflessIndexVector paramsShape = dataIn.shape();
        paramsShape.t = numParams;
        this->m_params.resize(paramsShape);
    }
    
    // Construct the design matrix
    ScalarMatrix A(dataIn.length(), numParams);
    A.col(0).setConstant(1); // intercept term
    if (this->linear_trend)
        A.col(1).setLinSpaced(static_cast<Scalar>(0), static_cast<Scalar>(dataIn.length() - 1));
    for (DataTensor::Index t = 0; t < dataIn.length(); t++)
    {
        DataTensor::Index offs = (this->linear_trend) ? 2 : 1;
        for (PeriodVector::const_iterator period = this->periods.begin(); period != this->periods.end(); ++period)
        {
            DataTensor::Index ind = (t / period->len) % period->num;
            A(t, offs + ind) = static_cast<Scalar>(1);
            if (this->linear_season_trend)
                A(t, offs + numSeasonCoeffs + ind) = static_cast<Scalar>(t) / static_cast<Scalar>(period->num);
            offs += period->num;
        }
    }
    // We use the Cholesky decomposition, because it is orders of magnitude faster than the ColPivHouseholderQR,
    // though not as accurate.
    Eigen::LLT<ScalarMatrix> llt(A.transpose() * A);
    
    // Perform least squares estimation for each spatial location and attribute
    auto tsOut = dataOut.asTemporalMatrix();
    Sample params(numParams);
    ScalarMatrix::Index ts;
    
    #pragma omp parallel for private(params)
    for (ts = 0; ts < tsOut.cols(); ++ts)
    {
        // Least squares solving
        params = llt.solve(A.transpose() * tsOut.col(ts));
        // Subtract trend
        tsOut.col(ts).noalias() -= A * params;
        // Store parameters
        if (this->m_storeParams)
            this->m_params.asTemporalMatrix().col(ts) = params;
    }
    
    return dataOut;
}


ZScoreDeseasonalization::ZScoreDeseasonalization(unsigned int period_len) : period_len(period_len) {};

DataTensor & ZScoreDeseasonalization::operator()(DataTensor & data) const
{
    if (data.empty())
        return data;
    
    DataTensor::Index nAttrib = data.numAttrib(),
                      cols = data.shape().prod(1),
                      rowStride = cols * this->period_len,
                      groups = data.length() / this->period_len,
                      overhang = data.length() % this->period_len;
    
    if (nAttrib == 1)
    {
        // Univariate case
        #pragma omp parallel for
        for (unsigned int p = 0; p < this->period_len; ++p)
        {
            Eigen::Map< ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > seasonData(
                data.raw() + p * cols,
                groups + ((p < overhang) ? 1 : 0), cols,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(rowStride, 1)
            );
            seasonData.rowwise() -= seasonData.colwise().mean();
            seasonData.array().rowwise() /= seasonData.cwiseAbs2().colwise().mean().cwiseSqrt().array();
        }
    }
    else
    {
        // Multivariate case
        #pragma omp parallel for
        for (unsigned int p = 0; p < this->period_len; ++p)
        {
            // Wrap the data for this seasonal group with an Eigen::Map
            Eigen::Map< ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > seasonData(
                data.raw() + p * cols,
                groups + ((p < overhang) ? 1 : 0), cols,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(rowStride, 1)
            );
            
            // Subtract the mean of the group
            seasonData.rowwise() -= seasonData.colwise().mean();
            
            // Normalize the time series at each spatial location using its covariance
            // matrix S: x' = S^(-1/2) * x
            ScalarMatrix cov;
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> es;
            for (DataTensor::Index offs = 0; offs < cols; offs += nAttrib)
            {
                auto locSeasonData = seasonData.block(0, offs, seasonData.rows(), nAttrib);
                cov.noalias() = (locSeasonData.transpose() * locSeasonData) / seasonData.rows();
                es.compute(cov);
                locSeasonData = locSeasonData * es.operatorInverseSqrt();
            }
        }
    }
    
    return data;
}
