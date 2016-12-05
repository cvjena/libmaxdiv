#include "preproc.h"
#include "math_utils.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
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

ReflessIndexVector PreprocessingPipeline::borderSize(const DataTensor & data) const
{
    ReflessIndexVector borderSize;
    for (const_iterator prep = this->begin(); prep != this->end(); ++prep)
        borderSize += (*prep)->borderSize(data);
    return borderSize;
}


void PreprocessingPipeline::enableProfiling(bool enabled)
{
    if (enabled)
        this->m_timing.assign(this->size(), 0);
    else
        this->m_timing.clear();
}


DataTensor & TimeDelayEmbedding::operator()(const DataTensor & dataIn, DataTensor & dataOut) const
{
    std::pair<int, int> params = this->getEmbeddingParams(dataIn);
    return (dataOut = time_delay_embedding(dataIn, params.first, params.second, this->borderPolicy));
};

ReflessIndexVector TimeDelayEmbedding::borderSize(const DataTensor & data) const
{
    ReflessIndexVector bs;
    if (this->borderPolicy == BorderPolicy::AUTO || this->borderPolicy == BorderPolicy::VALID)
    {
        std::pair<int, int> params = this->getEmbeddingParams(data);
        bs.t = (params.first - 1) * params.second;
        if (bs.t >= data.length() || (this->borderPolicy == BorderPolicy::AUTO && bs.t * 20 > data.length()))
            bs.t = 0;
    }
    return bs;
};

std::pair<int, int> TimeDelayEmbedding::getEmbeddingParams(const DataTensor & data) const
{
    int k = this->k;
    int T = this->T;
    if (k < 1 || T < 1)
    {
        if (k == 1)
            return std::make_pair(k, 1);
        int contextSize = this->determineContextWindowSize(data);
        if (k < 1 && T < 1)
        {
            int max_k = std::max(5, static_cast<int>(50 / data.numAttrib()));
            T = contextSize / max_k + 1;
        }
        if (k < 1)
            k = std::max(1, static_cast<int>(static_cast<float>(contextSize) / T + 0.5));
        else
            T = std::max(1, static_cast<int>(static_cast<float>(contextSize) / k + 0.5));
    }
    return std::make_pair(k, T);
};

int TimeDelayEmbedding::determineContextWindowSize(const DataTensor & data) const
{
    DataTensor::Index na = data.numAttrib();
    
    // Compute sum along time dimension
    ReflessIndexVector flatShape = data.shape();
    flatShape.t = 1;
    DataTensor sum(flatShape);
    sum.asTemporalMatrix() = data.asTemporalMatrix().colwise().sum();
    
    // Get missing sample mask
    DataTensor::Mask mask;
    if (data.hasMissingSamples())
        data.getMask(mask);
    
    // Compute mutual information for each location and various distances
    flatShape.d = std::min(data.length() / 20, this->maxContextWindowSize);
    DataTensor mi(flatShape);
    mi.channel(0).setConstant(1); // valid location indicator
    DataTensor::Index numValidSamples, numValidLeft, numValidRight, numValidJoint;
    Sample sumLeft(na), sumRight(na), mean(2 * na), centered(2 * na);
    ScalarMatrix cov(2 * na, 2 * na);
    ScalarMatrix indepCov = ScalarMatrix::Zero(2 * na, 2 * na);
    Scalar covLogDet, indepCovLogDet;
    Eigen::LLT<ScalarMatrix> indepCovChol;
    for (IndexVector loc = mi.makeIndexVector(); loc.t < loc.shape.t; ++loc)
        if (loc.d == 0)
        {
            sumLeft.setZero();
            sumRight.setZero();
            numValidSamples = (mask.empty()) ? data.length() : data.length() - mask.asTemporalMatrix().col(loc.prod(1, 3)).count();
            numValidLeft = numValidRight = 0;
        }
        else
        {
            // Sum up samples in cropped regions
            if (mask.empty() || !mask({ loc.d - 1, loc.x, loc.y, loc.z, 0 }))
            {
                sumLeft += data(loc.d - 1, loc.x, loc.y, loc.z);
                ++numValidLeft;
            }
            if (mask.empty() || !mask({ data.length() - loc.d, loc.x, loc.y, loc.z, 0 }))
            {
                sumRight += data(data.length() - loc.d, loc.x, loc.y, loc.z);
                ++numValidRight;
            }
            
            if (numValidSamples <= numValidLeft || numValidSamples <= numValidRight)
            {
                mi(loc) = 1;
                mi({ loc.t, loc.x, loc.y, loc.z, 0 }) = 0; // indicate invalid location
                continue;
            }
            
            // Compute mean and covariance of joint distribution
            mean.head(na) = (sum(0, loc.x, loc.y, loc.z) - sumLeft) / (numValidSamples - numValidLeft);
            mean.tail(na) = (sum(0, loc.x, loc.y, loc.z) - sumRight) / (numValidSamples - numValidRight);
            cov.setZero();
            numValidJoint = 0;
            for (DataTensor::Index t = loc.d; t < data.length(); ++t)
                if (mask.empty() || (!mask({ t, loc.x, loc.y, loc.z, 0 }) && !mask({ t - loc.d, loc.x, loc.y, loc.z, 0 })))
                {
                    centered.head(na) = data(t, loc.x, loc.y, loc.z);
                    centered.tail(na) = data(t - loc.d, loc.x, loc.y, loc.z);
                    centered -= mean;
                    cov.noalias() += centered * centered.transpose();
                    ++numValidJoint;
                }
            if (numValidJoint < 2)
            {
                mi(loc) = 1;
                mi({ loc.t, loc.x, loc.y, loc.z, 0 }) = 0; // indicate invalid location
                continue;
            }
            cov /= numValidJoint - 1;
            
            // Set up covariance of independent distribution
            indepCov.block(0, 0, na, na) = cov.block(0, 0, na, na);
            indepCov.block(na, na, na, na) = cov.block(na, na, na, na);
            
            // Compute KL divergence between p(x_t, x_(t-d)) and p(x_t)*p(x_(t-d))
            cholesky(cov, static_cast<Eigen::LLT<ScalarMatrix>*>(nullptr), &covLogDet);
            cholesky(indepCov, &indepCovChol, &indepCovLogDet);
            mi(loc) = (indepCovChol.solve(cov).trace() + indepCovLogDet - covLogDet - 2 * na) / 2;
        }
    assert((mi.data().array() >= 0).all());
    
    // Normalize MI by dividing by the first value
    for (DataTensor::Index d = 2; d < mi.numAttrib(); ++d)
        mi.channel(d) = mi.channel(d).cwiseQuotient(mi.channel(1));
    
    // Compute negative gradient of relative mutual information
    ScalarMatrix tmpChannel1 = ScalarMatrix::Constant(1, mi.numSamples(), 1);
    ScalarMatrix tmpChannel2(1, mi.numSamples());
    for (DataTensor::Index d = 1; d < mi.numAttrib() - 1; ++d)
    {
        tmpChannel2 = tmpChannel1 - mi.channel(d+1);
        tmpChannel1 = mi.channel(d);
        mi.channel(d) = tmpChannel2;
    }
    
    // Select first context window size below threshold for each location
    std::vector<int> cws;
    cws.reserve(mi.numSamples());
    for (DataTensor::Index loc = 0; loc < mi.numSamples(); ++loc)
    {
        const auto sample = mi.sample(loc);
        if (sample(0) != 0)
        {
            DataTensor::Index minInd = 0, d;
            for (d = 1; d < mi.numAttrib() - 1; ++d)
            {
                if (sample(d) <= opt_th)
                {
                    cws.push_back(d + 1);
                    break;
                }
                if (sample(d) < sample(minInd))
                    minInd = d;
            }
            if (d >= mi.numAttrib() - 1)
                cws.push_back(minInd + 1);
        }
    }
    
    // Return median context window size
    if (cws.empty())
        return 1;
    else if (cws.size() == 1)
        return cws[0];
    else
    {
        std::nth_element(cws.begin(), cws.begin() + cws.size() / 2, cws.end());
        return cws[cws.size() / 2];
    }
};


Normalizer::Normalizer() : scaleBySD(false) {};
Normalizer::Normalizer(bool scaleBySD) : scaleBySD(scaleBySD) {};

DataTensor & Normalizer::operator()(DataTensor & data) const
{
    if (!data.empty())
    {
        data -= data.data().colwise().sum() / static_cast<Scalar>(data.numValidSamples());
        if (this->scaleBySD)
            data /= (data.data().cwiseAbs2().colwise().sum() / static_cast<Scalar>(data.numValidSamples())).cwiseSqrt();
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
        if (d > 1)
            A.col(d) = A.col(d).cwiseProduct(A.col(d - 1));
    }
    Eigen::ColPivHouseholderQR<ScalarMatrix> qr(A);
    
    // Construct selector matrix for exclusion of missing values
    DataTensor::Mask mask;
    if (dataIn.hasMissingSamples())
        dataIn.getMask(mask);
    
    // Perform least squares estimation for each spatial location and attribute
    auto tsOut = dataOut.asTemporalMatrix();
    auto tsMask = mask.asTemporalMatrix();
    Sample params(this->degree + 1);
    ScalarMatrix::Index ts;
    
    #pragma omp parallel for private(ts,params)
    for (ts = 0; ts < tsOut.cols(); ++ts)
    {
        // Fit polynomial
        if (mask.empty() || !tsMask.col(ts / dataIn.numAttrib()).any())
            params = qr.solve(tsOut.col(ts));
        else if (tsMask.rows() - tsMask.col(ts / dataIn.numAttrib()).count() > params.size())
            params = Eigen::ColPivHouseholderQR<ScalarMatrix>(
                        tsMask.col(ts / dataIn.numAttrib()).replicate(1, A.cols()).select(static_cast<Scalar>(0), A)
                     ).solve(tsOut.col(ts));
        else
            params.setZero();
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
    ScalarMatrix A = ScalarMatrix::Zero(dataIn.length(), numParams);
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
    
    // Construct selector matrix for exclusion of missing values
    DataTensor::Mask mask;
    if (dataIn.hasMissingSamples())
        dataIn.getMask(mask);
    
    // Perform least squares estimation for each spatial location and attribute
    auto tsOut = dataOut.asTemporalMatrix();
    auto tsMask = mask.asTemporalMatrix();
    Sample params(numParams);
    ScalarMatrix::Index ts;
    
    #pragma omp parallel for private(params)
    for (ts = 0; ts < tsOut.cols(); ++ts)
    {
        // Least squares solving
        if (mask.empty() || !tsMask.col(ts / dataIn.numAttrib()).any())
            params = llt.solve(A.transpose() * tsOut.col(ts));
        else if (tsMask.rows() - tsMask.col(ts / dataIn.numAttrib()).count() > params.size())
        {
            ScalarMatrix maskedA = tsMask.col(ts / dataIn.numAttrib()).replicate(1, A.cols()).select(static_cast<Scalar>(0), A);
            params = Eigen::LLT<ScalarMatrix>(maskedA.transpose() * maskedA).solve(maskedA.transpose() * tsOut.col(ts));
        }
        else
            params.setZero();
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
                      locs = data.shape().prod(1, 3),
                      cols = locs * nAttrib,
                      rowStride = cols * this->period_len,
                      maskStride = locs * this->period_len,
                      groups = data.length() / this->period_len,
                      overhang = data.length() % this->period_len;
    
    DataTensor::Mask mask;
    if (data.hasMissingSamples())
        data.getMask(mask);
    mask.data() = mask.data().cwiseEqual(false); // cwise not
    
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
            if (mask.empty())
            {
                seasonData.rowwise() -= seasonData.colwise().mean();
                seasonData.array().rowwise() /= seasonData.cwiseAbs2().colwise().mean().cwiseSqrt().array();
            }
            else
            {
                Eigen::Map< DataTensor::Mask::ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > seasonMask(
                    mask.raw() + p * locs,
                    groups + ((p < overhang) ? 1 : 0), locs,
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(maskStride, 1)
                );
                Sample numValidSamples = seasonMask.colwise().count().cwiseMax(1).cast<Scalar>();
                seasonData.rowwise() -= seasonData.colwise().sum().cwiseQuotient(numValidSamples);
                data.setMissingValues();
                seasonData.array().rowwise() /= seasonData.cwiseAbs2().colwise().sum().cwiseQuotient(numValidSamples).cwiseSqrt().array();
            }
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
            
            // Determine number of valid samples per column
            Sample numValidSamples(cols);
            if (mask.empty())
                numValidSamples.setConstant(seasonData.rows());
            else
            {
                Eigen::Map< DataTensor::Mask::ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > seasonMask(
                    mask.raw() + p * locs,
                    groups + ((p < overhang) ? 1 : 0), locs,
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(maskStride, 1)
                );
                for (DataTensor::Index d = 0; d < locs; ++d)
                    numValidSamples.segment(d * nAttrib, nAttrib).setConstant(
                        std::max(static_cast<Scalar>(seasonMask.col(d).count()), static_cast<Scalar>(1))
                    );
            }
            
            // Subtract the mean of the group
            seasonData.rowwise() -= seasonData.colwise().sum().cwiseQuotient(numValidSamples);
            if (!mask.empty())
                data.setMissingValues();
            
            // Normalize the time series at each spatial location using its covariance
            // matrix S: x' = S^(-1/2) * x
            ScalarMatrix cov;
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> es;
            for (DataTensor::Index offs = 0; offs < cols; offs += nAttrib)
            {
                auto locSeasonData = seasonData.block(0, offs, seasonData.rows(), nAttrib);
                cov.noalias() = locSeasonData.transpose() * locSeasonData;
                cov /= numValidSamples(offs);
                es.compute(cov);
                locSeasonData = locSeasonData * es.operatorInverseSqrt();
            }
        }
    }
    
    return data;
}


DataTensor & PCAProjection::operator()(const DataTensor & dataIn, DataTensor & dataOut) const
{
    // Center data
    ScalarMatrix data = dataIn.data();
    data.rowwise() -= data.colwise().sum() / static_cast<Scalar>(dataIn.numValidSamples());
    for (const DataTensor::Index & missing : dataIn.getMissingSampleIndices())
        data.row(missing).setZero();
    
    // Compute covariance matrix
    ScalarMatrix cov;
    cov.noalias() = data.transpose() * data;
    cov /= static_cast<Scalar>(dataIn.numValidSamples() - 1);
    
    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<ScalarMatrix> eigensolver(cov);
    
    // Determine number of principal components to be kept
    DataTensor::Index k = (this->k > 0) ? this->k : 1;
    if (this->variability < 1.0)
    {
        Scalar totalEnergy = eigensolver.eigenvalues().sum(), curEnergy = 0.0;
        for (k = 1; k <= this->k && k <= dataIn.numAttrib(); ++k)
        {
            curEnergy += eigensolver.eigenvalues()(dataIn.numAttrib() - k);
            if (curEnergy >= this->variability * totalEnergy)
                break;
        }
    }
    if (k > dataIn.numAttrib())
        k = dataIn.numAttrib();
    
    // Compose projection matrix of eigenvectors.
    // Eigenvectors are sorted by eigenvalue in increasing order, so we take the last right columns and reverse their order:
    ScalarMatrix proj = eigensolver.eigenvectors().rightCols(k) * Eigen::PermutationMatrix<Eigen::Dynamic>(Eigen::VectorXi::LinSpaced(k, k - 1, 0));
    
    // Project data
    ReflessIndexVector shape = dataIn.shape();
    shape.d = k;
    dataOut.resize(shape);
    dataOut.data().noalias() = data * proj;
    dataOut.copyMask(dataIn);
    
    return dataOut;
}


DataTensor & SparseRandomProjection::operator()(const DataTensor & dataIn, DataTensor & dataOut) const
{
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    
    ReflessIndexVector shape = dataIn.shape();
    
    // Generate random projection vectors
    SparseMatrix proj(this->k, shape.d);
    DataTensor::Index numNonZero = static_cast<DataTensor::Index>(std::sqrt(shape.d) + 0.5);
    proj.reserve(Eigen::VectorXi::Constant(this->k, numNonZero));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> udis(0, shape.d - 1);
    std::normal_distribution<Scalar> ndis;
    
    DataTensor::Index i, j;
    int d;
    for (i = 0; i < this->k; ++i)
    {
        for (j = 0; j < numNonZero; ++j)
        {
            do {
                d = udis(gen);
            } while (proj.coeff(i, d) != 0);
            proj.insert(i, d) = ndis(gen);
        }
        proj.row(i) /= proj.row(i).norm();
    }
    
    // Project data
    shape.d = this->k;
    dataOut.resize(shape);
    dataOut.data().noalias() = dataIn.data() * proj.transpose();
    dataOut.copyMask(dataIn);
    
    return dataOut;
}
