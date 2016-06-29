#include "estimators.h"
#include "config.h"
#include <cassert>
#include <cmath>
#include <limits>
using namespace MaxDiv;


//-----------//
// Utilities //
//-----------//

/**
* Computes the sum over a specific attribute of all elements in a given sub-block of a DataTensor based
* on a pre-computed tensor of cumulative sums over all dimensions except the attribute dimension.
*
* @param[in] cumsum A DataTensor with the cumulative sums along all dimensions except the attribute dimension.
*
* @param[in] range The sub-block to compute the sum of. The attribute dimension will be ignored.
*
* @param[in] d The index of the attribute dimension to sum over.
*
* @return Sum over the given attribute over all elements in the given sub-block.
*/
static Scalar sumFromCumsum(const DataTensor & cumsum, const IndexRange & range, DataTensor::Index d)
{
    assert(!range.empty());
    assert(d >= 0 && d < cumsum.shape().d);
    
    // Extracting the sum of a block from a tensor of cumulative sums follows the Inclusion-Exclusion Principle.
    // For example, for two dimensions we have:
    // sum([a1,b1), [a2,b2)) = cumsum(b1, b2) - cumsum(a1 - 1, b2) - cumsum(b1, a2 - 2) + cumsum(a1 - 1, a2 - 1)
    Scalar sum = 0;
    ReflessIndexVector ind;
    ind.d = d;
    unsigned int i, s, numSummands = 1 << (MAXDIV_INDEX_DIMENSION - 1);
    bool isZeroBlock;
    Eigen::Array<bool, MAXDIV_INDEX_DIMENSION - 1, 1> state; // Switches between first and last point of each range
    state.setConstant(false);
    for (s = 0; s < numSummands; ++s)
    {
        
        // Determine index of the bottom right corner of the current block
        for (i = 0, isZeroBlock = false; i < MAXDIV_INDEX_DIMENSION - 1 && !isZeroBlock; ++i)
        {
            ind.ind[i] = (state(i)) ? range.a.ind[i] : range.b.ind[i];
            if (ind.ind[i] == 0)
                isZeroBlock = true;
            else
                ind.ind[i] -= 1;
        }
        
        // Add or subtract value of the block
        if (!isZeroBlock)
        {
            if (state.count() % 2 == 0)
                sum += cumsum(ind);
            else
                sum -= cumsum(ind);
        }
        
        // Move on to next block
        for (i = 0; state(i) && i < MAXDIV_INDEX_DIMENSION - 2; ++i)
            state(i) = false;
        state(i) = true;
        
    }
    return sum;
}

/**
* Computes the sum of all samples in a given sub-block of a DataTensor based on a pre-computed
* tensor of cumulative sums over all dimensions except the attribute dimension.
*
* @param[in] cumsum A DataTensor with the cumulative sums along all dimensions except the attribute dimension.
*
* @param[in] range The sub-block to compute the sum of. The attribute dimension will be ignored.
*
* @return Sum over all samples in the given sub-block.
*/
static Sample sumFromCumsum(const DataTensor & cumsum, const IndexRange & range)
{
    assert(!range.empty());
    
    // Extracting the sum of a block from a tensor of cumulative sums follows the Inclusion-Exclusion Principle.
    // For example, for two dimensions we have:
    // sum([a1,b1), [a2,b2)) = cumsum(b1, b2) - cumsum(a1 - 1, b2) - cumsum(b1, a2 - 2) + cumsum(a1 - 1, a2 - 1)
    Sample sum = Sample::Zero(cumsum.shape().d);
    IndexVector ind = cumsum.makeIndexVector();
    ind.shape.d = 1;
    unsigned int i, s, numSummands = 1 << (MAXDIV_INDEX_DIMENSION - 1);
    bool isZeroBlock;
    Eigen::Array<bool, MAXDIV_INDEX_DIMENSION - 1, 1> state; // Switches between first and last point of each range
    state.setConstant(false);
    for (s = 0; s < numSummands; ++s)
    {
        
        // Determine index of the bottom right corner of the current block
        for (i = 0, isZeroBlock = false; i < MAXDIV_INDEX_DIMENSION - 1 && !isZeroBlock; ++i)
        {
            ind.ind[i] = (state(i)) ? range.a.ind[i] : range.b.ind[i];
            if (ind.ind[i] == 0)
                isZeroBlock = true;
            else
                ind.ind[i] -= 1;
        }
        
        // Add or subtract value of the block
        if (!isZeroBlock)
        {
            if (state.count() % 2 == 0)
                sum += cumsum.sample(ind.linear());
            else
                sum -= cumsum.sample(ind.linear());
        }
        
        // Move on to next block
        for (i = 0; state(i) && i < MAXDIV_INDEX_DIMENSION - 2; ++i)
            state(i) = false;
        state(i) = true;
        
    }
    return sum;
}


//------------------//
// DensityEstimator //
//------------------//

DensityEstimator::DensityEstimator() : m_data(nullptr), m_singletonDim(-1) {}

DensityEstimator::DensityEstimator(const DensityEstimator & other) : m_data(other.m_data), m_singletonDim(other.m_singletonDim) {}

void DensityEstimator::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_data = data;
    this->m_singletonDim = -1;
    for (int d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
        if (data->shape().ind[d] > 1)
        {
            if (this->m_singletonDim == -1)
                this->m_singletonDim = d;
            else
            {
                this->m_singletonDim = -1;
                break;
            }
        }
}

void DensityEstimator::reset()
{
    this->m_data.reset();
}

DataTensor DensityEstimator::pdf() const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return DataTensor();
    
    ReflessIndexVector shape = this->m_data->shape();
    shape.d = 2;
    DataTensor pdf(shape);
    std::pair<Scalar, Scalar> singlePDF;
    IndexVector ind = pdf.makeIndexVector();
    ind.shape.d = 1;
    for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ++ind)
    {
        singlePDF = this->pdf(ind);
        pdf.data()(sample, 0) = singlePDF.first;
        pdf.data()(sample, 1) = singlePDF.second;
    }
    return pdf;
}

DataTensor DensityEstimator::pdf(const IndexRange & range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return DataTensor();
    
    ReflessIndexVector shape = range.shape();
    shape.d = 2;
    DataTensor pdf(shape);
    std::pair<Scalar, Scalar> singlePDF;
    IndexVector ind = pdf.makeIndexVector();
    ind.shape.d = 1;
    for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ++ind)
    {
        singlePDF = this->pdf(range.a + ind);
        pdf.data()(sample, 0) = singlePDF.first;
        pdf.data()(sample, 1) = singlePDF.second;
    }
    return pdf;
}

ScalarMatrix DensityEstimator::pdfOutsideRange(IndexRange range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return Sample();
    
    range.a.d = 0;
    range.b.d = this->m_data->numAttrib();
    ReflessIndexVector rangeShape = range.shape();
    
    ScalarMatrix pdf(this->m_data->numSamples() - rangeShape.prod(0, MAXDIV_INDEX_DIMENSION - 2), 2);
    std::pair<Scalar, Scalar> singlePDF;
    
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (DataTensor::Index pdfInd = 0; ind.t < ind.shape.t; ++ind)
        if (!range.contains(ind))
        {
            singlePDF = this->pdf(ind);
            pdf(pdfInd, 0) = singlePDF.first;
            pdf(pdfInd, 1) = singlePDF.second;
            ++pdfInd;
        }
    return pdf;
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihood() const
{
    std::pair<Scalar, Scalar> ll(0, 0), pdf;
    if (this->m_data == nullptr || this->m_data->empty())
        return ll;
    
    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (; ind.t < ind.shape.t; ++ind)
    {
        pdf = this->pdf(ind);
        ll.first += std::log(pdf.first + eps);
        ll.second += std::log(pdf.second + eps);
    }
    
    return ll;
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihood(const IndexRange & range) const
{
    std::pair<Scalar, Scalar> ll(0, 0), pdf;
    if (this->m_data == nullptr || this->m_data->empty())
        return ll;
    
    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    ReflessIndexVector shape = range.shape();
    shape.d = 1;
    for (IndexVector ind(shape, 0); ind.t < ind.shape.t; ++ind)
    {
        pdf = this->pdf(range.a + ind);
        ll.first += std::log(pdf.first + eps);
        ll.second += std::log(pdf.second + eps);
    }
    
    return ll;
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihoodOutsideRange(IndexRange range) const
{
    std::pair<Scalar, Scalar> ll(0, 0), pdf;
    if (this->m_data == nullptr || this->m_data->empty())
        return ll;
    
    range.a.d = 0;
    range.b.d = this->m_data->numAttrib();
    
    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (; ind.t < ind.shape.t; ++ind)
        if (!range.contains(ind))
        {
            pdf = this->pdf(ind);
            ll.first += std::log(pdf.first + eps);
            ll.second += std::log(pdf.second + eps);
        }
    
    return ll;
}


//------------------------//
// KernelDensityEstimator //
//------------------------//

KernelDensityEstimator::KernelDensityEstimator()
: DensityEstimator(), m_sigma_sq(1.0), m_normed(false), m_kernel(nullptr), m_cumKernel(nullptr), m_extremeRange() {}

KernelDensityEstimator::KernelDensityEstimator(Scalar kernel_sigma_sq, bool normed)
: DensityEstimator(), m_sigma_sq(kernel_sigma_sq), m_normed(normed), m_kernel(nullptr), m_cumKernel(nullptr), m_extremeRange() {}

KernelDensityEstimator::KernelDensityEstimator(const std::shared_ptr<const DataTensor> & data, Scalar kernel_sigma_sq, bool normed)
: DensityEstimator(), m_sigma_sq(kernel_sigma_sq), m_normed(normed), m_kernel(nullptr), m_cumKernel(nullptr), m_extremeRange()
{
    this->init(data);
}

KernelDensityEstimator::KernelDensityEstimator(const KernelDensityEstimator & other)
: DensityEstimator(other),
  m_sigma_sq(other.m_sigma_sq), m_normed(other.m_normed),
  m_kernel(other.m_kernel), m_cumKernel(other.m_cumKernel),
  m_extremeRange(other.m_extremeRange), m_numExtremes(other.m_numExtremes)
{}

KernelDensityEstimator & KernelDensityEstimator::operator=(const KernelDensityEstimator & other)
{
    this->m_data = other.m_data;
    this->m_singletonDim = other.m_singletonDim;
    this->m_sigma_sq = other.m_sigma_sq;
    this->m_normed = other.m_normed;
    this->m_kernel = other.m_kernel;
    this->m_cumKernel = other.m_cumKernel;
    this->m_extremeRange = other.m_extremeRange;
    this->m_numExtremes = other.m_numExtremes;
    return *this;
}

std::shared_ptr<DensityEstimator> KernelDensityEstimator::clone() const
{
    return std::make_shared<KernelDensityEstimator>(*this);
}

void KernelDensityEstimator::init(const std::shared_ptr<const DataTensor> & data)
{
    DensityEstimator::init(data);
    
    this->m_kernel.reset();
    this->m_cumKernel.reset();
    
    if (this->m_data && !this->m_data->empty())
    {
        this->m_kernel.reset(new GaussKernel(*(this->m_data), this->m_sigma_sq, this->m_normed));
        this->m_numExtremes = 0;
        this->m_extremeRange = IndexRange();
        if (this->m_data->numSamples() <= MAXDIV_KDE_CUMULATIVE_SIZE_LIMIT)
        {
            ReflessIndexVector cumShape = this->m_data->shape();
            cumShape.d = this->m_data->numSamples();
            this->m_cumKernel.reset(new DataTensor(cumShape));
            this->m_cumKernel->data() = this->m_kernel->materialize();
            this->m_cumKernel->cumsum(0, MAXDIV_INDEX_DIMENSION - 2);
        }
    }
}

void KernelDensityEstimator::fit(const IndexRange & range)
{
    assert(this->m_data != nullptr);
    assert((range.b.vec() <= this->m_data->shape().vec()).all());
    this->m_extremeRange = range;
    this->m_extremeRange.a.d = 0;
    this->m_extremeRange.b.d = this->m_data->numAttrib();
    this->m_numExtremes = range.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2);
}

void KernelDensityEstimator::reset()
{
    DensityEstimator::reset();
    this->m_kernel.reset();
    this->m_cumKernel.reset();
}

std::pair<Scalar, Scalar> KernelDensityEstimator::pdf(const ReflessIndexVector & ind) const
{
    assert(this->m_data != nullptr && !this->m_data->empty() && !this->m_extremeRange.empty());
    assert((ind.vec() < this->m_data->shape().vec()).all());
    
    // Compute the sum over the kernelized distances from this sample to all samples in the extremal
    // and non-extremal range
    ReflessIndexVector dataShape = this->m_data->shape();
    dataShape.d = 1;
    DataTensor::Index sampleIndex = IndexVector(dataShape, ind).linear();
    Scalar sum_extremes, sum_non_extremes;
    if (this->m_cumKernel)
    {
        ReflessIndexVector lastIndex = dataShape;
        lastIndex.vec() -= 1;
        lastIndex.d = sampleIndex;
        
        if (this->m_singletonDim >= 0)
        {
            // Shortcut for data with only one non-singleton dimension to avoid the rather expensive sumFromCumsum()
            sum_extremes = this->m_cumKernel->data()(this->m_extremeRange.b.ind[this->m_singletonDim] - 1, sampleIndex);
            if (this->m_extremeRange.a.ind[this->m_singletonDim] > 0)
                sum_extremes -= this->m_cumKernel->data()(this->m_extremeRange.a.ind[this->m_singletonDim] - 1, sampleIndex);
        }
        else
            sum_extremes = sumFromCumsum(*(this->m_cumKernel), this->m_extremeRange, sampleIndex);
        sum_non_extremes = (*(this->m_cumKernel))(lastIndex) - sum_extremes;
    }
    else
    {
        sum_extremes = sum_non_extremes = 0;
        Sample kernelCol = this->m_kernel->column(sampleIndex);
        
        IndexVector otherInd = this->m_data->makeIndexVector();
        otherInd.shape.d = 1;
        for (DataTensor::Index otherLinearInd = 0; otherInd.t < otherInd.shape.t; ++otherInd, ++otherLinearInd)
            if (this->m_extremeRange.contains(otherInd))
                sum_extremes += kernelCol(otherLinearInd);
            else
                sum_non_extremes += kernelCol(otherLinearInd);
    }
    
    // Divide the two sums by the number of samples used for their computation
    sum_extremes /= this->m_numExtremes;
    sum_non_extremes /= this->m_data->numSamples() - this->m_numExtremes;
    
    return std::make_pair(sum_extremes, sum_non_extremes);
}


//--------------------------//
// GaussianDensityEstimator //
//--------------------------//

GaussianDensityEstimator::GaussianDensityEstimator()
: DensityEstimator(), m_covMode(CovMode::FULL) {}

GaussianDensityEstimator::GaussianDensityEstimator(CovMode mode)
: DensityEstimator(), m_covMode(mode) {}

GaussianDensityEstimator::GaussianDensityEstimator(const std::shared_ptr<const DataTensor> & data, CovMode mode)
: DensityEstimator(), m_covMode(mode)
{
    this->init(data);
}

GaussianDensityEstimator::GaussianDensityEstimator(const GaussianDensityEstimator & other)
: DensityEstimator(other),
  m_covMode(other.m_covMode), m_cumsum(other.m_cumsum), m_cumOuter(other.m_cumOuter),
  m_innerMean(other.m_innerMean), m_outerMean(other.m_outerMean),
  m_innerCov(other.m_innerCov), m_outerCov(other.m_outerCov),
  m_innerCovChol(other.m_innerCovChol), m_outerCovChol(other.m_outerCovChol),
  m_innerCovLogDet(other.m_innerCovLogDet), m_outerCovLogDet(other.m_outerCovLogDet),
  m_normalizer(other.m_normalizer), m_innerNormalizer(other.m_innerNormalizer), m_outerNormalizer(other.m_outerNormalizer)
{}

GaussianDensityEstimator & GaussianDensityEstimator::operator=(const GaussianDensityEstimator & other)
{
    this->m_data = other.m_data;
    this->m_singletonDim = other.m_singletonDim;
    this->m_covMode = other.m_covMode;
    this->m_cumsum = other.m_cumsum;
    this->m_cumOuter = other.m_cumOuter;
    this->m_innerMean = other.m_innerMean;
    this->m_outerMean = other.m_outerMean;
    this->m_innerCov = other.m_innerCov;
    this->m_outerCov = other.m_outerCov;
    this->m_innerCovChol = other.m_innerCovChol;
    this->m_outerCovChol = other.m_outerCovChol;
    this->m_innerCovLogDet = other.m_innerCovLogDet;
    this->m_outerCovLogDet = other.m_outerCovLogDet;
    this->m_normalizer = other.m_normalizer;
    this->m_innerNormalizer = other.m_innerNormalizer;
    this->m_outerNormalizer = other.m_outerNormalizer;
    return *this;
}

std::shared_ptr<DensityEstimator> GaussianDensityEstimator::clone() const
{
    return std::make_shared<GaussianDensityEstimator>(*this);
}

void GaussianDensityEstimator::init(const std::shared_ptr<const DataTensor> & data)
{
    DensityEstimator::init(data);
    
    this->m_cumOuter.reset();
    
    if (this->m_data && !this->m_data->empty())
    {
        this->m_cumsum.reset(new DataTensor(*(this->m_data)));
        this->m_cumsum->cumsum(0, MAXDIV_INDEX_DIMENSION - 2);
        
        this->m_innerMean.resize(this->m_data->numAttrib());
        this->m_outerMean.resize(this->m_data->numAttrib());
        
        this->m_normalizer = std::pow(2 * M_PI, 0.5 * this->m_data->numAttrib());
        
        if (this->m_covMode == CovMode::FULL)
        {
            this->computeCumOuter(*(this->m_data));
            this->m_innerCov.resize(this->m_data->numAttrib(), this->m_data->numAttrib());
            this->m_outerCov.resize(this->m_data->numAttrib(), this->m_data->numAttrib());
        }
        else if (this->m_covMode == CovMode::SHARED)
        {
            this->m_outerCov = ScalarMatrix();
            this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
            
            // Compute global covariance matrix
            Sample mean = this->m_cumsum->sample(this->m_cumsum->numSamples() - 1) / static_cast<Scalar>(this->m_cumsum->numSamples());
            DataTensor centered = *(this->m_data) - mean;
            this->m_innerCov.noalias() = centered.data().transpose() * centered.data();
            this->m_innerCov /= static_cast<Scalar>(this->m_data->numSamples());
            cholesky(this->m_innerCov, &(this->m_innerCovChol), &(this->m_innerCovLogDet));
            
            // Compute normalizing constant
            this->m_innerNormalizer = this->m_outerNormalizer = this->m_normalizer * std::exp(this->m_innerCovLogDet / 2);
        }
        else
        {
            this->m_innerCov = this->m_outerCov = ScalarMatrix();
            this->m_innerCovChol = Eigen::LLT<ScalarMatrix>();
            this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
            this->m_innerNormalizer = this->m_outerNormalizer = this->m_normalizer;
        }
    }
    else
    {
        this->m_cumsum.reset();
        this->m_innerCov = this->m_outerCov = ScalarMatrix();
        this->m_innerCovChol = Eigen::LLT<ScalarMatrix>();
        this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
    }
}

void GaussianDensityEstimator::computeCumOuter(const DataTensor & data)
{
    ReflessIndexVector outerShape = data.shape();
    outerShape.d *= outerShape.d;
    this->m_cumOuter.reset(new DataTensor(outerShape));
    
    ScalarMatrix singleProd(data.numAttrib(), data.numAttrib());
    Eigen::Map<const Sample> singleProdVec(singleProd.data(), outerShape.d);
    for (DataTensor::Index sample = 0; sample < data.numSamples(); ++sample)
    {
        singleProd.noalias() = data.sample(sample) * data.sample(sample).transpose();
        this->m_cumOuter->sample(sample) = singleProdVec;
    }
        
    this->m_cumOuter->cumsum(0, MAXDIV_INDEX_DIMENSION - 2);
}

void GaussianDensityEstimator::fit(const IndexRange & range)
{
    assert(this->m_data && !this->m_data->empty());
    assert(!range.empty() && (range.b.vec() <= this->m_data->shape().vec()).all());
    
    // Compute the mean of the samples inside and outside of the given range
    DataTensor::Index numExtremes = range.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2);
    DataTensor::Index numNonExtremes = this->m_data->numSamples() - numExtremes;
    if (this->m_singletonDim >= 0)
    {
        // Shortcut for data with only one non-singleton dimension to avoid the rather expensive sumFromCumsum()
        this->m_innerMean = this->m_cumsum->sample(range.b.ind[this->m_singletonDim] - 1);
        if (range.a.ind[this->m_singletonDim] > 0)
            this->m_innerMean -= this->m_cumsum->sample(range.a.ind[this->m_singletonDim] - 1);
    }
    else
        this->m_innerMean = sumFromCumsum(*(this->m_cumsum), range);
    this->m_outerMean = this->m_cumsum->sample(this->m_cumsum->numSamples() - 1) - this->m_innerMean;
    this->m_innerMean /= static_cast<Scalar>(numExtremes);
    this->m_outerMean /= static_cast<Scalar>(numNonExtremes);
    
    // Compute covariance matrices
    if (this->m_covMode == CovMode::FULL)
    {
        Scalar eps = std::numeric_limits<Scalar>::epsilon();
        
        Eigen::Map<Sample> innerCovVec(this->m_innerCov.data(), this->m_cumOuter->numAttrib(), 1);
        Eigen::Map<Sample> outerCovVec(this->m_outerCov.data(), this->m_cumOuter->numAttrib(), 1);
        if (this->m_singletonDim >= 0)
        {
            // Shortcut for data with only one non-singleton dimension to avoid the rather expensive sumFromCumsum()
            innerCovVec = this->m_cumOuter->sample(range.b.ind[this->m_singletonDim] - 1);
            if (range.a.ind[this->m_singletonDim] > 0)
                innerCovVec -= this->m_cumOuter->sample(range.a.ind[this->m_singletonDim] - 1);
        }
        else
            innerCovVec = sumFromCumsum(*(this->m_cumOuter), range);
        outerCovVec = this->m_cumOuter->sample(this->m_cumOuter->numSamples() - 1) - innerCovVec;
        
        this->m_innerCov /= static_cast<Scalar>(numExtremes);
        this->m_outerCov /= static_cast<Scalar>(numNonExtremes);
        
        this->m_innerCov -= this->m_innerMean * this->m_innerMean.transpose();
        this->m_outerCov -= this->m_outerMean * this->m_outerMean.transpose();
        
        // Compute cholesky decomposition and log-determinant
        cholesky(this->m_innerCov, &(this->m_innerCovChol), &(this->m_innerCovLogDet));
        cholesky(this->m_outerCov, &(this->m_outerCovChol), &(this->m_outerCovLogDet));
        
        // Compute normalizing constant
        this->m_innerNormalizer = this->m_normalizer * std::exp(this->m_innerCovLogDet / 2) + eps;
        this->m_outerNormalizer = this->m_normalizer * std::exp(this->m_outerCovLogDet / 2) + eps;
    }
}

void GaussianDensityEstimator::reset()
{
    DensityEstimator::reset();
    this->m_cumsum.reset();
    this->m_cumOuter.reset();
    this->m_innerMean = this->m_outerMean = Sample();
    this->m_innerCov = this->m_outerCov = ScalarMatrix();
    this->m_innerCovChol = Eigen::LLT<ScalarMatrix>();
    this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::pdf(const ReflessIndexVector & ind) const
{
    IndexVector sampleInd = IndexVector(this->m_data->shape(), ind);
    sampleInd.shape.d = 1;
    sampleInd.d = 0;
    const auto sample = this->m_data->sample(sampleInd.linear());
    
    // Compute (x - mu)^T * S^-1 * (x - mu)
    Sample x1 = sample - this->m_innerMean, x2 = sample - this->m_outerMean;
    std::pair<Scalar, Scalar> pdf;
    switch (this->m_covMode)
    {
        case CovMode::FULL:
            pdf.first  = x1.dot(this->m_innerCovChol.solve(x1));
            pdf.second = x2.dot(this->m_outerCovChol.solve(x2));
            break;
        case CovMode::SHARED:
            pdf.first  = x1.dot(this->m_innerCovChol.solve(x1));
            pdf.second = x2.dot(this->m_innerCovChol.solve(x2));
            break;
        default:
            pdf.first  = x1.dot(x1);
            pdf.second = x2.dot(x2);
            break;
    }
    
    // Compute (2*pi)^(-D/2) * |S|^(-1/2) * exp((x - mu)^T * S^-1 * (x - mu))
    pdf.first  = (pdf.first < 1400) ? std::exp(pdf.first / -2) / this->m_innerNormalizer : 0;
    pdf.second = (pdf.first < 1400) ? std::exp(pdf.second / -2) / this->m_outerNormalizer : 0;
    return pdf;
}

DataTensor GaussianDensityEstimator::pdf() const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return DataTensor();
    
    ReflessIndexVector shape = this->m_data->shape();
    shape.d = 2;
    DataTensor pdf(shape);
    
    // Compute (x - mu)^T * S^-1 * (x - mu)
    {
        DataTensor centered = *(this->m_data) - this->m_innerMean;
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(0) = centered.data().rowwise().squaredNorm();
        else
            pdf.data().col(0) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        
        centered = *(this->m_data) - this->m_outerMean;
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(1) = centered.data().rowwise().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        else
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_outerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
    }
    
    // Compute (2*pi)^(-D/2) * |S|^(-1/2) * exp((x - mu)^T * S^-1 * (x - mu))
    pdf.data() = (pdf.data().array().cwiseMin(1400) / -2).exp(); // taking the minimum is just to prevent underflow
    pdf.data().col(0) /= this->m_innerNormalizer;
    pdf.data().col(1) /= this->m_outerNormalizer;
    return pdf;
}

DataTensor GaussianDensityEstimator::pdf(const IndexRange & range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return DataTensor();
    
    ReflessIndexVector shape = range.shape();
    shape.d = 2;
    DataTensor pdf(shape);
    
    // Compute (x - mu)^T * S^-1 * (x - mu)
    {
        shape.d = this->m_data->numAttrib();
        DataTensor centered(range.shape());
        IndexVector ind = pdf.makeIndexVector();
        for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ++ind)
            centered(sample) = (*(this->m_data))(range.a.t + ind.t, range.a.x + ind.x, range.a.y + ind.y, range.a.z + ind.z) - this->m_innerMean;
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(0) = centered.data().rowwise().squaredNorm();
        else
            pdf.data().col(0) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        
        ind = pdf.makeIndexVector();
        for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ++ind)
            centered(sample) = (*(this->m_data))(range.a.t + ind.t, range.a.x + ind.x, range.a.y + ind.y, range.a.z + ind.z) - this->m_outerMean;
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(1) = centered.data().rowwise().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        else
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_outerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
    }
    
    // Compute (2*pi)^(-D/2) * |S|^(-1/2) * exp((x - mu)^T * S^-1 * (x - mu))
    pdf.data() = (pdf.data().array().cwiseMin(1400) / -2).exp(); // taking the minimum is just to prevent underflow
    pdf.data().col(0) /= this->m_innerNormalizer;
    pdf.data().col(1) /= this->m_outerNormalizer;
    return pdf;
}

ScalarMatrix GaussianDensityEstimator::pdfOutsideRange(IndexRange range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return Sample();
    
    range.a.d = 0;
    range.b.d = this->m_data->numAttrib();
    ReflessIndexVector rangeShape = range.shape();
    
    ScalarMatrix pdf(this->m_data->numSamples() - rangeShape.prod(0, MAXDIV_INDEX_DIMENSION - 2), 2);
    
    // Compute (x - mu)^T * S^-1 * (x - mu)
    {
        ScalarMatrix centered(pdf.rows(), this->m_data->numAttrib());
        
        IndexVector ind = this->m_data->makeIndexVector();
        ind.shape.d = 1;
        for (DataTensor::Index pdfInd = 0; ind.t < ind.shape.t; ++ind)
            if (!range.contains(ind))
                centered.row(pdfInd++) = this->m_data->sample(ind.linear()) - this->m_innerMean;
        
        if (this->m_covMode == CovMode::ID)
            pdf.col(0) = centered.rowwise().squaredNorm();
        else
            pdf.col(0) = centered.cwiseProduct(this->m_innerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
        
        ind = this->m_data->makeIndexVector();
        ind.shape.d = 1;
        for (DataTensor::Index pdfInd = 0; ind.t < ind.shape.t; ++ind)
            if (!range.contains(ind))
                centered.row(pdfInd++) = this->m_data->sample(ind.linear()) - this->m_outerMean;
        
        if (this->m_covMode == CovMode::ID)
            pdf.col(1) = centered.rowwise().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            pdf.col(1) = centered.cwiseProduct(this->m_innerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
        else
            pdf.col(1) = centered.cwiseProduct(this->m_outerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
    }
    
    // Compute (2*pi)^(-D/2) * |S|^(-1/2) * exp((x - mu)^T * S^-1 * (x - mu))
    pdf = (pdf.array().cwiseMin(1400) / -2).exp(); // taking the minimum is just to prevent underflow
    pdf.col(0) /= this->m_innerNormalizer;
    pdf.col(1) /= this->m_outerNormalizer;
    return pdf;
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::logLikelihood() const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return std::pair<Scalar, Scalar>(0, 0);
    
    Sample ll(2);
    // Compute sum over distances: (x - mu)^T * S^-1 * (x - mu)
    {
        // inner distribution
        DataTensor centered = *(this->m_data) - this->m_innerMean;
        if (this->m_covMode == CovMode::ID)
            ll(0) = centered.data().squaredNorm();
        else
            ll(0) = centered.data().transpose().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose())).sum();
        
        // outer distribution
        centered = *(this->m_data) - this->m_outerMean;
        if (this->m_covMode == CovMode::ID)
            ll(1) = centered.data().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            ll(1) = centered.data().transpose().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose())).sum();
        else
            ll(1) = centered.data().transpose().cwiseProduct(this->m_outerCovChol.solve(centered.data().transpose())).sum();
    }
    
    // Compute sum over log-probabilities: log((2*pi)^(-D/2) * |S|^(-1/2)) - (x - mu)^T * S^-1 * (x - mu) / 2
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 - this->m_data->numSamples() * std::log(this->m_innerNormalizer),
        ll(1) / -2 - this->m_data->numSamples() * std::log(this->m_outerNormalizer)
    );
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::logLikelihood(const IndexRange & range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return std::pair<Scalar, Scalar>(0, 0);
    
    // Compute sum over distances: (x - mu)^T * S^-1 * (x - mu)
    Sample ll = Sample::Zero(2), centered(this->m_data->numAttrib());
    ReflessIndexVector shape = range.shape();
    shape.d = 1;
    for (IndexVector ind(shape, 0); ind.t < ind.shape.t; ++ind)
    {
        const auto sample = (*(this->m_data))(range.a.t + ind.t, range.a.x + ind.x, range.a.y + ind.y, range.a.z + ind.z);
        
        // inner distribution
        centered = sample - this->m_innerMean;
        if (this->m_covMode == CovMode::ID)
            ll(0) += centered.squaredNorm();
        else
            ll(0) += centered.dot(this->m_innerCovChol.solve(centered));
        
        // outer distribution
        centered = sample - this->m_outerMean;
        if (this->m_covMode == CovMode::ID)
            ll(1) += centered.squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            ll(1) += centered.dot(this->m_innerCovChol.solve(centered));
        else
            ll(1) += centered.dot(this->m_outerCovChol.solve(centered));
    }
    
    // Compute sum over log-probabilities: log((2*pi)^(-D/2) * |S|^(-1/2)) - (x - mu)^T * S^-1 * (x - mu) / 2
    ReflessIndexVector::Index numSamples = shape.prod();
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 - numSamples * std::log(this->m_innerNormalizer),
        ll(1) / -2 - numSamples * std::log(this->m_outerNormalizer)
    );
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::logLikelihoodOutsideRange(IndexRange range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return std::pair<Scalar, Scalar>(0, 0);
    
    range.a.d = 0;
    range.b.d = this->m_data->numAttrib();
    
    // Compute sum over distances: (x - mu)^T * S^-1 * (x - mu)
    Sample ll = Sample::Zero(2), centered(this->m_data->numAttrib());
    ReflessIndexVector::Index numSamples = 0;
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (; ind.t < ind.shape.t; ++ind)
        if (!range.contains(ind))
        {
            ++numSamples;
            const auto sample = this->m_data->sample(ind.linear());
            
            // inner distribution
            centered = sample - this->m_innerMean;
            if (this->m_covMode == CovMode::ID)
                ll(0) += centered.squaredNorm();
            else
                ll(0) += centered.dot(this->m_innerCovChol.solve(centered));
            
            // outer distribution
            centered = sample - this->m_outerMean;
            if (this->m_covMode == CovMode::ID)
                ll(1) += centered.squaredNorm();
            else if (this->m_covMode == CovMode::SHARED)
                ll(1) += centered.dot(this->m_innerCovChol.solve(centered));
            else
                ll(1) += centered.dot(this->m_outerCovChol.solve(centered));
        }
    
    // Compute sum over log-probabilities: log((2*pi)^(-D/2) * |S|^(-1/2)) - (x - mu)^T * S^-1 * (x - mu) / 2
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 - numSamples * std::log(this->m_innerNormalizer),
        ll(1) / -2 - numSamples * std::log(this->m_outerNormalizer)
    );
}

const Scalar GaussianDensityEstimator::mahalanobisDistance(const Eigen::Ref<const Sample> & x1, const Eigen::Ref<const Sample> & x2, bool innerDist) const
{
    Sample diff = x1 - x2;
    if (this->m_covMode == CovMode::ID)
        return diff.squaredNorm();
    else
    {
        const Eigen::LLT<ScalarMatrix> * llt = (innerDist || this->m_covMode == CovMode::SHARED) ? &(this->m_innerCovChol) : &(this->m_outerCovChol);
        return diff.dot(llt->solve(diff));
    }
}
