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

#include "estimators.h"
#include "config.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>
using namespace MaxDiv;


//------------------//
// DensityEstimator //
//------------------//

DensityEstimator::DensityEstimator() : m_data(nullptr), m_extremeRange() {}

DensityEstimator::DensityEstimator(const DensityEstimator & other)
: m_data(other.m_data), m_extremeRange(other.m_extremeRange), m_numExtremes(other.m_numExtremes)
{}

void DensityEstimator::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_data = data;
    this->m_extremeRange = IndexRange();
    this->m_numExtremes = 0;
}

void DensityEstimator::fit(const IndexRange & range)
{
    assert(this->m_data != nullptr);
    assert((range.b.vec() <= this->m_data->shape().vec()).all());
    this->m_extremeRange = range;
    this->m_extremeRange.a.d = 0;
    this->m_extremeRange.b.d = this->m_data->numAttrib();
    this->m_numExtremes = range.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2) - this->m_data->numMissingSamplesInRange(range);
}

void DensityEstimator::reset()
{
    this->m_data.reset();
    this->m_extremeRange = IndexRange();
    this->m_numExtremes = 0;
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

std::pair<Scalar, Scalar> DensityEstimator::logpdf(const ReflessIndexVector & ind) const
{
    if (this->m_data == nullptr || this->m_data->isMissingSample(ind))
        return std::pair<Scalar, Scalar>(0, 0);
    
    Scalar eps = std::numeric_limits<Scalar>::epsilon();
    std::pair<Scalar, Scalar> pdf = this->pdf(ind);
    pdf.first = std::log(pdf.first + eps);
    pdf.second = std::log(pdf.second + eps);
    return pdf;
}

DataTensor DensityEstimator::logpdf() const
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
        singlePDF = this->logpdf(ind);
        pdf.data()(sample, 0) = singlePDF.first;
        pdf.data()(sample, 1) = singlePDF.second;
    }
    return pdf;
}

DataTensor DensityEstimator::logpdf(const IndexRange & range) const
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
        singlePDF = this->logpdf(range.a + ind);
        pdf.data()(sample, 0) = singlePDF.first;
        pdf.data()(sample, 1) = singlePDF.second;
    }
    return pdf;
}

ScalarMatrix DensityEstimator::logpdfOutsideRange(IndexRange range) const
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
            singlePDF = this->logpdf(ind);
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
    
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (; ind.t < ind.shape.t; ++ind)
    {
        pdf = this->logpdf(ind);
        ll.first += pdf.first;
        ll.second += pdf.second;
    }
    
    return ll;
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihoodInner() const
{
    assert(!this->m_extremeRange.empty());
    return this->logLikelihood(this->m_extremeRange);
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihoodOuter() const
{
    assert(!this->m_extremeRange.empty());
    return this->logLikelihoodOutsideRange(this->m_extremeRange);
}

std::pair<Scalar, Scalar> DensityEstimator::logLikelihood(const IndexRange & range) const
{
    std::pair<Scalar, Scalar> ll(0, 0), pdf;
    if (this->m_data == nullptr || this->m_data->empty())
        return ll;
    
    ReflessIndexVector shape = range.shape();
    shape.d = 1;
    for (IndexVector ind(shape, 0); ind.t < ind.shape.t; ++ind)
    {
        pdf = this->logpdf(range.a + ind);
        ll.first += pdf.first;
        ll.second += pdf.second;
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
    
    IndexVector ind = this->m_data->makeIndexVector();
    ind.shape.d = 1;
    for (; ind.t < ind.shape.t; ++ind)
        if (!range.contains(ind))
        {
            pdf = this->logpdf(ind);
            ll.first += pdf.first;
            ll.second += pdf.second;
        }
    
    return ll;
}


//------------------------//
// KernelDensityEstimator //
//------------------------//

KernelDensityEstimator::KernelDensityEstimator()
: DensityEstimator(), m_sigma_sq(1.0), m_normed(false), m_kernel(nullptr), m_cumKernel(nullptr) {}

KernelDensityEstimator::KernelDensityEstimator(Scalar kernel_sigma_sq, bool normed)
: DensityEstimator(), m_sigma_sq(kernel_sigma_sq), m_normed(normed), m_kernel(nullptr), m_cumKernel(nullptr) {}

KernelDensityEstimator::KernelDensityEstimator(const std::shared_ptr<const DataTensor> & data, Scalar kernel_sigma_sq, bool normed)
: DensityEstimator(), m_sigma_sq(kernel_sigma_sq), m_normed(normed), m_kernel(nullptr), m_cumKernel(nullptr)
{
    this->init(data);
}

KernelDensityEstimator::KernelDensityEstimator(const KernelDensityEstimator & other)
: DensityEstimator(other),
  m_sigma_sq(other.m_sigma_sq), m_normed(other.m_normed),
  m_kernel(other.m_kernel), m_cumKernel(other.m_cumKernel)
{}

KernelDensityEstimator & KernelDensityEstimator::operator=(const KernelDensityEstimator & other)
{
    this->m_data = other.m_data;
    this->m_extremeRange = other.m_extremeRange;
    this->m_numExtremes = other.m_numExtremes;
    this->m_sigma_sq = other.m_sigma_sq;
    this->m_normed = other.m_normed;
    this->m_kernel = other.m_kernel;
    this->m_cumKernel = other.m_cumKernel;
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
    
    if (this->m_data->isMissingSample(ind))
        return std::pair<Scalar, Scalar>(1, 1);
    
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
        sum_extremes = this->m_cumKernel->sumFromCumsum(this->m_extremeRange, sampleIndex);
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
    sum_non_extremes /= this->m_data->numValidSamples() - this->m_numExtremes;
    
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
  m_cumOuter_offset(other.m_cumOuter_offset), m_cumOuter_maxLen(other.m_cumOuter_maxLen),
  m_innerMean(other.m_innerMean), m_outerMean(other.m_outerMean),
  m_innerCov(other.m_innerCov), m_outerCov(other.m_outerCov), m_outerProdSum(other.m_outerProdSum),
  m_innerCovChol(other.m_innerCovChol), m_outerCovChol(other.m_outerCovChol),
  m_innerCovLogDet(other.m_innerCovLogDet), m_outerCovLogDet(other.m_outerCovLogDet),
  m_logNormalizer(other.m_logNormalizer), m_innerLogNormalizer(other.m_innerLogNormalizer), m_outerLogNormalizer(other.m_outerLogNormalizer)
{}

GaussianDensityEstimator & GaussianDensityEstimator::operator=(const GaussianDensityEstimator & other)
{
    this->m_data = other.m_data;
    this->m_extremeRange = other.m_extremeRange;
    this->m_numExtremes = other.m_numExtremes;
    this->m_covMode = other.m_covMode;
    this->m_cumsum = other.m_cumsum;
    this->m_cumOuter = other.m_cumOuter;
    this->m_cumOuter_offset = other.m_cumOuter_offset;
    this->m_cumOuter_maxLen = other.m_cumOuter_maxLen;
    this->m_innerMean = other.m_innerMean;
    this->m_outerMean = other.m_outerMean;
    this->m_innerCov = other.m_innerCov;
    this->m_outerCov = other.m_outerCov;
    this->m_outerProdSum = other.m_outerProdSum;
    this->m_innerCovChol = other.m_innerCovChol;
    this->m_outerCovChol = other.m_outerCovChol;
    this->m_innerCovLogDet = other.m_innerCovLogDet;
    this->m_outerCovLogDet = other.m_outerCovLogDet;
    this->m_logNormalizer = other.m_logNormalizer;
    this->m_innerLogNormalizer = other.m_innerLogNormalizer;
    this->m_outerLogNormalizer = other.m_outerLogNormalizer;
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
        
        this->m_logNormalizer = -0.5 * this->m_data->numAttrib() * std::log(2 * M_PI);
        
        if (this->m_covMode == CovMode::FULL)
        {
            // Determine maximum allowed size of cumulative sum of outer products
            this->m_cumOuter_maxLen = MAXDIV_GAUSSIAN_CUMULATIVE_SIZE_LIMIT / (data->shape().prod(1) * data->numAttrib() * sizeof(Scalar));
            if (this->m_cumOuter_maxLen < 20)
                this->m_cumOuter_maxLen = 0;
            
            // Compute (first) cumulative sum of outer products
            this->computeCumOuter();
            
            // Compute sum of outer products of all samples
            this->m_outerProdSum.resize(this->m_data->numAttrib(), this->m_data->numAttrib());
            if (this->m_cumOuter->numSamples() == this->m_data->numSamples())
            {
                Eigen::Map<Sample> outerSumVec(this->m_outerProdSum.data(), this->m_outerProdSum.rows() * this->m_outerProdSum.cols());
                outerSumVec = this->m_cumOuter->sample(this->m_cumOuter->numSamples() - 1);
            }
            else
                this->m_outerProdSum.noalias() = this->m_data->data().transpose() * this->m_data->data();
            
            // Resize covariance matrices
            this->m_innerCov.resize(this->m_data->numAttrib(), this->m_data->numAttrib());
            this->m_outerCov.resize(this->m_data->numAttrib(), this->m_data->numAttrib());
        }
        else if (this->m_covMode == CovMode::SHARED)
        {
            this->m_outerCov = this->m_outerProdSum = ScalarMatrix();
            this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
            
            // Compute global covariance matrix
            Sample mean = this->m_cumsum->sample(this->m_cumsum->numSamples() - 1) / static_cast<Scalar>(this->m_data->numValidSamples());
            DataTensor centered = *(this->m_data) - mean;
            this->m_innerCov.noalias() = centered.data().transpose() * centered.data();
            this->m_innerCov /= static_cast<Scalar>(this->m_data->numValidSamples());
            cholesky(this->m_innerCov, &(this->m_innerCovChol), &(this->m_innerCovLogDet));
            
            // Compute normalizing constant
            this->m_innerLogNormalizer = this->m_outerLogNormalizer = this->m_logNormalizer - this->m_innerCovLogDet / 2;
        }
        else
        {
            this->m_innerCov = this->m_outerCov = this->m_outerProdSum = ScalarMatrix();
            this->m_innerCovChol = Eigen::LLT<ScalarMatrix>();
            this->m_outerCovChol = Eigen::LLT<ScalarMatrix>();
            this->m_innerLogNormalizer = this->m_outerLogNormalizer = this->m_logNormalizer;
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

void GaussianDensityEstimator::computeCumOuter(DataTensor::Index offset)
{
    assert(this->m_data && !this->m_data->empty());
    if (this->m_data)
    {
        // Determine shape of tensor with cumulative sums of outer products for a sub-block of the data
        ReflessIndexVector outerShape = this->m_data->shape();
        outerShape.d *= outerShape.d;
        outerShape.t = (offset < outerShape.t) ? outerShape.t - offset : 0;
        if (outerShape.t > this->m_cumOuter_maxLen)
            outerShape.t = this->m_cumOuter_maxLen;
        
        // Create new tensor
        this->m_cumOuter.reset(new DataTensor(outerShape));
        this->m_cumOuter_offset = offset;
        
        // Compute outer products of the samples in the given sub-block
        ScalarMatrix singleProd(this->m_data->numAttrib(), this->m_data->numAttrib());
        Eigen::Map<const Sample> singleProdVec(singleProd.data(), outerShape.d);
        for (DataTensor::Index cumSample = 0, sample = offset * this->m_data->shape().prod(1, 3); cumSample < this->m_cumOuter->numSamples(); ++cumSample, ++sample)
            if (!this->m_data->isMissingSample(sample))
            {
                singleProd.noalias() = this->m_data->sample(sample) * this->m_data->sample(sample).transpose();
                this->m_cumOuter->sample(cumSample) = singleProdVec;
            }
            else
                this->m_cumOuter->sample(cumSample).setZero();
        
        // Compute cumulative sum of outer products
        this->m_cumOuter->cumsum(0, MAXDIV_INDEX_DIMENSION - 2);
    }
    else
        this->m_cumOuter.reset();
}

ScalarMatrix GaussianDensityEstimator::computeOuterSum(const IndexRange & range)
{
    assert(this->m_data && !this->m_data->empty());
    if (this->m_data)
    {
        ScalarMatrix outerSum(this->m_data->numAttrib(), this->m_data->numAttrib());
        ReflessIndexVector shape = range.shape(), ind;
        shape.d = 1;
        IndexVector offs(range.shape(), 0);
        for (; offs.t < offs.shape.t; ++offs)
        {
            ind = range.a + offs;
            if (!this->m_data->isMissingSample(ind))
            {
                const auto sample = this->m_data->sample(ind);
                outerSum.noalias() += sample * sample.transpose();
            }
        }
        return outerSum;
    }
    else
        return ScalarMatrix();
}

void GaussianDensityEstimator::fit(const IndexRange & range)
{
    DensityEstimator::fit(range);
    
    // Compute the mean of the samples inside and outside of the given range
    DataTensor::Index numNonExtremes = this->m_data->numValidSamples() - this->m_numExtremes;
    assert(this->m_numExtremes > 0 && numNonExtremes > 0);
    this->m_innerMean = this->m_cumsum->sumFromCumsum(range);
    this->m_outerMean = this->m_cumsum->sample(this->m_cumsum->numSamples() - 1) - this->m_innerMean;
    this->m_innerMean /= static_cast<Scalar>(this->m_numExtremes);
    this->m_outerMean /= static_cast<Scalar>(numNonExtremes);
    
    // Compute covariance matrices
    if (this->m_covMode == CovMode::FULL)
    {
        DataTensor::Index rangeLen = range.b.t - range.a.t, cumEnd = this->m_cumOuter_offset + this->m_cumOuter->length();
        
        if (!this->m_cumOuter || this->m_cumOuter->empty() || (rangeLen > this->m_cumOuter_maxLen && (range.b.t <= this->m_cumOuter_offset || range.a.t >= cumEnd)))
            this->m_innerCov = this->computeOuterSum(range);
        else
        {
            // Flat wrapper around m_innerCov
            Eigen::Map<Sample> innerCovVec(this->m_innerCov.data(), this->m_cumOuter->numAttrib(), 1);
            
            // Adjust range covered by partial cumulative sum if it could cover the requested range, but currently doesn't.
            if (rangeLen <= this->m_cumOuter_maxLen && (this->m_cumOuter_offset > range.a.t || cumEnd < range.b.t))
            {
                this->computeCumOuter(range.a.t);
                cumEnd = this->m_cumOuter_offset + this->m_cumOuter->length();
            }
            
            // Determine sub-range which overlaps with the partial cumulative sum
            IndexRange cumRange = range;
            cumRange.a.t = std::max(range.a.t, this->m_cumOuter_offset) - this->m_cumOuter_offset;
            cumRange.b.t = std::min(range.b.t, cumEnd) - this->m_cumOuter_offset;
            assert(range.b.t > this->m_cumOuter_offset);
            assert(cumRange.b.t <= this->m_cumOuter->length());
            
            // Extract sum from the cumulative sum tensor
            innerCovVec = this->m_cumOuter->sumFromCumsum(cumRange);
            
            // Add sum over sub-range which is not covered by the cumulative sum
            if (range.a.t < this->m_cumOuter_offset)
            {
                cumRange.a.t = range.a.t;
                cumRange.b.t = this->m_cumOuter_offset;
                this->m_innerCov += this->computeOuterSum(cumRange);
            }
            if (range.b.t > cumEnd)
            {
                cumRange.a.t = cumEnd;
                cumRange.b.t = range.b.t;
                this->m_innerCov += this->computeOuterSum(cumRange);
            }
        }
        this->m_outerCov = this->m_outerProdSum - this->m_innerCov;
        
        this->m_innerCov /= static_cast<Scalar>(this->m_numExtremes);
        this->m_outerCov /= static_cast<Scalar>(numNonExtremes);
        
        this->m_innerCov -= this->m_innerMean * this->m_innerMean.transpose();
        this->m_outerCov -= this->m_outerMean * this->m_outerMean.transpose();
        
        // Compute cholesky decomposition and log-determinant
        cholesky(this->m_innerCov, &(this->m_innerCovChol), &(this->m_innerCovLogDet));
        cholesky(this->m_outerCov, &(this->m_outerCovChol), &(this->m_outerCovLogDet));
        
        // Compute normalizing constant
        this->m_innerLogNormalizer = this->m_logNormalizer - this->m_innerCovLogDet / 2;
        this->m_outerLogNormalizer = this->m_logNormalizer - this->m_outerCovLogDet / 2;
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
    std::pair<Scalar, Scalar> pdf = this->logpdf(ind);
    pdf.first = (pdf.first > -700) ? std::exp(pdf.first) : 0;
    pdf.first = (pdf.second > -700) ? std::exp(pdf.second) : 0;
    return pdf;
}

DataTensor GaussianDensityEstimator::pdf() const
{
    DataTensor pdf = this->logpdf();
    pdf.data() = pdf.data().array().cwiseMax(-700).exp(); // taking the maximum is just to prevent underflow
    pdf.missingValuePlaceholder(1);
    return pdf;
}

DataTensor GaussianDensityEstimator::pdf(const IndexRange & range) const
{
    DataTensor pdf = this->logpdf(range);
    pdf.data() = pdf.data().array().cwiseMax(-700).exp(); // taking the maximum is just to prevent underflow
    pdf.missingValuePlaceholder(1);
    return pdf;
}

ScalarMatrix GaussianDensityEstimator::pdfOutsideRange(IndexRange range) const
{
    ScalarMatrix pdf = this->logpdfOutsideRange(range);
    pdf = pdf.array().cwiseMax(-700).exp(); // taking the maximum is just to prevent underflow
    return pdf;
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::logpdf(const ReflessIndexVector & ind) const
{
    if (this->m_data->isMissingSample(ind))
        return std::pair<Scalar, Scalar>(0, 0);
    
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
    
    // Compute (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    pdf.first  = (pdf.first / -2) + this->m_innerLogNormalizer;
    pdf.second = (pdf.second / -2) + this->m_outerLogNormalizer;
    return pdf;
}

DataTensor GaussianDensityEstimator::logpdf() const
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
    
    // Compute (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    pdf.data() /= static_cast<Scalar>(-2);
    pdf.data().array().col(0) += this->m_innerLogNormalizer;
    pdf.data().array().col(1) += this->m_outerLogNormalizer;
    pdf.copyMask(*(this->m_data));
    return pdf;
}

DataTensor GaussianDensityEstimator::logpdf(const IndexRange & range) const
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
        for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ind += ind.shape.d)
        {
            if (!this->m_data->isMissingSample(range.a + ind))
                centered(sample) = this->m_data->sample(range.a + ind) - this->m_innerMean;
            else
            {
                centered(sample).setZero();
                pdf.setMissingSample(sample);
            }
        }
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(0) = centered.data().rowwise().squaredNorm();
        else
            pdf.data().col(0) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        
        ind = pdf.makeIndexVector();
        for (DataTensor::Index sample = 0; sample < pdf.numSamples(); ++sample, ind += ind.shape.d)
        {
            if (!this->m_data->isMissingSample(range.a + ind))
                centered(sample) = this->m_data->sample(range.a + ind) - this->m_outerMean;
            else
                centered(sample).setZero();
        }
        if (this->m_covMode == CovMode::ID)
            pdf.data().col(1) = centered.data().rowwise().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_innerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
        else
            pdf.data().col(1) = centered.data().cwiseProduct(this->m_outerCovChol.solve(centered.data().transpose()).transpose()).rowwise().sum();
    }
    
    // Compute (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    pdf.data() /= static_cast<Scalar>(-2);
    pdf.data().array().col(0) += this->m_innerLogNormalizer;
    pdf.data().array().col(1) += this->m_outerLogNormalizer;
    return pdf;
}

ScalarMatrix GaussianDensityEstimator::logpdfOutsideRange(IndexRange range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return Sample();
    
    range.a.d = 0;
    range.b.d = this->m_data->numAttrib();
    ReflessIndexVector rangeShape = range.shape();
    
    ScalarMatrix pdf(this->m_data->numSamples() - rangeShape.prod(0, MAXDIV_INDEX_DIMENSION - 2), 2);
    std::vector<DataTensor::Index> missingSamples;
    
    // Compute (x - mu)^T * S^-1 * (x - mu)
    {
        ScalarMatrix centered(pdf.rows(), this->m_data->numAttrib());
        
        IndexVector ind = this->m_data->makeIndexVector();
        ind.shape.d = 1;
        for (DataTensor::Index pdfInd = 0, sampleInd = 0; ind.t < ind.shape.t; ++ind, ++sampleInd)
            if (!range.contains(ind))
            {
                if (!this->m_data->isMissingSample(sampleInd))
                    centered.row(pdfInd) = this->m_data->sample(sampleInd) - this->m_innerMean;
                else
                {
                    centered.row(pdfInd).setZero();
                    missingSamples.push_back(pdfInd);
                }
                ++pdfInd;
            }
        
        if (this->m_covMode == CovMode::ID)
            pdf.col(0) = centered.rowwise().squaredNorm();
        else
            pdf.col(0) = centered.cwiseProduct(this->m_innerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
        
        ind = this->m_data->makeIndexVector();
        ind.shape.d = 1;
        for (DataTensor::Index pdfInd = 0, sampleInd = 0; ind.t < ind.shape.t; ++ind, ++sampleInd)
            if (!range.contains(ind))
            {
                if (!this->m_data->isMissingSample(sampleInd))
                    centered.row(pdfInd) = this->m_data->sample(sampleInd) - this->m_outerMean;
                else
                    centered.row(pdfInd).setZero();
                ++pdfInd;
            }
        
        if (this->m_covMode == CovMode::ID)
            pdf.col(1) = centered.rowwise().squaredNorm();
        else if (this->m_covMode == CovMode::SHARED)
            pdf.col(1) = centered.cwiseProduct(this->m_innerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
        else
            pdf.col(1) = centered.cwiseProduct(this->m_outerCovChol.solve(centered.transpose()).transpose()).rowwise().sum();
    }
    
    // Compute (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    pdf /= static_cast<Scalar>(-2);
    pdf.array().col(0) += this->m_innerLogNormalizer;
    pdf.array().col(1) += this->m_outerLogNormalizer;
    
    // Set log-pdf of missing samples to 0
    for (const DataTensor::Index & missing : missingSamples)
        pdf.row(missing).setZero();
    
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
    
    // Compute sum over log-probabilities: (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 + this->m_data->numValidSamples() * this->m_innerLogNormalizer,
        ll(1) / -2 + this->m_data->numValidSamples() * this->m_outerLogNormalizer
    );
}

std::pair<Scalar, Scalar> GaussianDensityEstimator::logLikelihood(const IndexRange & range) const
{
    if (this->m_data == nullptr || this->m_data->empty())
        return std::pair<Scalar, Scalar>(0, 0);
    
    // Compute sum over distances: (x - mu)^T * S^-1 * (x - mu)
    Sample ll = Sample::Zero(2), centered(this->m_data->numAttrib());
    ReflessIndexVector::Index numSamples = 0;
    ReflessIndexVector shape = range.shape();
    shape.d = 1;
    for (IndexVector ind(shape, 0); ind.t < ind.shape.t; ++ind)
        if (!this->m_data->isMissingSample(range.a + ind))
        {
            ++numSamples;
            const auto sample = this->m_data->sample(range.a + ind);
            
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
    
    // Compute sum over log-probabilities: (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 + numSamples * this->m_innerLogNormalizer,
        ll(1) / -2 + numSamples * this->m_outerLogNormalizer
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
    for (ReflessIndexVector::Index sampleInd = 0; ind.t < ind.shape.t; ++ind, ++sampleInd)
        if (!range.contains(ind) && !this->m_data->isMissingSample(sampleInd))
        {
            ++numSamples;
            const auto sample = this->m_data->sample(sampleInd);
            
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
    
    // Compute sum over log-probabilities: (D * log(2*pi) + log(|S|) + (x - mu)^T * S^-1 * (x - mu)) / -2
    return std::pair<Scalar, Scalar>(
        ll(0) / -2 + numSamples * this->m_innerLogNormalizer,
        ll(1) / -2 + numSamples * this->m_outerLogNormalizer
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


//--------------------------------------//
// EnsembleOfRandomProjectionHistograms //
//--------------------------------------//

EnsembleOfRandomProjectionHistograms::EnsembleOfRandomProjectionHistograms()
: EnsembleOfRandomProjectionHistograms(100, 0, 1) {}

EnsembleOfRandomProjectionHistograms::EnsembleOfRandomProjectionHistograms(DataTensor::Index num_hist, DataTensor::Index num_bins, Scalar discount)
: DensityEstimator(), m_num_hist(num_hist), m_num_bins(num_bins), m_discount(std::max(discount, 1e-7)),
  m_hist_bins(IntTensor::Sample::Constant(num_hist, num_bins)), m_hist_offsets(num_hist),
  m_logprob_normalized(false)
{
    if (this->m_num_hist == 0)
        throw std::invalid_argument("Ensemble must contain at least 1 histogram.");
    
    if (this->m_num_bins > 0)
    {
        this->m_hist_offsets.setLinSpaced(0, this->m_num_hist - 1);
        this->m_hist_offsets *= this->m_num_bins;
    }
}

EnsembleOfRandomProjectionHistograms::EnsembleOfRandomProjectionHistograms(const std::shared_ptr<const DataTensor> & data)
: EnsembleOfRandomProjectionHistograms()
{
    this->init(data);
}

EnsembleOfRandomProjectionHistograms::EnsembleOfRandomProjectionHistograms(const std::shared_ptr<const DataTensor> & data, DataTensor::Index num_hist, DataTensor::Index num_bins, Scalar discount)
: EnsembleOfRandomProjectionHistograms(num_hist, num_bins, discount)
{
    this->init(data);
}

EnsembleOfRandomProjectionHistograms::EnsembleOfRandomProjectionHistograms(const EnsembleOfRandomProjectionHistograms & other)
: DensityEstimator(other),
  m_num_hist(other.m_num_hist), m_num_bins(other.m_num_bins), m_discount(other.m_discount),
  m_hist_bins(other.m_hist_bins), m_hist_offsets(other.m_hist_offsets),
  m_proj(other.m_proj), m_indices(other.m_indices), m_counts(other.m_counts),
  m_hist_inner(other.m_hist_inner), m_hist_outer(other.m_hist_outer),
  m_logprob_inner(other.m_logprob_inner), m_logprob_outer(other.m_logprob_outer),
  m_log_cache(other.m_log_cache), m_log_denom_cache(other.m_log_denom_cache), m_logprob_normalized(other.m_logprob_normalized)
{}

EnsembleOfRandomProjectionHistograms & EnsembleOfRandomProjectionHistograms::operator=(const EnsembleOfRandomProjectionHistograms & other)
{
    this->m_data = other.m_data;
    this->m_extremeRange = other.m_extremeRange;
    this->m_numExtremes = other.m_numExtremes;
    this->m_num_hist = other.m_num_hist;
    this->m_num_bins = other.m_num_bins;
    this->m_discount = other.m_discount;
    this->m_hist_bins = other.m_hist_bins;
    this->m_hist_offsets = other.m_hist_offsets;
    this->m_proj = other.m_proj;
    this->m_indices = other.m_indices;
    this->m_counts = other.m_counts;
    this->m_hist_inner = other.m_hist_inner;
    this->m_hist_outer = other.m_hist_outer;
    this->m_logprob_inner = other.m_logprob_inner;
    this->m_logprob_outer = other.m_logprob_outer;
    this->m_log_cache = other.m_log_cache;
    this->m_log_denom_cache = other.m_log_denom_cache;
    this->m_logprob_normalized = other.m_logprob_normalized;
    return *this;
}

std::shared_ptr<DensityEstimator> EnsembleOfRandomProjectionHistograms::clone() const
{
    return std::make_shared<EnsembleOfRandomProjectionHistograms>(*this);
}

void EnsembleOfRandomProjectionHistograms::init(const std::shared_ptr<const DataTensor> & data)
{
    DensityEstimator::init(data);
    
    this->m_indices.reset();
    this->m_counts.reset();
    
    if (this->m_data && !this->m_data->empty())
    {
        ReflessIndexVector shape = this->m_data->shape();
        
        // Cache some logarithms
        if (!this->m_log_cache || static_cast<DataTensor::Index>(this->m_log_cache->size()) < data->numValidSamples() + 1)
        {
            DataTensor::Index n;
            if (this->m_log_cache)
            {
                n = this->m_log_cache->size();
                this->m_log_cache->conservativeResize(data->numValidSamples() + 1);
            }
            else
            {
                n = 0;
                this->m_log_cache.reset(new Sample(data->numValidSamples() + 1));
            }
            this->m_log_cache->segment(n, this->m_log_cache->size() - n)
                = (Sample::LinSpaced(this->m_log_cache->size() - n, n, data->numValidSamples()).array() + this->m_discount).log();
        }
        
        // Generate random projection vectors
        if (!this->m_proj || static_cast<DataTensor::Index>(this->m_proj->cols()) != shape.d)
        {
            this->m_proj.reset(new SparseMatrix(this->m_num_hist, shape.d));
            DataTensor::Index numNonZero = static_cast<DataTensor::Index>(std::sqrt(shape.d) + 0.5);
            this->m_proj->reserve(Eigen::VectorXi::Constant(this->m_num_hist, numNonZero));
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> udis(0, shape.d - 1);
            std::normal_distribution<Scalar> ndis;
            
            DataTensor::Index i, j;
            int d;
            for (i = 0; i < this->m_num_hist; ++i)
            {
                for (j = 0; j < numNonZero; ++j)
                {
                    do {
                        d = udis(gen);
                    } while (this->m_proj->coeff(i, d) != 0);
                    this->m_proj->insert(i, d) = ndis(gen);
                }
                this->m_proj->row(i) /= this->m_proj->row(i).norm();
            }
        }
        
        // Project data onto 1d spaces
        shape.d = this->m_num_hist;
        DataTensor projectedData(shape);
        projectedData.data().noalias() = this->m_data->data() * this->m_proj->transpose();
        projectedData.copyMask(*this->m_data);
        
        // Transform projected attributes to be in range [0,1]
        projectedData.missingValuePlaceholder(std::numeric_limits<Scalar>::max());
        projectedData -= projectedData.data().colwise().minCoeff();
        projectedData.missingValuePlaceholder(0);
        projectedData /= projectedData.data().colwise().maxCoeff();
        
        // Determine optimal number of bins for each histogram
        if (this->m_num_bins == 0)
        {
            this->m_hist_bins = EnsembleOfRandomProjectionHistograms::getOptimalBinNum(projectedData);
            for (DataTensor::Index i = 0, offs = 0; i < this->m_num_hist; offs += this->m_hist_bins(i++))
                this->m_hist_offsets(i) = offs;
        }
        this->m_hist_inner = this->m_hist_outer = IntTensor::Sample(this->m_hist_offsets(this->m_num_hist - 1) + this->m_hist_bins(this->m_num_hist - 1));
        this->m_logprob_inner = this->m_logprob_outer = Sample(this->m_hist_inner.size());
        
        // Determine indices of all samples and compute cumulative counts for all bins
        this->m_indices.reset(new IntTensor(shape));
        shape.d = this->m_hist_inner.size();
        this->m_counts.reset(new IntTensor(shape, 0));
        DataTensor::Index i, j;
        for (i = 0; i < projectedData.numSamples(); ++i)
        {
            const auto sample = projectedData.sample(i);
            auto ind = this->m_indices->sample(i);
            auto c = this->m_counts->sample(i);
            if (!projectedData.isMissingSample(i))
                for (j = 0; j < this->m_num_hist; ++j)
                {
                    ind(j) = std::min(static_cast<DataTensor::Index>(sample(j) * this->m_hist_bins(j)), this->m_hist_bins(j) - 1);
                    c(this->m_hist_offsets(j) + ind(j)) = 1;
                }
        }
        this->m_counts->cumsum(0, MAXDIV_INDEX_DIMENSION - 2);
    }
}

void EnsembleOfRandomProjectionHistograms::fit(const IndexRange & range)
{
    DensityEstimator::fit(range);
    
    // Compute the histograms of the samples inside and outside of the given range
    this->m_hist_inner = this->m_counts->sumFromCumsum(range);
    this->m_hist_outer = this->m_counts->sample(this->m_counts->numSamples() - 1) - this->m_hist_inner;
    
    // Compute logarithm of probability density estimates:
    // log( bins * (n_i + discount) / (N + bins * discount) ) = log(n_i + discount) - log( N/bins + discount )
    // We only compute log(n_i + discount) here and leave the denominator for being added later, since it does not depend on n_i.
    for (DataTensor::Index i = 0; i < static_cast<DataTensor::Index>(this->m_hist_inner.size()); ++i)
    {
        this->m_logprob_inner(i) = (*this->m_log_cache)(this->m_hist_inner(i));
        this->m_logprob_outer(i) = (*this->m_log_cache)(this->m_hist_outer(i));
    }
    this->m_logprob_normalized = false;
}

void EnsembleOfRandomProjectionHistograms::reset()
{
    DensityEstimator::reset();
    this->m_indices.reset();
    this->m_counts.reset();
    this->m_hist_inner = this->m_hist_outer = IntTensor::Sample();
    this->m_logprob_inner = this->m_logprob_outer = Sample();
}

std::pair<Scalar, Scalar> EnsembleOfRandomProjectionHistograms::pdf(const ReflessIndexVector & ind) const
{
    std::pair<Scalar, Scalar> pdf = this->logpdf(ind);
    pdf.first = std::exp(pdf.first);
    pdf.second = std::exp(pdf.second);
    return pdf;
}

std::pair<Scalar, Scalar> EnsembleOfRandomProjectionHistograms::logpdf(const ReflessIndexVector & ind) const
{
    assert(this->m_data != nullptr && !this->m_data->empty() && !this->m_extremeRange.empty());
    assert(this->m_indices != nullptr && this->m_logprob_inner.size() > 0 && this->m_logprob_outer.size() > 0);
    assert((ind.vec() < this->m_data->shape().vec()).all());
    
    if (this->m_data->isMissingSample(ind))
        return std::pair<Scalar, Scalar>(0, 0);
    
    // Check if we still have to normalize the log-probabilities first, which we haven't done in fit().
    if (!this->m_logprob_normalized)
    {
        const Sample & logDenomInner = this->logDenomFromCache(this->m_numExtremes);
        const Sample & logDenomOuter = this->logDenomFromCache(this->m_data->numValidSamples() - this->m_numExtremes);
        for (DataTensor::Index i = 0; i < this->m_num_hist; ++i)
        {
            this->m_logprob_inner.segment(this->m_hist_offsets(i), this->m_hist_bins(i)).array() -= logDenomInner(i);
            this->m_logprob_outer.segment(this->m_hist_offsets(i), this->m_hist_bins(i)).array() -= logDenomOuter(i);
        }
        this->m_logprob_normalized = true;
    }
    
    // Compute average log-likelihood over all histograms
    const auto bins = this->m_indices->sample(ind);
    DataTensor::Index bin;
    Scalar sum_inner = 0, sum_outer = 0;
    for (DataTensor::Index i = 0; i < this->m_num_hist; ++i)
    {
        bin = this->m_hist_offsets(i) + bins(i);
        sum_inner += this->m_logprob_inner(bin);
        sum_outer += this->m_logprob_outer(bin);
    }
    return std::make_pair(sum_inner / this->m_num_hist, sum_outer / this->m_num_hist);
}

std::pair<Scalar, Scalar> EnsembleOfRandomProjectionHistograms::logLikelihoodInner() const
{
    assert(this->m_data != nullptr && !this->m_data->empty() && !this->m_extremeRange.empty());
    assert(this->m_hist_inner.size() > 0 && this->m_logprob_inner.size() > 0 && this->m_logprob_outer.size() > 0);
    
    std::pair<Scalar, Scalar> ll(
        this->m_hist_inner.cast<Scalar>().cwiseProduct(this->m_logprob_inner).sum(),
        this->m_hist_inner.cast<Scalar>().cwiseProduct(this->m_logprob_outer).sum()
    );
    
    if (!this->m_logprob_normalized)
    {
        ll.first  -= this->m_numExtremes * this->logDenomFromCache(this->m_numExtremes).sum();
        ll.second -= this->m_numExtremes * this->logDenomFromCache(this->m_data->numValidSamples() - this->m_numExtremes).sum();
    }
    
    ll.first /= this->m_num_hist;
    ll.second /= this->m_num_hist;
    return ll;
}

std::pair<Scalar, Scalar> EnsembleOfRandomProjectionHistograms::logLikelihoodOuter() const
{
    assert(this->m_data != nullptr && !this->m_data->empty() && !this->m_extremeRange.empty());
    assert(this->m_hist_outer.size() > 0 && this->m_logprob_inner.size() > 0 && this->m_logprob_outer.size() > 0);
    
    std::pair<Scalar, Scalar> ll(
        this->m_hist_outer.cast<Scalar>().cwiseProduct(this->m_logprob_inner).sum(),
        this->m_hist_outer.cast<Scalar>().cwiseProduct(this->m_logprob_outer).sum()
    );
    
    if (!this->m_logprob_normalized)
    {
        DataTensor::Index numNonExtremes = this->m_data->numValidSamples() - this->m_numExtremes;
        ll.first  -= numNonExtremes * this->logDenomFromCache(this->m_numExtremes).sum();
        ll.second -= numNonExtremes * this->logDenomFromCache(numNonExtremes).sum();
    }
    
    ll.first /= this->m_num_hist;
    ll.second /= this->m_num_hist;
    return ll;
}

const Sample & EnsembleOfRandomProjectionHistograms::logDenomFromCache(DataTensor::Index n) const
{
    try
    {
        return this->m_log_denom_cache.at(n);
    }
    catch (const std::out_of_range & e)
    {
        this->m_log_denom_cache.insert(std::pair<DataTensor::Index, Sample>(
            n,
            (static_cast<Scalar>(n) / this->m_hist_bins.array().cast<Scalar>() + this->m_discount).log())
        );
        return this->m_log_denom_cache.at(n);
    }
}

EnsembleOfRandomProjectionHistograms::IntTensor::Sample EnsembleOfRandomProjectionHistograms::getOptimalBinNum(const DataTensor & data)
{
    assert(data.data().minCoeff() >= 0.0 && data.data().maxCoeff() <= 1.0);
    
    static std::vector<Scalar> penalties; // cache for penalty terms
    
    // Optimize bins separately for each histogram in order to allow for early exit
    IntTensor::Sample bins = IntTensor::Sample::Constant(data.numAttrib(), 1);
    DataTensor::Index maxBins = std::ceil(data.numValidSamples() / std::log(data.numValidSamples()));
    Scalar logLikelihood, penalty, pml, max_pml, eps = std::numeric_limits<Scalar>::epsilon();
    Sample last_ll_cache(20), last_pen_cache(20);
    DataTensor::Index h, b;
    Sample::Index i, j;
    for (h = 0; h < data.numAttrib(); ++h)
    {
        const auto channel = data.channel(h);
        max_pml = 0.0; // maximum penalized likelihood is always 0 for only 1 bin
        
        for (b = 2; b <= maxBins; ++b)
        {
            // Count entries in each bin
            IntTensor::Sample counts = IntTensor::Sample::Zero(b);
            for (i = 0; i < channel.rows(); ++i)
                for (j = 0; j < channel.cols(); ++j)
                    if (!data.hasMissingSamples() || !data.isMissingSample(i * channel.cols() + j))
                        counts(std::min(static_cast<DataTensor::Index>(b * channel(i, j)), b - 1)) += 1;
            
            // Compute penalized maximum likelihood
            Sample logprob = counts.cast<Scalar>() * static_cast<Scalar>(b) / data.numValidSamples();
            logprob = (logprob.array() + eps).log();
            logLikelihood = counts.cast<Scalar>().cwiseProduct(logprob).sum();
            if (penalties.size() <= b - 2)
            {
                assert(penalties.size() == b - 2);
                penalties.push_back(b - 1 + std::pow(std::log(b), 2.5));
            }
            penalty = penalties[b - 2];
            pml = logLikelihood - penalty;
            
            // Check for new maximum and early termination criterion
            last_ll_cache(b % 20) = logLikelihood;
            last_pen_cache(b % 20) = penalty;
            if (pml > max_pml)
            {
                bins(h) = b;
                max_pml = pml;
            }
            else if (b >= 22 && b - bins(h) > 20
                     && (last_ll_cache.tail(19) - last_ll_cache.head(19)).mean() < (last_pen_cache.tail(19) - last_pen_cache.head(19)).mean())
                break;
        }
    }
    return bins;
}
