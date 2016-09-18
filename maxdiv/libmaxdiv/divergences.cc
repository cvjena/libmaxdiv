#include "divergences.h"
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cassert>
using namespace MaxDiv;


//-----------------------------//
// Kullback-Leibler Divergence //
//-----------------------------//

KLDivergence::KLDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, KLMode mode)
: m_mode(mode),
  m_densityEstimator(densityEstimator),
  m_gaussDensityEstimator(std::dynamic_pointer_cast<GaussianDensityEstimator>(densityEstimator)),
  m_data(nullptr), m_chiMean(0), m_chiSD(1)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

KLDivergence::KLDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode)
: m_mode(mode),
  m_densityEstimator(densityEstimator),
  m_gaussDensityEstimator(std::dynamic_pointer_cast<GaussianDensityEstimator>(densityEstimator)),
  m_data(nullptr), m_chiMean(0), m_chiSD(1)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

KLDivergence::KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, KLMode mode)
: m_mode(mode), m_densityEstimator(densityEstimator), m_gaussDensityEstimator(densityEstimator),
  m_data(nullptr), m_chiMean(0), m_chiSD(1)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

KLDivergence::KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode)
: m_mode(mode), m_densityEstimator(densityEstimator), m_gaussDensityEstimator(densityEstimator),
  m_data(nullptr), m_chiMean(0), m_chiSD(1)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

KLDivergence::KLDivergence(const KLDivergence & other)
: m_mode(other.m_mode), m_data(other.m_data), m_chiMean(other.m_chiMean), m_chiSD(other.m_chiSD)
{
    if (other.m_gaussDensityEstimator != nullptr)
        this->m_densityEstimator = this->m_gaussDensityEstimator = std::make_shared<GaussianDensityEstimator>(*(other.m_gaussDensityEstimator));
    else
    {
        this->m_densityEstimator = other.m_densityEstimator->clone();
        this->m_gaussDensityEstimator = nullptr;
    }
}

KLDivergence & KLDivergence::operator=(const KLDivergence & other)
{
    this->m_mode = other.m_mode;
    this->m_data = other.m_data;
    this->m_chiMean = other.m_chiMean;
    this->m_chiSD = other.m_chiSD;
    if (other.m_gaussDensityEstimator != nullptr)
        this->m_densityEstimator = this->m_gaussDensityEstimator = std::make_shared<GaussianDensityEstimator>(*(other.m_gaussDensityEstimator));
    else
    {
        this->m_densityEstimator = other.m_densityEstimator->clone();
        this->m_gaussDensityEstimator = nullptr;
    }
    return *this;
}

std::shared_ptr<Divergence> KLDivergence::clone() const
{
    return std::make_shared<KLDivergence>(*this);
}

void KLDivergence::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_densityEstimator->init(data);
    this->m_data = data;
    this->m_chiMean = (this->m_data->numAttrib() * (this->m_data->numAttrib() + 3)) / 2;
    this->m_chiSD = std::sqrt(2 * this->m_chiMean);
}

void KLDivergence::reset()
{
    this->m_densityEstimator->reset();
    this->m_data.reset();
}

Scalar KLDivergence::operator()(const IndexRange & innerRange)
{
    assert(this->m_data != nullptr);
    
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    DataTensor::Index numExtremes = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2) - this->m_data->numMissingSamplesInRange(innerRange);
    
    // Compute divergence
    Scalar score = 0;
    if (this->m_gaussDensityEstimator)
    {
        // There is a closed form solution for the KL divergence for normal distributions:
        // KL(I, Omega) = (trace(S_Omega^-1 * S_I) + (mu_I - mu_Omega)^T * S_Omega^-1 * (mu_I - mu_Omega) - D + log(|S_Omega|) - log(|S_I|)) / 2
        GaussianDensityEstimator * gde = this->m_gaussDensityEstimator.get();
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM || this->m_mode == KLMode::UNBIASED)
        {
            score += gde->mahalanobisDistance(gde->getInnerMean(), gde->getOuterMean(), false);
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
            {
                score += gde->getOuterCovChol().solve(gde->getInnerCov()).trace()
                         + gde->getOuterCovLogDet() - gde->getInnerCovLogDet()
                         - this->m_data->numAttrib();
            }
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            score += gde->mahalanobisDistance(gde->getOuterMean(), gde->getInnerMean(), true);
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
            {
                score += gde->getInnerCovChol().solve(gde->getOuterCov()).trace()
                         + gde->getInnerCovLogDet() - gde->getOuterCovLogDet()
                         - this->m_data->numAttrib();
            }
        }
        if (this->m_mode == KLMode::UNBIASED)
        {
            score *= numExtremes;
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
                score = (score - this->m_chiMean) / this->m_chiSD;
        }
    }
    else
    {
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM || this->m_mode == KLMode::UNBIASED)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihoodInner();
            score += (ll.first - ll.second) / numExtremes;
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihoodOuter();
            score += (ll.second - ll.first) / (this->m_data->numValidSamples() - numExtremes);
        }
        if (this->m_mode == KLMode::UNBIASED)
            score *= numExtremes;
    }
    return score;
}


//---------------//
// Cross-Entropy //
//---------------//

std::shared_ptr<Divergence> CrossEntropy::clone() const
{
    return std::make_shared<CrossEntropy>(*this);
}

Scalar CrossEntropy::operator()(const IndexRange & innerRange)
{
    assert(this->m_data != nullptr);
    
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    DataTensor::Index numExtremes = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2) - this->m_data->numMissingSamplesInRange(innerRange);
    
    // Compute cross-entropy
    Scalar score = 0;
    if (this->m_gaussDensityEstimator)
    {
        // There is a closed form solution for the cross-entropy for normal distributions:
        // H(I, Omega) = (trace(S_Omega^-1 * S_I) + (mu_I - mu_Omega)^T * S_Omega^-1 * (mu_I - mu_Omega) + log(|S_Omega|) + D * log(2*pi)) / 2
        GaussianDensityEstimator * gde = this->m_gaussDensityEstimator.get();
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM || this->m_mode == KLMode::UNBIASED)
        {
            score += gde->mahalanobisDistance(gde->getInnerMean(), gde->getOuterMean(), false) - 2 * gde->getLogNormalizer();
            switch (gde->getMode())
            {
                case GaussianDensityEstimator::CovMode::FULL:
                    score += gde->getOuterCovChol().solve(gde->getInnerCov()).trace() + gde->getOuterCovLogDet();
                    break;
                case GaussianDensityEstimator::CovMode::SHARED:
                    score += this->m_data->numAttrib() + gde->getOuterCovLogDet();
                    break;
                case GaussianDensityEstimator::CovMode::ID:
                    score += this->m_data->numAttrib();
                    break;
            }
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            score += gde->mahalanobisDistance(gde->getOuterMean(), gde->getInnerMean(), true) - 2 * gde->getLogNormalizer();
            switch (gde->getMode())
            {
                case GaussianDensityEstimator::CovMode::FULL:
                    score += gde->getInnerCovChol().solve(gde->getOuterCov()).trace() + gde->getInnerCovLogDet();
                    break;
                case GaussianDensityEstimator::CovMode::SHARED:
                    score += this->m_data->numAttrib() + gde->getInnerCovLogDet();
                    break;
                case GaussianDensityEstimator::CovMode::ID:
                    score += this->m_data->numAttrib();
                    break;
            }
        }
        if (this->m_mode == KLMode::UNBIASED)
        {
            score *= numExtremes;
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
                score = (score - this->m_chiMean) / this->m_chiSD;
        }
    }
    else
    {
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM || this->m_mode == KLMode::UNBIASED)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihoodInner();
            score -= ll.second / ((this->m_mode != KLMode::UNBIASED) ? numExtremes : 1);
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihoodOuter();
            score -= ll.first / (this->m_data->numValidSamples() - numExtremes);
        }
    }
    return score;
}


//---------------------------//
// Jensen-Shannon Divergence //
//---------------------------//

JSDivergence::JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator)
: m_densityEstimator(densityEstimator), m_data(nullptr)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

JSDivergence::JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data)
: m_densityEstimator(densityEstimator), m_data(nullptr)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

JSDivergence::JSDivergence(const JSDivergence & other)
: m_densityEstimator(other.m_densityEstimator->clone()), m_data(other.m_data)
{}

JSDivergence & JSDivergence::operator=(const JSDivergence & other)
{
    this->m_densityEstimator = other.m_densityEstimator->clone();
    this->m_data = other.m_data;
    return *this;
}

std::shared_ptr<Divergence> JSDivergence::clone() const
{
    return std::make_shared<JSDivergence>(*this);
}

void JSDivergence::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_densityEstimator->init(data);
    this->m_data = data;
}

void JSDivergence::reset()
{
    this->m_densityEstimator->reset();
    this->m_data.reset();
}

Scalar JSDivergence::operator()(const IndexRange & innerRange)
{
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    
    // Determine number of samples within and outside of the range
    DataTensor::Index numExtremes = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2) - this->m_data->numMissingSamplesInRange(innerRange);
    DataTensor::Index numNonExtremes = this->m_data->numValidSamples() - numExtremes;
    
    // Compute divergence
    DataTensor pdf = this->m_densityEstimator->pdf();
    Scalar scoreInner = 0, scoreOuter = 0, eps = std::numeric_limits<Scalar>::epsilon(), combined;
    IndexVector ind = pdf.makeIndexVector();
    ind.shape.d = 1;
    for (DataTensor::Index sampleIndex = 0; sampleIndex < this->m_data->numSamples(); ++ind, ++sampleIndex)
        if (!this->m_data->isMissingSample(sampleIndex))
        {
            const auto sample = pdf.sample(sampleIndex);
            combined = std::log(sample.mean() + eps);
            if (innerRange.contains(ind))
                scoreInner += std::log(sample(0) + eps) - combined;
            else
                scoreOuter += std::log(sample(1) + eps) - combined;
        }
    
    return (scoreInner / numExtremes + scoreOuter / numNonExtremes) / (2 * std::log(2));
}
