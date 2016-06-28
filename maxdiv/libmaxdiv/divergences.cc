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
  m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

KLDivergence::KLDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode)
: m_mode(mode),
  m_densityEstimator(densityEstimator),
  m_gaussDensityEstimator(std::dynamic_pointer_cast<GaussianDensityEstimator>(densityEstimator)),
  m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

KLDivergence::KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, KLMode mode)
: m_mode(mode), m_densityEstimator(densityEstimator), m_gaussDensityEstimator(densityEstimator), m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

KLDivergence::KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode)
: m_mode(mode), m_densityEstimator(densityEstimator), m_gaussDensityEstimator(densityEstimator), m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

KLDivergence::KLDivergence(const KLDivergence & other)
: m_mode(other.m_mode), m_numSamples(other.m_numSamples)
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
    this->m_numSamples = other.m_numSamples;
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
    this->m_numSamples = data->numSamples();
}

void KLDivergence::reset()
{
    this->m_densityEstimator->reset();
}

Scalar KLDivergence::operator()(const IndexRange & innerRange)
{
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    
    // Compute divergence
    Scalar score = 0;
    if (this->m_gaussDensityEstimator)
    {
        // There is a closed form solution for the KL divergence for normal distributions:
        // KL(I, Omega) = (trace(S_Omega^-1 * S_I) + (mu_I - mu_Omega)^T * S_Omega^-1 * (mu_I - mu_Omega) - D + log(|S_Omega|) - log(|S_I|)) / 2
        GaussianDensityEstimator * gde = this->m_gaussDensityEstimator.get();
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM)
        {
            score += gde->mahalanobisDistance(gde->getInnerMean(), gde->getOuterMean(), false);
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
            {
                score += gde->getOuterCovChol().solve(gde->getInnerCov()).trace()
                         + gde->getOuterCovLogDet() - gde->getInnerCovLogDet();
            }
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            score += gde->mahalanobisDistance(gde->getOuterMean(), gde->getInnerMean(), true);
            if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
            {
                score += gde->getInnerCovChol().solve(gde->getOuterCov()).trace()
                         + gde->getInnerCovLogDet() - gde->getOuterCovLogDet();
            }
        }
    }
    else
    {
        DataTensor::Index numExtremes = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2);
        if (this->m_mode == KLMode::I_OMEGA || this->m_mode == KLMode::SYM)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihood(innerRange);
            score += (ll.first - ll.second) / numExtremes;
        }
        if (this->m_mode == KLMode::OMEGA_I || this->m_mode == KLMode::SYM)
        {
            std::pair<Scalar, Scalar> ll = this->m_densityEstimator->logLikelihoodOutsideRange(innerRange);
            score += (ll.second - ll.first) / (this->m_numSamples - numExtremes);
        }
    }
    return score;
}


//-------------------------//
// Gaussian Test Statistic //
//-------------------------//

GaussianTestStatisticDivergence::GaussianTestStatisticDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator)
: KLDivergence(densityEstimator, KLMode::I_OMEGA), m_scoreMean(0), m_scoreSD(1)
{}

GaussianTestStatisticDivergence::GaussianTestStatisticDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data)
: KLDivergence(densityEstimator, KLMode::I_OMEGA), m_scoreMean(0), m_scoreSD(1)
{
    this->init(data);
}

GaussianTestStatisticDivergence::GaussianTestStatisticDivergence(const GaussianTestStatisticDivergence & other)
: KLDivergence(other), m_scoreMean(other.m_scoreMean), m_scoreSD(other.m_scoreSD)
{}

GaussianTestStatisticDivergence & GaussianTestStatisticDivergence::operator=(const GaussianTestStatisticDivergence & other)
{
    KLDivergence::operator=(other);
    this->m_scoreMean = other.m_scoreMean;
    this->m_scoreSD = other.m_scoreSD;
    return *this;
}

std::shared_ptr<Divergence> GaussianTestStatisticDivergence::clone() const
{
    return std::make_shared<GaussianTestStatisticDivergence>(*this);
}

void GaussianTestStatisticDivergence::init(const std::shared_ptr<const DataTensor> & data)
{
    KLDivergence::init(data);
    
    DataTensor::Index na = data->numAttrib();
    this->m_scoreMean = (na * (na + 3)) / 2;
    this->m_scoreSD = std::sqrt(2 * this->m_scoreMean);
}

Scalar GaussianTestStatisticDivergence::operator()(const IndexRange & innerRange)
{
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    IndexVector::Index len = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 1);
    IndexVector::Index d = innerRange.d().length();
    
    // Compute test statistic
    Scalar score = 0;
    GaussianDensityEstimator * gde = this->m_gaussDensityEstimator.get();
    score += gde->mahalanobisDistance(gde->getInnerMean(), gde->getOuterMean(), false);
    if (gde->getMode() == GaussianDensityEstimator::CovMode::FULL)
    {
        score += gde->getOuterCovLogDet() - (gde->getInnerCovLogDet() + d * std::log(len)) + d * (std::log(len) - 1);
        score = gde->getOuterCovChol().solve(gde->getInnerCov() * static_cast<Scalar>(len)).trace() + len * score;
    }
    
    // Normalize score regarding the number of attributes in the time-series
    score = (score - this->m_scoreMean) / this->m_scoreSD;
    return score;
}


//---------------------------//
// Jensen-Shannon Divergence //
//---------------------------//

JSDivergence::JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator)
: m_densityEstimator(densityEstimator), m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
}

JSDivergence::JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data)
: m_densityEstimator(densityEstimator), m_numSamples(0)
{
    if (densityEstimator == nullptr)
        throw std::invalid_argument("densityEstimator must not be NULL.");
    
    this->init(data);
}

JSDivergence::JSDivergence(const JSDivergence & other)
: m_densityEstimator(other.m_densityEstimator->clone()), m_numSamples(other.m_numSamples)
{}

JSDivergence & JSDivergence::operator=(const JSDivergence & other)
{
    this->m_densityEstimator = other.m_densityEstimator->clone();
    this->m_numSamples = other.m_numSamples;
    return *this;
}

std::shared_ptr<Divergence> JSDivergence::clone() const
{
    return std::make_shared<JSDivergence>(*this);
}

void JSDivergence::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_densityEstimator->init(data);
    this->m_numSamples = data->numSamples();
}

void JSDivergence::reset()
{
    this->m_densityEstimator->reset();
}

Scalar JSDivergence::operator()(const IndexRange & innerRange)
{
    // Estimate distributions
    this->m_densityEstimator->fit(innerRange);
    
    // Determine number of samples within and outside of the range
    DataTensor::Index numExtremes = innerRange.shape().prod(0, MAXDIV_INDEX_DIMENSION - 2);
    DataTensor::Index numNonExtremes = this->m_numSamples - numExtremes;
    
    // Compute divergence
    DataTensor pdf = this->m_densityEstimator->pdf();
    Scalar scoreInner = 0, scoreOuter = 0, eps = std::numeric_limits<Scalar>::epsilon(), combined;
    IndexVector ind = pdf.makeIndexVector();
    ind.shape.d = 1;
    for (DataTensor::Index sampleIndex = 0; sampleIndex < this->m_numSamples; ++ind, ++sampleIndex)
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
