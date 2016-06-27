#ifndef MAXIDV_DIVERGENCES_H
#define MAXIDV_DIVERGENCES_H

#include <memory>
#include "DataTensor.h"
#include "estimators.h"

namespace MaxDiv
{

/**
* @brief Abstract base class for divergences measuring the difference between two probability distributions
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class Divergence
{
public:

    virtual ~Divergence() {};
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<Divergence> clone() const =0;
    
    /**
    * Initializes this divergence with a given DataTensor @p data.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) =0;
    
    /**
    * Resets this divergence to its uninitialized state and releases any memory allocated by `init()`.
    */
    virtual void reset() =0;
    
    /**
    * Approximates the divergence between a given sub-block of the data passed to `init()` and the rest
    * of that data tensor.
    *
    * `init()` has to be called before this can be used.
    *
    * @param[in] innerRange The sub-block to compare against the rest of the data passed to `init()`.
    *
    * @return Returns a measure of divergence which is high if the two data segments are very dissimilar,
    * but near zero if they are similar.
    */
    virtual Scalar operator()(const IndexRange & innerRange) =0;

};


/**
* @brief Kullback-Leibler Divergence
*
* The Kullback-Leibler Divergence is defined as:
*
* \f[
*     \text{KL}(p_I, p_\Omega) = \integral p_I(x_t) \cdot \log \left ( \frac{p_I(x_t)}{p_\Omega(x_t)} \right ) dx_t
* \f]
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class KLDivergence : public Divergence
{
public:

    enum class KLMode
    {
        I_OMEGA, /**< Integrate over the inner (extremal) range. */
        OMEGA_I, /**< Integrate over the outer (nominal) range. */
        SYM      /**< Symmetric version of the KL divergence: I_OMEGA + OMEGA_I */
    };
    

    KLDivergence() = delete;
    
    /**
    * Constructs an uninitialized KL divergence using a given density estimator.
    * `init()` has to be called before this divergence can be used.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    *
    * @param[in] mode Specifies the polarity of the KL divergence.
    */
    KLDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, KLMode mode = KLMode::I_OMEGA);
    
    /**
    * Constructs and initializes a KL divergence using a given density estimator.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    *
    * @param[in] data The data tensor to initialize the density estimator for.
    *
    * @param[in] mode Specifies the polarity of the KL divergence.
    */
    KLDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode = KLMode::I_OMEGA);
    
    /**
    * Constructs an uninitialized KL divergence using a given GaussianDensityEstimator.
    * `init()` has to be called before this divergence can be used.
    *
    * Compared to the more general constructor which takes a pointer to a `DensityEstimator`,
    * this one does not need to perform a `dynamic_cast` to determine the type of the density
    * estimator at run-time.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    *
    * @param[in] mode Specifies the polarity of the KL divergence.
    */
    KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, KLMode mode = KLMode::I_OMEGA);
    
    /**
    * Constructs and initializes a KL divergence using a given GaussianDensityEstimator.
    *
    * Compared to the more general constructor which takes a pointer to a `DensityEstimator`,
    * this one does not need to perform a `dynamic_cast` to determine the type of the density
    * estimator at run-time.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    *
    * @param[in] data The data tensor to initialize the density estimator for.
    *
    * @param[in] mode Specifies the polarity of the KL divergence.
    */
    KLDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data, KLMode mode = KLMode::I_OMEGA);
    
    /**
    * Copy constructor.
    */
    KLDivergence(const KLDivergence & other);
    
    KLDivergence & operator=(const KLDivergence & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<Divergence> clone() const override;
    
    /**
    * Initializes the density estimator used by this divergence with a given DataTensor @p data.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Resets this divergence and the density estimator to their uninitialized state and releases any
    * memory allocated by `init()`.
    */
    virtual void reset() override;
    
    /**
    * Approximates the KL divergence between a given sub-block of the data passed to `init()` and the rest
    * of that data tensor by evaluating one of the following formulas:
    *
    * - For `I_OMEGA` mode: \f$KL_{I,\Omega} = \frac{1}{\left | I \right |}\sum_{t \in I} \left ( \log \left ( p_I(x_t) \right ) - \log \left ( p_\Omega(x_t) \right ) \right )\f$
    * - For `OMEGA_I` mode: \f$KL_{\Omega,I} = \frac{1}{\left | \Omega \right |}\sum_{t \in \Omega} \left ( \log \left ( p_\Omega(x_t) \right ) - \log \left ( p_I(x_t) \right ) \right )\f$
    * - For `SYM` mode: \f$KL_{\text{SYM}} = KL_{I,\Omega} + KL_{\Omega,I}\f$
    *
    * `init()` has to be called before this can be used.
    *
    * @param[in] innerRange The sub-block to compare against the rest of the data passed to `init()`.
    *
    * @return Returns an approximation of the KL divergence which is high if the distributions of the two data
    * segments are very dissimilar, but zero if their distributions are identical.
    */
    virtual Scalar operator()(const IndexRange & innerRange) override;


protected:

    KLMode m_mode;
    std::shared_ptr<DensityEstimator> m_densityEstimator;
    std::shared_ptr<GaussianDensityEstimator> m_gaussDensityEstimator;
    DataTensor::Index m_numSamples; /**< Number of samples in the DataTensor passed to `init()`. */

};


/**
* @brief A length-normalized variant of the KL divergence for Gaussian Data
*
* This divergence is based on the test statistic proposed by Anderson (1984) on page 442.
* Theoretically, it should not be biased towards larger or smaller ranges as opposed to the KL divergence.
* However, this unbiasedness does only hold asymptotically. Thus, it does not need to perform better than
* the plain KL divergence in practice.
*
* It can be defined as follows:
*
* \f[
*       \text{TS}(p_I, p_\Omega) = \text{trace}(\Sigma^{-1} B) + n \cdot \left [
*           \left( \bar{x} - \mu \right )^T \Sigma^{-1} \left( \bar{x} - \mu \right)
*           + \text{log} \left | \Sigma \right | - \text{log} \left | B \right |
*           + D \cdot \left( \text{log}(n) - 1 \right )
*       \right ]
* \f]
*
* Where \f$ B \f$ is the outer product of the data after subtracting the estimated mean \f$ \bar{x} \f$.
*
* Though this divergence measure should not be biased regarding the length of the intervals, it still
* systematically yields higher scores for time-series with more attributes. Thus, we subtract the
* theoretical mean from the scores and divide them by the standard deviation. Those are known, since
* the scores should be distributed according to a chi^2 distribution when applied to time-series
* consisting of white noise.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class GaussianTestStatisticDivergence : public KLDivergence
{
public:

    GaussianTestStatisticDivergence() = delete;
    
    /**
    * Constructs an uninitialized divergence using a given GaussianDensityEstimator.
    * `init()` has to be called before this divergence can be used.
    *
    * @param[in] densityEstimator The gaussian density estimator to be used. Must not be `NULL`.
    */
    GaussianTestStatisticDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator);
    
    /**
    * Constructs and initializes a divergence using a given GaussianDensityEstimator.
    *
    * @param[in] densityEstimator The gaussian density estimator to be used. Must not be `NULL`.
    *
    * @param[in] data The data tensor to initialize the density estimator for.
    */
    GaussianTestStatisticDivergence(const std::shared_ptr<GaussianDensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Copy constructor.
    */
    GaussianTestStatisticDivergence(const GaussianTestStatisticDivergence & other);
    
    GaussianTestStatisticDivergence & operator=(const GaussianTestStatisticDivergence & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<Divergence> clone() const override;
    
    /**
    * Initializes the density estimator used by this divergence with a given DataTensor @p data.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Computes the test statistic for a given sub-block of the data passed to `init()`, while considering
    * the distribution of the rest of the time-series as "true distribution".
    *
    * `init()` has to be called before this can be used.
    *
    * @param[in] innerRange The sub-block to compare against the rest of the data passed to `init()`.
    *
    * @return Returns the value of the test statistic which is high if the distributions of the two data
    * segments are very dissimilar, but zero if their distributions are identical.
    */
    virtual Scalar operator()(const IndexRange & innerRange) override;


protected:

    Scalar m_scoreMean; /**< The theoretical mean of the scores. */
    Scalar m_scoreSD; /**< The theoretical standard deviation of the scores. */

};


/**
* @brief Jensen-Shannon Divergence
*
* The Jensen-Shannon Divergence can be defined as:
* \f[
*     \text{JSD}(p_I, p_\Omega)
+     = \frac{1}{2} \cdot \text{KL}\left(p_I, \frac{p_I + p_\Omega}{2}\right)
*     + \frac{1}{2} \cdot \text{KL}\left(p_\Omega, \frac{p_I + p_\Omega}{2}\right)
* \f]
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class JSDivergence : public Divergence
{
public:

    JSDivergence() = delete;
    
    /**
    * Constructs an uninitialized JS divergence using a given density estimator.
    * `init()` has to be called before this divergence can be used.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    */
    JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator);
    
    /**
    * Constructs and initializes a JS divergence using a given density estimator.
    *
    * @param[in] densityEstimator The density estimator to be used. Must not be `NULL`.
    *
    * @param[in] data The data tensor to initialize the density estimator for.
    */
    JSDivergence(const std::shared_ptr<DensityEstimator> & densityEstimator, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Copy constructor.
    */
    JSDivergence(const JSDivergence & other);
    
    JSDivergence & operator=(const JSDivergence & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<Divergence> clone() const override;
    
    /**
    * Initializes the density estimator used by this divergence with a given DataTensor @p data.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Resets this divergence and the density estimator to their uninitialized state and releases any
    * memory allocated by `init()`.
    */
    virtual void reset() override;
    
    /**
    * Approximates the JS divergence between a given sub-block of the data passed to `init()` and the rest
    * of that data tensor by evaluating the following formula:
    *
    * \f[
    *   \text{JSD}_{I,\Omega}
    *   = \frac{1}{\left | I \right |} \left (
    *       \sum_{t \in I} \log \left ( p_I(x_t) \right )
    *       - \log \left ( \frac{p_I(x_t) + p_\Omega(x_t)}{2} \right )
    *   \right )
    *   + \frac{1}{\left | \Omega \right |} \left (
    *       \sum_{t \in \Omega} \log \left ( p_\Omega(x_t) \right )
    *       - \log \left ( \frac{p_I(x_t) + p_\Omega(x_t)}{2} \right )
    *   \right )
    * \f]
    *
    * `init()` has to be called before this can be used.
    *
    * @param[in] innerRange The sub-block to compare against the rest of the data passed to `init()`.
    *
    * @return Returns a value between 0 and 1 which is near 1 if the distributions of the two data
    * segments are very dissimilar, but zero if their distributions are identical.
    */
    virtual Scalar operator()(const IndexRange & innerRange) override;


protected:

    std::shared_ptr<DensityEstimator> m_densityEstimator;
    DataTensor::Index m_numSamples; /**< Number of samples in the DataTensor passed to `init()`. */

};

}

#endif