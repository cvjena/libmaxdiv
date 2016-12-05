#ifndef MAXIDV_ESTIMATORS_H
#define MAXIDV_ESTIMATORS_H

#include <memory>
#include <unordered_map>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "DataTensor.h"
#include "math_utils.h"

namespace MaxDiv
{

/**
* @brief Base interface for probability density estimators
*
* The probability density estimators used in MaxDiv are always bound to a DataTensor,
* wich is passed to `init()`. The parameters of the inner and the outer distributions
* are fit to a sub-block of that tensor using `fit()` and the probability density
* function for a specific sample or another sub-block of data can be computed using `pdf()`.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class DensityEstimator
{
public:

    DensityEstimator();
    
    DensityEstimator(const DensityEstimator & other);

    virtual ~DensityEstimator() {};
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<DensityEstimator> clone() const =0;

    /**
    * Initializes this density estimator with a given DataTensor @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
    *
    * The result of this function is undefined if the inner or the outer range consists of
    * missing samples only.
    */
    virtual void fit(const IndexRange & range);
    
    /**
    * Resets this density estimator to its uninitialized state and releases any memory allocated
    * by `init()` and `fit()`.
    */
    virtual void reset();
    
    /**
    * Computes the value of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the pdf values for the given sample under the inner and the outer
    * distribution. If the sample is a missing sample, it will be assigned a pdf of 1.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const =0;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples. A value of 1 will be assigned to
    * the pdf of missing samples.
    */
    virtual DataTensor pdf() const;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples in the given sub-block. A value of 1 will
    * be assigned to the pdf of missing samples.
    */
    virtual DataTensor pdf(const IndexRange & range) const;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples outside of a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A matrix with as many rows as there are samples outside of the given sub-block and exactly two
    * columns which specify the values of the pdf of the inner and the outer distribution for all samples outside
    * of the given sub-block. A value of 1 will be assigned to the pdf of missing samples.
    */
    virtual ScalarMatrix pdfOutsideRange(IndexRange range) const;

    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the log-pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the logarithm of the pdf for the given sample under the inner and the
    * outer distribution. If the sample is a missing sample, it will be assigned a log-pdf of 0.
    */
    virtual std::pair<Scalar, Scalar> logpdf(const ReflessIndexVector & ind) const;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the logarithms of the pdf of the
    * inner and the outer distribution for all samples. A value of 0 will be assigned to the
    * log-pdf of missing samples.
    */
    virtual DataTensor logpdf() const;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples in a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A DataTensor with two attributes which specify the logarithms of the pdf of the
    * inner and the outer distribution for all samples in the given sub-block. A value of 0 will
    * be assigned to the log-pdf of missing samples.
    */
    virtual DataTensor logpdf(const IndexRange & range) const;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples outside of a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A matrix with as many rows as there are samples outside of the given sub-block and exactly two
    * columns which specify the logarithms of the pdf of the inner and the outer distribution for all samples outside
    * of the given sub-block. A value of 0 will be assigned to the log-pdf of missing samples.
    */
    virtual ScalarMatrix logpdfOutsideRange(IndexRange range) const;
    
    /**
    * Computes the log-likelihood of all samples in the DataTensor passed to `init()` for the inner
    * and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihood() const;
    
    /**
    * Computes the log-likelihood of all samples in the sub-block passed to `fit()` of the DataTensor
    * passed to `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodInner() const;

    /**
    * Computes the log-likelihood of all samples outside of the sub-block passed to `fit()` of the DataTensor
    * passed to `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodOuter() const;
    
    /**
    * Computes the log-likelihood of all samples in a given sub-block of the DataTensor passed to
    * `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the log-likelihood for. The attribute dimension will be ignored.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution. If the given range consists of missing samples only, 0 will be returned.
    */
    virtual std::pair<Scalar, Scalar> logLikelihood(const IndexRange & range) const;
    
    /**
    * Computes the log-likelihood of all samples outside of a given sub-block of the DataTensor passed to
    * `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the log-likelihood for.
    * The attribute dimension will be ignored.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution. If there are only missing samples outside of the given range, 0 will be returned.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodOutsideRange(IndexRange range) const;


protected:

    std::shared_ptr<const DataTensor> m_data; /**< Pointer to the DataTensor passed to `init()`. */
    IndexRange m_extremeRange; /**< Range of inner block passed to `fit()`. */
    DataTensor::Index m_numExtremes; /**< Number of non-missing samples in the block passed to `fit()`. */

};


/**
* @brief Estimates the distribution of given data by Kernel Density Estimation using a Gaussian kernel
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class KernelDensityEstimator : public DensityEstimator
{
public:

    /**
    * Constructs an un-initialized KernelDensityEstimator using a Gaussian kernel with default parameters.
    * `init()` has to be called before this density estimator can be used.
    */
    KernelDensityEstimator();
    
    /**
    * Constructs an un-initialized KernelDensityEstimator using a Gaussian kernel with given parameters.
    * `init()` has to be called before this density estimator can be used.
    *
    * @param[in] kernel_sigma_sq The "standard deviation" of the gaussian kernel.
    *
    * @param[in] normed Specifies whether to normalize the kernel in order to get a proper distribution
    * (not necessary for use with KL divergence).
    */
    KernelDensityEstimator(Scalar kernel_sigma_sq, bool normed = false);
    
    /**
    * Constructs and initializes a KernelDensityEstimator for a given data tensor.
    *
    * @param[in] data Pointer to the DataTensor.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    *
    * @param[in] kernel_sigma_sq The "standard deviation" of the gaussian kernel.
    *
    * @param[in] normed Specifies whether to normalize the kernel in order to get a proper distribution
    * (not necessary for use with KL divergence).
    */
    KernelDensityEstimator(const std::shared_ptr<const DataTensor> & data, Scalar kernel_sigma_sq = 1.0, bool normed = false);
    
    /**
    * Makes a flat copy of another KernelDensityEstimator. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The KernelDensityEstimator to be copied.
    */
    KernelDensityEstimator(const KernelDensityEstimator & other);
    
    /**
    * Makes a flat copy of another KernelDensityEstimator. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The KernelDensityEstimator to be copied.
    *
    * @return A reference to this object.
    */
    virtual KernelDensityEstimator & operator=(const KernelDensityEstimator & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<DensityEstimator> clone() const override;
    
    /**
    * Initializes this density estimator with a given DataTensor @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Resets this density estimator to its uninitialized state and releases any memory allocated
    * by `init()` and `fit()`.
    */
    virtual void reset() override;
    
    /**
    * Computes the value of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the pdf values for the given sample under the inner and the outer
    * distribution. If the given sample is a missing sample, it will be assigned a pdf of 1.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const override;


protected:

    Scalar m_sigma_sq; /**< Standard deviation of the Gaussian kernel. */
    bool m_normed; /**< Whether to normalize the kernel. */
    std::shared_ptr<GaussKernel> m_kernel; /**< Pointer to the Gaussian kernel instance. */
    std::shared_ptr<DataTensor> m_cumKernel; /**< Materialized kernel matrix with cumulated rows. */

};


/**
* @brief Estimates a multivariate normal distribution from given data
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class GaussianDensityEstimator : public DensityEstimator
{
public:

    enum class CovMode
    {
        FULL,   /**< Compute the covariance matrix on every fit to a sub-block of the data. */
        SHARED, /**< Assume a global covariance matrix which is computed once during `init()`. */
        ID      /**< Assume the identity matrix as covariance matrix. */
    };


    /**
    * Constructs an un-initialized GaussianDensityEstimator with no restrictions on the covariance matrix.
    * `init()` has to be called before this density estimator can be used.
    */
    GaussianDensityEstimator();
    
    /**
    * Constructs an un-initialized GaussianDensityEstimator with a specific covariance estimation mode.
    * `init()` has to be called before this density estimator can be used.
    *
    * @param[in] mode Specifies how the covariance matrix should be estimated.
    */
    GaussianDensityEstimator(CovMode mode);
    
    /**
    * Constructs and initializes a GaussianDensityEstimator for a given data tensor.
    *
    * @param[in] data Pointer to the DataTensor.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    *
    * @param[in] mode Specifies how the covariance matrix should be estimated.
    */
    GaussianDensityEstimator(const std::shared_ptr<const DataTensor> & data, CovMode mode = CovMode::FULL);
    
    /**
    * Makes a flat copy of another GaussianDensityEstimator. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The GaussianDensityEstimator to be copied.
    */
    GaussianDensityEstimator(const GaussianDensityEstimator & other);
    
    /**
    * Makes a flat copy of another GaussianDensityEstimator. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The GaussianDensityEstimator to be copied.
    *
    * @return A reference to this object.
    */
    virtual GaussianDensityEstimator & operator=(const GaussianDensityEstimator & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<DensityEstimator> clone() const override;
    
    /**
    * Initializes this density estimator with a given DataTensor @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
    *
    * The result of this function is undefined if the inner or the outer range consists of
    * missing samples only.
    */
    virtual void fit(const IndexRange & range) override;
    
    /**
    * Resets this density estimator to its uninitialized state and releases any memory allocated
    * by `init()` and `fit()`.
    */
    virtual void reset() override;
    
    /**
    * Computes the value of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the pdf values for the given sample under the inner and the outer
    * distribution. If the given sample is a missing sample, it will be assigned a pdf of 1.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const override;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples. A value of 1 will be assigned to
    * the pdf of missing samples.
    */
    virtual DataTensor pdf() const override;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples in the given sub-block. A value of 1 will
    * be assigned to the pdf of missing samples.
    */
    virtual DataTensor pdf(const IndexRange & range) const override;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples outside of a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A matrix with as many rows as there are samples outside of the given sub-block and exactly two
    * columns which specify the values of the pdf of the inner and the outer distribution for all samples outside
    * of the given sub-block. A value of 1 will be assigned to the pdf of missing samples.
    */
    virtual ScalarMatrix pdfOutsideRange(IndexRange range) const override;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the log-pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the logarithm of the pdf for the given sample under the inner and the
    * outer distribution. If the given sample is a missing sample, it will be assigned a log-pdf
    * of 0.
    */
    virtual std::pair<Scalar, Scalar> logpdf(const ReflessIndexVector & ind) const override;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the logarithms of the pdf of the
    * inner and the outer distribution for all samples. A value of 0 will be assigned to the
    * log-pdf of missing samples.
    */
    virtual DataTensor logpdf() const override;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples in a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A DataTensor with two attributes which specify the logarithms of the pdf of the
    * inner and the outer distribution for all samples in the given sub-block.
    * A value of 0 will be assigned to the log-pdf of missing samples.
    */
    virtual DataTensor logpdf(const IndexRange & range) const override;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for all samples outside of a given sub-block of the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the pdf for. The attribute dimension will be ignored.
    *
    * @return A matrix with as many rows as there are samples outside of the given sub-block and exactly two
    * columns which specify the logarithms of the pdf of the inner and the outer distribution for all samples outside
    * of the given sub-block. A value of 0 will be assigned to the log-pdf of missing samples.
    */
    virtual ScalarMatrix logpdfOutsideRange(IndexRange range) const override;
    
    /**
    * Computes the log-likelihood of all samples in the DataTensor passed to `init()` for the inner
    * and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihood() const override;
    
    /**
    * Computes the log-likelihood of all samples in a given sub-block of the DataTensor passed to
    * `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the log-likelihood for. The attribute dimension will be ignored.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution. If the given range consists of missing samples only, 0 will be returned.
    */
    virtual std::pair<Scalar, Scalar> logLikelihood(const IndexRange & range) const override;
    
    /**
    * Computes the log-likelihood of all samples outside of a given sub-block of the DataTensor passed to
    * `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block of samples *not* to compute the log-likelihood for.
    * The attribute dimension will be ignored.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution. If there are only missing samples outside of the given range, 0 will be returned.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodOutsideRange(IndexRange range) const override;
    
    /**
    * @returns Returns the covariance computation mode of this estimator.
    */
    CovMode getMode() const { return this->m_covMode; };
    
    /**
    * @return Returns a reference to the estimated mean of the inner distribution.
    */
    const Sample & getInnerMean() const { return this->m_innerMean; };
    
    /**
    * @return Returns a reference to the estimated mean of the outer distribution.
    */
    const Sample & getOuterMean() const { return this->m_outerMean; };
    
    /**
    * @return Returns a reference to the estimated covariance matrix of the inner distribution.
    */
    const ScalarMatrix & getInnerCov() const { return this->m_innerCov; };
    
    /**
    * @return Returns a reference to the estimated covariance matrix of the outer distribution.
    */
    const ScalarMatrix & getOuterCov() const { return this->m_outerCov; };
    
    /**
    * @return Returns the Cholesky decomposition of the estimated covariance matrix of the inner distribution.
    */
    const Eigen::LLT<ScalarMatrix> & getInnerCovChol() const { return this->m_innerCovChol; };
    
    /**
    * @return Returns the Cholesky decomposition of the estimated covariance matrix of the outer distribution.
    */
    const Eigen::LLT<ScalarMatrix> & getOuterCovChol() const { return this->m_outerCovChol; };
    
    /**
    * @return Returns the natural logarithm of the determinant of the estimated covariance matrix of the inner distribution.
    */
    const Scalar getInnerCovLogDet() const { return this->m_innerCovLogDet; };
    
    /**
    * @return Returns the natural logarithm of the determinant of the estimated covariance matrix of the outer distribution.
    */
    const Scalar getOuterCovLogDet() const { return this->m_outerCovLogDet; };
    
    /**
    * @return Returns the cached value of `-D/2 * log(2 * pi)`.
    */
    const Scalar getLogNormalizer() const { return this->m_logNormalizer; };
    
    /**
    * @return Returns the cached value of `this->getLogNormalizer() - this->getInnerCovLogDet() / 2`.
    */
    const Scalar getInnerLogNormalizer() const { return this->m_innerLogNormalizer; };
    
    /**
    * @return Returns the cached value of `this->getLogNormalizer() - this->getOuterCovLogDet() / 2`.
    */
    const Scalar getOuterLogNormalizer() const { return this->m_outerLogNormalizer; };
    
    /**
    * Computes the Mahalanobis distance between two vectors using the covariance matrix estimated during the last call to `fit()`.
    *
    * @param[in] x1 First vector.
    *
    * @param[in] x2 Second vector.
    *
    * @param[in] innerDist Specifies whether to use the inner or the outer distribution.
    *
    * @return Returns the Mahalanobis distance between x1 and x2.
    */
    const Scalar mahalanobisDistance(const Eigen::Ref<const Sample> & x1, const Eigen::Ref<const Sample> & x2, bool innerDist = true) const;


protected:

    CovMode m_covMode; /**< Specifies how the covariance matrix should be estimated. */
    std::shared_ptr<DataTensor> m_cumsum; /**< Cumulative sum of the data passed to `init()`. */
    std::shared_ptr<DataTensor> m_cumOuter; /**< Cumulative sum of the outer products of the samples passed to `init()`. */
    DataTensor::Index m_cumOuter_offset; /**< Offset of the first time step in `m_cumOuter` from the first time step in the data (used for partial cumulative sums). */
    DataTensor::Index m_cumOuter_maxLen; /**< Maximum number of time steps covered by `m_cumOuter` for memory's sake (used for partial cumulative sums). */
    Sample m_innerMean; /**< Mean of the inner or the shared distribution. */
    Sample m_outerMean; /**< Mean of the outer distribution. */
    ScalarMatrix m_innerCov; /**< Covariance matrix of the inner or the shared distribution. */
    ScalarMatrix m_outerCov; /**< Covariance matrix of the outer distribution. */
    ScalarMatrix m_outerProdSum; /**< Sum of outer products of the samples in the data tensor passed to `init()`. */
    Eigen::LLT<ScalarMatrix> m_innerCovChol; /**< Cholesky decomposition of the covariance matrix of the inner or the shared distribution. */
    Eigen::LLT<ScalarMatrix> m_outerCovChol; /**< Cholesky decomposition of the covariance matrix of the outer distribution. */
    Scalar m_innerCovLogDet; /**< Natural logarithm of the determinant of the covariance matrix of the inner or the shared distribution. */
    Scalar m_outerCovLogDet; /**< Natural logarithm of the determinant of the covariance matrix of the outer distribution. */
    Scalar m_logNormalizer; /**< `-D/2 * log(2 * pi)` */
    Scalar m_innerLogNormalizer; /**< `-D/2 * log(2 * pi) - this->m_innerCovLogDet / 2` */
    Scalar m_outerLogNormalizer; /**< `-D/2 * log(2 * pi) - this->m_outerCovLogDet / 2` */
    
    /**
    * Computes the cumulative sum of the outer products of the samples in the data tensor passed to `init()`
    * to speed up computation of covariance matrices later on. The result will be stored in `m_cumOuter`.
    *
    * The number of timesteps which the cumulative sums are computed for is limited by `m_cumOuter_maxLen`.
    * The @p offset parameter specifies the time step to start summation with.
    */
    void computeCumOuter(DataTensor::Index offset = 0);
    
    /**
    * Explicitely computes the sum of the outer products of the samples in a given @p range in the data tensor
    * passed to `init()`. This function will *not* use the cumulative sums of outer products in `m_cumOuter`,
    * but compute the sum explicitely. It is intended to be used when cumulative sums can't be used due to
    * high-dimensional data.
    *
    * @return Returns a square matrix with the sum of the outer products of the samples in the given range.
    */
    ScalarMatrix computeOuterSum(const IndexRange & range);
    

};


/**
* @brief Estimates the probability density of given data using an ensemble of histograms over random sparse
* 1d projections of the data.
* 
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class EnsembleOfRandomProjectionHistograms : public DensityEstimator
{
public:

    typedef DataTensor_<DataTensor::Index> IntTensor;
    
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;


    /**
    * Constructs an un-initialized EnsembleOfRandomProjectionHistograms with default parameters.
    * `init()` has to be called before this density estimator can be used.
    */
    EnsembleOfRandomProjectionHistograms();
    
    /**
    * Constructs an un-initialized EnsembleOfRandomProjectionHistograms with given parameters.
    * `init()` has to be called before this density estimator can be used.
    *
    * @param[in] num_hist The number of histograms.
    *
    * @param[in] num_bins The number of bins per histogram. If set to 0, the number of bins
    * will be determined automatically for each histogram individually.
    *
    * @param[in] discount Discount to be added to all histogram bins in order to make unseen values not
    * completely unlikely.
    */
    EnsembleOfRandomProjectionHistograms(DataTensor::Index num_hist, DataTensor::Index num_bins = 0, Scalar discount = 1);
    
    /**
    * Constructs and initializes a EnsembleOfRandomProjectionHistograms for a given data tensor with default parameters.
    *
    * @param[in] data Pointer to the DataTensor.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    EnsembleOfRandomProjectionHistograms(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs and initializes a EnsembleOfRandomProjectionHistograms for a given data tensor.
    *
    * @param[in] data Pointer to the DataTensor.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    *
    * @param[in] num_hist The number of histograms.
    *
    * @param[in] num_bins The number of bins per histogram. If set to 0, the number of bins
    * will be determined automatically for each histogram individually.
    *
    * @param[in] discount Discount to be added to all histogram bins in order to make unseen values not
    * completely unlikely.
    */
    EnsembleOfRandomProjectionHistograms(const std::shared_ptr<const DataTensor> & data, DataTensor::Index num_hist, DataTensor::Index num_bins = 0, Scalar discount = 1);
    
    /**
    * Makes a flat copy of another EnsembleOfRandomProjectionHistograms. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The EnsembleOfRandomProjectionHistograms to be copied.
    */
    EnsembleOfRandomProjectionHistograms(const EnsembleOfRandomProjectionHistograms & other);
    
    /**
    * Makes a flat copy of another EnsembleOfRandomProjectionHistograms. Most internal structures will be shared
    * between the original object and the copy, so this is a cheap operation.
    *
    * @param[in] other The EnsembleOfRandomProjectionHistograms to be copied.
    *
    * @return A reference to this object.
    */
    virtual EnsembleOfRandomProjectionHistograms & operator=(const EnsembleOfRandomProjectionHistograms & other);
    
    /**
    * Creates a copy of this object by calling the copy constructor of the actual derived class.
    *
    * @return Returns a pointer to a copy of this object.
    */
    virtual std::shared_ptr<DensityEstimator> clone() const override;
    
    /**
    * Initializes this density estimator with a given DataTensor @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
    *
    * The result of this function is undefined if the inner or the outer range consists of
    * missing samples only.
    */
    virtual void fit(const IndexRange & range) override;
    
    /**
    * Resets this density estimator to its uninitialized state and releases any memory allocated
    * by `init()` and `fit()`.
    */
    virtual void reset() override;
    
    /**
    * Computes the value of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the pdf values for the given sample under the inner and the outer
    * distribution. If the given sample is a missing sample, it will be assigned a pdf of 1.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const override;
    
    /**
    * Computes the logarithm of the probability density function of the inner and the outer
    * distribution for the sample at a given position in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] ind The index of the sample to compute the log-pdf for. The attribute dimension
    * will be ignored.
    * 
    * @return A pair with the logarithm of the pdf for the given sample under the inner and the
    * outer distribution. If the given sample is a missing sample, it will be assigned a log-pdf
    * of 0.
    */
    virtual std::pair<Scalar, Scalar> logpdf(const ReflessIndexVector & ind) const override;

    /**
    * Computes the log-likelihood of all samples in the sub-block passed to `fit()` of the DataTensor
    * passed to `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodInner() const override;

    /**
    * Computes the log-likelihood of all samples outside of the sub-block passed to `fit()` of the DataTensor
    * passed to `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodOuter() const override;
    
    /**
    * @return Returns the number of histograms in the ensemble.
    */
    DataTensor::Index numHist() const { return this->m_num_hist; };
    
    /**
    * Determines a suitable number of bins for 1-dimensional histograms for each dimension of the given data.
    *
    * The optimal number of bins `b` is determined by maximizing *penalized maximum likelihood*:
    *
    * \f[
    *     \sum_{i = 1}^{k}{ \left( n_i \cdot \log \left ( \frac{kn_i}{n} \right ) \right) } - k + 1 - (\log k)^{2.5}
    * \f]
    *
    * @param[in] data The data. The values for each attribute **must** be in the range [0,1].
    *
    * @return Returns a vector with the optimal number of bins for each attribute.
    */
    static IntTensor::Sample getOptimalBinNum(const DataTensor & data);


protected:

    DataTensor::Index m_num_hist; /**< Number of histograms in the ensemble. */
    DataTensor::Index m_num_bins; /**< The number of bins of each histogram. May be 0 to determine the number of bins automatically for each histogram indivdually. */
    Scalar m_discount; /**< Discount to be added to all histogram bins. */
    IntTensor::Sample m_hist_bins; /**< Number of bins in each individual histogram. */
    IntTensor::Sample m_hist_offsets; /**< Offsets of the first bin of each histogram in flat vectors. */
    std::shared_ptr<SparseMatrix> m_proj; /**< Sparse random projection vectors, one per row. */
    std::shared_ptr<IntTensor> m_indices; /**< Indices of the bins which the samples passed to `init()` fall into. */
    std::shared_ptr<IntTensor> m_counts; /**< Cumulative counts for the bins of all histograms. */
    IntTensor::Sample m_hist_inner; /**< Flat vector of histogram bins for the data in the range passed to `fit()`. */
    IntTensor::Sample m_hist_outer; /**< Flat vector of histogram bins for the data outside of the range passed to `fit()`. */
    mutable Sample m_logprob_inner; /**< Flat vector of log-PDF estimates over the inner range for each bin of all histograms. */
    mutable Sample m_logprob_outer; /**< Flat vector of log-PDF estimates over the outer range for each bin of all histograms. */
    mutable std::shared_ptr<Sample> m_log_cache; /**< Cached values for `log(n + discount)` for `0 <= n <=N`. */
    mutable std::unordered_map<DataTensor::Index, Sample> m_log_denom_cache; /**< Cached values for `log(N/bins + discount)` for each histogram. */
    mutable bool m_logprob_normalized; /**< Indicates if the log-denominator `log(N/bins + discount)` has already been subtracted from m_logprob_inner/outer. */
    
    /**
    * Retrieves the log-denominator `log(n/bins + discount)` from a cache, adding it if it does not exist yet.
    *
    * @param[in] n The number of samples in the histogram.
    *
    * @return Returns a vector with the log-demoninator for each histogram.
    */
    const Sample & logDenomFromCache(DataTensor::Index n) const;

};

}

#endif