#ifndef MAXIDV_ESTIMATORS_H
#define MAXIDV_ESTIMATORS_H

#include <memory>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Cholesky>
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
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
    */
    virtual void fit(const IndexRange & range) =0;
    
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
    * distribution.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const =0;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples.
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
    * inner and the outer distribution for all samples in the given sub-block.
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
    * of the given sub-block.
    */
    virtual ScalarMatrix pdfOutsideRange(IndexRange range) const;
    
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
    * Computes the log-likelihood of all samples in a given sub-block of the DataTensor passed to
    * `init()` for the inner and the outer distribution.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @param[in] range The block to compute the log-likelihood for. The attribute dimension will be ignored.
    *
    * @return A pair with the log-likelihood of the samples for the inner (first) and the outer
    * (second) distribution.
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
    * (second) distribution.
    */
    virtual std::pair<Scalar, Scalar> logLikelihoodOutsideRange(IndexRange range) const;


protected:

    std::shared_ptr<const DataTensor> m_data; /**< Pointer to the DataTensor passed to `init()`. */
    int m_singletonDim; /**< Index of the non-singleton dimension if there is only one in the data, otherwise -1. */

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
    * @param[in] normed Specifies whether to normalize the kernel in order to get a proper distribution.
    */
    KernelDensityEstimator(Scalar kernel_sigma_sq, bool normed = true);
    
    /**
    * Constructs and initializes a KernelDensityEstimator for a given data tensor.
    *
    * @param[in] data Pointer to the DataTensor.
    *
    * @param[in] kernel_sigma_sq The "standard deviation" of the gaussian kernel.
    *
    * @param[in] normed Specifies whether to normalize the kernel in order to get a proper distribution.
    */
    KernelDensityEstimator(const std::shared_ptr<const DataTensor> & data, Scalar kernel_sigma_sq = 1.0, bool normed = true);
    
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
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
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
    * distribution.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const override;


protected:

    Scalar m_sigma_sq; /**< Standard deviation of the Gaussian kernel. */
    bool m_normed; /**< Whether to normalize the kernel. */
    std::shared_ptr<GaussKernel> m_kernel; /**< Pointer to the Gaussian kernel instance. */
    std::shared_ptr<DataTensor> m_cumKernel; /**< Materialized kernel matrix with cumulated rows. */
    IndexRange m_extremeRange; /**< Range of inner block passed to `fit()`. */
    DataTensor::Index m_numExtremes; /**< Number of samples in the block passed to `fit()`. */

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
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data) override;
    
    /**
    * Fits the parameters of the inner and outer distribution to a sub-block of the
    * DataTensor passed to `init()` specified by the given @p range.
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
    * distribution.
    */
    virtual std::pair<Scalar, Scalar> pdf(const ReflessIndexVector & ind) const override;
    
    /**
    * Computes the values of the probability density function of the inner and the outer
    * distribution for all samples in the DataTensor passed to `init()`.
    *
    * `init()` and `fit()` have to be called before this can be used.
    *
    * @return A DataTensor with two attributes which specify the values of the pdf of the
    * inner and the outer distribution for all samples.
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
    * inner and the outer distribution for all samples in the given sub-block.
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
    * of the given sub-block.
    */
    virtual ScalarMatrix pdfOutsideRange(IndexRange range) const override;
    
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
    * (second) distribution.
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
    * (second) distribution.
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
    Sample m_innerMean; /**< Mean of the inner or the shared distribution. */
    Sample m_outerMean; /**< Mean of the outer distribution. */
    ScalarMatrix m_innerCov; /**< Covariance matrix of the inner or the shared distribution. */
    ScalarMatrix m_outerCov; /**< Covariance matrix of the outer distribution. */
    Eigen::LLT<ScalarMatrix> m_innerCovChol; /**< Cholesky decomposition of the covariance matrix of the inner or the shared distribution. */
    Eigen::LLT<ScalarMatrix> m_outerCovChol; /**< Cholesky decomposition of the covariance matrix of the outer distribution. */
    Scalar m_innerCovLogDet; /**< Natural logarithm of the determinant of the covariance matrix of the inner or the shared distribution. */
    Scalar m_outerCovLogDet; /**< Natural logarithm of the determinant of the covariance matrix of the outer distribution. */
    Scalar m_normalizer; /**< `(2 * pi)^(-D/2)` */
    Scalar m_innerNormalizer; /**< `(2 * pi)^(-D/2) * sqrt(this->m_innerCovLogDet)` */
    Scalar m_outerNormalizer; /**< `(2 * pi)^(-D/2) * sqrt(this->m_outerCovLogDet)` */
    
    /**
    * Computes the cumulative sum of the outer products of the samples in @p data to speed up
    * computation of covariance matrices later on. The result will be stored in `m_cumOuter`.
    */
    void computeCumOuter(const DataTensor & data);
    

};

}

#endif