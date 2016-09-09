#ifndef MAXDIV_POINTWISE_DETECTORS_H
#define MAXDIV_POINTWISE_DETECTORS_H

#include "DataTensor.h"

namespace MaxDiv
{

/**
* @brief Hotelling's T^2 Outlier Detection
* 
* Scores every sample in the time series @p data by it's Mahalanobis distance to the mean.
*
* @return Returns a DataTensor with the same shape as `data`, but with only one attribute,
* which is the outlier score of the corresponding sample.
*
* @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
*/
DataTensor hotellings_t(const DataTensor & data);


/**
* Scores every sample in the time series @p data by it's unlikelihood under Kernel Density Estimation.
*
* @param[in] data The data tensor to perform point-wise outlier detectin on.
* If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
*
* @param[in] kernel_sigma_sq The variance of the Gauss kernel used by KDE.
*
* @return Returns a DataTensor with the same shape as `data`, but with only one attribute,
* which is the outlier score of the corresponding sample.
*
* @note The values of `data` should be in range [-1,1] to avoid numerical problems.
*/
DataTensor pointwise_kde(const DataTensor & data, Scalar kernel_sigma_sq = 1.0);

}

#endif