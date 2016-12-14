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