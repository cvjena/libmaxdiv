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

#include "pointwise_detectors.h"
#include "math_utils.h"
using namespace MaxDiv;


DataTensor MaxDiv::hotellings_t(const DataTensor & data)
{
    if (data.empty())
        return DataTensor();
    
    // Subtract the mean from the data
    DataTensor centered = data - (data.data().colwise().sum() / data.numValidSamples());
    
    if (data.shape().d == 1)
    {
        // Univariate case: score = (x - mu)^2 / variance
        centered.data() = centered.data().cwiseAbs2();
        centered.data() /= centered.data().sum() / centered.numValidSamples();
        return centered;
    }
    else
    {
        // Multivariate case: score = (x - mu)^T * S^-1 * (x - mu)
        ScalarMatrix normed;
        
        {
            // Compute covariance matrix S
            ScalarMatrix cov;
            cov.noalias() = centered.data().transpose() * centered.data();
            cov /= static_cast<Scalar>(centered.numValidSamples());
            
            // Compute S^-1 * (x - mu)
            Eigen::LLT<ScalarMatrix> llt;
            cholesky(cov, &llt);
            normed = llt.solve(centered.data().transpose());
        }
        
        // Create tensor for scores and compute (x - mu)^T * (S^-1 * (x - mu))
        ReflessIndexVector scoresShape = data.shape();
        scoresShape.d = 1;
        DataTensor scores(scoresShape);
        scores.data() = centered.data().transpose().cwiseProduct(normed).colwise().sum().transpose();
        scores.copyMask(data);
        return scores;
    }
}


DataTensor MaxDiv::pointwise_kde(const DataTensor & data, Scalar kernel_sigma_sq)
{
    if (data.empty())
        return DataTensor();
    
    ReflessIndexVector scoresShape = data.shape();
    scoresShape.d = 1;
    DataTensor scores(scoresShape);
    scores.data() = GaussKernel(data, kernel_sigma_sq, false).rowwiseMean();
    scores.data().array() -= static_cast<Scalar>(1);
    scores.data() *= -1;
    scores.copyMask(data);
    return scores;
}
