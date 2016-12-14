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

#ifndef MAXDIV_CONFIG_H
#define MAXDIV_CONFIG_H

#ifdef MAXDIV_FLOAT
typedef float MaxDivScalar;
#else
typedef double MaxDivScalar;
#endif

/**
* Number of dimensions of a DataTensor. By default, we have 5 dimensions: time, x, y, z and
* attribute. Better do not change this, unless you know what you're doing and are aware, that
* many parts of the MaxDiv code would have to be adjusted as well.
*/
#define MAXDIV_INDEX_DIMENSION 5

#ifndef MAXDIV_KDE_CUMULATIVE_SIZE_LIMIT
/**
* Usually, Kernel Density Estimation can be sped up by using tensors with cumulative sums of
* the kernel matrix to avoid redundant summations. But since those matrices consume a lot of
* memory (quadratic in the number of samples), that approach may be unfeasible for a large
* number of samples.  
* This constant sets a limit on the number of samples which cumulative sums will be used
* for during in Kernel Density Estimation. For a larger number of samples, cumulative sums will
* be disabled in favor of redundant summations.
*
* You may want to adjust this limit in case you have more or less memory available. The default
* value has been chosen under the assumption of 4 GiB of RAM.
*/
#define MAXDIV_KDE_CUMULATIVE_SIZE_LIMIT 20000
#endif

#ifndef MAXDIV_GAUSSIAN_CUMULATIVE_SIZE_LIMIT
/**
* Usually, the estimation of the parameters of a Gaussian distribution can be sped up by using
* tensors with cumulative sums of the samples and cumulative sums of their outer products to avoid
* redundant summations. But since those matrices, especially the matrix of outer product sums,
* consume a lot of memory, that approach may be unfeasible for a large number of samples and attributes.
* 
* This constant sets a limit on the size of the tensor with cumulative sums of outer products in bytes.
* If the size of the entire tensor would exceed this limit, it will be split up into partial cumulative sums
* which will be computed piecewise.  
* Note that if multiple processors are available to process the data in parallel, each thread will allocate
* a tensor with size up to this limit.
*
* You may want to adjust this limit in case you have more or less memory available. The default
* value limits the size of the tensor of cumulative outer products to 2 GiB.
*/
#define MAXDIV_GAUSSIAN_CUMULATIVE_SIZE_LIMIT 2147483648
#endif

#ifndef MAXDIV_NMP_LIMIT
/**
* For offline non-maximum suppression, the scores of all sub-blocks in the data have to be
* retrieved and stored first. Since the number of sub-blocks grows quadratically with the
* data size, those scores may consume a lot of memory. In this case, an online non-maximum
* suppression algorithm would be much more efficient regarding space and time, but its output
* will depend on the order which the scores are retrieved in.
*
* This constant sets a limit on the number of samples which offline non-maximum suppression
* is feasible for. Online non-maximum suppression will be used for data sets with more samples.
*/
#define MAXDIV_NMP_LIMIT 10000
#endif

#endif