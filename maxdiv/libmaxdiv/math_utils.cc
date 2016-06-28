#include "math_utils.h"
using namespace MaxDiv;


ScalarMatrix MaxDiv::gauss_kernel(const DataTensor & data, Scalar kernel_sigma_sq, bool normed)
{
    return gauss_kernel(data.data(), kernel_sigma_sq, normed);
}


GaussKernel::GaussKernel(const DataTensor & data, Scalar kernel_sigma_sq, bool normed)
: m_data(data), m_sigma_sq(-2 * kernel_sigma_sq), m_norm(std::pow(2 * M_PI * kernel_sigma_sq, data.numAttrib() / 2.0)), m_normed(normed) {}
    
Scalar GaussKernel::operator()(DataTensor::Index x, DataTensor::Index y) const
{
    Scalar k = std::exp((this->m_data.sample(x) - this->m_data.sample(y)).squaredNorm() / this->m_sigma_sq);
    if (this->m_normed)
        k /= this->m_norm;
    return k;
}

ScalarMatrix GaussKernel::materialize() const
{
    return gauss_kernel(this->m_data.data(), -0.5 * this->m_sigma_sq, this->m_normed);
}

Sample GaussKernel::column(DataTensor::Index col) const
{
    Sample column(this->m_data.numSamples());
    
    // Compute distances
    Sample sample = this->m_data.sample(col);
    for (DataTensor::Index row = 0; row < static_cast<DataTensor::Index>(column.size()); ++row)
        column(row) = (sample - this->m_data.sample(row)).squaredNorm();
    
    // Compute kernel values
    column = (column.array() / this->m_sigma_sq).exp();
    if (this->m_normed)
        column /= this->m_norm;
    
    return column;
}

Sample GaussKernel::rowwiseSum() const
{
    if (this->m_data.numSamples() < 10000)
        // explicit materialization of the kernel matrix is faster for few samples
        return this->materialize().rowwise().sum();
    
    Sample sum = Sample::Zero(this->m_data.numSamples());
    DataTensor::Index x, y;
    
    #if defined(_OPENMP) and _OPENMP >= 201307
    #pragma omp parallel for private(x,y) reduction(+:sum) schedule(static,1)
    #endif
    for (x = 0; x < static_cast<DataTensor::Index>(sum.size()); ++x)
    {
        for (y = x + 1; y < static_cast<DataTensor::Index>(sum.size()); ++y)
        {
            Scalar k = std::exp((this->m_data.sample(x) - this->m_data.sample(y)).squaredNorm() / this->m_sigma_sq);
            sum(x) += k;
            sum(y) += k;
        }
    }
    
    sum.array() += 1; // diagonal elements
    if (this->m_normed)
        sum /= this->m_norm;
    
    return sum;
}

Sample GaussKernel::rowwiseMean() const
{
    Sample mean = this->rowwiseSum();
    mean /= mean.size();
    return mean;
}