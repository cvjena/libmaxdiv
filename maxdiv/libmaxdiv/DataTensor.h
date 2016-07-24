#ifndef MAXDIV_DATATENSOR_H
#define MAXDIV_DATATENSOR_H

#include <cstddef>
#include <cstring>
#include <cassert>
#include <set>
#include <Eigen/Core>
#include "config.h"
#include "indexing.h"

namespace MaxDiv
{

typedef MaxDivScalar Scalar; /**< Default scalar type used throughout MaxDiv. */


/**
* @brief Container for multivariate spatio-temporal data
*
* Objects of this class store scalar values along 5 dimensions: time, x, y, z, attribute.  
* The first dimension is the temporal dimension, i.e. the time axis.  
* The second, third and fourth dimensions are spatial dimensions. The distance between each
* pair of consecutive spatial indices is assumed to be constant.  
* The last dimension is the feature or attribute dimension for multivariate time series.
* 
* The memory will be layed out in such a way that the last dimension is changing fastest.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
template<typename Scalar>
class DataTensor_
{

public:

    typedef IndexVector::Index Index; /**< Index types for the dimensions of the tensor. */
    
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Sample; /**< Feature vector type of a single timestamp at a single location. */
    
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ScalarMatrix; /**< A matrix of scalar values. */
    
    typedef Eigen::Map< ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > ScalarMatrixMap;
    typedef Eigen::Map< const ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > ConstScalarMatrixMap;

    /**
    * Constructs an empty tensor with 0 elements.
    */
    DataTensor_() : m_shape(), m_size(0), m_numEl(0), m_numSamples(0), m_data_p(NULL), m_data(NULL, 0, 0), m_allocated(false) {};
    
    /**
    * Constructs a tensor with given the shape.
    *
    * @param[in] shape IndexVector with the size of each dimension.
    */
    DataTensor_(const ReflessIndexVector & shape)
    : m_shape(shape), m_size(shape.prod()), m_numEl(m_size), m_numSamples(shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)),
      m_data_p((m_size > 0) ? new Scalar[m_size] : NULL),
      m_data(m_data_p, m_numSamples, shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(m_size > 0)
    {};
    
    /**
    * Constructs a tensor with the given shape and initializes all elments with a
    * given scalar value.
    *
    * @param[in] shape IndexVector with the size of each dimension.
    *
    * @param[in] value The value to initialize every element with.
    */
    DataTensor_(const ReflessIndexVector & shape, Scalar value)
    : DataTensor_(shape)
    { this->setConstant(value); };
    
    /**
    * Constructs a tensor with the given shape and initializes all data points
    * with a constant vector.
    *
    * @param[in] shape IndexVector with the size of each dimension.
    *
    * @param[in] value The initial value for all spatio-temporal samples.
    * This vector must have the same number of elements as specified in `shape.d`.
    */
    DataTensor_(const ReflessIndexVector & shape, const Sample & value)
    : DataTensor_(shape)
    { this->setConstant(value); };
    
    /**
    * Constructs a tensor wrapping an external data container.
    *
    * This version of the constructor does not allocate an own data storage and the
    * destructor will not free any memory.
    *
    * @param[in] data Pointer to external data.
    *
    * @param[in] shape IndexVector with the size of each dimension.
    */
    DataTensor_(Scalar * data, const ReflessIndexVector & shape)
    : m_shape(shape), m_size(shape.prod()), m_numEl(m_size), m_numSamples(shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)),
      m_data_p(data), m_data(data, m_numSamples, shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(false)
    {};
    
    /**
    * Copies another DataTensor.
    *
    * @param[in] other The tensor to be copied.
    */
    DataTensor_(const DataTensor_ & other)
    : DataTensor_(other.m_shape)
    {
        if (other.m_data_p != NULL && this->m_data_p != NULL)
            std::memcpy(reinterpret_cast<void*>(this->m_data_p), reinterpret_cast<const void*>(other.m_data_p), sizeof(Scalar) * this->m_size);
    };
    
    /**
    * Moves the data of another DataTensor to this one and leaves the other
    * one empty.
    *
    * @param[in] other The tensor whose data is to be moved.
    */
    DataTensor_(DataTensor_ && other)
    : m_shape(other.m_shape), m_size(other.m_size), m_numEl(other.m_numEl), m_numSamples(other.m_numSamples),
      m_data_p(other.m_data_p), m_data(m_data_p, m_numSamples, m_shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(other.m_allocated)
    {
        other.m_shape = ReflessIndexVector();
        other.m_size = 0;
        other.m_data_p = NULL;
        other.m_allocated = false;
    };
    
    /**
    * Copies another DataTensor based on another scalar type by casting its elements
    * to the type of this DataTensor.
    *
    * @param[in] other The tensor to be copied.
    */
    template<typename OtherScalar>
    DataTensor_(const DataTensor_<OtherScalar> & other)
    : DataTensor_(other.shape())
    {
        if (!other.empty() && this->m_data_p != NULL)
        {
            Scalar * myData = this->m_data_p, * myDataEnd = myData + this->m_size;
            const OtherScalar * otherData = other.raw();
            for (; myData != myDataEnd; myData++, otherData++)
                *myData = static_cast<Scalar>(*otherData);
        }
    };
    
    virtual ~DataTensor_() { if (this->m_allocated) delete[] this->m_data_p; };
    
    /**
    * Copies the contents of another DataTensor to this one.
    *
    * @param[in] other The tensor to be copied.
    */
    virtual DataTensor_ & operator=(const DataTensor_ & other)
    {
        if (this == &other)
            return *this;
        
        this->resize(other.m_shape);
        if (this->numEl() > 0)
            std::memcpy(reinterpret_cast<void*>(this->m_data_p), reinterpret_cast<const void*>(other.m_data_p), sizeof(Scalar) * this->numEl());
        return *this;
    };
    
    /**
    * Moves the contents of another DataTensor to this one and leaves the other one empty.
    *
    * @param[in] other The tensor whose data is to be moved.
    */
    virtual DataTensor_ & operator=(DataTensor_ && other)
    {
        if (this == &other)
            return *this;
        
        if (this->m_allocated)
            delete[] this->m_data_p;
        
        this->m_shape = other.m_shape;
        this->m_size = other.m_size;
        this->m_numEl = other.m_numEl;
        this->m_numSamples = other.m_numSamples;
        this->m_data_p = other.m_data_p;
        this->m_allocated = other.m_allocated;
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
        
        other.m_shape = ReflessIndexVector();
        other.m_size = 0;
        other.m_data_p = NULL;
        other.m_allocated = false;
        
        return *this;
    };
    
    /**
    * Copies the contents of another DataTensor based on another scalar type by casting
    * its elements to the scalar type of this DataTensor.
    *
    * @param[in] other The tensor to be copied.
    */
    template<typename OtherScalar>
    DataTensor_ & operator=(const DataTensor_<OtherScalar> & other)
    {
        this->resize(other.shape());
        if (!other.empty() && this->m_data_p != NULL)
        {
            Scalar * myData = this->m_data_p, * myDataEnd = myData + this->numEl();
            const OtherScalar * otherData = other.raw();
            for (; myData != myDataEnd; myData++, otherData++)
                *myData = static_cast<Scalar>(*otherData);
        }
        return *this;
    };
    
    /**
    * @return Returns true if this tensor has no elements.
    */
    bool empty() const { return (this->m_data_p == NULL || this->m_numEl == 0); };
    
    /**
    * @return Returns the shape of this tensor as vector with the size of each dimension.
    */
    const ReflessIndexVector & shape() const { return this->m_shape; };
    
    /**
    * @return Returns the number of time steps in this tensor. Equivalent to `shape().t`.
    */
    Index length() const { return this->m_shape.t; };
    
    /**
    * @return Returns the spatial extent of this data tensor along the x-axis. Equivalent to `shape().x`.
    */
    Index width() const { return this->m_shape.x; };
    
    /**
    * @return Returns the spatial extent of this data tensor along the y-axis. Equivalent to `shape().y`.
    */
    Index height() const { return this->m_shape.y; };
    
    /**
    * @return Returns the spatial extent of this data tensor along the z-axis. Equivalent to `shape().z`.
    */
    Index depth() const { return this->m_shape.z; };
    
    /**
    * @return Returns the number of attributes/features of the samples in this tensor. Equivalent to `shape().d`.
    */
    Index numAttrib() const { return this->m_shape.d; };
    
    /**
    * @return Returns the number of elements in this tensor, i.e. `shape().prod()`.
    */
    Index numEl() const { return this->m_numEl; };
    
    /**
    * @return Returns the number of samples in this tensor, i.e. `length * width * height * depth`.
    */
    Index numSamples() const { return this->m_numSamples; };
    
    /**
    * Creates an index vector for a specific element and the shape of this tensor.
    * @param[in] t Index along the time axis.
    * @param[in] x Index in the 1st spatial dimension.
    * @param[in] y Index in the 2nd spatial dimension.
    * @param[in] z Index in the 3rd spatial dimension.
    * @param[in] d Index of the attribute/feature.
    * @return `IndexVector(this->shape(), t, x, y, z, d)`
    */
    IndexVector makeIndexVector(Index t = 0, Index x = 0, Index y = 0, Index z = 0, Index d = 0) const
    { return IndexVector(this->m_shape, t, x, y, z, d); };
    
    /**
    * Changes the shape of this tensor.
    *
    * New memory will only be allocated if the number of elements is greater than before.
    * In that case, existing data will be lost.
    *
    * @param[in] shape IndexVector with the new size of each dimension.
    */
    void resize(const ReflessIndexVector & shape)
    {
        Index numEl = shape.prod();
        if (numEl > this->m_size)
        {
            if (this->m_allocated)
                delete[] this->m_data_p;
            this->m_data_p = new Scalar[numEl];
            this->m_allocated = true;
            this->m_size = numEl;
        }
        this->m_shape = shape;
        this->m_numEl = numEl;
        this->m_numSamples = shape.prod(0, MAXDIV_INDEX_DIMENSION - 2);
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
    };
    
    /**
    * Shrink the memory allocated by this tensor to its current size.
    *
    * Normally, when resize() is called to resize the tensor to a smaller size, memory won't be
    * re-allocated, so that a portion of the allocated memory won't be used with the new size.
    * This method shrinks the allocated memory to the actual size of the tensor.
    */
    void shrink()
    {
        if (this->m_size > this->numEl())
        {
            Scalar * newData = new Scalar[this->numEl()];
            std::memcpy(reinterpret_cast<void*>(newData), reinterpret_cast<const void*>(this->m_data_p), sizeof(Scalar) * this->numEl());
            if (this->m_allocated)
                delete[] this->m_data_p;
            this->m_data_p = newData;
            this->m_allocated = true;
            this->m_size = this->numEl();
            new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
        }
    }
    
    /**
    * Releases any memory allocated by this tensor and resets its size to 0.
    */
    void release()
    {
        if (this->m_allocated)
            delete[] this->m_data_p;
        this->m_allocated = false;
        this->m_data_p = NULL;
        this->m_size = this->m_numEl = this->m_numSamples = 0;
        this->m_shape.vec().setZero();
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(NULL, 0, 0);
    }
    
    /**
    * Crops this feature matrix to a sub-block.
    *
    * @param[in] range The range of indices to be contained in the sub-block.
    */
    void crop(const IndexRange & range)
    {
        // Do nothing if shape does not change
        if (range.a == 0 && range.b == this->m_shape)
            return;
        
        // Validate cropping region
        assert((range.a.vec() <= range.b.vec()).all() && (range.b.vec() <= this->m_shape.vec()).all());
        
        // Shortcut for cutting of some trailing time steps and leaving everything else unchanged
        if (range.a == 0 && range.b.vec().tail(MAXDIV_INDEX_DIMENSION - 1) == this->m_shape.vec().tail(MAXDIV_INDEX_DIMENSION - 1))
        {
            this->resize(range.b);
            return;
        }
        
        // Compute shape of cropped block
        ReflessIndexVector newShape = range.shape();
        Index newNumEl = newShape.prod();
        
        // Shortcut for an empty cropping region
        if ((newShape.vec() == 0).any())
        {
            this->resize(newShape);
            return;
        }
        
        // Move data from sub-block to new positions
        Scalar * data = this->m_data_p;
        IndexVector ind(newShape, 0);
        for (Index i = 0; i < newNumEl; ++i, ++ind, ++data)
            *data = (*this)(range.a + ind);
        this->resize(newShape);
    };
    
    /**
    * Sets all elements in the tensor to a constant value.
    *
    * @param[in] val The new value for all elements of the tensor.
    */
    void setConstant(const Scalar val) { this->m_data.setConstant(val); };
    
    /**
    * Sets all data points in the tensor to a constant vector.
    *
    * @param[in] value The new value for all spatio-temporal samples.
    * This vector must have the same number of elements as specified in `shape.d`.
    */
    void setConstant(const Sample & value)
    {
        assert(value.size() == this->numAttrib());
        this->m_data.rowwise() = value.transpose();
    };
    
    /**
    * Sets all elements in the tensor to 0.
    */
    void setZero() { this->setConstant(static_cast<Scalar>(0)); };
        
    /**
    * @return Returns a pointer to the raw data storage of this tensor.
    * The returned pointer may be NULL if the tensor is empty.
    */
    Scalar * raw() { return this->m_data_p; };
    
    /**
    * @return Returns a const pointer to the raw data storage of this tensor.
    * The returned pointer may be NULL if the tensor is empty.
    */
    const Scalar * raw() const { return this->m_data_p; };
    
    /**
    * Returns an Eigen::Map object wrapping the raw data of this tensor, where the samples
    * are stored in a contiguous way, i.e. the returned object will have as many rows as
    * there are data points and as many columns as each data points has attributes.
    */
    Eigen::Map<ScalarMatrix> data() { return this->m_data; };
    
    /**
    * Returns a constant Eigen::Map object wrapping the raw data of this tensor, where the
    * samples are stored in a contiguous way, i.e. the returned object will have as rows
    * as there are data points and as many columns as each data points has attributes.
    */
    Eigen::Map<const ScalarMatrix> data() const
    { return Eigen::Map<const ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib()); };
    
    /**
    * Provides a linear view on this tensor as by concatenating all elements.
    *
    * @return Returns a Eigen::Map object with `numEl()` rows and 1 column.
    */
    Eigen::Map<Sample> asVector() { return Eigen::Map<Sample>(this->m_data_p, this->numEl(), 1); };
    
    /**
    * Provides a linear view on this tensor as by concatenating all elements.
    *
    * @return Returns a constant Eigen::Map object with `numEl()` rows and 1 column.
    */
    Eigen::Map<const Sample> asVector() const { return Eigen::Map<const Sample>(this->m_data_p, this->numEl(), 1); };
    
    /**
    * Provides a view on this tensor as as a matrix where each row is the concatenation of all samples
    * at a single time step. Thus, the matrix has `width * height * depth * numAttrib` columns.
    *
    * @return Eigen::Map object
    */
    Eigen::Map<ScalarMatrix> asTemporalMatrix()
    { return Eigen::Map<ScalarMatrix>(this->m_data_p, this->length(), this->m_shape.prod(1)); };
    
    /**
    * Provides a view on this tensor as as a matrix where each row is the concatenation of all samples
    * at a single time step. Thus, the matrix has `width * height * depth * numAttrib` columns.
    *
    * @return constant Eigen::Map object
    */
    Eigen::Map<const ScalarMatrix> asTemporalMatrix() const
    { return Eigen::Map<const ScalarMatrix>(this->m_data_p, this->length(), this->m_shape.prod(1)); };
    
    /**
    * Returns an `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    */
    Eigen::Map<Sample> operator()(Index t, Index x = 0, Index y = 0, Index z = 0)
    {
        assert(this->m_data_p != NULL);
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        return Eigen::Map<Sample>(this->m_data_p + this->makeIndexVector(t, x, y, z).linear(), this->m_shape.d);
    };
    
    /**
    * Returns a constant `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    */
    Eigen::Map<const Sample> operator()(Index t, Index x = 0, Index y = 0, Index z = 0) const
    {
        assert(this->m_data_p != NULL);
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        return Eigen::Map<const Sample>(this->m_data_p + this->makeIndexVector(t, x, y, z).linear(), this->m_shape.d);
    };
    
    /**
    * @return Returns a reference to the element at the given @p index in this tensor.
    */
    Scalar & operator()(const ReflessIndexVector & index)
    {
        assert(this->m_data_p != NULL);
        assert((index.vec() < this->m_shape.vec()).all());
        return *(this->m_data_p + IndexVector(this->m_shape, index).linear());
    };
    
    /**
    * @return Returns a const reference to the element at the given @p index in this tensor.
    */
    const Scalar & operator()(const ReflessIndexVector & index) const
    {
        assert(this->m_data_p != NULL);
        assert((index.vec() < this->m_shape.vec()).all());
        return *(this->m_data_p + IndexVector(this->m_shape, index).linear());
    };
    
    /**
    * Returns an `Eigen::Map<Sample>` object wrapping a single sample in this tensor,
    * addressed by a linear index.
    *
    * @param[in] s The linear index of the sample (`0 <= s < length * width * height * depth`).
    */
    Eigen::Map<Sample> sample(Index s)
    {
        assert(this->m_data_p != NULL);
        assert(s >= 0 && s < this->numSamples());
        return Eigen::Map<Sample>(this->m_data_p + s * this->numAttrib(), this->numAttrib());
    };
    
    /**
    * Returns a constant `Eigen::Map<Sample>` object wrapping a single sample in this tensor,
    * addressed by a linear index.
    *
    * @param[in] s The linear index of the sample (`0 <= s < length * width * height * depth`).
    */
    Eigen::Map<const Sample> sample(Index s) const
    {
        assert(this->m_data_p != NULL);
        assert(s >= 0 && s < this->numSamples());
        return Eigen::Map<const Sample>(this->m_data_p + s * this->numAttrib(), this->numAttrib());
    };
    
    /**
    * Returns an `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    *
    * @param[in] index The index vector with the coordinates of the sample.
    * The attribute dimension will be ignored.
    */
    Eigen::Map<Sample> sample(const ReflessIndexVector & index)
    {
        return (*this)(index.t, index.x, index.y, index.z);
    };
    
    /**
    * Returns a constant `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    *
    * @param[in] index The index vector with the coordinates of the sample.
    * The attribute dimension will be ignored.
    */
    Eigen::Map<const Sample> sample(const ReflessIndexVector & index) const
    {
        return (*this)(index.t, index.x, index.y, index.z);
    };
    
    /**
    * Provides a view on the time series for a single location in this tensor.
    *
    * @param[in] x The index of the location along the 1st spatial dimension.
    *
    * @param[in] y The index of the location along the 2nd spatial dimension.
    *
    * @param[in] z The index of the location along the 3rd spatial dimension.
    *
    * @return Returns an `Eigen::Map` object where each row corresponds to a time step
    * and contains as many columns as there are attributes/features.
    */
    ScalarMatrixMap location(Index x, Index y, Index z)
    {
        assert(this->m_data_p != NULL);
        assert(x >= 0 && y >= 0 && z >= 0 && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        return ScalarMatrixMap(
            this->m_data_p + this->makeIndexVector(0, x, y, z).linear(),
            this->m_shape.t, this->m_shape.d,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_shape.prod(1), 1)
        );
    };
    
    /**
    * Provides a view on the time series for a single location in this tensor.
    *
    * @param[in] x The index of the location along the 1st spatial dimension.
    *
    * @param[in] y The index of the location along the 2nd spatial dimension.
    *
    * @param[in] z The index of the location along the 3rd spatial dimension.
    *
    * @return Returns a constant `Eigen::Map` object where each row corresponds
    * to a time step and contains as many columns as there are attributes/features.
    */
    ConstScalarMatrixMap location(Index x, Index y, Index z) const
    {
        assert(this->m_data_p != NULL);
        assert(x >= 0 && y >= 0 && z >= 0 && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        return ConstScalarMatrixMap(
            this->m_data_p + this->makeIndexVector(0, x, y, z).linear(),
            this->m_shape.t, this->m_shape.d,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_shape.prod(1), 1)
        );
    };
    
    /**
    * Provides a view on the data for a single attribute in this tensor.
    *
    * @param[in] d The index of the attribute.
    *
    * @return Returns an `Eigen::Map` object where each row corresponds to a time step
    * and contains `width * height * depth` columns, one for each spatial location.
    */
    ScalarMatrixMap channel(Index d)
    {
        assert(this->m_data_p != NULL);
        assert(d >= 0 && d < this->m_shape.d);
        return ScalarMatrixMap(
            this->m_data_p + d,
            this->m_shape.t, this->m_shape.prod(1, 3),
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_shape.prod(1), this->m_shape.d)
        );
    };
    
    /**
    * Provides a view on the data for a single attribute in this tensor.
    *
    * @param[in] d The index of the attribute.
    *
    * @return Returns a constant `Eigen::Map` object where each row corresponds to a time step
    * and contains `width * height * depth` columns, one for each spatial location.
    */
    ConstScalarMatrixMap channel(Index d) const
    {
        assert(this->m_data_p != NULL);
        assert(d >= 0 && d < this->m_shape.d);
        return ConstScalarMatrixMap(
            this->m_data_p + d,
            this->m_shape.t, this->m_shape.prod(1, 3),
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_shape.prod(1), this->m_shape.d)
        );
    };
    
    /**
    * Adds a constant feature vector to all samples in this DataTensor and
    * returns the result.
    *
    * @param[in] sample Feature vector to be added to each sample.
    *
    * @returns Returns a reference to the new tensor which forms the sum.
    */
    DataTensor_ operator+(const Sample & sample) const
    {
        DataTensor_ sum(*this);
        return sum += sample;
    };
    
    /**
    * Adds a constant feature vector to all samples in this DataTensor.
    *
    * @param[in] sample Feature vector to be added to each sample.
    *
    * @returns Returns a reference to this DataTensor.
    */
    DataTensor_ & operator+=(const Sample & sample)
    {
        assert(sample.size() == this->m_shape.d);
        this->m_data.rowwise() += sample.transpose();
        return *this;
    };
    
    /**
    * Subtracts a constant feature vector from all samples in this DataTensor
    * and returns the result.
    *
    * @param[in] sample Feature vector to be subtracted from each sample.
    *
    * @returns Returns a reference to the new tensor which forms the difference.
    */
    DataTensor_ operator-(const Sample & sample) const
    {
        DataTensor_ sum(*this);
        return sum -= sample;
    };
    
    /**
    * Subtracts a constant feature vector from all samples in this DataTensor.
    *
    * @param[in] sample Feature vector to be subtracted from each sample.
    *
    * @returns Returns a reference to this DataTensor.
    */
    DataTensor_ & operator-=(const Sample & sample)
    {
        assert(static_cast<Index>(sample.size()) == this->m_shape.d);
        this->m_data.rowwise() -= sample.transpose();
        return *this;
    };

    /**
    * Multiplies the values of all samples in this DataTensor with a scalar factor
    * depending on the feature channel.
    *
    * @param[in] sample Feature vector with the scalar factors to multiply each feature channel with.
    *
    * @returns Returns a reference to the new tensor which forms the product.
    */
    DataTensor_ operator*(const Sample & sample) const
    {
        DataTensor_ prod(*this);
        return prod *= sample;
    };

    /**
    * Multiplies the values of all samples in this DataTensor with a scalar factor
    * depending on the feature channel.
    *
    * @param[in] sample Feature vector with the scalar factors to multiply each feature channel with.
    *
    * @returns Returns a reference to this DataTensor.
    */
    DataTensor_ & operator*=(const Sample & sample)
    {
        assert(sample.size() == this->m_shape.d);
        this->m_data.array().rowwise() *= sample.transpose().array();
        return *this;
    };

    /**
    * Divides the values of all samples in this DataTensor by a scalar factor
    * depending on the feature channel.
    *
    * @param[in] sample Feature vector with the scalar factors to divide each feature channel by.
    *
    * @returns Returns a reference to the new tensor which forms the quotient.
    */
    DataTensor_ operator/(const Sample & sample) const
    {
        DataTensor_ quot(*this);
        return quot /= sample;
    };

    /**
    * Divides the values of all samples in this DataTensor by a scalar factor
    * depending on the feature channel.
    *
    * @param[in] sample Feature vector with the scalar factors to divide each feature channel by.
    *
    * @returns Returns a reference to this DataTensor.
    */
    DataTensor_ & operator/=(const Sample & sample)
    {
        assert(static_cast<Index>(sample.size()) == this->m_shape.d);
        this->m_data.array().rowwise() /= sample.transpose().array();
        return *this;
    };
    
    
    /**
    * Computes the cumulative sum along a given dimension of this data tensor *in-place*.
    *
    * @param[in] dim The dimension which to compute the cumulative sum along.
    */
    void cumsum(unsigned int dim)
    {
        assert(dim < MAXDIV_INDEX_DIMENSION);
        assert(this->m_data_p != NULL);
        
        if (this->m_shape.ind[dim] <= 1)
            return;
        
        if (dim == 0)
        {
            auto tm = this->asTemporalMatrix();
            for (Index t = 1; t < this->length(); ++t)
                tm.row(t) += tm.row(t - 1);
        }
        else if (dim == MAXDIV_INDEX_DIMENSION - 1)
        {
            // Process matrix row by row to leverage row-major storage order.
            // Parallelization did not pay off here.
            Index row, col;
            for (row = 0; row < static_cast<Index>(this->m_data.rows()); ++row)
                for (col = 1; col < this->numAttrib(); ++col)
                    this->m_data(row, col) += this->m_data(row, col - 1);
        }
        else
        {
            auto tm = this->asTemporalMatrix();
            Index blockSize = this->m_shape.prod(dim + 1);
            Index chunkSize = this->m_shape.ind[dim] * blockSize;
            Index rowInd, bs, ps;
            // Again, we could sum blocks, but we process the matrix row-by-row to exploit
            // the row-major storage order.
            #pragma omp parallel for private(rowInd, bs, ps)
            for (rowInd = 0; rowInd < static_cast<Index>(tm.rows()); ++rowInd)
            {
                auto row = tm.row(rowInd);
                for (bs = blockSize, ps = 0; bs < static_cast<Index>(tm.cols()); bs += blockSize, ps += blockSize)
                    if (bs % chunkSize != 0)
                        row.segment(bs, blockSize) += row.segment(ps, blockSize);
            }
        }
    };
    
    /**
    * Computes the cumulative sum along given dimensions of this data tensor *in-place*.
    *
    * @param[in] dimensions Set with the dimensions which to compute the cumulative sum along.
    */
    void cumsum(const std::set<unsigned int> & dimensions)
    {
        for (std::set<unsigned int>::const_iterator dim = dimensions.begin(); dim != dimensions.end(); ++dim)
            this->cumsum(*dim);
    };
    
    /**
    * Computes the cumulative sum along given dimensions of this data tensor *in-place*.
    *
    * @param[in] fromDim The first dimension.
    *
    * @param[in] toDim The last dimension.
    */
    void cumsum(unsigned int fromDim, unsigned int toDim)
    {
        for (unsigned int d = fromDim; d <= toDim; ++d)
            this->cumsum(d);
    };
    
    /**
    * Computes the cumulative sum along all dimensions of this data tensor *in-place*.
    */
    void cumsum()
    {
        this->cumsum(0, MAXDIV_INDEX_DIMENSION - 1);
    };


protected:
    
    ReflessIndexVector m_shape; /**< Size of each dimension of this tensor. */
    
    Index m_size; /**< Size of the allocated array. */
    Index m_numEl; /**< Cached value of `m_shape.prod()`. */
    Index m_numSamples; /**< Cached value of `m_shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)`. */
    
    Scalar * m_data_p; /**< Pointer to the raw data. */
    
    Eigen::Map<ScalarMatrix> m_data; /**< Eigen wrapper around the data. */
    
    bool m_allocated; /**< True if this object has allocated the data storage by itself. */

};


typedef DataTensor_<Scalar> DataTensor; /**< Data tensor using the default scalar type. */
typedef DataTensor::Sample Sample; /**< Feature vector type of a single timestamp at a single location. */
typedef DataTensor::ScalarMatrix ScalarMatrix; /**< A matrix of scalar values. */


#if defined(_OPENMP) and _OPENMP >= 201307
#pragma omp declare reduction( + : Sample : omp_out += omp_in ) initializer( omp_priv = Sample::Zero(omp_orig.size()) )
#endif

}

#endif
