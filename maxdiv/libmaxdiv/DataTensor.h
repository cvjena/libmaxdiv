#ifndef MAXDIV_DATATENSOR_H
#define MAXDIV_DATATENSOR_H

#include <cstddef>
#include <cstring>
#include <cassert>
#include <limits>
#include <set>
#include <unordered_set>
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
    
    typedef DataTensor_<bool> Mask; /**< Mask indicating missing samples. */
    

    /**
    * Constructs an empty tensor with 0 elements.
    */
    DataTensor_()
    : m_shape(), m_size(0), m_numEl(0), m_numSamples(0), m_nonSingletonDim(-1),
      m_data_p(NULL), m_data(NULL, 0, 0), m_allocated(false),
      m_missingValues(), m_missingValuePlaceholder(0), m_cumMissingCounts(NULL), m_dirty(false) {};
    
    /**
    * Constructs a tensor with given the shape.
    *
    * @param[in] shape IndexVector with the size of each dimension.
    */
    DataTensor_(const ReflessIndexVector & shape)
    : m_shape(shape), m_size(shape.prod()), m_numEl(m_size), m_numSamples(shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)), m_nonSingletonDim(-1),
      m_data_p((m_size > 0) ? new Scalar[m_size] : NULL),
      m_data(m_data_p, m_numSamples, shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(m_size > 0), m_missingValues(), m_missingValuePlaceholder(0), m_cumMissingCounts(NULL), m_dirty(false)
    {
        for (int d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
            if (shape.ind[d] > 1)
            {
                if (this->m_nonSingletonDim == -1)
                    this->m_nonSingletonDim = d;
                else
                {
                    this->m_nonSingletonDim = -1;
                    break;
                }
            }
    };
    
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
    : m_shape(shape), m_size(shape.prod()), m_numEl(m_size), m_numSamples(shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)), m_nonSingletonDim(-1),
      m_data_p(data), m_data(data, m_numSamples, shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(false), m_missingValues(), m_missingValuePlaceholder(0), m_cumMissingCounts(NULL), m_dirty(false)
    {
        for (int d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
            if (shape.ind[d] > 1)
            {
                if (this->m_nonSingletonDim == -1)
                    this->m_nonSingletonDim = d;
                else
                {
                    this->m_nonSingletonDim = -1;
                    break;
                }
            }
    };
    
    /**
    * Copies another DataTensor.
    *
    * @param[in] other The tensor to be copied.
    */
    DataTensor_(const DataTensor_ & other)
    : DataTensor_(other.m_shape)
    {
        if (other.m_data_p != NULL && this->m_data_p != NULL)
        {
            std::memcpy(reinterpret_cast<void*>(this->m_data_p), reinterpret_cast<const void*>(other.m_data_p), sizeof(Scalar) * this->m_size);
            this->m_missingValues = other.m_missingValues;
            this->m_dirty = other.m_dirty;
        }
        this->m_missingValuePlaceholder = other.m_missingValuePlaceholder;
        if (other.m_cumMissingCounts)
            this->m_cumMissingCounts = new DataTensor_<Index>(*(other.m_cumMissingCounts));
    };
    
    /**
    * Moves the data of another DataTensor to this one and leaves the other
    * one empty.
    *
    * @param[in] other The tensor whose data is to be moved.
    */
    DataTensor_(DataTensor_ && other)
    : m_shape(other.m_shape), m_size(other.m_size), m_numEl(other.m_numEl), m_numSamples(other.m_numSamples), m_nonSingletonDim(other.m_nonSingletonDim),
      m_data_p(other.m_data_p), m_data(m_data_p, m_numSamples, m_shape.ind[MAXDIV_INDEX_DIMENSION - 1]),
      m_allocated(other.m_allocated), m_missingValues(std::move(other.m_missingValues)),
      m_missingValuePlaceholder(other.m_missingValuePlaceholder),
      m_cumMissingCounts(other.m_cumMissingCounts), m_dirty(other.m_dirty)
    {
        other.m_shape = ReflessIndexVector();
        other.m_size = 0;
        other.m_nonSingletonDim = -1;
        other.m_data_p = NULL;
        other.m_allocated = false;
        other.m_cumMissingCounts = NULL;
        other.m_dirty = false;
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
            
            this->m_missingValues = other.m_missingValues;
            if (other.m_cumMissingCounts != NULL)
                this->m_cumMissingCounts = new DataTensor_<Index>(*(other.m_cumMissingCounts));
            this->m_dirty = other.m_dirty;
        }
        this->m_missingValuePlaceholder = other.m_missingValuePlaceholder;
    };
    
    virtual ~DataTensor_()
    {
        if (this->m_allocated)
            delete[] this->m_data_p;
        if (this->m_cumMissingCounts != NULL)
            delete this->m_cumMissingCounts;
    };
    
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
        this->m_missingValues = other.m_missingValues;
        this->m_missingValuePlaceholder = other.m_missingValuePlaceholder;
        if (other.m_cumMissingCounts != NULL)
        {
            if (this->m_cumMissingCounts != NULL)
                *(this->m_cumMissingCounts) = *(other.m_cumMissingCounts);
            else
                this->m_cumMissingCounts = new DataTensor_<Index>(*(other.m_cumMissingCounts));
        }
        else if (this->m_cumMissingCounts != NULL)
        {
            delete this->m_cumMissingCounts;
            this->m_cumMissingCounts = NULL;
        }
        this->m_dirty = other.m_dirty;
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
        if (this->m_cumMissingCounts != NULL)
            delete this->m_cumMissingCounts;
        
        this->m_shape = other.m_shape;
        this->m_size = other.m_size;
        this->m_numEl = other.m_numEl;
        this->m_numSamples = other.m_numSamples;
        this->m_nonSingletonDim = other.m_nonSingletonDim;
        this->m_data_p = other.m_data_p;
        this->m_allocated = other.m_allocated;
        this->m_missingValues = std::move(other.m_missingValues);
        this->m_missingValuePlaceholder = other.m_missingValuePlaceholder;
        this->m_cumMissingCounts = other.m_cumMissingCounts;
        this->m_dirty = other.m_dirty;
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
        
        other.m_shape = ReflessIndexVector();
        other.m_size = 0;
        other.m_nonSingletonDim = -1;
        other.m_data_p = NULL;
        other.m_allocated = false;
        other.m_cumMissingCounts = NULL;
        other.m_dirty = false;
        
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
            this->m_missingValues = other.m_missingValues;
            if (other.m_cumMissingCounts != NULL)
            {
                if (this->m_cumMissingCounts != NULL)
                    *(this->m_cumMissingCounts) = *(other.m_cumMissingCounts);
                else
                    this->m_cumMissingCounts = new DataTensor_<Index>(*(other.m_cumMissingCounts));
            }
            else if (this->m_cumMissingCounts != NULL)
            {
                delete this->m_cumMissingCounts;
                this->m_cumMissingCounts = NULL;
            }
            this->m_dirty = other.m_dirty;
        }
        this->m_missingValuePlaceholder = other.m_missingValuePlaceholder;
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
    * @return Returns the index of the non-singleton dimension if there is only one, otherwise -1.
    */
    int nonSingletonDim() const { return this->m_nonSingletonDim; };
    
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
    * @return Returns the number of samples without missing values in this tensor.
    * `mask()` has to be called initially to initialize missing samples.
    */
    Index numValidSamples() const { return this->m_numSamples - this->m_missingValues.size(); };
    
    /**
    * @return Returns the number of samples with missing values in this tensor.
    * `mask()` has to be called initially to initialize missing samples.
    */
    Index numMissingSamples() const { return this->m_missingValues.size(); };
    
    /**
    * Counts the samples with missing values in a given range.
    *
    * `mask()` has to be called initially to initialize missing samples.
    *
    * @param[in] range The sub-block of this tensor.
    * 
    * @return Returns the number of samples with missing values in the given range.
    */
    Index numMissingSamplesInRange(const IndexRange & range) const
    {
        if (this->hasMissingSamples())
        {
            #pragma omp critical
            {
                if (this->m_cumMissingCounts == NULL)
                    this->m_cumMissingCounts = new DataTensor_<Index>();
                if (this->m_cumMissingCounts->empty())
                {
                    this->getMask(*(this->m_cumMissingCounts));
                    this->m_cumMissingCounts->cumsum();
                }
            }
            return this->m_cumMissingCounts->sumFromCumsum(range, 0);
        }
        else
            return 0;
    };
    
    /**
    * @return Returns `true` if there are samples with missing values in this tensor, otherwise `false`.
    * `mask()` has to be called initially to initialize missing samples.
    */
    bool hasMissingSamples() const { return !this->m_missingValues.empty(); };
    
    /**
    * @return Returns `true` if any element in this tensor is `NaN`. Note that this won't be the case after
    * a call to `mask()`, even if there are missing samples.
    */
    bool hasMissingValues() const { return (!this->empty() && this->m_data.cwiseNotEqual(this->m_data).any()); };
    
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
            this->m_missingValues.clear();
            this->m_dirty = false;
        }
        
        this->m_shape = shape;
        this->m_numEl = numEl;
        this->m_numSamples = shape.prod(0, MAXDIV_INDEX_DIMENSION - 2);
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
        
        this->m_nonSingletonDim = -1;
        for (int d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
            if (shape.ind[d] > 1)
            {
                if (this->m_nonSingletonDim == -1)
                    this->m_nonSingletonDim = d;
                else
                {
                    this->m_nonSingletonDim = -1;
                    break;
                }
            }
        
        if (this->empty())
            this->m_missingValues.clear();
        else
        {
            for (std::unordered_set<Index>::iterator missingIt = this->m_missingValues.begin(); missingIt != this->m_missingValues.end();)
                if (*missingIt > this->numSamples())
                    missingIt = this->m_missingValues.erase(missingIt);
                else
                    ++missingIt;
        }
        if (this->m_cumMissingCounts != NULL)
            this->m_cumMissingCounts->resize(ReflessIndexVector());
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
        this->m_nonSingletonDim = -1;
        this->m_shape.vec().setZero();
        this->m_missingValues.clear();
        if (this->m_cumMissingCounts != NULL)
        {
            delete this->m_cumMissingCounts;
            this->m_cumMissingCounts = NULL;
        }
        this->m_dirty = false;
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
        
        // Shortcut for cutting off some trailing time steps and leaving everything else unchanged
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
            this->m_missingValues.clear();
            return;
        }
        
        // Adjust indices of missing samples
        if (!this->m_missingValues.empty())
        {
            std::unordered_set<Index> missingValues;
            IndexVector ind;
            for (std::unordered_set<Index>::const_iterator missingIt = this->m_missingValues.begin(); missingIt != this->m_missingValues.end(); ++missingIt)
            {
                ind = IndexVector(this->m_shape, *missingIt);
                if (range.contains(ind))
                {
                    ind -= range.a;
                    ind.shape = newShape;
                    missingValues.insert(ind.linear());
                }
            }
            this->m_missingValues = std::move(missingValues);
        }
        if (this->m_cumMissingCounts != NULL)
            this->m_cumMissingCounts->resize(ReflessIndexVector());
        
        // Move data from sub-block to new positions
        Scalar * data = this->m_data_p;
        IndexVector ind(newShape, 0);
        for (Index i = 0; i < newNumEl; ++i, ++ind, ++data)
            *data = (*this)(range.a + ind);
        this->resize(newShape);
    };
    
    /**
    * Sets all elements in the tensor to a constant value and removes any missing sample mask.
    *
    * @param[in] val The new value for all elements of the tensor.
    */
    void setConstant(const Scalar val)
    {
        this->m_data.setConstant(val);
        this->m_missingValues.clear();
        if (this->m_cumMissingCounts != NULL)
        {
            delete this->m_cumMissingCounts;
            this->m_cumMissingCounts = NULL;
        }
        this->m_dirty = false;
    };
    
    /**
    * Sets all data points in the tensor to a constant vector and removes any missing sample mask.
    *
    * @param[in] value The new value for all spatio-temporal samples.
    * This vector must have the same number of elements as specified in `shape.d`.
    */
    void setConstant(const Sample & value)
    {
        assert(value.size() == this->numAttrib());
        this->m_data.rowwise() = value.transpose();
        this->m_missingValues.clear();
        if (this->m_cumMissingCounts != NULL)
        {
            delete this->m_cumMissingCounts;
            this->m_cumMissingCounts = NULL;
        }
        this->m_dirty = false;
    };
    
    /**
    * Sets all elements in the tensor to 0 and removes any missing sample mask.
    */
    void setZero() { this->setConstant(static_cast<Scalar>(0)); };
        
    /**
    * @return Returns a pointer to the raw data storage of this tensor.
    * The returned pointer may be NULL if the tensor is empty.
    */
    Scalar * raw()
    {
        this->setMissingValues();
        this->m_dirty = true;
        return this->m_data_p;
    };
    
    /**
    * @return Returns a const pointer to the raw data storage of this tensor.
    * The returned pointer may be NULL if the tensor is empty.
    */
    const Scalar * raw() const
    {
        this->setMissingValues();
        return this->m_data_p;
    };
    
    /**
    * Returns an Eigen::Map object wrapping the raw data of this tensor, where the samples
    * are stored in a contiguous way, i.e. the returned object will have as many rows as
    * there are data points and as many columns as each data points has attributes.
    */
    Eigen::Map<ScalarMatrix> data()
    {
        this->setMissingValues();
        this->m_dirty = true;
        return this->m_data;
    };
    
    /**
    * Returns a constant Eigen::Map object wrapping the raw data of this tensor, where the
    * samples are stored in a contiguous way, i.e. the returned object will have as many rows
    * as there are data points and as many columns as each data points has attributes.
    */
    Eigen::Map<const ScalarMatrix> data() const
    {
        this->setMissingValues();
        return Eigen::Map<const ScalarMatrix>(this->m_data_p, this->numSamples(), this->numAttrib());
    };
    
    /**
    * Provides a linear view on this tensor as by concatenating all elements.
    *
    * @return Returns a Eigen::Map object with `numEl()` rows and 1 column.
    */
    Eigen::Map<Sample> asVector()
    {
        this->setMissingValues();
        this->m_dirty = true;
        return Eigen::Map<Sample>(this->m_data_p, this->numEl(), 1);
    };
    
    /**
    * Provides a linear view on this tensor as by concatenating all elements.
    *
    * @return Returns a constant Eigen::Map object with `numEl()` rows and 1 column.
    */
    Eigen::Map<const Sample> asVector() const
    {
        this->setMissingValues();
        return Eigen::Map<const Sample>(this->m_data_p, this->numEl(), 1);
    };
    
    /**
    * Provides a view on this tensor as as a matrix where each row is the concatenation of all samples
    * at a single time step. Thus, the matrix has `width * height * depth * numAttrib` columns.
    *
    * @return Eigen::Map object
    */
    Eigen::Map<ScalarMatrix> asTemporalMatrix()
    {
        this->setMissingValues();
        this->m_dirty = true;
        return Eigen::Map<ScalarMatrix>(this->m_data_p, this->length(), this->m_shape.prod(1));
    };
    
    /**
    * Provides a view on this tensor as as a matrix where each row is the concatenation of all samples
    * at a single time step. Thus, the matrix has `width * height * depth * numAttrib` columns.
    *
    * @return constant Eigen::Map object
    */
    Eigen::Map<const ScalarMatrix> asTemporalMatrix() const
    {
        this->setMissingValues();
        return Eigen::Map<const ScalarMatrix>(this->m_data_p, this->length(), this->m_shape.prod(1));
    };
    
    /**
    * Returns an `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    */
    Eigen::Map<Sample> operator()(Index t, Index x = 0, Index y = 0, Index z = 0)
    {
        assert(this->m_data_p != NULL);
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        if (this->isMissingSample(t, x, y, z))
        {
            this->setMissingValues();
            this->m_dirty = true;
        }
        return Eigen::Map<Sample>(this->m_data_p + this->makeIndexVector(t, x, y, z).linear(), this->m_shape.d);
    };
    
    /**
    * Returns a constant `Eigen::Map<Sample>` object wrapping a single sample in this tensor.
    */
    Eigen::Map<const Sample> operator()(Index t, Index x = 0, Index y = 0, Index z = 0) const
    {
        assert(this->m_data_p != NULL);
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        if (this->isMissingSample(t, x, y, z))
            this->setMissingValues();
        return Eigen::Map<const Sample>(this->m_data_p + this->makeIndexVector(t, x, y, z).linear(), this->m_shape.d);
    };
    
    /**
    * @return Returns a reference to the element at the given @p index in this tensor.
    */
    Scalar & operator()(const ReflessIndexVector & index)
    {
        assert(this->m_data_p != NULL);
        assert((index.vec() < this->m_shape.vec()).all());
        if (this->isMissingSample(index))
        {
            this->setMissingValues();
            this->m_dirty = true;
        }
        return *(this->m_data_p + IndexVector(this->m_shape, index).linear());
    };
    
    /**
    * @return Returns a const reference to the element at the given @p index in this tensor.
    */
    const Scalar & operator()(const ReflessIndexVector & index) const
    {
        assert(this->m_data_p != NULL);
        assert((index.vec() < this->m_shape.vec()).all());
        if (this->isMissingSample(index))
            this->setMissingValues();
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
        if (this->isMissingSample(s))
        {
            this->setMissingValues();
            this->m_dirty = true;
        }
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
        if (this->isMissingSample(s))
            this->setMissingValues();
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
        this->setMissingValues();
        this->m_dirty = true;
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
        this->setMissingValues();
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
        this->setMissingValues();
        this->m_dirty = true;
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
        this->setMissingValues();
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
        this->m_dirty = true;
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
        this->m_dirty = true;
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
        this->m_dirty = true;
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
        this->m_dirty = true;
        return *this;
    };
    
    
    /**
    * Marks all samples whose attributes contain a `NaN` as *missing sample* and sets
    * *all* attributes of those samples to a place-holder value, which defaults to zero, but
    * can be changed using `missingValuePlaceholder()`.  
    * That place-holder will be restored each time one of the data access functions is called
    * after a non-const version of them has been called before, so that the internal data may
    * have been changed.
    *
    * If such a mask does already exist, this function will only add new missing samples,
    * but does not remove old ones if their values have been changed. Use `setMissingSample()`
    * if you want to assign valid values to a missing sample.
    * 
    * @return Returns the number of missing samples added to the mask.
    */
    Index mask()
    {
        Index numAdded = 0;
        if (this->hasMissingValues())
        {
            for (Index s = 0; s < this->numSamples(); ++s)
                // NaNs are never equal to themselves
                if (this->m_data.row(s).cwiseNotEqual(this->m_data.row(s)).any() && this->m_missingValues.insert(s).second)
                    ++numAdded;
            if (this->m_cumMissingCounts != NULL)
                this->m_cumMissingCounts->resize(ReflessIndexVector());
            this->m_dirty = true;
            this->setMissingValues();
        }
        return numAdded;
    };
    
    /**
    * Marks all samples whose attributes contain a given value as *missing sample* and sets
    * *all* attributes of those samples to a place-holder value, which defaults to zero, but
    * can be changed using `missingValuePlaceholder()`.  
    * That place-holder will be restored each time one of the data access functions is called
    * after a non-const version of them has been called before, so that the internal data may
    * have been changed.
    *
    * If such a mask does already exist, this function will only add new missing samples,
    * but does not remove old ones if their values have been changed. Use `setMissingSample()`
    * if you want to assign valid values to a missing sample.
    *
    * @param[in] missingValue A scalar value that is used to indicate a *missing value* in the data.
    * 
    * @return Returns the number of missing samples added to the mask.
    */
    Index mask(Scalar missingValue)
    {
        Index numAdded = 0;
        for (Index s = 0; s < this->numSamples(); ++s)
            if (this->m_data.row(s).cwiseEqual(missingValue).any() && this->m_missingValues.insert(s).second)
                ++numAdded;
        if (numAdded > 0)
        {
            if (this->m_cumMissingCounts != NULL)
                this->m_cumMissingCounts->resize(ReflessIndexVector());
            this->m_dirty = true;
            this->setMissingValues();
        }
        return numAdded;
    };
    
    /**
    * Copies a mask indicating missing samples from another tensor.
    *
    * @param[in] other The tensor whose missing sample mask should be copied. All dimensions of that tensor,
    * except the attribute dimension, must have the same size as the dimensions of this tensor.
    *
    * @param[in] add Specifies how to deal with an existing mask of this tensor. If set to `true`, both masks
    * will be merged. If set to `false`, `unmask()` will be called before copying the new mask.
    */
    template<typename OtherScalar>
    void copyMask(const DataTensor_<OtherScalar> & other, bool add = false)
    {
        assert((this->shape().vec().head(MAXDIV_INDEX_DIMENSION - 1) == other.shape().vec().head(MAXDIV_INDEX_DIMENSION - 1)).all());
        if (add)
            this->m_missingValues.insert(other.getMissingSampleIndices().begin(), other.getMissingSampleIndices().end());
        else
        {
            this->unmask();
            this->m_missingValues = other.getMissingSampleIndices();
        }
        if (this->m_cumMissingCounts != NULL)
            this->m_cumMissingCounts->resize(ReflessIndexVector());
        if (!this->m_missingValues.empty())
        {
            this->m_dirty = true;
            this->setMissingValues();
        }
    };
    
    /**
    * Sets the attributes of all missing samples to `NaN` and removes the mask indicating missing
    * samples created by `mask()` from this tensor.
    */
    void unmask()
    {
        this->setMissingValues(std::numeric_limits<Scalar>::quiet_NaN());
        this->removeMask();
    };
    
    /**
    * Sets the attributes of all missing samples to a given value and removes the mask indicating
    * missing samples created by `mask()` from this tensor.
    *
    * @param[in] missingValue The value to be assigned to the attributes of missing samples.
    */
    void unmask(Scalar missingValue)
    {
        this->setMissingValues(missingValue);
        this->removeMask();
    };
    
    /**
    * Removes the mask indicating missing samples from this tensor without changing the attributes
    * of those samples.
    */
    void removeMask()
    {
        this->setMissingValues();
        this->m_missingValues.clear();
        if (this->m_cumMissingCounts != NULL)
        {
            delete this->m_cumMissingCounts;
            this->m_cumMissingCounts = NULL;
        }
        this->m_dirty = false;
    };
    
    /**
    * @return Returns the place-holder value used for all attributes of missing samples indicated by
    * the mask created by `mask()`.
    */
    Scalar missingValuePlaceholder() const { return this->m_missingValuePlaceholder; };
    
    /**
    * Changes the place-holder value used for all attributes of missing samples indicated by the mask
    * created by `mask()`.
    *
    * This function will not remove the mask. Use `setMissingSample()` if you want to assign valid attributes
    * to a currently missing sample or `unmask()` if you want to remove the mask completely.
    *
    * @param[in] value The new place-holder value.
    */
    void missingValuePlaceholder(Scalar value)
    {
        if (this->m_missingValuePlaceholder != value)
        {
            this->m_missingValuePlaceholder = value;
            this->m_dirty = true;
            this->setMissingValues();
        }
    };
    
    /**
    * Restores the place-holder values of all missing samples indicated by the mask created by `mask()`.
    *
    * This is usually called automatically whenever a data view is retrieved if a non-const view has been
    * retrieved before. However, you might want to call this function manually if you perform multiple operations
    * with a single, stored view.
    */
    void setMissingValues() const
    {
        if (this->m_dirty)
        {
            DataTensor_<Scalar> * me = const_cast<DataTensor_<Scalar>*>(this);
            if (!this->m_missingValues.empty())
            {
                assert(this->m_data_p != NULL);
                for (std::unordered_set<Index>::const_iterator missingIt = this->m_missingValues.begin(); missingIt != this->m_missingValues.end(); ++missingIt)
                {
                    assert(*missingIt < this->numSamples());
                    me->m_data.row(*missingIt).setConstant(this->m_missingValuePlaceholder);
                }
            }
            me->m_dirty = false;
        }
    };
    
    /**
    * Sets the attributes of all missing samples indicated by the mask created by `mask()` to a given
    * value. This function will neither remove the mask nor change the default place-holder value.
    * Use `setMissingSample()` if you want to assign valid attributes to a currently missing sample or
    * `missingValuePlaceholder()` if you want to change the default place-holder value.
    *
    * @param[in] missingValue The value to be assigned to the attributes of missing samples.
    */
    void setMissingValues(Scalar missingValue)
    {
        for (std::unordered_set<Index>::const_iterator missingIt = this->m_missingValues.begin(); missingIt != this->m_missingValues.end(); ++missingIt)
        {
            assert(*missingIt < this->numSamples());
            this->m_data.row(*missingIt).setConstant(missingValue);
        }
    };
    
    /**
    * Marks a specific sample as *missing sample* and replaces the values of its attributes with zero.
    * 
    * @param[in] s The linear index of the sample (`0 <= s < length * width * height * depth`).
    */
    void setMissingSample(Index s)
    {
        assert(s >= 0 && s < this->numSamples());
        if (this->m_missingValues.insert(s).second)
        {
            this->m_data.row(s).setZero();
            if (this->m_cumMissingCounts != NULL)
                this->m_cumMissingCounts->resize(ReflessIndexVector());
        }
    };
    
    /**
    * Marks a specific sample as *missing sample* and replaces the values of its attributes with zero.
    * 
    * @param[in] t The index of the sample along the temporal axis.
    *
    * @param[in] x The index of the sample along the first spatial axis.
    *
    * @param[in] y The index of the sample along the second spatial axis.
    *
    * @param[in] z The index of the sample along the third spatial axis.
    */
    void setMissingSample(Index t, Index x, Index y, Index z)
    {
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        IndexVector ind(this->m_shape, t, x, y, z, 0);
        ind.shape.d = 1;
        this->setMissingSample(ind.linear());
    };
    
    /**
    * Marks a specific sample as *missing sample* and replaces the values of its attributes with zero.
    * 
    * @param[in] index The index vector with the coordinates of the sample.
    * The attribute dimension will be ignored.
    */
    void setMissingSample(const ReflessIndexVector & index)
    {
        IndexVector ind(this->m_shape, index);
        ind.shape.d = 1;
        ind.d = 0;
        this->setMissingSample(ind.linear());
    };
    
    /**
    * Assigns values to the attributes of a *missing sample* and removes it from the mask.
    * 
    * @param[in] s The linear index of the sample (`0 <= s < length * width * height * depth`).
    *
    * @param[in] values The new values for the sample's attributes.
    */
    void setMissingSample(Index s, const Sample & values)
    {
        assert(s >= 0 && s < this->numSamples());
        this->m_missingValues.erase(s);
        this->m_data.row(s) = values.transpose();
        if (this->m_cumMissingCounts != NULL)
            this->m_cumMissingCounts->resize(ReflessIndexVector());
    };
    
    /**
    * Assigns values to the attributes of a *missing sample* and removes it from the mask.
    * 
    * @param[in] t The index of the sample along the temporal axis.
    *
    * @param[in] x The index of the sample along the first spatial axis.
    *
    * @param[in] y The index of the sample along the second spatial axis.
    *
    * @param[in] z The index of the sample along the third spatial axis.
    *
    * @param[in] values The new values for the sample's attributes.
    */
    void setMissingSample(Index t, Index x, Index y, Index z, const Sample & values)
    {
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        IndexVector ind(this->m_shape, t, x, y, z, 0);
        ind.shape.d = 1;
        this->setMissingSample(ind.linear(), values);
    };
    
    /**
    * Assigns values to the attributes of a *missing sample* and removes it from the mask.
    * 
    * @param[in] index The index vector with the coordinates of the sample.
    * The attribute dimension will be ignored.
    *
    * @param[in] values The new values for the sample's attributes.
    */
    void setMissingSample(const ReflessIndexVector & index, const Sample & values)
    {
        IndexVector ind(this->m_shape, index);
        ind.shape.d = 1;
        ind.d = 0;
        this->setMissingSample(ind.linear(), values);
    };
    
    /**
    * Checks if a sample given by a linear index is masked due to missing values.
    *
    * @param[in] s The linear index of the sample (`0 <= s < length * width * height * depth`).
    *
    * @return Returns `true` if the given sample has missing values.
    *
    * @note The mask has to be initialized by calling `mask()`.
    */
    bool isMissingSample(Index s) const
    {
        assert(s >= 0 && s < this->numSamples());
        return (!this->m_missingValues.empty() && this->m_missingValues.find(s) != this->m_missingValues.end());
    };
    
    /**
    * Checks if the sample at a given position is masked due to missing values.
    *
    * @param[in] t The index of the sample along the temporal axis.
    *
    * @param[in] x The index of the sample along the first spatial axis.
    *
    * @param[in] y The index of the sample along the second spatial axis.
    *
    * @param[in] z The index of the sample along the third spatial axis.
    *
    * @return Returns `true` if the given sample has missing values.
    *
    * @note The mask has to be initialized by calling `mask()`.
    */
    bool isMissingSample(Index t, Index x, Index y, Index z) const
    {
        assert(t >= 0 && x >= 0 && y >= 0 && z >= 0 && t < this->m_shape.t && x < this->m_shape.x && y < this->m_shape.y && z < this->m_shape.z);
        if (this->m_missingValues.empty())
            return false;
        
        IndexVector ind(this->m_shape, t, x, y, z, 0);
        ind.shape.d = 1;
        return this->isMissingSample(ind.linear());
    };
    
    /**
    * Checks if a given sample is masked due to missing values.
    *
    * @param[in] index The index vector with the coordinates of the sample.
    * The attribute dimension will be ignored.
    *
    * @return Returns `true` if the given sample has missing values.
    *
    * @note The mask has to be initialized by calling `mask()`.
    */
    bool isMissingSample(const ReflessIndexVector & index) const
    {
        if (this->m_missingValues.empty())
            return false;
        
        IndexVector ind(this->m_shape, index);
        ind.shape.d = 1;
        ind.d = 0;
        return this->isMissingSample(ind.linear());
    };
    
    /**
    * @return Returns a const reference to the list of indices of missing samples.
    */
    const std::unordered_set<Index> & getMissingSampleIndices() const { return this->m_missingValues; };
    
    /**
    * Retrieves a mask for this tensor indicating whether a sample is a *missing sample* or not.
    *
    * @return Returns a boolean tensor with 1 attribute dimension which has the value `true` if the
    * sample at the respective position is missing, and `false` if it is valid.
    */
    Mask getMask() const
    {
        Mask mask;
        this->getMask(mask);
        return mask;
    }
    
    /**
    * Retrieves a mask for this tensor indicating whether a sample is a *missing sample* or not.
    *
    * @param[out] mask A tensor to be filled with indicators which are `true` if the sample at the
    * respective position is missing, and `false` if it is valid. The tensor will be resized to have
    * the same shape as this tensor, but with only 1 attribute dimension.
    *
    * @return Returns a reference to `mask`.
    */
    template<typename IndicatorType>
    DataTensor_<IndicatorType> & getMask(DataTensor_<IndicatorType> & mask) const
    {
        ReflessIndexVector maskShape = this->m_shape;
        maskShape.d = 1;
        mask.resize(maskShape);
        mask.setZero();
        for (std::unordered_set<Index>::const_iterator missingIt = this->m_missingValues.begin(); missingIt != this->m_missingValues.end(); ++missingIt)
            mask.sample(*missingIt)(0) = static_cast<IndicatorType>(1);
        return mask;
    }
    
    /**
    * Checks if a given sub-block of this tensor has a border which consists of missing samples only and
    * could, thus, be reduced to a smaller block without any loss of information.
    *
    * @param[in] range The range to be checked for reducability.
    *
    * @return Returns true if any border of `range` consists of missing samples only.
    */
    bool isRangeReducable(const IndexRange & range) const
    {
        // Quick return if there aren't any missing samples at all
        if (this->m_missingValues.empty())
            return false;
        
        // Shortcut for the simple case of 1d data
        if (this->m_nonSingletonDim >= 0)
            return (this->isMissingSample(range.a.ind[this->m_nonSingletonDim]) || this->isMissingSample(range.b.ind[this->m_nonSingletonDim] - 1));
        
        // Check if there are any missing samples in that range at all
        Index numMissing = this->numMissingSamplesInRange(range);
        if (numMissing == 0)
            return false;
        
        // Check for empty borders
        IndexRange border;
        Index borderSize;
        for (unsigned int d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
            if (range.b.ind[d] > range.a.ind[d] + 1)
            {
                border = range;
                border.b.ind[d] = border.a.ind[d] + 1;
                border.a.d = 0;
                border.b.d = 1;
                borderSize = border.volume();
                if (borderSize <= numMissing)
                {
                    if (this->numMissingSamplesInRange(border) == borderSize)
                        return true;
                    border.b.ind[d] = range.b.ind[d];
                    border.a.ind[d] = border.b.ind[d] - 1;
                    if (this->numMissingSamplesInRange(border) == borderSize)
                        return true;
                }
            }
        
        return false;
    };
    
    
    /**
    * Computes the cumulative sum along a given dimension of this data tensor *in-place*.
    *
    * If there are missing samples, their attribute values will be set to 0 before computing the
    * cumulative sums and the mask will be removed.
    *
    * @param[in] dim The dimension which to compute the cumulative sum along.
    */
    void cumsum(unsigned int dim)
    {
        assert(dim < MAXDIV_INDEX_DIMENSION);
        
        if (this->m_shape.ind[dim] <= 1 || this->empty())
            return;
        
        this->unmask(static_cast<Scalar>(0));
        
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
    * If there are missing samples, their attribute values will be set to 0 before computing the
    * cumulative sums and the mask will be removed.
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
    * If there are missing samples, their attribute values will be set to 0 before computing the
    * cumulative sums and the mask will be removed.
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
    *
    * If there are missing samples, their attribute values will be set to 0 before computing the
    * cumulative sums and the mask will be removed.
    */
    void cumsum()
    {
        this->cumsum(0, MAXDIV_INDEX_DIMENSION - 1);
    };
    
    
    /**
    * Assuming that this tensor contains cumulative sums over all dimensions of the data except the attribute dimension,
    * this function restores the sum of a specific attribute of the samples in a given sub-block of the data in constant time.
    *
    * @param[in] range The sub-block to compute the sum of. The attribute dimension will be ignored.
    *
    * @param[in] d The index of the attribute dimension to sum over.
    *
    * @return Sum over the given attribute over all elements in the given sub-block.
    */
    Scalar sumFromCumsum(const IndexRange & range, Index d) const
    {
        assert(!range.empty());
        assert(d >= 0 && d < this->numAttrib());
        
        if (this->m_nonSingletonDim >= 0)
        {
            // Shortcut for data with only one non-singleton dimension
            Scalar sum = this->m_data(range.b.ind[this->m_nonSingletonDim] - 1, d);
            if (range.a.ind[this->m_nonSingletonDim] > 0)
                sum -= this->m_data(range.a.ind[this->m_nonSingletonDim] - 1, d);
            return sum;
        }
        else
        {
            // Extracting the sum of a block from a tensor of cumulative sums follows the Inclusion-Exclusion Principle.
            // For example, for two dimensions we have:
            // sum([a1,b1), [a2,b2)) = cumsum(b1, b2) - cumsum(a1 - 1, b2) - cumsum(b1, a2 - 2) + cumsum(a1 - 1, a2 - 1)
            Scalar sum = 0;
            ReflessIndexVector ind;
            ind.d = d;
            unsigned int i, s, numSummands = 1 << (MAXDIV_INDEX_DIMENSION - 1);
            bool isZeroBlock;
            Eigen::Array<bool, MAXDIV_INDEX_DIMENSION - 1, 1> state; // Switches between first and last point of each range
            state.setConstant(false);
            for (s = 0; s < numSummands; ++s)
            {
                
                // Determine index of the bottom right corner of the current block
                for (i = 0, isZeroBlock = false; i < MAXDIV_INDEX_DIMENSION - 1 && !isZeroBlock; ++i)
                {
                    ind.ind[i] = (state(i)) ? range.a.ind[i] : range.b.ind[i];
                    if (ind.ind[i] == 0)
                        isZeroBlock = true;
                    else
                        ind.ind[i] -= 1;
                }
                
                // Add or subtract value of the block
                if (!isZeroBlock)
                {
                    if (state.count() % 2 == 0)
                        sum += (*this)(ind);
                    else
                        sum -= (*this)(ind);
                }
                
                // Move on to next block
                for (i = 0; state(i) && i < MAXDIV_INDEX_DIMENSION - 2; ++i)
                    state(i) = false;
                state(i) = true;
                
            }
            return sum;
        }
    }

    /**
    * Assuming that this tensor contains cumulative sums over all dimensions of the data except the attribute dimension,
    * this function restores the sum of the samples in a given sub-block of the data in constant time.
    *
    * @param[in] range The sub-block to compute the sum of. The attribute dimension will be ignored.
    *
    * @return Sum over all samples in the given sub-block.
    */
    Sample sumFromCumsum(const IndexRange & range) const
    {
        assert(!range.empty());
        
        if (this->m_nonSingletonDim >= 0)
        {
            // Shortcut for data with only one non-singleton dimension
            Sample sum = this->sample(range.b.ind[this->m_nonSingletonDim] - 1);
            if (range.a.ind[this->m_nonSingletonDim] > 0)
                sum -= this->sample(range.a.ind[this->m_nonSingletonDim] - 1);
            return sum;
        }
        else
        {
            // Extracting the sum of a block from a tensor of cumulative sums follows the Inclusion-Exclusion Principle.
            // For example, for two dimensions we have:
            // sum([a1,b1), [a2,b2)) = cumsum(b1, b2) - cumsum(a1 - 1, b2) - cumsum(b1, a2 - 2) + cumsum(a1 - 1, a2 - 1)
            Sample sum = Sample::Zero(this->numAttrib());
            IndexVector ind = this->makeIndexVector();
            ind.shape.d = 1;
            unsigned int i, s, numSummands = 1 << (MAXDIV_INDEX_DIMENSION - 1);
            bool isZeroBlock;
            Eigen::Array<bool, MAXDIV_INDEX_DIMENSION - 1, 1> state; // Switches between first and last point of each range
            state.setConstant(false);
            for (s = 0; s < numSummands; ++s)
            {
                
                // Determine index of the bottom right corner of the current block
                for (i = 0, isZeroBlock = false; i < MAXDIV_INDEX_DIMENSION - 1 && !isZeroBlock; ++i)
                {
                    ind.ind[i] = (state(i)) ? range.a.ind[i] : range.b.ind[i];
                    if (ind.ind[i] == 0)
                        isZeroBlock = true;
                    else
                        ind.ind[i] -= 1;
                }
                
                // Add or subtract value of the block
                if (!isZeroBlock)
                {
                    if (state.count() % 2 == 0)
                        sum += this->sample(ind);
                    else
                        sum -= this->sample(ind);
                }
                
                // Move on to next block
                for (i = 0; state(i) && i < MAXDIV_INDEX_DIMENSION - 2; ++i)
                    state(i) = false;
                state(i) = true;
                
            }
            return sum;
        }
    }


protected:
    
    ReflessIndexVector m_shape; /**< Size of each dimension of this tensor. */
    
    Index m_size; /**< Size of the allocated array. */
    Index m_numEl; /**< Cached value of `m_shape.prod()`. */
    Index m_numSamples; /**< Cached value of `m_shape.prod(0, MAXDIV_INDEX_DIMENSION - 2)`. */
    int m_nonSingletonDim; /**< Index of the non-singleton dimension if there is only one in the data (except the attribute dimension), otherwise -1. */
    
    Scalar * m_data_p; /**< Pointer to the raw data. */
    
    Eigen::Map<ScalarMatrix> m_data; /**< Eigen wrapper around the data. */
    
    bool m_allocated; /**< True if this object has allocated the data storage by itself. */
    
    std::unordered_set<Index> m_missingValues; /**< Set of linear indices of samples with missing values. */
    Scalar m_missingValuePlaceholder; /**< Placeholder value assigned to all attributes of missing samples. */
    mutable DataTensor_<Index> * m_cumMissingCounts; /**< Cumulative counts of missing values (used by `numMissingSamplesInRange()`). */
    mutable bool m_dirty; /**< True if a non-const access to the data has happened, so that missing values may have been changed. */

};


typedef DataTensor_<Scalar> DataTensor; /**< Data tensor using the default scalar type. */
typedef DataTensor::Sample Sample; /**< Feature vector type of a single timestamp at a single location. */
typedef DataTensor::ScalarMatrix ScalarMatrix; /**< A matrix of scalar values. */


#if defined(_OPENMP) and _OPENMP >= 201307
#pragma omp declare reduction( + : Sample : omp_out += omp_in ) initializer( omp_priv = Sample::Zero(omp_orig.size()) )
#endif

}

#endif
