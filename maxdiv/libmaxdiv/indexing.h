#ifndef MAXDIV_INDEXING_H
#define MAXDIV_INDEXING_H

#include <cstddef>
#include <Eigen/Core>
#include "config.h"

namespace MaxDiv
{

/**
* @brief Simple struct with indices for DataTensor objects.
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct ReflessIndexVector
{

    typedef std::size_t Index;
    typedef Eigen::Array<Index, MAXDIV_INDEX_DIMENSION, 1> EigenIndexVector;
    
    union
    {
        Index ind[MAXDIV_INDEX_DIMENSION];
        struct
        {
            Index t; /**< time */
            Index x; /**< 1st spatial dimension */
            Index y; /**< 2nd spatial dimension */
            Index z; /**< 3rd spatial dimension */
            Index d; /**< attribute */
        };
    };
    
    /**
    * Constructs a new index vector with all indices being 0.
    */
    ReflessIndexVector() : t(0), x(0), y(0), z(0), d(0) {};
    
    /**
    * Constructs a new index vector from given indices.
    */
    ReflessIndexVector(Index t, Index x, Index y, Index z, Index d = 0) : t(t), x(x), y(y), z(z), d(d) {};
    
    /**
    * Copy constructor
    */
    ReflessIndexVector(const ReflessIndexVector & other) : t(other.t), x(other.x), y(other.y), z(other.z), d(other.d) {};
    
    virtual ~ReflessIndexVector() {};
    
    /**
    * Checks if this index vectors contains the same indices as @p other.
    */
    bool operator==(const ReflessIndexVector & other) const
    {
        bool eq = true;
        for (unsigned int i = 0; i < MAXDIV_INDEX_DIMENSION; i++)
            if (this->ind[i] != other.ind[i])
            {
                eq = false;
                break;
            }
        return eq;
    };
    
    /**
    * Checks two index vectors for inequality.
    */
    bool operator!=(const ReflessIndexVector & other) const { return !(*this == other); };
    
    /**
    * Checks if all elements of this index vector are equal to a given scalar @p value.
    */
    bool operator==(const Index value) const { return (this->vec() == value).all(); };
    
    /**
    * Checks if any element of this index vector is different from a given scalar @p value.
    */
    bool operator!=(const Index value) const { return (this->vec() != value).any(); };
    
    /**
    * Checks if this index vector refers to an element before @p other.
    */
    bool operator<(const ReflessIndexVector & other) const
    {
        int cmp = 0;
        for (unsigned int i = 0; i < MAXDIV_INDEX_DIMENSION && cmp == 0; i++)
            if (this->ind[i] < other.ind[i])
                cmp = -1;
            else if (this->ind[i] > other.ind[i])
                cmp = 1;
        return (cmp < 0);
    };
    
    bool operator>(const ReflessIndexVector & other) const { return (other < *this); };
    bool operator<=(const ReflessIndexVector & other) const { return !(other < *this); };
    bool operator>=(const ReflessIndexVector & other) const { return !(*this < other); };
    
    /**
    * Element-wise sum of two index vectors
    *
    * @param[in] other The index vector whose elements are to be added to this vector.
    *
    * @note This does not forward the index vector by the linear index of `other`, but
    * adds the indices of each vector independently.
    */
    ReflessIndexVector & operator+=(const ReflessIndexVector & other)
    {
        this->vec() += other.vec();
        return *this;
    };
    
    /**
    * Element-wise sum of two index vectors
    *
    * @param[in] other The index vector whose elements are to be added to this vector.
    *
    * @note This does not forward the index vector by the linear index of `other`, but
    * adds the indices of each vector independently.
    */
    ReflessIndexVector operator+(const ReflessIndexVector & other) const
    {
        ReflessIndexVector sum(*this);
        return sum += other;
    };
    
    /**
    * Element-wise difference of two index vectors
    *
    * @param[in] other The index vector whose elements are to be subtracted from this vector.
    *
    * @note If a minuend is less than the corresponding subtrahend, the result will be 0.
    *
    * @note This does not rewind the index vector by the linear index of `other`, but
    * subtracts the indices of each vector independently.
    */
    ReflessIndexVector & operator-=(const ReflessIndexVector & other)
    {
        this->vec() -= other.vec().min(this->vec());
        return *this;
    };
    
    /**
    * Element-wise difference of two index vectors
    *
    * @param[in] other The index vector whose elements are to be subtracted from this vector.
    *
    * @note If a minuend is less than the corresponding subtrahend, the result will be 0.
    *
    * @note This does not rewind the index vector by the linear index of `other`, but
    * subtracts the indices of each vector independently.
    */
    ReflessIndexVector operator-(const ReflessIndexVector & other) const
    {
        ReflessIndexVector sum(*this);
        sum -= other;
        return sum;
    };
    
    /**
    * Computes the product of the elements of the vector.
    *
    * @param[in] from Index of the first element to include in the product.
    *
    * @param[in] to Index of the last element to include in the product.
    *
    * @return The product of all indices in the specified range.
    */
    Index prod(unsigned int from = 0, unsigned int to = MAXDIV_INDEX_DIMENSION - 1) const
    {
        if (from >= MAXDIV_INDEX_DIMENSION || from > to)
            return 0;
        Index p = this->ind[from];
        for (unsigned int i = from + 1; i <= to && i < MAXDIV_INDEX_DIMENSION; i++)
            p *= this->ind[i];
        return p;
    };
    
    /**
    * @return An Eigen array wrapping the elements of this index vector.
    */
    Eigen::Map<EigenIndexVector> vec()
    {
        return Eigen::Map<EigenIndexVector>(this->ind);
    };
    
    /**
    * @return A constant Eigen array wrapping the elements of this index vector.
    */
    Eigen::Map<const EigenIndexVector> vec() const
    {
        return Eigen::Map<const EigenIndexVector>(this->ind);
    };

};


/**
* @brief Simple struct with indices for DataTensor objects of a specific shape.
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct IndexVector : public ReflessIndexVector
{
    
    ReflessIndexVector shape; /**< Shape of the DataTensor which the indices in this vector refer to */
    
    /**
    * Constructs a new index vector with all indices being 0.
    */
    IndexVector() : ReflessIndexVector(), shape() {};
    
    /**
    * Constructs a new index vector from given indices.
    */
    IndexVector(Index t, Index x, Index y, Index z, Index d = 0) : ReflessIndexVector(t, x, y, z, d), shape() {};
    
    /**
    * Copy constructor
    */
    IndexVector(const IndexVector & other) : ReflessIndexVector(other.t, other.x, other.y, other.z, other.d), shape(other.shape) {};
    
    /**
    * Copies the indices from a ReflessIndexVector and sets the shape of the referred tensor to 0.
    */
    IndexVector(const ReflessIndexVector & other) : ReflessIndexVector(other), shape() {};
    
    /**
    * Constructs a new index vector from given indices and sets the @p shape of the corresponding tensor.
    */
    IndexVector(const ReflessIndexVector & shape, Index t, Index x, Index y, Index z, Index d)
    : ReflessIndexVector(t, x, y, z, d), shape(shape) {};
    
    /**
    * Constructs a new index vector with the given @p shape and @p indices.
    */
    IndexVector(const ReflessIndexVector & shape, const ReflessIndexVector & indices)
    : ReflessIndexVector(indices), shape(shape) {};
    
    /**
    * Constructs a new index vector from a linear index @p k referring to a tensor of the given @p shape.
    */
    IndexVector(const ReflessIndexVector & shape, Index k) : shape(shape)
    {
        Index s;
        this->ind[0] = k;
        for (unsigned int i = 0; i < MAXDIV_INDEX_DIMENSION - 1; i++)
            if (this->ind[i] == 0)
                this->ind[i+1] = 0;
            else
            {
                s = shape.ind[i+1];
                for (unsigned int j = i + 2; j < MAXDIV_INDEX_DIMENSION; j++)
                    s *= shape.ind[j];
                this->ind[i+1] = this->ind[i] % s;
                this->ind[i] /= s;
            }
    };
    
    /**
    * Returns the linear index of this index vector.
    *
    * The shape of the corresponding tensor has to be set.
    */
    Index linear() const
    {
        Index ind = this->ind[0];
        for (unsigned int i = 1; i < MAXDIV_INDEX_DIMENSION; i++)
            ind = (ind * this->shape.ind[i]) + this->ind[i];
        return ind;
    };
    
    /**
    * Copies the contents of @p other to this index vector.
    */
    IndexVector & operator=(const IndexVector & other)
    {
        this->vec() = other.vec();
        this->shape = other.shape;
        return *this;
    };
    
    /**
    * Copies the indices of @p other to this index vector and leaves the shape of the
    * referred tensor unchanged.
    */
    IndexVector & operator=(const ReflessIndexVector & other)
    {
        this->vec() = other.vec();
        return *this;
    };
    
    /**
    * Forwards this index vector by @p n elements.
    */
    IndexVector & operator+=(Index n)
    {
        unsigned int i = MAXDIV_INDEX_DIMENSION - 1;
        this->ind[i] += n;
        for (; i > 0 && this->ind[i] >= this->shape.ind[i]; i--)
        {
            this->ind[i-1] += this->ind[i] / this->shape.ind[i];
            this->ind[i] = this->ind[i] % this->shape.ind[i];
        }
        return *this;
    };
    
    /**
    * Returns a copy of this index vector forwarded by @p n elements.
    */
    IndexVector operator+(Index n) const
    {
        IndexVector sum(*this);
        return sum += n;
    };
    
    /**
    * Element-wise sum of two index vectors
    *
    * @param[in] other The index vector whose elements are to be added to this vector.
    *
    * @note This does not forward the index vector by the linear index of `other`, but
    * adds the indices of each vector independently.
    */
    IndexVector & operator+=(const ReflessIndexVector & other)
    {
        this->vec() += other.vec();
        return *this;
    };
    
    /**
    * Element-wise sum of two index vectors
    *
    * @param[in] other The index vector whose elements are to be added to this vector.
    *
    * @note This does not forward the index vector by the linear index of `other`, but
    * adds the indices of each vector independently.
    */
    IndexVector operator+(const ReflessIndexVector & other) const
    {
        IndexVector sum(*this);
        return sum += other;
    };
    
    /**
    * Element-wise difference of two index vectors
    *
    * @param[in] other The index vector whose elements are to be subtracted from this vector.
    *
    * @note If a minuend is less than the corresponding subtrahend, the result will be 0.
    *
    * @note This does not rewind the index vector by the linear index of `other`, but
    * subtracts the indices of each vector independently.
    */
    IndexVector operator-(const ReflessIndexVector & other) const
    {
        IndexVector sum(*this);
        sum -= other;
        return sum;
    };
    
    /**
    * Forwards this vector by 1 element.  
    * This is semantically equivalent to `*this += 1`, but slightly faster.  
    * The pre-condition is that all indices of the vector must be less than the size of the
    * corresponding dimension.
    */
    IndexVector & operator++()
    {
        unsigned int i = MAXDIV_INDEX_DIMENSION - 1;
        this->ind[i] += 1;
        for (; i > 0 && this->ind[i] >= this->shape.ind[i]; i--)
        {
            this->ind[i-1] += 1;
            this->ind[i] = 0;
        }
        return *this;
    };
    
    IndexVector operator++(int) { IndexVector tmp(*this); this->operator++(); return tmp; };

};


/**
* @brief Simple 1D range `[a, b)`
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct Range
{
    typedef ReflessIndexVector::Index Index;
    
    Index a; /**< First point in the range. */
    Index b; /**< First point after the end of the range. */
    
    Range() : a(0), b(0) {};
    Range(Index a, Index b) : a(a), b(b) {};
    
    /**
    * @return Returns the length of the range, i.e. `b - a`.
    */
    Index length() const { return (this->b > this->a) ? this->b - this->a : 0; };
    
    /**
    * Checks if this range contains a given @p index.
    */
    bool contains(Index index) const { return (index >= this->a && index < this->b); };
};


/**
* @brief Indices specifying a sub-block of a multi-dimensional tensor
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct IndexRange
{
    
    IndexVector a; /**< First point in the range. */
    IndexVector b; /**< First point after the end of the range. */
    
    IndexRange() : a(), b() {};
    IndexRange(const IndexVector & a, const IndexVector & b) : a(a), b(b) {};
    
    Range t() const { return Range(this->a.t, this->b.t); }; /**< Range along the time axis. */
    Range x() const { return Range(this->a.x, this->b.x); }; /**< Range along the 1st spatial dimension. */
    Range y() const { return Range(this->a.y, this->b.y); }; /**< Range along the 2nd spatial dimension. */
    Range z() const { return Range(this->a.z, this->b.z); }; /**< Range along the 3rd spatial dimension. */
    Range d() const { return Range(this->a.d, this->b.d); }; /**< Range along the feature dimension. */
    
    /**
    * Range along a given dimension.
    * @param[in] i Index of the dimension.
    * @return `Range(a.ind[i], b.ind[i])`
    */
    Range ind(unsigned int i) const { return Range(a.ind[i], b.ind[i]); };
    
    /**
    * Computes the shape of this range.
    * @return Returns an index vector with the size of each dimension according to this range.
    */
    ReflessIndexVector shape() const
    {
        ReflessIndexVector shape;
        shape.vec() = this->b.vec() - this->a.vec().min(this->b.vec());
        return shape;
    };
    
    /**
    * @return Returns true if any dimension of this range is of size zero, otherwise false.
    */
    bool empty() const { return (this->b.vec() <= this->a.vec()).any(); };
    
    /**
    * Checks if this range contains a given @p index.
    */
    bool contains(const ReflessIndexVector & index) const
    { return ((index.vec() >= this->a.vec()).all() && (index.vec() < this->b.vec()).all()); };
    
    /**
    * @return The volume of this block, i.e. the number of elements in the range.
    */
    IndexVector::Index volume() const { return this->shape().prod(); };
    
    /**
    * Computes the *intersection over union* between this block and another one.
    * @return The quotient of the intersection and the union of the two blocks.
    */
    double IoU(const IndexRange & other) const
    {
        IndexVector intA, intB;
        intA.vec() = this->a.vec().max(other.a.vec());
        intB.vec() = this->b.vec().min(other.b.vec());
        IndexVector::Index intVol = IndexRange(intA, intB).volume();
        return static_cast<double>(intVol) / static_cast<double>(this->volume() + other.volume() - intVol);
    };
    
    /**
    * @return Returns true iff both the first and the last point in this range equal
    * the first and the last point of @p other.
    */
    bool operator==(const IndexRange & other) const
    {
        return (this->a == other.a && this->b == other.b);
    };
    
    bool operator!=(const IndexRange & other) const { return !(*this == other); };
    
};

}

#endif