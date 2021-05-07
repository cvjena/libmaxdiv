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

#ifndef MAXDIV_PROPOSALS_H
#define MAXDIV_PROPOSALS_H

#include <iterator>
#include <memory>
#include <functional>
#include "DataTensor.h"

namespace MaxDiv
{

class ProposalIterator;

/**
* @brief Abstract base class for generators which propose possibly anomalous intervals.
*
* Basic usage example:
*
*     SomeProposalGenerator proposals();
*     proposals.init(data);
*     for (const IndexRange & range : proposals) {
*         // do something with range
*     }
*
* This could also be parallelized by splitting the range of start points:
*
*     SomeProposalGenerator proposals();
*     proposals.init(data);
*     #pragma omp parallel
*     {
*         for (ProposalIterator rangeIt = proposals.iteratePartial(omp_get_num_threads(), omp_get_thread_num()); rangeIt != proposals.end(); ++rangeIt)
*         {
*             IndexRange & range = *rangeIt;
*             // do something with range
*         }
*     }
*
* Derived classes should override `init()` and `next(state)`.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ProposalGenerator
{
public:

    /**
    * Constructs an un-initialized proposal generator. `init()` has to be called before it may be used.
    */
    ProposalGenerator();
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    */
    ProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    */
    ProposalGenerator(IndexRange lengthRange);
    
    virtual ~ProposalGenerator();
    
    /**
    * Initializes this proposal generator to make proposals for the data in @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Resets this proposal generator to its un-initialized state and releases any memory allocated
    * by `init()`.
    */
    virtual void reset();
    
    /**
    * Fetches the next proposal for a specific start point, based on a given state of iteration.
    * `init()` has to be called before this can be used.
    *
    * @param[in] startIndex The start point of the proposed ranges.
    *
    * @param[in,out] state Pointer to a state object which contains information about the state
    * of iteration. This way, proposals can be made from any point and in parallel. The given
    * state will be updated accordingly or initialized if a reference to a null-pointer is passed.
    *
    * @return Returns a proposal for a possibly anomalous range or an empty range if there aren't
    * any more proposals or init() hasn't been called yet.
    *
    * @note The proposed range is not guaranteed to actually begin at the position given by `startIndex`.
    */
    virtual IndexRange next(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const =0;
    
    /**
    * Fetches the next proposal. init() has to be called before this can be used.
    *
    * This function essentially calls `next(state)` with an internal state which is
    * reset on every call to `init()`.
    *
    * @return Returns a proposal for a possibly anomalous range or an empty range if
    * there aren't any more proposals or init() hasn't been called yet.
    */
    virtual IndexRange next();
    
    /**
    * Returns an iterator of the ranges proposed by this proposal generator.
    * init() has to be called before this may be used.
    *
    * @return A ProposalIterator pointing to the next proposed range.
    */
    ProposalIterator begin();
    
    /**
    * @return A past-the-end ProposalIterator.
    */
    ProposalIterator end();
    
    /**
    * Returns an iterator over proposals within a given range of start points.
    * init() has to be called before this may be used.
    *
    * @param[in] startIndex Beginning of the range of start indices (inclusively). The index of the
    * attribute dimension will be set to 0.
    *
    * @param[in] endIndex End of the range of start indices (exclusively). May be empty to iterate till
    * the end of the range.
    *
    * @return A ProposalIterator pointing to the next proposed range.
    */
    ProposalIterator iterateFromTo(const ReflessIndexVector & startIndex, const ReflessIndexVector & endIndex) const;
    
    /**
    * Returns an iterator over proposals within a subset of possible start points.
    * This is indented to be used for multi-threading.
    * init() has to be called before this may be used.
    *
    * @param[in] num_groups Total number of subsets of start points.
    *
    * @param[in] group_num Index of the subset to iterate over.
    *
    * @return A ProposalIterator pointing to the next proposed range.
    */
    ProposalIterator iteratePartial(unsigned int num_groups, unsigned int group_num) const;


protected:
    
    std::shared_ptr<const DataTensor> m_data; /**< Pointer to the data tensor passed to `init()`. */
    IndexRange m_lengthRange; /**< Minimum and maximum length of the proposed ranges. */
    IndexVector m_curStartPoint; /**< The current start point for internal iteration used by next(). */


private:

    std::shared_ptr<void> m_internalState; /**< Internal state of iteration used by next(). */

};


/**
* @brief Iterates over interval proposals generated by a ProposalGenerator
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ProposalIterator : public std::iterator< std::input_iterator_tag, IndexRange >
{
public:

    /**
    * Constructs an past-the-end iterator.
    */
    ProposalIterator() : m_pg(nullptr), m_curStartIndex(), m_endIndex(), m_state(), m_range() {};
    
    /**
    * Constructs a ProposalIterator for the given ProposalGenerator @p pg.
    */
    ProposalIterator(const ProposalGenerator * pg)
    : m_pg(const_cast<ProposalGenerator*>(pg)), m_curStartIndex(), m_endIndex(), m_state(), m_range()
    {
        ++(*this);
    };
    
    /**
    * Constructs a ProposalIterator for the given ProposalGenerator @p pg which iterates over ranges
    * with start points ranging from @p startIndex (inclusively) to @p endIndex (exclusively).
    * `endIndex` may be empty to iterate till the end of the range.
    */
    ProposalIterator(const ProposalGenerator * pg, const IndexVector & startIndex, const IndexVector & endIndex)
    : m_pg(const_cast<ProposalGenerator*>(pg)), m_curStartIndex(startIndex), m_endIndex(endIndex), m_state(), m_range()
    {
        ++(*this);
    };
    
    /**
    * Copy constructor.
    */
    ProposalIterator(const ProposalIterator & other)
    : m_pg(other.m_pg), m_curStartIndex(other.m_curStartIndex), m_endIndex(other.m_endIndex), m_state(other.m_state), m_range(other.m_range) {};
    
    /**
    * Copy assignment operator.
    */
    ProposalIterator & operator=(const ProposalIterator & other)
    {
        this->m_pg = other.m_pg;
        this->m_curStartIndex = other.m_curStartIndex;
        this->m_endIndex = other.m_endIndex;
        this->m_state = other.m_state;
        this->m_range = other.m_range;
        return *this;
    };
    
    reference operator*() { return this->m_range; };
    pointer operator->() { return &(this->m_range); };
    
    ProposalIterator & operator++()
    {
        if (this->m_pg)
        {
            if (this->m_curStartIndex.shape == 0)
                this->m_range = this->m_pg->next();
            else
            {
                while (this->m_curStartIndex.t < this->m_curStartIndex.shape.t && (this->m_endIndex == 0 || this->m_curStartIndex < this->m_endIndex))
                {
                    // Fetch next proposal for current start point
                    this->m_range = this->m_pg->next(this->m_curStartIndex, this->m_state);
                    if (!this->m_range.empty())
                        break;
                    // No proposals left -> move to next start point
                    this->m_curStartIndex += this->m_curStartIndex.shape.d;
                    this->m_state.reset();
                }
            }
        }
        return *this;
    };
    
    ProposalIterator operator++(int) { ProposalIterator tmp(*this); this->operator++(); return tmp; };
    
    /**
    * @return Returns true if both this and the other iterator are past-the-end iterators or both
    * refer to the same range proposed from the same ProposalGenerator instance.
    */
    bool operator==(const ProposalIterator & other) const
    {
        return ((!(*this) && !other) || (this->m_pg == other.m_pg && this->m_range == other.m_range));
    };
    
    bool operator!=(const ProposalIterator & other) const { return !(*this == other); };
    
    /**
    * @return Returns false if this is a past-the-end iterator, otherwise true.
    */
    operator bool() const { return (this->m_pg != nullptr && !this->m_range.empty()); };


private:
    
    ProposalGenerator * m_pg;
    IndexVector m_curStartIndex;
    IndexVector m_endIndex;
    std::shared_ptr<void> m_state;
    IndexRange m_range;

};


/**
* @brief Proposes each possible range (essentially the equivalent of nested for-loops)
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class DenseProposalGenerator : public ProposalGenerator
{
public:

    /**
    * Constructs an un-initialized proposal generator. `init()` has to be called before it may be used.
    */
    DenseProposalGenerator();
    
    /**
    * Constructs and initializes a proposal generator by calling `init(data)`.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    DenseProposalGenerator(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    * 
    * @param[in] stride Offset between two consecutive proposals in each dimension.
    */
    DenseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, DataTensor::Index stride = 1);
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    DenseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const std::shared_ptr<const DataTensor> & data);

    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    * 
    * @param[in] stride Offset between two consecutive proposals in each dimension.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    DenseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, DataTensor::Index stride, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    * 
    * @param[in] stride A vector specifying the dimension-wise offset between two consecutive proposals.
    */
    DenseProposalGenerator(IndexRange lengthRange, ReflessIndexVector stride = {1, 1, 1, 1, 1});
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    DenseProposalGenerator(IndexRange lengthRange, const std::shared_ptr<const DataTensor> & data);

    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    * 
    * @param[in] stride A vector specifying the dimension-wise offset between two consecutive proposals.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    DenseProposalGenerator(IndexRange lengthRange, ReflessIndexVector stride, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Fetches the next proposal for a specific start point, based on a given state of iteration.
    * init() has to be called before this can be used.
    *
    * @param[in] startIndex The start point of the proposed ranges. The index of the attribute
    * dimension will be set to 0.
    *
    * @param[in,out] state Pointer to a state object which contains information about the state
    * of iteration. This way, proposals can be made from any point and in parallel. The given
    * state will be updated accordingly or initialized if a reference to a null-pointer is passed.
    *
    * @return Returns a proposal for a possibly anomalous range or an empty range if there aren't
    * any more proposals or init() hasn't been called yet.
    *
    * @note The proposed range is not guaranteed to actually begin at the position given by `startIndex`.
    */
    virtual IndexRange next(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const override;


protected:

    ReflessIndexVector m_stride; /**< Vector with dimension-wise offsets between proposals. */
    
    /**
    * Resets @p state (an IndexVector) to the smallest possible offset from @p startIndex and adjusts its associated
    * shape to the minimum of the remaining space and the maximum range length.
    */
    virtual void initState(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const;

};


/**
* @brief Proposes intervals between peaks of point-wise scores
*
* By default, Hotelling's T^2 is used as point-wise scoring method, but an alternative scoring function
* may be passed to the constructor.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class PointwiseProposalGenerator : public DenseProposalGenerator
{
public:

    typedef std::function<DataTensor(const DataTensor&)> PointwiseScorer;
    
    struct Params
    {
        PointwiseScorer scorer; /**< The point-wise scoring function used to obtain scores. */
        bool gradientFilter; /**< Specifies whether to apply a gradient filter to the obtained scores. */
        bool mad; /**< Specifies whether to use *Median Absolute Deviation (MAD)* for a robust computation of mean and standard deviation of the scores. */
        Scalar sd_th; /**< Thresholds for scores will be `mean + sd_th * standard_deviation`. */
    };
    
    static const Params defaultParams;
    

    /**
    * Constructs an un-initialized proposal generator. `init()` has to be called before it may be used.
    */
    PointwiseProposalGenerator();
    
    /**
    * Constructs an un-initialized proposal generator. `init()` has to be called before it may be used.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    */
    PointwiseProposalGenerator(const Params & params);
    
    /**
    * Constructs and initializes a proposal generator by calling `init(data)`.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs and initializes a proposal generator by calling `init(data)`.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(const Params & params, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    */
    PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    */
    PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const Params & params);
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] minLength The minimum length of proposed ranges in each dimension.
    *
    * @param[in] maxLength The maximum length of proposed ranges in each dimension.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const Params & params, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    */
    PointwiseProposalGenerator(IndexRange lengthRange);
    
    /**
    * Constructs an un-initialized proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges. `init()` has to be called before proposals can be made.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    */
    PointwiseProposalGenerator(IndexRange lengthRange, const Params & params);
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(IndexRange lengthRange, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Constructs and initializes proposal generator and specifies a minimum and a maximum length for
    * the proposed ranges.
    *
    * @param[in] lengthRange A range whose start specifies the minimum length of the proposed ranges
    * for each dimension and whose end specifies the maximum length. The attribute dimension will be
    * ignored.
    *
    * @param[in] params A structure with parameters for this proposal generator. Default parameters
    * can be found in `PointwiseProposalGenerator::defaultParams`.
    *
    * @param[in] data Pointer to the data to generate range proposals for.  
    * If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    PointwiseProposalGenerator(IndexRange lengthRange, const Params & params, const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Initializes this proposal generator to make proposals for the data in @p data.
    *
    * @note If the data contain missing samples, they must have been masked by calling `DataTensor::mask()`.
    */
    virtual void init(const std::shared_ptr<const DataTensor> & data);
    
    /**
    * Resets this proposal generator to its un-initialized state and releases any memory allocated
    * by `init()`.
    */
    virtual void reset() override;
    
    /**
    * Fetches the next proposal for a specific start point, based on a given state of iteration.
    * init() has to be called before this can be used.
    *
    * @param[in] startIndex The start point of the proposed ranges. The index of the attribute
    * dimension will be set to 0.
    *
    * @param[in,out] state Pointer to a state object which contains information about the state
    * of iteration. This way, proposals can be made from any point and in parallel. The given
    * state will be updated accordingly or initialized if a reference to a null-pointer is passed.
    *
    * @return Returns a proposal for a possibly anomalous range or an empty range if there aren't
    * any more proposals or init() hasn't been called yet.
    *
    * @note The proposed range is not guaranteed to actually begin at the position given by `startIndex`.
    */
    virtual IndexRange next(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const override;


protected:

    Params m_params; /**< Parameters of this instance */
    DataTensor m_scores; /**< Point-wise scores */
    Scalar m_th; /**< Threshold */
    Scalar m_fallback_th; /**< Lower threshold for isolated peaks */
    
    struct State : public IndexVector
    {
        IndexVector::Index isolated_peak;
        IndexVector::Index isolated_pos;
        IndexVector::Index isolated_end;
        IndexVector::Index isolated_minLength;
    };
    
    virtual void initState(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const override;
    
    /**
    * Approximates the gradient magnitude of a DataTensor using the centralized gradient filter [-1, 0, 1]
    * (with constant padding on the borders). 
    *
    * @param[in,out] data The DataTensor which is to be transformed into the magnitude of its gradient.
    * The attribute dimension of this tensor must be of size 1.
    */
    void computeGradient(DataTensor & data) const;

};

}

#endif