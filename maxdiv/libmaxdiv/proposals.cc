#include "proposals.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include "pointwise_detectors.h"
using namespace MaxDiv;


//-------------------//
// ProposalGenerator //
//-------------------//

ProposalGenerator::ProposalGenerator() : m_lengthRange(), m_curStartPoint(), m_internalState() {}

ProposalGenerator::ProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength)
: m_lengthRange(IndexVector(minLength, minLength, minLength, minLength), IndexVector(maxLength, maxLength, maxLength, maxLength)),
  m_curStartPoint(), m_internalState()
{}

ProposalGenerator::ProposalGenerator(IndexRange lengthRange)
: m_lengthRange(lengthRange), m_curStartPoint(), m_internalState()
{}

ProposalGenerator::~ProposalGenerator() {}

void ProposalGenerator::init(const std::shared_ptr<const DataTensor> & data)
{
    this->m_data = data;
    this->m_curStartPoint = data->makeIndexVector();
    this->m_internalState.reset();
}

void ProposalGenerator::reset()
{
    this->m_data.reset();
    this->m_curStartPoint = IndexVector();
    this->m_internalState.reset();
}

IndexRange ProposalGenerator::next()
{
    if (this->m_curStartPoint.shape.t == 0) // not initialized
        return IndexRange();
    
    IndexRange range;
    while (this->m_curStartPoint.t < this->m_curStartPoint.shape.t)
    {
        // Fetch next proposal for current start point
        range = this->next(this->m_curStartPoint, this->m_internalState);
        if (!range.empty())
            break;
        // No proposals left -> move to next start point
        this->m_curStartPoint += this->m_curStartPoint.shape.d;
        this->m_internalState.reset();
    }
    return range;
}

ProposalIterator ProposalGenerator::begin()
{
    return ProposalIterator(this);
}

ProposalIterator ProposalGenerator::end()
{
    return ProposalIterator();
}

ProposalIterator ProposalGenerator::iterateFromTo(const ReflessIndexVector & startIndex, const ReflessIndexVector & endIndex) const
{
    IndexVector start = IndexVector(this->m_curStartPoint.shape, startIndex);
    IndexVector end = IndexVector(this->m_curStartPoint.shape, endIndex);
    start.d = 0;
    end.d = end.shape.d - 1;
    return ProposalIterator(this, start, end);
}

ProposalIterator ProposalGenerator::iteratePartial(unsigned int num_groups, unsigned int group_num) const
{
    if (group_num >= num_groups)
        return ProposalIterator();
    
    ReflessIndexVector shape = this->m_curStartPoint.shape;
    shape.d = 1;
    ReflessIndexVector::Index numSamples = shape.prod();
    if (numSamples == 0)
        return ProposalIterator();
    
    ReflessIndexVector::Index batchSize = numSamples / num_groups;
    return ProposalIterator(
        this,
        IndexVector(shape, group_num * batchSize),
        (group_num < num_groups - 1) ? IndexVector(shape, (group_num+1) * batchSize) : IndexVector()
    );
}


//------------------------//
// DenseProposalGenerator //
//------------------------//

DenseProposalGenerator::DenseProposalGenerator() : ProposalGenerator() {}

DenseProposalGenerator::DenseProposalGenerator(const std::shared_ptr<const DataTensor> & data) : ProposalGenerator()
{
    this->init(data);
}

DenseProposalGenerator::DenseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength)
: ProposalGenerator(minLength, maxLength)
{}

DenseProposalGenerator::DenseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const std::shared_ptr<const DataTensor> & data)
: ProposalGenerator(minLength, maxLength)
{
    this->init(data);
}

DenseProposalGenerator::DenseProposalGenerator(IndexRange lengthRange) : ProposalGenerator(lengthRange) {}

DenseProposalGenerator::DenseProposalGenerator(IndexRange lengthRange, const std::shared_ptr<const DataTensor> & data)
: ProposalGenerator(lengthRange)
{
    this->init(data);
}

void DenseProposalGenerator::initState(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const
{
    // Start with smallest length
    IndexVector * rangeEndOffs = (state) ? reinterpret_cast<IndexVector*>(state.get()) : new IndexVector();
    rangeEndOffs->vec() = this->m_lengthRange.a.vec().max(1).min(this->m_curStartPoint.shape.vec()) - 1;
    rangeEndOffs->d = this->m_curStartPoint.shape.d - 1;
    
    // Set shape to the available or maximum allowed length
    rangeEndOffs->shape = this->m_lengthRange.b;
    rangeEndOffs->shape.d = this->m_curStartPoint.shape.d;
    
    ReflessIndexVector::Index maxLen;
    for (unsigned int i = 0; i < MAXDIV_INDEX_DIMENSION - 1; i++)
    {
        maxLen = this->m_curStartPoint.shape.ind[i] - startIndex.ind[i];
        if (rangeEndOffs->shape.ind[i] == 0 || rangeEndOffs->shape.ind[i] > maxLen)
            rangeEndOffs->shape.ind[i] = maxLen;
    }
    
    if (!state)
        state.reset(rangeEndOffs);
}

IndexRange DenseProposalGenerator::next(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const
{
    if (this->m_curStartPoint.shape.t == 0 || startIndex.t >= this->m_curStartPoint.shape.t)
        return IndexRange();
    
    IndexVector rangeStart, rangeEnd;
    IndexVector * rangeEndOffs;
    IndexRange range;
    do
    {
        if (!state)
        {
            // first proposal
            this->initState(startIndex, state);
            rangeEndOffs = reinterpret_cast<IndexVector*>(state.get());
        }
        else
        {
            // Forward offset
            rangeEndOffs = reinterpret_cast<IndexVector*>(state.get());
            *rangeEndOffs += rangeEndOffs->shape.d;
            // Respect minimum length
            for (unsigned int i = 0; i < MAXDIV_INDEX_DIMENSION - 1; i++)
                if (this->m_curStartPoint.shape.ind[i] > 1 && this->m_lengthRange.a.ind[i] > 0 && rangeEndOffs->ind[i] < this->m_lengthRange.a.ind[i] - 1)
                    rangeEndOffs->ind[i] = this->m_lengthRange.a.ind[i] - 1;
        }
        
        if ((rangeEndOffs->vec() >= rangeEndOffs->shape.vec()).any())
            return IndexRange(); // Maximum length reached -> no proposals left
        
        rangeStart = startIndex;
        rangeStart.d = 0;
        rangeEnd = rangeStart + *rangeEndOffs;
        rangeEnd.vec() += 1;
        range = IndexRange(rangeStart, rangeEnd);
    }
    while (this->m_data->isRangeReducable(range));
    
    return range;
}


//----------------------------//
// PointwiseProposalGenerator //
//----------------------------//

const PointwiseProposalGenerator::Params PointwiseProposalGenerator::defaultParams = {
    &hotellings_t,
    true,
    false,
    1.5
};

PointwiseProposalGenerator::PointwiseProposalGenerator()
: DenseProposalGenerator(), m_params(defaultParams), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(const Params & params)
: DenseProposalGenerator(), m_params(params), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(), m_params(defaultParams), m_scores(), m_th()
{
    this->init(data);
}

PointwiseProposalGenerator::PointwiseProposalGenerator(const Params & params, const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(), m_params(params), m_scores(), m_th()
{
    this->init(data);
}

PointwiseProposalGenerator::PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength)
: DenseProposalGenerator(minLength, maxLength), m_params(defaultParams), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const Params & params)
: DenseProposalGenerator(minLength, maxLength), m_params(params), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(minLength, maxLength), m_params(defaultParams), m_scores(), m_th()
{
    this->init(data);
}

PointwiseProposalGenerator::PointwiseProposalGenerator(DataTensor::Index minLength, DataTensor::Index maxLength, const Params & params, const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(minLength, maxLength), m_params(params), m_scores(), m_th()
{
    this->init(data);
}

PointwiseProposalGenerator::PointwiseProposalGenerator(IndexRange lengthRange)
: DenseProposalGenerator(lengthRange), m_params(defaultParams), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(IndexRange lengthRange, const Params & params)
: DenseProposalGenerator(lengthRange), m_params(params), m_scores(), m_th()
{}

PointwiseProposalGenerator::PointwiseProposalGenerator(IndexRange lengthRange, const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(lengthRange), m_params(defaultParams), m_scores(), m_th()
{
    this->init(data);
}

PointwiseProposalGenerator::PointwiseProposalGenerator(IndexRange lengthRange, const Params & params, const std::shared_ptr<const DataTensor> & data)
: DenseProposalGenerator(lengthRange), m_params(params), m_scores(), m_th()
{
    this->init(data);
}

void PointwiseProposalGenerator::init(const std::shared_ptr<const DataTensor> & data)
{
    DenseProposalGenerator::init(data);
    
    if (!data->empty())
    {
        // Compute point-wise scores
        this->m_scores = this->m_params.scorer(*data);
        
        // Apply gradient filter
        if (this->m_params.gradientFilter && this->m_scores.numEl() > 1)
            this->computeGradient(this->m_scores);
        
        // Determine threshold based on mean and standard deviation
        Scalar mean, sd, sd_th = this->m_params.sd_th;
        if (this->m_params.mad)
        {
            Sample tmp;
            if (!this->m_scores.hasMissingSamples())
                tmp = this->m_scores.asVector();
            else
            {
                tmp.resize(this->m_scores.numValidSamples());
                for (DataTensor::Index sample = 0, tmpSample = 0; tmpSample < static_cast<DataTensor::Index>(tmp.size()); ++sample)
                    if (!this->m_scores.isMissingSample(sample))
                        tmp(tmpSample++) = this->m_scores.sample(sample)(0);
            }
            Sample::Index medianIndex = tmp.size() / 2;
            std::nth_element(tmp.data(), tmp.data() + medianIndex, tmp.data() + tmp.size());
            mean = tmp(medianIndex);
            tmp = (tmp.array() - mean).cwiseAbs();
            std::nth_element(tmp.data(), tmp.data() + medianIndex, tmp.data() + tmp.size());
            sd = 1.4826 * tmp(medianIndex);
        }
        else
        {
            mean = this->m_scores.data().sum() / this->m_scores.numValidSamples();
            if (!this->m_scores.hasMissingSamples())
                sd = std::sqrt((this->m_scores.data().array() - mean).cwiseAbs2().mean());
            else
            {
                DataTensor centered = this->m_scores;
                centered.data().array() -= mean;
                sd = std::sqrt(centered.data().cwiseAbs2().sum() / centered.numValidSamples());
            }
        }
        
        this->m_th = mean + sd_th * sd;
        while (!(this->m_scores.data().array() >= this->m_th).any())
        {
            sd_th *= 0.8;
            this->m_th = mean + sd_th * sd;
        }
        
        this->m_fallback_th = mean;
    }
}

void PointwiseProposalGenerator::reset()
{
    DenseProposalGenerator::reset();
    this->m_scores.release();
}

void PointwiseProposalGenerator::initState(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const
{
    if (!state)
        state.reset(new State());
    DenseProposalGenerator::initState(startIndex, state);
    State * searchState = reinterpret_cast<State*>(state.get());
    searchState->isolated_end = 0;
}

IndexRange PointwiseProposalGenerator::next(const ReflessIndexVector & startIndex, std::shared_ptr<void> & state) const
{
    // Check if initialized and start index within range
    if (this->m_curStartPoint.shape.t == 0 || startIndex.t >= this->m_curStartPoint.shape.t)
        return IndexRange();
    
    // Check if score of start index is above threshold
    ReflessIndexVector rangeStart = startIndex;
    rangeStart.d = 0;
    if (this->m_scores(rangeStart) < this->m_th)
        return IndexRange();
    
    // Search next end index above threshold
    bool firstIteration = !state;
    IndexRange range;
    IndexVector rangeEnd;
    if (firstIteration || reinterpret_cast<State*>(state.get())->isolated_end == 0)
    {
        do
        {
            range = DenseProposalGenerator::next(startIndex, state);
            if (range.empty())
                break;
            
            rangeEnd = range.b;
            rangeEnd.vec() -= 1;
            rangeEnd.d = 0;
        } while (this->m_scores(rangeEnd) < this->m_th);
    }
    
    // If we did not find an end point above the threshold, look behind and check if this
    // start index is an isolated peak. If so, propose ranges based on a lower threshold.
    // (Currently available for data with 1 non-singleton spatio-temporal dimension only.)
    State * searchState = reinterpret_cast<State*>(state.get());
    if (range.empty() && firstIteration && this->m_data->nonSingletonDim() >= 0)
    {
        int d = this->m_data->nonSingletonDim();
        bool isolated = false;
        
        DataTensor::Index peakIndex = startIndex.ind[d],
                          minLength = this->m_lengthRange.a.ind[d],
                          maxLength = this->m_lengthRange.b.ind[d],
                          firstPrev, lastPrev;
        if (peakIndex == 0 || (minLength > 0 && peakIndex + 1 < minLength))
        {
            isolated = true;
            firstPrev = peakIndex;
        }
        else
        {
            firstPrev = (maxLength > 0 && maxLength <= peakIndex) ? peakIndex - (maxLength - 1) : 0;
            lastPrev = (minLength > 0) ? peakIndex - (minLength - 1) : peakIndex - 1;
            if ((this->m_scores.asVector().segment(firstPrev, lastPrev - firstPrev + 1).array() < this->m_th).all())
                isolated = true;
        }
        
        if (isolated)
        {
            searchState->isolated_peak = peakIndex;
            searchState->isolated_pos = firstPrev;
            searchState->isolated_end = (maxLength > 0 && peakIndex + maxLength < this->m_curStartPoint.shape.ind[d]) ? peakIndex + maxLength : this->m_curStartPoint.shape.ind[d];
            searchState->isolated_minLength = minLength;
        }
    }
    
    if (searchState->isolated_end > 0)
    {
        while (searchState->isolated_pos < searchState->isolated_end
                && (std::max(searchState->isolated_pos, searchState->isolated_peak) - std::min(searchState->isolated_pos, searchState->isolated_peak) + 1 < searchState->isolated_minLength
                || this->m_scores.asVector()(searchState->isolated_pos) < this->m_fallback_th))
        {
            searchState->isolated_pos += 1;
        }
        
        if (searchState->isolated_pos < searchState->isolated_end)
        {
            if (searchState->isolated_pos < searchState->isolated_peak)
            {
                range.a = IndexVector(this->m_curStartPoint.shape, searchState->isolated_pos * this->m_curStartPoint.shape.d);
                range.b = IndexVector(this->m_curStartPoint.shape, searchState->isolated_peak * this->m_curStartPoint.shape.d);
            }
            else
            {
                range.a = IndexVector(this->m_curStartPoint.shape, searchState->isolated_peak * this->m_curStartPoint.shape.d);
                range.b = IndexVector(this->m_curStartPoint.shape, searchState->isolated_pos * this->m_curStartPoint.shape.d);
            }
            range.b.vec() += 1;
            searchState->isolated_pos += 1;
        }
    }
    
    return range;
}

void PointwiseProposalGenerator::computeGradient(DataTensor & data) const
{
    assert(data.shape().d == 1);
    
    if (this->m_data->nonSingletonDim() >= 0)
    {
        // Efficient shortcut for 1-d data
        DataTensor::Index numEl = data.numEl();
        Sample vec(numEl + 2);
        // Remove missing sample mask, but remember it
        std::vector<DataTensor::Index> missingSamples(data.getMissingSampleIndices().begin(), data.getMissingSampleIndices().end());
        data.removeMask();
        // Add padding
        vec.segment(1, numEl) = data.asVector();
        vec(0) = vec(1);
        vec(numEl + 1) = vec(numEl);
        // Compute centralized gradient
        data.asVector() = (vec.tail(numEl) - vec.head(numEl)).cwiseAbs();
        // Restore and adjust missing samples
        for (const DataTensor::Index & missing : missingSamples)
        {
            data.setMissingSample((missing > 0) ? missing - 1 : 0);
            data.setMissingSample(std::min(missing + 1, data.numSamples()));
        }
    }
    else
    {
        // Compute the gradient magnitude for each point
        DataTensor grad(data.shape());
        IndexVector ind, prev, next;
        DataTensor::Index sampleInd;
        unsigned int d;
        Scalar mag, val;
        for (ind = grad.makeIndexVector(), sampleInd = 0; ind.t < ind.shape.t; ++ind, ++sampleInd)
        {
            mag = 0.0;
            for (d = 0; d < MAXDIV_INDEX_DIMENSION - 1; ++d)
                if (ind.shape.ind[d] > 1)
                {
                    prev = next = ind;
                    if (prev.ind[d] > 0)
                        prev.ind[d] -= 1;
                    if (next.ind[d] + 1 < ind.shape.ind[d])
                        next.ind[d] += 1;
                    if (!data.isMissingSample(prev) && !data.isMissingSample(next))
                    {
                        val = data(next) - data(prev);
                        mag += val * val;
                    }
                    else
                    {
                        grad.setMissingSample(sampleInd);
                        break;
                    }
                }
            grad.asVector()(sampleInd) = mag;
        }
        data.data() = grad.data().cwiseSqrt();
    }
}
