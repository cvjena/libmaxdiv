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

#include "search_strategies.h"
#include "config.h"
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace MaxDiv;


SearchStrategy::SearchStrategy()
: autoReset(true), m_divergence(new KLDivergence(std::make_shared<GaussianDensityEstimator>())), m_preproc(nullptr), m_overlap_th(0.0) {}

SearchStrategy::SearchStrategy(const std::shared_ptr<Divergence> & divergence)
: autoReset(true), m_divergence(divergence), m_preproc(nullptr), m_overlap_th(0.0)
{
    if (divergence == nullptr)
        throw std::invalid_argument("divergence must not be NULL.");
}

SearchStrategy::SearchStrategy(const std::shared_ptr<Divergence> & divergence, const std::shared_ptr<const PreprocessingPipeline> & preprocessing)
: autoReset(true), m_divergence(divergence), m_preproc(preprocessing), m_overlap_th(0.0)
{
    if (divergence == nullptr)
        throw std::invalid_argument("divergence must not be NULL.");
}

DetectionList SearchStrategy::operator()(const std::shared_ptr<DataTensor> & data, unsigned int numDetections)
{
    DetectionList detections;
    if (data)
    {
        // Mask missing values
        data->mask();
        
        // Apply pre-processing
        ReflessIndexVector borderSize;
        if (this->m_preproc && !this->m_preproc->empty())
        {
            borderSize = this->m_preproc->borderSize(*data);
            (*(this->m_preproc))(*data);
        }
        
        // Detect anomalous intervals
        detections = this->detect(data, numDetections);
        
        // Add offset to the detected ranges if a border has been cut off from the original data
        if (borderSize != 0)
            for (Detection & detection : detections)
            {
                detection.a += borderSize;
                detection.b += borderSize;
            }
    }
    return detections;
}

DetectionList SearchStrategy::operator()(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections)
{
    DetectionList detections;
    if (data)
    {
        // Apply pre-processing
        ReflessIndexVector borderSize;
        std::shared_ptr<const DataTensor> modData;
        bool hasMissingValues = data->hasMissingValues();
        if ((this->m_preproc && !this->m_preproc->empty()) || hasMissingValues)
        {
            DataTensor * md = nullptr;
            if (hasMissingValues)
            {
                md = new DataTensor(*data);
                md->mask();
            }
            if (this->m_preproc && !this->m_preproc->empty())
            {
                borderSize = this->m_preproc->borderSize(*data);
                if (md == nullptr)
                {
                    md = new DataTensor();
                    (*(this->m_preproc))(*data, *md);
                }
                else
                    (*(this->m_preproc))(*md);
            }
            modData = std::shared_ptr<const DataTensor>(md);
        }
        else
            modData = data;
        
        // Detect anomalous intervals
        detections = this->detect(modData, numDetections);
        
        // Add offset to the detected ranges if a border has been cut off from the original data
        if (borderSize != 0)
            for (Detection & detection : detections)
            {
                detection.a += borderSize;
                detection.b += borderSize;
            }
    }
    return detections;
}

void SearchStrategy::scoreIntervals(const std::shared_ptr<DataTensor> & data, DetectionList & intervals)
{
    if (!data)
        return;
    
    // Mask missing values
    data->mask();
    
    // Apply pre-processing
    ReflessIndexVector borderSize;
    if (this->m_preproc && !this->m_preproc->empty())
    {
        borderSize = this->m_preproc->borderSize(*data);
        (*(this->m_preproc))(*data);
    }
    
    // Score provided intervals
    this->doScoreIntervals(data, intervals, borderSize);
}

void SearchStrategy::scoreIntervals(const std::shared_ptr<const DataTensor> & data, DetectionList & intervals)
{
    if (!data)
        return;
    
    // Apply pre-processing
    ReflessIndexVector borderSize;
    std::shared_ptr<const DataTensor> modData;
    bool hasMissingValues = data->hasMissingValues();
    if ((this->m_preproc && !this->m_preproc->empty()) || hasMissingValues)
    {
        DataTensor * md = nullptr;
        if (hasMissingValues)
        {
            md = new DataTensor(*data);
            md->mask();
        }
        if (this->m_preproc && !this->m_preproc->empty())
        {
            borderSize = this->m_preproc->borderSize(*data);
            if (md == nullptr)
            {
                md = new DataTensor();
                (*(this->m_preproc))(*data, *md);
            }
            else
                (*(this->m_preproc))(*md);
        }
        modData = std::shared_ptr<const DataTensor>(md);
    }
    else
        modData = data;
    
    // Score provided intervals
    this->doScoreIntervals(modData, intervals, borderSize);
}

void SearchStrategy::doScoreIntervals(const std::shared_ptr<const DataTensor> & data, DetectionList & intervals, const ReflessIndexVector & offset)
{
    if (!data)
        return;

    bool hasOffset = (offset.vec() > 0).any();
    
    // Initialize divergence metric
    this->m_divergence->init(data);

    // Score provided intervals
    #ifdef _OPENMP
    Eigen::setNbThreads(1);
    bool first_loop = true;
    std::shared_ptr<Divergence> divergence;
    #pragma omp parallel for private(divergence) firstprivate(first_loop)
    for (size_t i = 0; i < intervals.size(); ++i)
    {
        if (first_loop)
        {
            divergence = this->m_divergence->clone();
            first_loop = false;
        }
        if ((intervals[i].a.vec() >= offset.vec()).all()
                && (intervals[i].a.vec().head(MAXDIV_INDEX_DIMENSION - 1) < intervals[i].b.vec().head(MAXDIV_INDEX_DIMENSION - 1)).all()
                && (intervals[i].a.vec() - offset.vec() < data->shape().vec()).all()
                && (intervals[i].b.vec() - offset.vec() <= data->shape().vec()).all())
        {
            if (hasOffset)
                intervals[i].score = (*divergence)(IndexRange(intervals[i].a - offset, intervals[i].b - offset));
            else
                intervals[i].score = (*divergence)(intervals[i]);
        }
        else
            intervals[i].score = std::numeric_limits<Scalar>::quiet_NaN();
    }
    Eigen::setNbThreads(0);
    #else
    for (size_t i = 0; i < intervals.size(); ++i)
    {
        if ((intervals[i].a.vec() >= offset.vec()).all()
                && (intervals[i].a.vec().head(MAXDIV_INDEX_DIMENSION - 1) < intervals[i].b.vec().head(MAXDIV_INDEX_DIMENSION - 1)).all()
                && (intervals[i].a.vec() - offset.vec() < data->shape().vec()).all()
                && (intervals[i].b.vec() - offset.vec() <= data->shape().vec()).all())
        {
            if (hasOffset)
                intervals[i].score = (*(this->m_divergence))(IndexRange(intervals[i].a - offset, intervals[i].b - offset));
            else
                intervals[i].score = (*(this->m_divergence))(intervals[i]);
        }
        else
            intervals[i].score = std::numeric_limits<Scalar>::quiet_NaN();
    }
    #endif
    
    // Release memory
    if (this->autoReset)
        this->m_divergence->reset();
}


ProposalSearch::ProposalSearch() : SearchStrategy(), m_proposals(new DenseProposalGenerator()) {}

ProposalSearch::ProposalSearch(const std::shared_ptr<Divergence> & divergence)
: SearchStrategy(divergence), m_proposals(new DenseProposalGenerator())
{}

ProposalSearch::ProposalSearch(const std::shared_ptr<Divergence> & divergence, const std::shared_ptr<ProposalGenerator> & generator)
: SearchStrategy(divergence), m_proposals(generator)
{
    if (generator == nullptr)
        throw std::invalid_argument("generator must not be NULL.");
}

ProposalSearch::ProposalSearch(const std::shared_ptr<Divergence> & divergence,
                               const std::shared_ptr<ProposalGenerator> & generator,
                               const std::shared_ptr<const PreprocessingPipeline> & preprocessing)
: SearchStrategy(divergence, preprocessing), m_proposals(generator)
{
    if (generator == nullptr)
        throw std::invalid_argument("generator must not be NULL.");
}

DetectionList ProposalSearch::detect(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections)
{
    DetectionList detections;
    if (data)
    {
        // Initialize density estimator and proposal generator
        this->m_divergence->init(data);
        this->m_proposals->init(data);
        
        // Score every proposed range
        if (data->numSamples() <= MAXDIV_NMP_LIMIT)
        {
            // Offline non-maximum suppression: Collect all scores first, then apply non-maximum suppression
            #ifdef _OPENMP
            Eigen::setNbThreads(1);
            #pragma omp parallel
            {
                std::shared_ptr<Divergence> divergence = this->m_divergence->clone();
                DetectionList localDetections;
                for (ProposalIterator range = this->m_proposals->iteratePartial(omp_get_num_threads(), omp_get_thread_num()); range != this->m_proposals->end(); ++range)
                    localDetections.push_back(Detection(*range, (*divergence)(*range)));
                #pragma omp critical
                detections.insert(detections.end(), localDetections.begin(), localDetections.end());
            }
            Eigen::setNbThreads(0);
            #else
            for (const IndexRange & range : *(this->m_proposals))
                detections.push_back(Detection(range, (*(this->m_divergence))(range)));
            #endif
            
            // Non-maximum suppression
            nonMaximumSuppression(detections, numDetections, this->m_overlap_th);
        }
        else
        {
            // Online non-maximum suppression: Apply non-maximum suppression concurrently while retrieving scores
            #ifdef _OPENMP
            std::vector<MaximumDetectionList> detectionLists(omp_get_max_threads(), MaximumDetectionList(numDetections, this->m_overlap_th));
            Eigen::setNbThreads(1);
            #pragma omp parallel
            {
                std::shared_ptr<Divergence> divergence = this->m_divergence->clone();
                MaximumDetectionList & localDetections = detectionLists[omp_get_thread_num()];
                for (ProposalIterator range = this->m_proposals->iteratePartial(omp_get_num_threads(), omp_get_thread_num()); range != this->m_proposals->end(); ++range)
                    localDetections.insert(Detection(*range, (*divergence)(*range)));
            }
            Eigen::setNbThreads(0);
            #else
            std::vector<MaximumDetectionList> detectionLists(1, MaximumDetectionList(numDetections, this->m_overlap_th));
            for (const IndexRange & range : *(this->m_proposals))
                detectionLists[0].insert(Detection(range, (*(this->m_divergence))(range)));
            #endif
            
            // Merge results from different threads
            if (detectionLists.size() > 1)
                detectionLists[0].merge(detectionLists.begin() + 1, detectionLists.end());
            detections.insert(detections.begin(), detectionLists[0].begin(), detectionLists[0].end());
        }
        
        // Release memory
        if (this->autoReset)
        {
            this->m_divergence->reset();
            this->m_proposals->reset();
        }
    }
    return detections;
}


MaximumDetectionList::MaximumDetectionList()
: m_detections(), m_maxDetections(0), m_overlap_th(0.0) {}

MaximumDetectionList::MaximumDetectionList(unsigned int maxDetections)
: m_detections(), m_maxDetections(maxDetections), m_overlap_th(0.0) {}

MaximumDetectionList::MaximumDetectionList(Scalar overlap_th)
: m_detections(), m_maxDetections(0), m_overlap_th(overlap_th) {}

MaximumDetectionList::MaximumDetectionList(unsigned int maxDetections, Scalar overlap_th)
: m_detections(), m_maxDetections(maxDetections), m_overlap_th(overlap_th) {}

MaximumDetectionList::MaximumDetectionList(const MaximumDetectionList & other)
: m_detections(other.m_detections), m_maxDetections(other.m_maxDetections), m_overlap_th(other.m_overlap_th) {}

MaximumDetectionList::MaximumDetectionList(MaximumDetectionList && other)
: m_detections(std::move(other.m_detections)), m_maxDetections(other.m_maxDetections), m_overlap_th(other.m_overlap_th) {}

MaximumDetectionList & MaximumDetectionList::operator=(const MaximumDetectionList & other)
{
    this->m_detections = other.m_detections;
    this->m_maxDetections = other.m_maxDetections;
    this->m_overlap_th = other.m_overlap_th;
    return *this;
}

MaximumDetectionList & MaximumDetectionList::operator=(MaximumDetectionList && other)
{
    this->m_detections = std::move(other.m_detections);
    this->m_maxDetections = other.m_maxDetections;
    this->m_overlap_th = other.m_overlap_th;
    return *this;
}

bool MaximumDetectionList::insert(const Detection & detection)
{
    if (this->m_detections.empty())
    {
        this->m_detections.push_back(detection);
        return true;
    }
    else if (this->m_maxDetections == 1)
    {
        if (detection < this->m_detections.front())
        {
            this->m_detections.front() = detection;
            return true;
        }
        else
            return false;
    }
    else
    {
        iterator pos;
        for (pos = this->m_detections.begin(); pos != this->m_detections.end() && *pos < detection; ++pos)
            if (this->m_overlap_th < 1.0 && pos->IoU(detection) > this->m_overlap_th)
                return false;
        
        if (pos != this->m_detections.end())
        {
            this->m_detections.insert(pos, detection);
            if (this->m_overlap_th < 1.0)
                while (pos != this->m_detections.end())
                {
                    if (pos->IoU(detection) > this->m_overlap_th)
                        pos = this->m_detections.erase(pos);
                    else
                        ++pos;
                }
            if (this->m_maxDetections > 0 && this->m_detections.size() > this->m_maxDetections)
                this->m_detections.pop_back();
            return true;
        }
        else if (this->m_maxDetections > 0 || this->m_detections.size() < this->m_maxDetections)
        {
            this->m_detections.push_back(detection);
            return true;
        }
        else
            return false;
    }
}

bool MaximumDetectionList::insert(Detection && detection)
{
    if (this->m_detections.empty())
    {
        this->m_detections.push_back(std::move(detection));
        return true;
    }
    else if (this->m_maxDetections == 1)
    {
        if (detection < this->m_detections.front())
        {
            this->m_detections.front() = std::move(detection);
            return true;
        }
        else
            return false;
    }
    else
    {
        iterator pos;
        for (pos = this->m_detections.begin(); pos != this->m_detections.end() && *pos < detection; ++pos)
            if (this->m_overlap_th < 1.0 && pos->IoU(detection) > this->m_overlap_th)
                return false;
        
        if (pos != this->m_detections.end())
        {
            iterator det = this->m_detections.insert(pos, std::move(detection));
            if (this->m_overlap_th < 1.0)
                while (pos != this->m_detections.end())
                {
                    if (pos->IoU(*det) > this->m_overlap_th)
                        pos = this->m_detections.erase(pos);
                    else
                        ++pos;
                }
            if (this->m_maxDetections > 0 && this->m_detections.size() > this->m_maxDetections)
                this->m_detections.pop_back();
            return true;
        }
        else if (this->m_maxDetections > 0 || this->m_detections.size() < this->m_maxDetections)
        {
            this->m_detections.push_back(std::move(detection));
            return true;
        }
        else
            return false;
    }
}

MaximumDetectionList::const_iterator MaximumDetectionList::erase(const_iterator pos)
{
#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 8))
    // gcc <= 4.8 does not support erasure of const_iterator
    iterator it = this->m_detections.begin();
    std::advance(it, std::distance(this->m_detections.cbegin(), pos));
    return this->m_detections.erase(it);
#else
    return this->m_detections.erase(pos);
#endif
}

void MaximumDetectionList::merge(MaximumDetectionList & other)
{
    this->m_detections.merge(other.m_detections);
    this->nonMaximumSuppression();
}

void MaximumDetectionList::merge(MaximumDetectionList && other)
{
    this->m_detections.merge(std::move(other.m_detections));
    this->nonMaximumSuppression();
}

void MaximumDetectionList::merge(std::vector<MaximumDetectionList>::iterator first, std::vector<MaximumDetectionList>::iterator last)
{
    for (; first != last; ++first)
        this->m_detections.merge(first->m_detections);
    this->nonMaximumSuppression();
}

MaximumDetectionList::const_iterator MaximumDetectionList::begin() const
{
    return this->m_detections.cbegin();
}

MaximumDetectionList::const_iterator MaximumDetectionList::end() const
{
    return this->m_detections.cend();
}

void MaximumDetectionList::nonMaximumSuppression()
{
    if (this->m_maxDetections != 1 && this->m_overlap_th < 1.0)
        for (iterator major = this->m_detections.begin(); major != this->m_detections.end(); ++major)
        {
            iterator minor = major;
            std::advance(minor, 1);
            while (minor != this->m_detections.end())
            {
                if (major->IoU(*minor) > this->m_overlap_th)
                    minor = this->m_detections.erase(minor);
                else
                    ++minor;
            }
        }
    if (this->m_maxDetections > 0 && this->m_detections.size() > this->m_maxDetections)
    {
        iterator eraseStart = this->m_detections.begin();
        std::advance(eraseStart, this->m_maxDetections);
        this->m_detections.erase(eraseStart, this->m_detections.end());
    }
}


void MaxDiv::nonMaximumSuppression(DetectionList & detections, unsigned int numDetections, double overlap_th)
{
    if (detections.empty())
        return;
    
    if (numDetections == 1)
    {
        // Shortcut if only the maximum is of interest
        detections = DetectionList{*(std::min_element(detections.begin(), detections.end()))};
    }
    else
    {
        // Sort detections by score in descending order
        std::sort(detections.begin(), detections.end());
        
        if (overlap_th < 1.0)
        {
            // Non-maximum suppression
            unsigned int i, j;
            std::vector<bool> include(detections.size(), true); // suppressed intervals will be set to False
            DetectionList newDetections;
            for (i = 0; i < detections.size() && (numDetections == 0 || newDetections.size() < numDetections); ++i)
                if (include[i])
                {
                    newDetections.push_back(detections[i]);
                    
                    // Exclude intervals with a lower score overlapping this one
                    for (j = i + 1; j < detections.size(); ++j)
                        if (include[j] && detections[i].IoU(detections[j]) > overlap_th)
                            include[j] = false;
                }
            
            detections = newDetections;
        }
        else if (numDetections > 0)
        {
            // If overlap_th is at least 1.0, non-maximum suppression is not needed.
            // Instead, we can just truncate the detection vector to the requested size.
            detections.resize(numDetections);
        }
    }
}
