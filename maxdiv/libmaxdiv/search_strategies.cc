#include "search_strategies.h"
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace MaxDiv;


SearchStrategy::SearchStrategy()
: autoReset(true), m_divergence(new KLDivergence(std::make_shared<GaussianDensityEstimator>())), m_preproc(nullptr) {}

SearchStrategy::SearchStrategy(const std::shared_ptr<Divergence> & divergence)
: autoReset(true), m_divergence(divergence), m_preproc(nullptr)
{
    if (divergence == nullptr)
        throw std::invalid_argument("divergence must not be NULL.");
}

SearchStrategy::SearchStrategy(const std::shared_ptr<Divergence> & divergence, const std::shared_ptr<const PreprocessingPipeline> & preprocessing)
: autoReset(true), m_divergence(divergence), m_preproc(preprocessing)
{
    if (divergence == nullptr)
        throw std::invalid_argument("divergence must not be NULL.");
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

DetectionList ProposalSearch::operator()(const std::shared_ptr<DataTensor> & data, unsigned int numDetections)
{
    DetectionList detections;
    if (data)
    {
        // Apply pre-processing
        if (this->m_preproc && !this->m_preproc->empty())
            (*(this->m_preproc))(*data);
        
        // Initialize density estimator and proposal generator
        this->m_divergence->init(data);
        this->m_proposals->init(*data);
        
        // Score every proposed range
        #ifdef _OPENMP
        #pragma omp parallel
        {
            std::shared_ptr<Divergence> divergence = this->m_divergence->clone();
            DetectionList localDetections;
            for (ProposalIterator range = this->m_proposals->iteratePartial(omp_get_num_threads(), omp_get_thread_num()); range != this->m_proposals->end(); ++range)
                localDetections.push_back(Detection(*range, (*divergence)(*range)));
            #pragma omp critical
            detections.insert(detections.end(), localDetections.begin(), localDetections.end());
        }
        #else
        for (const IndexRange & range : *(this->m_proposals))
            detections.push_back(Detection(range, (*(this->m_divergence))(range)));
        #endif
        
        // Release memory
        if (this->autoReset)
        {
            this->m_divergence->reset();
            this->m_proposals->reset();
        }
        
        // Non-maximum suppression
        nonMaximumSuppression(detections, numDetections);
    }
    return detections;
}

DetectionList ProposalSearch::operator()(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections)
{
    DetectionList detections;
    if (data)
    {
        // Apply pre-processing
        std::shared_ptr<const DataTensor> modData;
        if (this->m_preproc && !this->m_preproc->empty())
        {
            DataTensor * md = new DataTensor();
            (*(this->m_preproc))(*data, *md);
            modData = std::shared_ptr<const DataTensor>(md);
        }
        else
            modData = data;
        
        // Initialize density estimator and proposal generator
        this->m_divergence->init(modData);
        this->m_proposals->init(*modData);
        
        // Score every proposed range
        #ifdef _OPENMP
        #pragma omp parallel
        {
            std::shared_ptr<Divergence> divergence = this->m_divergence->clone();
            DetectionList localDetections;
            for (ProposalIterator range = this->m_proposals->iteratePartial(omp_get_num_threads(), omp_get_thread_num()); range != this->m_proposals->end(); ++range)
                localDetections.push_back(Detection(*range, (*divergence)(*range)));
            #pragma omp critical
            detections.insert(detections.end(), localDetections.begin(), localDetections.end());
        }
        #else
        for (const IndexRange & range : *(this->m_proposals))
            detections.push_back(Detection(range, (*(this->m_divergence))(range)));
        #endif
        
        // Release memory
        if (this->autoReset)
        {
            this->m_divergence->reset();
            this->m_proposals->reset();
        }
        
        // Non-maximum suppression
        nonMaximumSuppression(detections, numDetections);
    }
    return detections;
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
                    if (include[j] && (detections[j].score < 1e-8 || detections[i].IoU(detections[j]) > overlap_th))
                        include[j] = false;
            }
        
        detections = newDetections;
    }
}
