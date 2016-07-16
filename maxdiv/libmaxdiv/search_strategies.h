#ifndef MAXDIV_SEARCH_STRATEGIES_H
#define MAXDIV_SEARCH_STRATEGIES_H

#include <memory>
#include <vector>
#include <list>
#include "DataTensor.h"
#include "proposals.h"
#include "divergences.h"
#include "preproc.h"

namespace MaxDiv
{

/**
* @brief Indices specifying a detected anomalous sub-block of a multi-dimensional tensor, along with a detection score
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct Detection : public IndexRange
{
    Scalar score; /**< The confidence score of the detection (higher is better). */
    
    Detection() : IndexRange(), score(0) {};
    Detection(const IndexVector & a, const IndexVector & b) : IndexRange(a, b), score(0) {};
    Detection(const IndexVector & a, const IndexVector & b, Scalar score) : IndexRange(a, b), score(score) {};
    Detection(const IndexRange & range) : IndexRange(range), score(0) {};
    Detection(const IndexRange & range, Scalar score) : IndexRange(range), score(score) {};
    Detection(const Detection & other) : IndexRange(other), score(other.score) {};
    
    /**
    * @return Returns `true` if this detection would be at an earlier position than @p other in
    * a sorted list of detections, i.e. if the score of this detection is *greater* than the score
    * of the other detection.
    */
    bool operator<(const Detection & other) const { return (this->score > other.score); };
};

typedef std::vector<Detection> DetectionList;


/**
* @brief A sorted list of detections which applies non-maximum suppression on insertion.
*
* The detections in this list are sorted in descending order by their score. Whenever a new
* detection is to be inserted, its size will be compared to all detections before and after
* the position of the new one: If there is an overlapping detection with a higher score, the
* new one won't be inserted at all. Otherwise, all overlapping detections with a lower score
* will be removed from the list.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class MaximumDetectionList
{
public:

    typedef std::list<Detection>::iterator iterator;
    typedef std::list<Detection>::const_iterator const_iterator;

    MaximumDetectionList();
    MaximumDetectionList(unsigned int maxDetections);
    MaximumDetectionList(Scalar overlap_th);
    MaximumDetectionList(unsigned int maxDetections, Scalar overlap_th);
    MaximumDetectionList(const MaximumDetectionList & other);
    MaximumDetectionList(MaximumDetectionList && other);
    
    virtual ~MaximumDetectionList() {};
    
    MaximumDetectionList & operator=(const MaximumDetectionList & other);
    MaximumDetectionList & operator=(MaximumDetectionList && other);
    
    virtual bool insert(const Detection & detection);
    virtual bool insert(Detection && detection);
    
    virtual const_iterator erase(const_iterator pos);
    
    virtual void merge(MaximumDetectionList & other);
    virtual void merge(MaximumDetectionList && other);
    virtual void merge(std::vector<MaximumDetectionList>::iterator first, std::vector<MaximumDetectionList>::iterator last);
    
    const_iterator begin() const;
    const_iterator end() const;
    
    std::size_t size() const { return this->m_detections.size(); };
    bool empty() const { return this->m_detections.empty(); };
    
    /**
    * @return Returns the maximum number of detections maintained in this list.
    * A value of 0 indicates that there isn't any limit.
    */
    unsigned int getMaxSize() const { return this->m_maxDetections; };
    
    /**
    * @return Returns the current overlap threshold used for non-maximum suppression:
    * Intervals with an Intersection over Union (IoU) greater than this threshold will be considered overlapping.
    */
    Scalar getOverlapTh() const { return this->m_overlap_th; };


protected:

    std::list<Detection> m_detections;
    unsigned int m_maxDetections;
    Scalar m_overlap_th;
    
    virtual void nonMaximumSuppression();

};


/**
* @brief Abstract base class for strategies to search for anomalous intervals
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class SearchStrategy
{
public:

    bool autoReset; /**< Specifies whether to reset associated data structures after each call to `operator()` automatically to release the memory. */
    

    /**
    * Constructs a SearchStrategy with the default divergence measure and no pre-processing.
    */
    SearchStrategy();
    
    /**
    * Constructs a SearchStrategy with a given divergence measure and no pre-processing.
    *
    * @param[in] divergence The divergence measure used to compare a sub-block of the data with the remaining data.
    * Must not be `NULL`.
    */
    SearchStrategy(const std::shared_ptr<Divergence> & divergence);
    
    /**
    * Constructs a SearchStrategy with a given divergence measure and pre-processing pipeline.
    *
    * @param[in] divergence The divergence measure used to compare a sub-block of the data with the remaining data.
    * Must not be `NULL`.
    *
    * @param[in] preprocessing The pre-processing pipeline to be applied to the data before searching for anomalous sub-blocks.
    */
    SearchStrategy(const std::shared_ptr<Divergence> & divergence, const std::shared_ptr<const PreprocessingPipeline> & preprocessing);
    
    virtual ~SearchStrategy() {};
    
    /**
    * Searches for anomalous sub-blocks in a given DataTensor.
    *
    * @param[in] data The spatio-temporal data. This function may modify the given data (e.g. during pre-processing,
    * but there also is a const version).
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList operator()(const std::shared_ptr<DataTensor> & data, unsigned int numDetections = 0);
    
    /**
    * Searches for anomalous sub-blocks in a given DataTensor.
    *
    * @param[in] data The spatio-temporal data.
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList operator()(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections = 0);
    
    /**
    * @return Returns a pointer to the divergence measure used by this strategy to compare a sub-block of data with the remaining data.
    */
    const std::shared_ptr<Divergence> & getDivergence() const { return this->m_divergence; };
    
    /**
    * @return Returns a pointer to the pre-processing pipeline applied to the data before searching for anomalies. May be `NULL`.
    */
    const std::shared_ptr<const PreprocessingPipeline> & getPreprocessingPipeline() const { return this->m_preproc; };
    
    /**
    * @return Returns the current overlap threshold used for non-maximum suppression:
    * Intervals with an Intersection over Union (IoU) greater than this threshold will be considered overlapping.
    */
    Scalar getOverlapTh() const { return this->m_overlap_th; };
    
    /**
    * Changes the divergence measure used by this strategy to compare a sub-block of data with the remaining data.
    *
    * @param[in] divergence Pointer to the new divergence. Must not be `NULL`.
    */
    void setDivergence(const std::shared_ptr<Divergence> & divergence) { this->m_divergence = divergence; };
    
    /**
    * Changes the pre-processing pipeline applied to the data before searching for anomalies.
    *
    * @param[in] preprocessing Pointer to the new preprocessing pipeline. May be `NULL`.
    */
    void setPreprocessingPipeline(const std::shared_ptr<const PreprocessingPipeline> & preprocessing) { this->m_preproc = preprocessing; };
    
    /**
    * Changes the overlap threshold used for non-maximum suppression.
    *
    * @param[in] overlap_th Intervals with an Intersection over Union (IoU) greater than this threshold will be considered overlapping.
    * Must be in the range `[0,1]`.
    */
    void setOverlapTh(Scalar overlap_th) { if (overlap_th >= 0 and overlap_th <= 1) this->m_overlap_th = overlap_th; };


protected:

    std::shared_ptr<Divergence> m_divergence; /**< The divergence measure used to compare a sub-block of the data with the remaining data. */
    std::shared_ptr<const PreprocessingPipeline> m_preproc; /**< The pre-processing pipeline to be applied to the data before searching for anomalous sub-blocks. */
    Scalar m_overlap_th; /**< Overlap threshold for non-maximum suppression: Intervals with a greater IoU will be considered overlapping. */
    
    /**
    * Searches for anomalous sub-blocks in a given pre-processed DataTensor.
    *
    * This function will be called by `operator()` after pre-processing to perform the actual detection.
    *
    * @param[in] data The pre-processed spatio-temporal data.
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList detect(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections = 0) =0;

};


/**
* @brief Uses a proposal generator to search for anomalous sub-blocks
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ProposalSearch : public SearchStrategy
{
public:

    /**
    * Constructs a ProposalSearch with the default divergence measure, dense proposals and no pre-processing.
    */
    ProposalSearch();
    
    /**
    * Constructs a ProposalSearch with a given divergence measure, dense proposals and no pre-processing.
    *
    * @param[in] divergence The divergence measure used to compare a sub-block of the data with the remaining data.
    * Must not be `NULL`.
    */
    ProposalSearch(const std::shared_ptr<Divergence> & divergence);
    
    /**
    * Constructs a ProposalSearch with a given divergence measure and proposal generator, but without pre-processing.
    *
    * @param[in] divergence The divergence measure used to compare a sub-block of the data with the remaining data.
    * Must not be `NULL`.
    *
    * @param[in] generator The proposal generator to be used to retrieve a list of possibly anomalous ranges.
    * Must not be `NULL`.
    */
    ProposalSearch(const std::shared_ptr<Divergence> & divergence, const std::shared_ptr<ProposalGenerator> & generator);
    
    /**
    * Constructs a ProposalSearch with a given divergence measure, proposal generator and pre-processing pipeline.
    *
    * @param[in] divergence The divergence measure used to compare a sub-block of the data with the remaining data.
    *  Must not be `NULL`.
    *
    * @param[in] generator The proposal generator to be used to retrieve a list of possibly anomalous ranges.
    * Must not be `NULL`.
    *
    * @param[in] preprocessing The pre-processing pipeline to be applied to the data before searching for anomalous sub-blocks.
    */
    ProposalSearch(const std::shared_ptr<Divergence> & divergence,
                   const std::shared_ptr<ProposalGenerator> & generator,
                   const std::shared_ptr<const PreprocessingPipeline> & preprocessing);
    
    /**
    * @return Returns a pointer to the proposal generator.
    */
    const std::shared_ptr<ProposalGenerator> & getProposalGenerator() const { return this->m_proposals; };
    
    /**
    * Changes the proposal generator used to retrieve a list of possibly anomalous ranges.
    *
    * @param[in] generator The new proposal generator.
    */
    void setProposalGenerator(const std::shared_ptr<ProposalGenerator> & generator) { this->m_proposals = generator; };


protected:

    std::shared_ptr<ProposalGenerator> m_proposals; /**< The proposal generator to be used to retrieve a list of possibly anomalous ranges. */
    
    /**
    * Searches for anomalous sub-blocks in a given pre-processed DataTensor.
    *
    * This function will be called by `operator()` after pre-processing to perform the actual detection.
    *
    * @param[in] data The pre-processed spatio-temporal data.
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList detect(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections = 0) override;

};


/**
* Orders a list of detected ranges by their score in decreasing order and removes overlapping intervals with lower scores (non-maxima).
*
* @param[in,out] The list of detected ranges. Will be modified in-place.
*
* @param[in] numDetections The maximum number of detections to keep. `detections` will have at most this number of
* elements after applying this function. Set this to `0` to keep all maximum detections.
*
* @param[in] overlap_th Threshold for non-maximum suppression: Intervals with an Intersection over Union (IoU) greater than this
* threshold will be considered overlapping.
*/
void nonMaximumSuppression(DetectionList & detections, unsigned int numDetections = 0, double overlap_th = 0.0);

}

#endif