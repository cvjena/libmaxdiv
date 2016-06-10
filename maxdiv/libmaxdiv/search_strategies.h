#ifndef MAXDIV_SEARCH_STRATEGIES_H
#define MAXDIV_SEARCH_STRATEGIES_H

#include <memory>
#include <vector>
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
    virtual DetectionList operator()(const std::shared_ptr<DataTensor> & data, unsigned int numDetections = 0) =0;
    
    /**
    * Searches for anomalous sub-blocks in a given DataTensor.
    *
    * @param[in] data The spatio-temporal data.
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList operator()(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections = 0) =0;
    
    /**
    * @return Returns a pointer to the divergence measure used by this strategy to compare a sub-block of data with the remaining data.
    */
    const std::shared_ptr<Divergence> & getDivergence() const { return this->m_divergence; };
    
    /**
    * @return Returns a pointer to the pre-processing pipeline applied to the data before searching for anomalies. May be `NULL`.
    */
    const std::shared_ptr<const PreprocessingPipeline> & getPreprocessingPipeline() const { return this->m_preproc; };
    
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


protected:

    std::shared_ptr<Divergence> m_divergence; /**< The divergence measure used to compare a sub-block of the data with the remaining data. */
    std::shared_ptr<const PreprocessingPipeline> m_preproc; /**< The pre-processing pipeline to be applied to the data before searching for anomalous sub-blocks. */

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
    * Searches for anomalous sub-blocks in a given DataTensor.
    *
    * @param[in] data The spatio-temporal data. This function may modify the given data (e.g. during pre-processing,
    * but there also is a const version).
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList operator()(const std::shared_ptr<DataTensor> & data, unsigned int numDetections = 0) override;
    
    /**
    * Searches for anomalous sub-blocks in a given DataTensor.
    *
    * @param[in] data The spatio-temporal data.
    *
    * @param[in] numDetections Maximum number of detections to return. Set this to `0` to retrieve all detections.
    *
    * @return Returns a list of detected ranges, sorted by detection score in decreasing order.
    */
    virtual DetectionList operator()(const std::shared_ptr<const DataTensor> & data, unsigned int numDetections = 0) override;
    
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