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

/**
* @file
*
* Command line interface to libmaxdiv, the Maximally Divergent Intervals anomaly detector.
*
* Time-series data can be read from CSV files, but tempo-spatial data is not supported by the CLI yet.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#include "search_strategies.h"
#include "pointwise_detectors.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <getopt.h>

//#undef __STRICT_ANSI__
//#include <float.h>

using namespace MaxDiv;
using namespace std;
using namespace std::chrono;


enum maxdiv_divergence_t { MAXDIV_KL_DIVERGENCE, MAXDIV_JS_DIVERGENCE, MAXDIV_CROSS_ENTROPY };
enum maxdiv_estimator_t { MAXDIV_KDE, MAXDIV_GAUSSIAN, MAXDIV_ERPH };
enum maxdiv_proposal_generator_t { MAXDIV_DENSE_PROPOSALS, MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST, MAXDIV_POINTWISE_PROPOSALS_KDE };


void printHelp(const char *);


DetectionList apply_maxdiv(const std::shared_ptr<DataTensor> & data,
                           maxdiv_divergence_t divergence, maxdiv_estimator_t estimator, maxdiv_proposal_generator_t proposals,
                           KLDivergence::KLMode kl_mode, GaussianDensityEstimator::CovMode gauss_cov_mode,
                           unsigned int min_len, unsigned int max_len, unsigned int stride, unsigned int num_intervals, Scalar overlap_th,
                           Scalar kernel_sigma_sq, unsigned int num_hist, unsigned int num_bins, Scalar discount,
                           Scalar prop_th, bool prop_mad, bool prop_filter,
                           bool normalize, unsigned int td_embed, unsigned int td_lag, BorderPolicy borders,
                           bool zscore_deseas, unsigned int period_num, unsigned int period_len, bool linear_trend, bool linear_season_trend,
                           unsigned int pca_dims, unsigned int random_projection_dims)
{
    DetectionList detections;
    
    // Create density estimator
    std::shared_ptr<DensityEstimator> densityEstimator;
    switch (estimator)
    {
        case MAXDIV_KDE:
            densityEstimator = std::make_shared<KernelDensityEstimator>(kernel_sigma_sq);
            break;
        case MAXDIV_GAUSSIAN:
            densityEstimator = std::make_shared<GaussianDensityEstimator>(gauss_cov_mode);
            break;
        case MAXDIV_ERPH:
            densityEstimator = std::make_shared<EnsembleOfRandomProjectionHistograms>(num_hist, num_bins, discount);
            break;
        default:
            return detections;
    }
    
    // Create divergence
    std::shared_ptr<Divergence> div;
    switch (divergence)
    {
        case MAXDIV_KL_DIVERGENCE:
            div = std::make_shared<KLDivergence>(densityEstimator, kl_mode);
            break;
        case MAXDIV_JS_DIVERGENCE:
            div = std::make_shared<JSDivergence>(densityEstimator);
            break;
        case MAXDIV_CROSS_ENTROPY:
            div = std::make_shared<CrossEntropy>(densityEstimator, kl_mode);
            break;
        default:
            return detections;
    }
    
    // Create proposal generator
    std::shared_ptr<ProposalGenerator> proposal_gen;
    
    PointwiseProposalGenerator::Params ppParams;
    ppParams.gradientFilter = prop_filter;
    ppParams.mad = prop_mad;
    ppParams.sd_th = prop_th;
    
    switch (proposals)
    {
        case MAXDIV_DENSE_PROPOSALS:
            proposal_gen = std::make_shared<DenseProposalGenerator>(min_len, max_len, stride);
            break;
        
        case MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST:
            ppParams.scorer = &hotellings_t;
            proposal_gen = std::make_shared<PointwiseProposalGenerator>(min_len, max_len, ppParams);
            break;
        
        case MAXDIV_POINTWISE_PROPOSALS_KDE:
            ppParams.scorer = [](const DataTensor & data) { return pointwise_kde(data); };
            proposal_gen = std::make_shared<PointwiseProposalGenerator>(min_len, max_len, ppParams);
            break;
        
        default:
            return detections;
    }
    
    // Create pre-processing pipeline
    std::shared_ptr<PreprocessingPipeline> preproc = std::make_shared<PreprocessingPipeline>();
    
    if (normalize)
        preproc->push_back(std::make_shared<Normalizer>());
    
    if (period_num > 1 && period_len > 0 && !zscore_deseas)
        preproc->push_back(std::make_shared<OLSDetrending>(
            OLSDetrending::Period{period_num, period_len}, linear_trend, linear_season_trend
        ));
    else if (linear_trend)
        preproc->push_back(std::make_shared<LinearDetrending>());
    
    if (period_num > 1 && zscore_deseas)
        preproc->push_back(std::make_shared<ZScoreDeseasonalization>(period_num));
    
    if (td_embed != 1)
        preproc->push_back(std::make_shared<TimeDelayEmbedding>(td_embed, td_lag, borders));
    
    if (pca_dims > 0)
        preproc->push_back(std::make_shared<PCAProjection>(pca_dims));
    if (random_projection_dims > 0)
        preproc->push_back(std::make_shared<SparseRandomProjection>(random_projection_dims));
    
    // Put everything together and construct the SearchStrategy
    ProposalSearch detector(div, proposal_gen, preproc);
    detector.setOverlapTh(overlap_th);
    detections = detector(data, num_intervals);
    return detections;
}


int main(int argc, char * argv[])
{
    // Default parameters
    maxdiv_divergence_t divergence = MAXDIV_KL_DIVERGENCE;
    maxdiv_estimator_t estimator = MAXDIV_GAUSSIAN;
    maxdiv_proposal_generator_t proposals = MAXDIV_DENSE_PROPOSALS;
    KLDivergence::KLMode kl_mode = KLDivergence::KLMode::UNBIASED;
    GaussianDensityEstimator::CovMode gauss_cov_mode = GaussianDensityEstimator::CovMode::FULL;
    BorderPolicy borders = BorderPolicy::AUTO;
    unsigned int min_len = 0, max_len = 0, stride = 1, num_intervals = 0,
                 num_hist = 100, num_bins = 0,
                 td_embed = 1, td_lag = 1, period_num = 0, period_len = 1,
                 pca_dims = 0, random_projection_dims = 0,
                 first_row = 0, first_col = 0, last_col = -1;
    Scalar overlap_th = 0.0, kernel_sigma_sq = 1.0, discount = 1.0, prop_th = 1.5, missing_value = std::numeric_limits<Scalar>::quiet_NaN();
    int prop_mad = 0, prop_filter = 1, normalize = 0, zscore_deseas = 0, linear_trend = 0, linear_season_trend = 0, timing = 0;
    char delimiter = ',';
    
    // Parse options
    int c;
    char * conv_end;
    string argstr;
    while (c != -1)
    {
        static struct option long_options[] =
        {
            // General options
            {"help",                no_argument,        NULL,       'h'},
            {"min_len",             required_argument,  NULL,       'a'},
            {"max_len",             required_argument,  NULL,       'b'},
            {"stride",              required_argument,  NULL,       'S'},
            {"num",                 required_argument,  NULL,       'n'},
            {"overlap_th",          required_argument,  NULL,       'o'},
            {"timing",              no_argument,        &timing,      1},
            
            // Divergence and distribution model
            {"divergence",          required_argument,  NULL,       'd'},
            {"estimator",           required_argument,  NULL,       'e'},
            {"kernel_sigma_sq",     required_argument,  NULL,       's'},
            {"num_hist",            required_argument,  NULL,       'f'},
            {"num_bins",            required_argument,  NULL,       'g'},
            {"discount",            required_argument,  NULL,       'w'},
            
            // Proposals
            {"proposals",           required_argument,  NULL,       'p'},
            {"prop_th",             required_argument,  NULL,       'q'},
            {"prop_mad",            no_argument,        &prop_mad,    1},
            {"prop_unfiltered",     no_argument,        &prop_filter, 0},
            
            // Preprocessing
            {"normalize",           no_argument,        &normalize,   1},
            {"td",                  optional_argument,  NULL,       't'},
            {"td_lag",              required_argument,  NULL,       'l'},
            {"borders",             required_argument,  NULL,       'x'},
            {"period_num",          required_argument,  NULL,       'i'},
            {"period_len",          required_argument,  NULL,       'j'},
            {"zscore",              no_argument,        &zscore_deseas, 1},
            {"linear_trend",        no_argument,        &linear_trend, 1},
            {"linear_season_trend", no_argument,        &linear_season_trend, 1},
            {"pca_dims",            required_argument,  NULL,       'P'},
            {"random_projection_dims", required_argument, NULL,     'Q'},
            
            // Data format
            {"delimiter",           required_argument,  NULL,       'u'},
            {"first_row",           required_argument,  NULL,       'r'},
            {"first_col",           required_argument,  NULL,       'c'},
            {"last_col",            required_argument,  NULL,       'z'},
            {"missing_value",       required_argument,  NULL,       'm'},
            
            {0, 0, 0, 0}
        };
        
        int option_index = 0;
        c = getopt_long(argc, argv, "a:b:c:d:e:f:g:hi:j:l:m:n:o:p:q:r:st::u:w:x:z:P:Q:S:", long_options, &option_index);
        switch (c)
        {
            case 'h':
                printHelp(argv[0]);
                return 0;
            case 'a':
                min_len = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --min_len" << endl;
                    return 1;
                }
                break;
            case 'b':
                max_len = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --max_len" << endl;
                    return 1;
                }
                break;
            case 'S':
                stride = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --stride" << endl;
                    return 1;
                }
                break;
            case 'n':
                num_intervals = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --num" << endl;
                    return 1;
                }
                break;
            case 'o':
                overlap_th = strtod(optarg, &conv_end);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --overlap_th" << endl;
                    return 1;
                }
                break;
            case 'd':
                argstr = strtoupper(optarg);
                if (argstr == "KL_I_OMEGA")
                {
                    divergence = MAXDIV_KL_DIVERGENCE;
                    kl_mode = KLDivergence::KLMode::I_OMEGA;
                }
                else if (argstr == "KL_OMEGA_I")
                {
                    divergence = MAXDIV_KL_DIVERGENCE;
                    kl_mode = KLDivergence::KLMode::OMEGA_I;
                }
                else if (argstr == "KL_SYM")
                {
                    divergence = MAXDIV_KL_DIVERGENCE;
                    kl_mode = KLDivergence::KLMode::SYM;
                }
                else if (argstr == "KL_UNBIASED")
                {
                    divergence = MAXDIV_KL_DIVERGENCE;
                    kl_mode = KLDivergence::KLMode::UNBIASED;
                }
                else if (argstr == "JS")
                    divergence = MAXDIV_JS_DIVERGENCE;
                else if (argstr == "CROSSENT")
                {
                    divergence = MAXDIV_CROSS_ENTROPY;
                    kl_mode = KLDivergence::KLMode::I_OMEGA;
                }
                else if (argstr == "CROSSENT_TS")
                {
                    divergence = MAXDIV_CROSS_ENTROPY;
                    kl_mode = KLDivergence::KLMode::UNBIASED;
                }
                else
                {
                    cerr << "Unknown divergence: " << argstr << endl << "See --help for a list of possible values." << endl;
                    return 1;
                }
                break;
            case 'e':
                argstr = strtoupper(optarg);
                if (argstr == "GAUSSIAN" || argstr == "GAUSSIAN_COV")
                {
                    estimator = MAXDIV_GAUSSIAN;
                    gauss_cov_mode = GaussianDensityEstimator::CovMode::FULL;
                }
                else if (argstr == "GAUSSIAN_GLOBAL_COV")
                {
                    estimator = MAXDIV_GAUSSIAN;
                    gauss_cov_mode = GaussianDensityEstimator::CovMode::SHARED;
                }
                else if (argstr == "GAUSSIAN_ID_COV")
                {
                    estimator = MAXDIV_GAUSSIAN;
                    gauss_cov_mode = GaussianDensityEstimator::CovMode::ID;
                }
                else if (argstr == "PARZEN" || argstr == "KDE")
                    estimator = MAXDIV_KDE;
                else if (argstr == "ERPH")
                    estimator = MAXDIV_ERPH;
                else
                {
                    cerr << "Unknown estimator: " << argstr << endl << "See --help for a list of possible values." << endl;
                    return 1;
                }
                break;
            case 's':
                kernel_sigma_sq = strtod(optarg, NULL);
                if (kernel_sigma_sq <= 0)
                {
                    cerr << "Invalid value specified for option --kernel_sigma_sq" << endl;
                    return 1;
                }
                break;
            case 'f':
                num_hist = strtoul(optarg, NULL, 10);
                if (num_hist == 0)
                {
                    cerr << "Invalid value specified for option --num_hist" << endl;
                    return 1;
                }
                break;
            case 'g':
                num_bins = strtoul(optarg, NULL, 10);
                if (num_bins == 0)
                {
                    cerr << "Invalid value specified for option --num_bins" << endl;
                    return 1;
                }
                break;
            case 'w':
                discount = strtod(optarg, NULL);
                if (discount < 0)
                {
                    cerr << "Invalid value specified for option --discount" << endl;
                    return 1;
                }
                break;
            case 'p':
                argstr = strtolower(optarg);
                if (argstr == "dense")
                    proposals = MAXDIV_DENSE_PROPOSALS;
                else if (argstr == "hotellings_t")
                    proposals = MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST;
                else if (argstr == "kde")
                    proposals = MAXDIV_POINTWISE_PROPOSALS_KDE;
                else
                {
                    cerr << "Unknown proposal generator: " << argstr << endl << "See --help for a list of possible values." << endl;
                    return 1;
                }
                break;
            case 'q':
                prop_th = strtod(optarg, &conv_end);
                if (prop_th < 0 || conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --prop_th" << endl;
                    return 1;
                }
                break;
            case 't':
                if (optarg)
                {
                    td_embed = strtoul(optarg, NULL, 10);
                    if (td_embed == 0)
                    {
                        cerr << "Invalid value specified for option --td" << endl;
                        return 1;
                    }
                }
                else
                    td_embed = 0;
                break;
            case 'l':
                td_lag = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --td_lag" << endl;
                    return 1;
                }
                break;
            case 'x':
                argstr = strtolower(optarg);
                if (argstr == "constant")
                    borders = BorderPolicy::CONSTANT;
                else if (argstr == "mirror")
                    borders = BorderPolicy::MIRROR;
                else if (argstr == "valid")
                    borders = BorderPolicy::VALID;
                else if (argstr == "auto")
                    borders = BorderPolicy::AUTO;
                else
                {
                    cerr << "Unknown border policy: " << argstr << endl << "See --help for a list of possible values." << endl;
                    return 1;
                }
                break;
            case 'i':
                period_num = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --period_num" << endl;
                    return 1;
                }
                break;
            case 'j':
                period_len = strtoul(optarg, NULL, 10);
                if (period_len == 0)
                {
                    cerr << "Invalid value specified for option --period_len" << endl;
                    return 1;
                }
                break;
            case 'P':
                pca_dims = strtoul(optarg, NULL, 10);
                if (pca_dims == 0)
                {
                    cerr << "Invalid value specified for option --pca_dims" << endl;
                    return 1;
                }
                break;
            case 'Q':
                random_projection_dims = strtoul(optarg, NULL, 10);
                if (random_projection_dims == 0)
                {
                    cerr << "Invalid value specified for option --random_projection_dims" << endl;
                    return 1;
                }
                break;
            case 'u':
                argstr = optarg;
                if (argstr.size() != 1)
                {
                    cerr << "Invalid value specified for option --delimiter" << endl;
                    return 1;
                }
                delimiter = argstr[0];
                break;
            case 'r':
                first_row = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --first_row" << endl;
                    return 1;
                }
                break;
            case 'c':
                first_col = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --first_col" << endl;
                    return 1;
                }
                break;
            case 'z':
                last_col = strtoul(optarg, &conv_end, 10);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --last_col" << endl;
                    return 1;
                }
                break;
            case 'm':
                missing_value = strtod(optarg, &conv_end);
                if (conv_end == NULL || *conv_end != '\0')
                {
                    cerr << "Invalid value specified for option --missing_value" << endl;
                    return 1;
                }
                break;
        }
    }
    
    if (optind >= argc)
    {
        cerr << "No CSV file has been specified." << endl << endl;
        printHelp(argv[0]);
        return 1;
    }
    if (max_len > 0 && max_len < min_len)
    {
        cerr << "--max_len must not be less than --min_len." << endl;
        return 1;
    }
    if (last_col < first_col)
    {
        cerr << "--last_col must not be less than --first_col." << endl;
        return 1;
    }
    
    /*_clearfp();
    unsigned unused_current_word = 0;
    _controlfp_s(&unused_current_word, 0, _EM_OVERFLOW | _EM_UNDERFLOW | _EM_INVALID | _EM_ZERODIVIDE);*/
    
    Eigen::initParallel();
    
    // Read data
    shared_ptr<DataTensor> data = make_shared<DataTensor>(readDataFromCSV(argv[optind], delimiter, first_row, first_col, last_col));
    if (data->empty())
    {
        cerr << "Could not read file: " << argv[optind] << endl;
        return 2;
    }
    if (!std::isnan(missing_value))
        data->mask(missing_value);
    
    // Apply MaxDiv algorithm
    auto start = high_resolution_clock::now();
    DetectionList detections = apply_maxdiv(data, divergence, estimator, proposals, kl_mode, gauss_cov_mode,
                                            min_len, max_len, stride, num_intervals, overlap_th, kernel_sigma_sq, num_hist, num_bins, discount,
                                            prop_th, prop_mad, prop_filter,
                                            normalize, td_embed, td_lag, borders,
                                            zscore_deseas, period_num, period_len, linear_trend, linear_season_trend,
                                            pca_dims, random_projection_dims);
    auto stop = high_resolution_clock::now();
    if (timing)
        cerr << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    
    // Print detections
    for (const Detection & det : detections)
        cout << det.a.t << "," << det.b.t << "," << det.score << endl;
    
    return 0;
}


void printHelp(const char * progName)
{
    cout << progName << " [options] <csv-file>" << endl
         << endl
         << "Searches for Maximally Divergent Intervals in multivariate time-series." << endl
         << "Spatio-temporal data is supported by libmaxdiv, but not by this command line interface." << endl
         << endl
         << "The detections will be written to stdout, one detection per line, where each line" << endl
         << "consists of 3 comma-separated values: The first point in the range, the first point" << endl
         << "after the end of the range and the detection score. The results will be sorted in" << endl
         << "decreasing order by their scores." << endl
         << endl
         << "OPTIONS:" << endl
         << endl
         << "    --help, -h" << endl
         << "        Print this message." << endl
         << endl
         << "    --min_len <int>, -a <int>" << endl
         << "        Minimum length of intervals to be taken into account." << endl
         << endl
         << "    --max_len <int>, -b <int>" << endl
         << "        Maximum length of intervals to be taken into account." << endl
         << endl
         << "    --stride <int>, -S <int> (default: 1)" << endl
         << "        Step size between two intervals to be taken into account (only applicable to dense proposals)." << endl
         << endl
         << "    --num <int>, -n <int>" << endl
         << "        Maximum number of detections to be returned." << endl
         << endl
         << "    --overlap_th <float>, -o <float> (default: 0.0)" << endl
         << "        Overlap threshold for non-maximum suppression: Intervals with a greater IoU will" << endl
         << "        be considered overlapping." << endl
         << endl
         << "    --timing" << endl
         << "        Print the time taken by the algorithm to stderr." << endl
         << endl
         << "    --divergence <str>, -d <str> (default: KL_UNBIASED)" << endl
         << "        Divergence measure. One of: KL_I_OMEGA, KL_OMEGA_I, KL_SYM, KL_UNBIASED, JS, CROSSENT, CROSSENT_TS" << endl
         << endl
         << "    --estimator <str>, -e <str> (default: GAUSSIAN)" << endl
         << "        Distribution model. One of: GAUSSIAN, GAUSSIAN_GLOBAL_COV, GAUSSIAN_ID_COV, PARZEN, ERPH" << endl
         << endl
         << "    --kernel_sigma_sq <float>, -s <float> (default: 1.0)" << endl
         << "        The variance of the Gaussian kernel used by the PARZEN density estimator." << endl
         << endl
         << "    --num_hist <int>, -f <int> (default: 100)" << endl
         << "        The number of histograms used by the ERPH estimator." << endl
         << endl
         << "    --num_bins <int>, -g <int>" << endl
         << "        The number of bins in the histograms used by the ERPH estimator. If not specified, an" << endl
         << "        individual number of bins will be determined automatically for each histogram." << endl
         << endl
         << "    --discount <float>, -w <float> (default: 1.0)" << endl
         << "        Discount added to all bins of the histograms of the ERPH estimator in order to make" << endl
         << "        unseen values not completely unlikely." << endl
         << endl
         << "    --proposals <str>, -p <str> (default: DENSE)" << endl
         << "        Interval proposal generation method. One of: DENSE, HOTELLINGS_T, KDE" << endl
         << endl
         << "    --prop_th <float>, -q <float> (default: 1.5)" << endl
         << "        Threshold for the point-wise proposal generators." << endl
         << endl
         << "    --prop_mad" << endl
         << "        Use the Median Absolute Deviation From Median as robust estimate for the" << endl
         << "        standard deviation for point-wise proposal generators." << endl
         << endl
         << "    --prop_unfiltered" << endl
         << "        Do not apply a gradient filter to the raw scores for point-wise proposal generation." << endl
         << endl
         << "    --normalize" << endl
         << "        Normalize the values in the time-series." << endl
         << endl
         << "    --td=[<int>], -t" << endl
         << "        Apply time-delay embedding to the time-series with the given embedding dimension." << endl
         << "        If no argument is given, an embedding dimension will be determined automatically." << endl
         << endl
         << "    --td_lag <int>, -l <int> (default: 1)" << endl
         << "        Distance between time steps for time-delay embedding. May be set to 0 for automatic" << endl
         << "        determination." << endl
         << endl
         << "    --borders <str>, -x <str> (default: AUTO)" << endl
         << "        Policy to be applied at the beginning of the time-series when performing time-delay" << endl
         << "        embedding. One of: CONSTANT, MIRROR, VALID, AUTO" << endl
         << endl
         << "    --period_num <int>, -i <int>" << endl
         << "        Apply deseasonalization by Ordinary Least Squares estimation assuming the given" << endl
         << "        number of seasonal units. Z-Score deseasonalization will be used instead if" << endl
         << "        --zscore is specified." << endl
         << endl
         << "    --period_len <int>, -j <int> (default: 1)" << endl
         << "        The length of each seasonal unit for OLS deseasonalization." << endl
         << endl
         << "    --zscore" << endl
         << "        Use Z-Score deseasonalization instead of OLS." << endl
         << endl
         << "    --linear_trend" << endl
         << "        Fit a line to the data to remove a simple linear trend from the data." << endl
         << endl
         << "    --linear_season_trend" << endl
         << "        Allow the seasonal coefficients of OLS deseasonalization to change linearly over time." << endl
         << endl
         << "    --pca_dims <int>, -P <int>" << endl
         << "        Reduce the data to the specified number of dimensions using PCA." << endl
         << endl
         << "    --random_projection_dims <int>, -Q <int>" << endl
         << "        Project the data onto the specified number of sparse random projection vectors." << endl
         << endl
         << "    --delimiter <char>, -u <char> (default: ,)" << endl
         << "        Delimiter used to separate the fields in the given CSV file." << endl
         << endl
         << "    --first_row <int>, -r <int> (default: 0)" << endl
         << "        The first row in the CSV file to be read." << endl
         << endl
         << "    --first_col <int>, -c <int> (default: 0)" << endl
         << "        The first column in the CSV file to be read." << endl
         << endl
         << "    --last_col <int>, -z <int>" << endl
         << "        The last column in the CSV file to be read." << endl
         << endl
         << "    --missing_value <float>, -m <float>" << endl
         << "        If missing values in the CSV file are not encoded as 'nan', but as a special floating" << endl
         << "        point value, that missing value may be specified using this option." << endl
         << endl;
}
