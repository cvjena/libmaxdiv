#include "libmaxdiv.h"


#define COASTDAT_PATH "/home/barz/anomaly-detection/CoastDat-raw/"
#define COASTDAT_FIRST_YEAR 1958
#define COASTDAT_NUM_YEARS 50


extern "C"
{

enum coastdat_deseasonalization_method_t
{
    COASTDAT_DESEAS_NONE,       /**< No deseasonalization. */
    COASTDAT_DESEAS_OLS_DAY,    /**< OLS deseasonalization with a period of 24 hours. */
    COASTDAT_DESEAS_OLS_YEAR,   /**< OLS deseasonalization with a period of 24 hours and 365 days. */
    COASTDAT_DESEAS_ZSCORE_DAY, /**< Z Score deseasonalization with a period of 24 hours. */
    COASTDAT_DESEAS_ZSCORE_YEAR /**< Z Score deseasonalization with a period of 24 * 365 hours. */
};

typedef struct {
    const char * variables; /**< Comma-separated list of the variables to be read. Available variables are: dd, ds, ff, hs, mp, tm1, tm2, tp, wd */
    unsigned int firstYear; /**< First year to include in the data (ranging from 1958 to 2007 or from 1 to 50). */
    unsigned int lastYear; /**< Last year to include in the data (ranging from 1958 to 2007 or from 1 to 50). */
    unsigned int firstLat; /**< Index of the first latitude to include in the data. */
    unsigned int lastLat; /**< Index of the last latitude to include in the data. */
    unsigned int firstLon; /**< Index of the first longitude to include in the data. */
    unsigned int lastLon; /**< Index of the last longitude to include in the data. */
    unsigned int spatialPoolingSize; /**< Number of spatial cells to be aggregated. */
    coastdat_deseasonalization_method_t deseasonalization; /**< Deseasonalization method to be applied after loading the data. */
} coastdat_params_t;

/**
* Initializes a given `coastdat_params_t` structure with the default parameters.
*
* @param[out] data_params Pointer to the parameter structure to be set to the default parameters.
*/
void coastdat_default_params(coastdat_params_t * data_params);

/**
* Loads data from the CoastDat data set and dumps it into a binary file, which can be used for
* faster re-loading of the data than from the NetCDF files.
*
* @param[in] data_params Pointer to a structure specifying the portion of the data set to be read.
* The default parameters can be retrieved by calling `maxdiv_coastdat_default_params()`.
*
* @param[in] dump_file Path of the file to write the data to.
*
* @return Returns 0 on success, a negative error code obtained from libnetcdf if the data could not be read
* or a positive error code if the dump could not be created.
*/
int coastdat_dump(const coastdat_params_t * data_params, const char * dump_file);

/**
* Loads data from the CoastDat data set and applies the MaxDiv anomaly detection algorithm to it.
*
* @param[in] params Pointer to a structure with the parameters for the algorithm.
*
* @param[in] data_params Pointer to a structure specifying the portion of the data set to be read.
* The default parameters can be retrieved by calling `maxdiv_coastdat_default_params()`.
*
* @param[out] detection_buf Pointer to a buffer where the detected sub-blocks will be stored.
*
* @param[in,out] detection_buf_size Pointer to the number of elements allocated for `detection_buf`. The integer
* pointed to will be set to the actual number of elements written to the buffer.
*
* @return Returns 0 on success, a negative error code obtained from libnetcdf if the data could not be read
* or a positive error code if an internal error occurred.
*/
int coastdat_maxdiv(const maxdiv_params_t * params, const coastdat_params_t * data_params,
                    detection_t * detection_buf, unsigned int * detection_buf_size);

/**
* Loads a binary dump of the CoastDat data set and applies the MaxDiv anomaly detection algorithm to it.
*
* @param[in] params Pointer to a structure with the parameters for the algorithm.
*
* @param[in] dump_file Path to the dump of the data created by `coastdat_dump()`.
*
* @param[out] detection_buf Pointer to a buffer where the detected sub-blocks will be stored.
*
* @param[in,out] detection_buf_size Pointer to the number of elements allocated for `detection_buf`. The integer
* pointed to will be set to the actual number of elements written to the buffer.
*
* @return Returns 0 on success, -1 if the dump could not be read or a positive error code if an internal error occurred.
*/
int coastdat_maxdiv_dump(const maxdiv_params_t * params, const char * dump_file,
                         detection_t * detection_buf, unsigned int * detection_buf_size);

/**
* Determines the size of window of relevant context for a given portion of the CoastDat data set.
*
* @param[in] data_params Pointer to a structure specifying the portion of the data set to be read.
* The default parameters can be retrieved by calling `maxdiv_coastdat_default_params()`.
*
* @return Returns the context window size
*
* @see MaxDiv::TimeDelayEmbedding::determineContextWindowSize
*/
int coastdat_context_window_size(const coastdat_params_t * data_params);

/**
* Determines the size of window of relevant context for a given portion of the CoastDat data set
* read from a dump.
*
* @param[in] dump_file Path to the dump of the data created by `coastdat_dump()`.
*
* @param[in] deseasonalization Deseasonalization method to be applied to the data.
*
* @return Returns the context window size
*
* @see MaxDiv::TimeDelayEmbedding::determineContextWindowSize
*/
int coastdat_context_window_size_dump(const char * dump_file, coastdat_deseasonalization_method_t deseasonalization = COASTDAT_DESEAS_NONE);

};

