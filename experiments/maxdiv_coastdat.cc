// g++ --std=c++11 -O3 -Wall -I../maxdiv/libmaxdiv -I/home/barz/lib/eigen-3.2.8 -I/home/barz/lib/anaconda3/include -L/home/barz/lib/anaconda3/lib -L../maxdiv/libmaxdiv/bin -Wl,-rpath,/home/barz/lib/anaconda3/lib,-rpath,/home/barz/anomaly-detection/extreme-interval-detection/maxdiv/libmaxdiv/bin -shared -fPIC -o maxdiv_coastdat.so maxdiv_coastdat.cc -lmaxdiv -lnetcdf -fopenmp

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <netcdf.h>
#include "libmaxdiv.h"
#include "DataTensor.h"
#include "preproc.h"
#include "utils.h"
using MaxDiv::DataTensor;
using MaxDiv::ReflessIndexVector;


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


void coastdat_deseasonalize(DataTensor & coastData, coastdat_deseasonalization_method_t method)
{
    if (method != COASTDAT_DESEAS_NONE)
    {
        auto start = std::chrono::high_resolution_clock::now();
        switch (method)
        {
            case COASTDAT_DESEAS_OLS_DAY:
                MaxDiv::OLSDetrending(24, false)(coastData, coastData);
                break;
            case COASTDAT_DESEAS_OLS_YEAR:
                MaxDiv::OLSDetrending(MaxDiv::OLSDetrending::PeriodVector{ {24, 1}, {365, 24} }, false)(coastData, coastData);
                break;
            case COASTDAT_DESEAS_ZSCORE_DAY:
                MaxDiv::ZScoreDeseasonalization(24)(coastData);
                break;
            case COASTDAT_DESEAS_ZSCORE_YEAR:
                MaxDiv::ZScoreDeseasonalization(24*365)(coastData);
                break;
            default:
                break;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::cerr << "Deseasonalization took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
                  << " s." << std::endl;
    }
}


int read_coastdat(const coastdat_params_t * data_params, DataTensor & coastData)
{
    int status, ncid, var_id, dim_id;
    char filename[512];
    
    // Check parameters
    if (data_params == NULL || data_params->spatialPoolingSize < 1)
        return 1;
    std::vector<std::string> variables;
    if (MaxDiv::splitString(MaxDiv::strtolower(data_params->variables), ",; ", variables) < 1)
        return 1;
    unsigned int firstYear = (data_params->firstYear >= COASTDAT_FIRST_YEAR) ? data_params->firstYear - COASTDAT_FIRST_YEAR + 1 : data_params->firstYear;
    unsigned int lastYear = (data_params->lastYear >= COASTDAT_FIRST_YEAR) ? data_params->lastYear - COASTDAT_FIRST_YEAR + 1 : data_params->lastYear;
    if (firstYear < 1 || lastYear < 1 || lastYear < firstYear || lastYear - firstYear + 1 > COASTDAT_NUM_YEARS)
        return 1;
    
    // Determine number of time steps
    std::size_t dim_len;
    ReflessIndexVector shape;
    shape.t = 0;
    shape.x = ceil((data_params->lastLon - data_params->firstLon + 1) / static_cast<float>(data_params->spatialPoolingSize));;
    shape.y = ceil((data_params->lastLat - data_params->firstLat + 1) / static_cast<float>(data_params->spatialPoolingSize));
    shape.z = 1;
    shape.d = variables.size();
    for (unsigned int year = firstYear; year <= lastYear; ++year)
    {
        // Open NetCDF file
        sprintf(filename, COASTDAT_PATH "%s/coastDat-1_Waves_%s_%03u.nc", variables[0].c_str(), variables[0].c_str(), year);
        status = nc_open(filename, 0, &ncid);
        if (status != NC_NOERR) return status;
        
        // Get handle to variable
        status = nc_inq_varid(ncid, variables[0].c_str(), &var_id);
        if (status != NC_NOERR) return status;
        
        // Query length of time dimension
        status = nc_inq_dimid(ncid, "time", &dim_id);
        if (status != NC_NOERR) return status;
        status = nc_inq_dimlen(ncid, dim_id, &dim_len);
        if (status != NC_NOERR) return status;
        shape.t += dim_len;
        
        nc_close(ncid);
    }
    
    std::cerr << "Data shape: " << shape.t << " x " << shape.x << " x " << shape.y << " x " << shape.z << " x " << shape.d << std::endl;
    std::cerr << "Memory usage: " << static_cast<float>(shape.prod() * sizeof(MaxDiv::Scalar)) / (1 << 30) << " GiB" << std::endl;
    
    // Read data
    coastData.resize(shape);
    DataTensor buffer;
    std::size_t dataStart[] = { 0, data_params->firstLat, data_params->firstLon };
    std::size_t dataLength[] = { 0, data_params->lastLat - data_params->firstLat + 1, data_params->lastLon - data_params->firstLon + 1 };
    DataTensor::Index timeOffset = 0;
    for (unsigned int year = firstYear; year <= lastYear; ++year)
    {
        for (DataTensor::Index d = 0; d < variables.size(); ++d)
        {
            // Open NetCDF file
            sprintf(filename, COASTDAT_PATH "%s/coastDat-1_Waves_%s_%03u.nc", variables[d].c_str(), variables[d].c_str(), year);
            std::cerr << "Reading " << filename << std::endl;
            status = nc_open(filename, 0, &ncid);
            if (status != NC_NOERR) return status;
            
            // Get handle to variable
            status = nc_inq_varid(ncid, variables[d].c_str(), &var_id);
            if (status != NC_NOERR) return status;
            
            // Query length of time dimension
            status = nc_inq_dimid(ncid, "time", &dim_id);
            if (status != NC_NOERR) return status;
            status = nc_inq_dimlen(ncid, dim_id, &dim_len);
            if (status != NC_NOERR) return status;
            dataLength[0] = dim_len;
            
            // Read block from NetCDF file
            buffer.resize({ dataLength[0], dataLength[1], dataLength[2], 1, 1 });
            #ifdef MAXDIV_FLOAT
            status = nc_get_vara_float(ncid, var_id, dataStart, dataLength, buffer.raw());
            #else
            status = nc_get_vara_double(ncid, var_id, dataStart, dataLength, buffer.raw());
            #endif
            if (status != NC_NOERR) return status;

            nc_close(ncid);

            // Average Pooling (and swapping of Lat/Lon)
            for (DataTensor::Index t = 0; t < dim_len; ++t)
            {
                DataTensor::ConstScalarMatrixMap timestep(buffer.raw() + t * buffer.width() * buffer.height(), buffer.width(), buffer.height(), Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(buffer.height(), 1));
                for (DataTensor::Index x = 0; x < shape.x; ++x)
                    for (DataTensor::Index y = 0; y < shape.y; ++y)
                    {
                        // Note that latitude is mapped to the y-axis in `coastData`,
                        // but to the x-axis in `buffer`.
                        DataTensor::Index firstX = y * data_params->spatialPoolingSize;
                        DataTensor::Index firstY = x * data_params->spatialPoolingSize;
                        coastData({ timeOffset + t, x, y, 0, d }) = timestep.block(
                            firstX,
                            firstY,
                            std::min(static_cast<DataTensor::Index>(data_params->spatialPoolingSize), buffer.width() - firstX),
                            std::min(static_cast<DataTensor::Index>(data_params->spatialPoolingSize), buffer.height() - firstY)
                        ).mean();
                    }
            }
        }
        timeOffset += dim_len;
    }
    buffer.release();
    
    // Deseasonalization
    coastdat_deseasonalize(coastData, data_params->deseasonalization);
    
    return 0;
}


void coastdat_default_params(coastdat_params_t * data_params)
{
    if (data_params == NULL)
        return;
    
    data_params->variables = "ff,hs,tp";
    data_params->firstYear = 1;
    data_params->lastYear = 50;
    data_params->firstLat = 53;
    data_params->lastLat = 100;
    data_params->firstLon = 30;
    data_params->lastLon = 98;
    data_params->spatialPoolingSize = 4;
    data_params->deseasonalization = COASTDAT_DESEAS_NONE;
}


int coastdat_dump(const coastdat_params_t * data_params, const char * dump_file)
{
    // Load data
    DataTensor coastData;
    int status = read_coastdat(data_params, coastData);
    if (status != 0) return status;
    
    // Open dump file
    std::ofstream dump(dump_file, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if (!dump.is_open())
        return 2;
    
    // Write shape of the data tensor
    dump.write(reinterpret_cast<const char*>(coastData.shape().ind), sizeof(DataTensor::Index) * MAXDIV_INDEX_DIMENSION);
    
    // Write data
    dump.write(reinterpret_cast<const char*>(coastData.raw()), sizeof(MaxDiv::Scalar) * coastData.numEl());
    
    // Clean up
    status = (dump.good()) ? 0 : 3;
    dump.close();
    return status;
}


bool read_coastdat_dump(const char * dump_file, DataTensor & coastData)
{
    // Open dump file
    std::ifstream dump(dump_file, std::ios_base::in | std::ios_base::binary);
    if (!dump.is_open())
        return false;
    
    // Read shape of the data
    MaxDiv::ReflessIndexVector shape;
    dump.read(reinterpret_cast<char*>(shape.ind), sizeof(DataTensor::Index) * MAXDIV_INDEX_DIMENSION);
    coastData.resize(shape);
    
    // Read data
    dump.read(reinterpret_cast<char*>(coastData.raw()), sizeof(MaxDiv::Scalar) * coastData.numEl());
    
    bool status = !dump.fail();
    dump.close();
    return status;
}


int coastdat_maxdiv(const maxdiv_params_t * params, const coastdat_params_t * data_params,
                    detection_t * detection_buf, unsigned int * detection_buf_size)
{
    if (params == NULL)
        return 1;
    
    // Read dataset
    DataTensor coastData;
    int status = read_coastdat(data_params, coastData);
    if (status != 0) return status;
    
    // Apply MaxDiv algorithm
    unsigned int shape[MAXDIV_INDEX_DIMENSION];
    for (int d = 0; d < MAXDIV_INDEX_DIMENSION; ++d)
        shape[d] = coastData.shape().ind[d];
    auto start = std::chrono::high_resolution_clock::now();
    maxdiv(params, coastData.raw(), shape, detection_buf, detection_buf_size, false);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << "MaxDiv algorithm took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
              << " s." << std::endl;
    
    return 0;
}


int coastdat_maxdiv_dump(const maxdiv_params_t * params, const char * dump_file,
                         detection_t * detection_buf, unsigned int * detection_buf_size)
{
    if (params == NULL)
        return 1;
    
    // Read dataset
    DataTensor coastData;
    if (!read_coastdat_dump(dump_file, coastData))
        return -1;
    
    // Apply MaxDiv algorithm
    unsigned int shape[MAXDIV_INDEX_DIMENSION];
    for (int d = 0; d < MAXDIV_INDEX_DIMENSION; ++d)
        shape[d] = coastData.shape().ind[d];
    auto start = std::chrono::high_resolution_clock::now();
    maxdiv(params, coastData.raw(), shape, detection_buf, detection_buf_size, false);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << "MaxDiv algorithm took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
              << " s." << std::endl;
    
    return 0;
}


int coastdat_context_window_size(const coastdat_params_t * data_params)
{
    DataTensor coastData;
    if (read_coastdat(data_params, coastData) != 0)
        return 0;
    
    return MaxDiv::TimeDelayEmbedding().determineContextWindowSize(coastData);
}

int coastdat_context_window_size_dump(const char * dump_file, coastdat_deseasonalization_method_t deseasonalization)
{
    DataTensor coastData;
    if (!read_coastdat_dump(dump_file, coastData))
        return 0;

    coastdat_deseasonalize(coastData, deseasonalization);
    
    return MaxDiv::TimeDelayEmbedding().determineContextWindowSize(coastData);
}
