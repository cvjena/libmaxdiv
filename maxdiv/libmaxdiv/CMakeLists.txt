CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

PROJECT(maxdiv)

set(BUILD_VERSION "1.0")
set(API_VERSION 1)


#### Set C++ standard ####
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#### Build options ####
OPTION(MAXDIV_FLOAT "Use single precision." OFF)
SET(MAXDIV_KDE_CUMULATIVE_SIZE_LIMIT 20000 CACHE STRING "Limit on the number of samples which cumulative sums will be used for during in Kernel Density Estimation.")

IF(MAXDIV_FLOAT)
  ADD_DEFINITIONS(-DMAXDIV_FLOAT)
ENDIF()
ADD_DEFINITIONS(-DMAXDIV_KDE_CUMULATIVE_SIZE_LIMIT=${MAXDIV_KDE_CUMULATIVE_SIZE_LIMIT})

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()


#### Build shared library ####

SET(SOURCES utils.cc math_utils.cc pointwise_detectors.cc preproc.cc estimators.cc divergences.cc proposals.cc search_strategies.cc libmaxdiv.cc)
ADD_LIBRARY(maxdiv SHARED ${SOURCES})
SET_TARGET_PROPERTIES(maxdiv PROPERTIES VERSION ${BUILD_VERSION} SOVERSION ${API_VERSION})


### Build command line interface ###

ADD_EXECUTABLE(maxdiv_cli maxdiv.cc)
TARGET_LINK_LIBRARIES(maxdiv_cli maxdiv)


#### Detect and enable some useful compiler features ####

# OpenMP (not required, but strongly recommended on multi-core systems)
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()
