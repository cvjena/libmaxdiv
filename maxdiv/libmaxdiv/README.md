libmaxdiv
=========

Welcome to `libmaxdiv`, an efficient C++ implementation of the **Maximally Divergent Intervals Algorithm**
described in the following publication:

> Björn Barz, Erik Rodner, Yanira Guanche Garcia, Joachim Denzler.  
> "Detecting Regions of Maximal Divergence for Spatio-Temporal Anomaly Detection."  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence* (accepted). 2018.

`libmaxdiv` supports spatio-temporal data, is easily extendable and provides a `C` interface for a
maximum of interoperability.


Dependencies
------------

- A C++ compiler capable of C++11 and OpenMP 4.0 (e.g. g++ >= 4.9)
- [Eigen](http://eigen.tuxfamily.org/) >= 3.2.7
- [CMake](https://cmake.org/) >= 3.1


Building
--------

Inside the `libmaxdiv` directory, execute the following commands:

    mkdir bin
    cd bin
    cmake ..
    make

Enjoy!


Pre-built Windows Binaries
--------------------------

We provide pre-built binaries for MS Windows (both 32 and 64 bit) in the [`win32`](..\..\win32) directory.