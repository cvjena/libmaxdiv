libmaxdiv
=========

Welcome to `libmaxdiv`, an efficient C++ implementation of the **Maximally Divergent Intervals Algorithm**
described in the following publication:

> Erik Rodner, Björn Barz, Yanira Guanche, Milan Flach, Miguel Mahecha, Paul Bodesheim, Markus Reichstein, Joachim Denzler.
> "Maximally Divergent Intervals for Anomaly Detection".
> *ICML Workshop on Anomaly Detection (ICML-WS)*. 2016

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