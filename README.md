Maximally Divergent Intervals for Anomaly Detection
===================================================

The **Maximally Divergent Intervals (MDI) Algorithm** can be used to detect anomalous *intervals* (as opposed to
anomalous points) in multi-variate spatio-temporal time-series. A description of the algorithm along with a variety
of application examples can be found in the following article:

> Björn Barz, Erik Rodner, Yanira Guanche Garcia, Joachim Denzler.  
> "Detecting Regions of Maximal Divergence for Spatio-Temporal Anomaly Detection."  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence* (accepted). 2018.

An efficient C++ implementation called `libmaxdiv` is provided in `maxdiv/libmaxdiv` and may be used stand-alone. If it has been
built in `maxdiv/libmaxdiv/bin`, it will be used automatically by the GUI and the `maxdiv` function in the `maxdiv.maxdiv` Python
package. See `maxdiv/libmaxdiv/README.md` for build instructions.  
Otherwise, the pure Python implementation of the MDI algorithm will be used, which is not recommended, since it is extremely slow
and lacks some features such as support for spatial data.

The directories `experiments` and `tools` contain some benchmarks, experiments and scripts we've been using for development and
evaluation of the algorithm. You don't need them if you just want to use it.


Dependencies of the Python implementation and the GUI
-----------------------------------------------------

- Python >= 2.7 or Python 3
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `PIL`/`Pillow`

`libmaxdiv` has its own dependencies in addition. Please refer to `maxdiv/libmaxdiv/README.md`.


Get started
-----------

After having installed the dependencies mentioned above and built `libmaxdiv` according to the instructions in `maxdiv/libmaxdiv/README.md`,
just run `python launch-gui.py` to start the interactive interface.

A comprehensive installation guide and user manual for the GUI can be found in `libmaxdiv user guide.pdf`.

You may also use the maxdiv algorithm programmatically via the `maxdiv` function in the `maxdiv.maxdiv` package or by using the `libmaxdiv`
library directly from your application. It provides a C-style procedural interface defined in `libmaxdiv.h` for maximum inter-operability.