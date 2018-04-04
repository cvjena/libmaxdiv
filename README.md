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
package. See [`maxdiv/libmaxdiv/README.md`](maxdiv/libmaxdiv/README.md) for build instructions.  
Otherwise, the pure Python implementation of the MDI algorithm will be used, which is not recommended, since it is extremely slow
and lacks some features such as support for spatial data.

The directories `experiments` and `tools` contain some benchmarks, experiments and scripts we've been using for development and
evaluation of the algorithm. You don't need them if you just want to use it.

Some examples of what can be done with the MDI algorithm can be found on the [project page](https://cvjena.github.io/libmaxdiv/).


Dependencies of the Python implementation and the GUI
-----------------------------------------------------

- Python >= 2.7 or Python 3
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `PIL`/`Pillow`

`libmaxdiv` has its own dependencies in addition. Please refer to [`maxdiv/libmaxdiv/README.md`](maxdiv/libmaxdiv/README.md) for build instructions.


Get started
-----------

After having installed the dependencies mentioned above and built `libmaxdiv` according to the instructions in `maxdiv/libmaxdiv/README.md`,
just run `python launch-gui.py` to start the interactive interface.

A comprehensive installation guide and user manual for the GUI can be found in [`libmaxdiv user guide.pdf`](libmaxdiv%20user%20guide.pdf).

You may also use the maxdiv algorithm programmatically via the [`maxdiv`](maxdiv/maxdiv.py#L733) function in the [`maxdiv.maxdiv`](maxdiv/maxdiv.py)
package or by using the `libmaxdiv` library directly from your application. It provides a C-style procedural interface defined in
[`libmaxdiv.h`](maxdiv/libmaxdiv/libmaxdiv.h) for maximum inter-operability.

Note that not all functions of `libmaxdiv` are made available through the high-level interface `maxdiv` python function in the `maxdiv.maxdiv` package.
In particular, it can only process temporal, but not spatio-temporal data.
However, you can still use python to interact with `libmaxdiv` by calling the C-style functions defined in [`libmaxdiv.h`](maxdiv/libmaxdiv/libmaxdiv.h)
from python using the wrapper provided in [`maxdiv.libmaxdiv_wrapper`](maxdiv/libmaxdiv_wrapper.py).