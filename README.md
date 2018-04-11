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
evaluation of the algorithm; for example, detecting severe North Sea storms like the Hamburg-Flut shown below.
You don't need those directories if you just want to use `libmaxdiv`.

![Hamburg-Flut](https://cvjena.github.io/libmaxdiv/coastdat_det_aggregated_00%20(Hamburg-Flut).gif)

More examples of what can be done with the MDI algorithm can be found on the [project page](https://cvjena.github.io/libmaxdiv/)
and in the article mentioned above.


GUI
---

![GUI](https://user-images.githubusercontent.com/7915048/38293859-49f3eeb4-37e9-11e8-9b67-00c0f487ec01.png)


Dependencies of the Python implementation and the GUI
-----------------------------------------------------

- Python >= 2.7 or Python 3
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `PIL`/`Pillow`

`libmaxdiv` has its own dependencies in addition. Please refer to [`maxdiv/libmaxdiv/README.md`](maxdiv/libmaxdiv/README.md) for build instructions.


Getting started
---------------

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

### A simple step-by-step example for using the GUI

In this simple example, we will use the graphical user interface (GUI) for detecting hurricanes in a non-spatial time-series comprising measurements
of significant wave height (Hs), sea level pressure (SLP), and wind speed (W). The measurements have been taken in 2012 at a location near the
Bahamas and are provided by [NOAA](http://www.ndbc.noaa.gov/).

The hurricane season was particularly active in that year and 3 major hurricanes passed the Bahamas: Isaac (August 22-25), Rafael (October 12-18),
and Sandy (October 22-29).

We have already converted the data to the correct format for the libmaxdiv GUI and provide it in the file
[`experiments/HPW_2012_41046_standardized.csv`](experiments/HPW_2012_41046_standardized.csv).

First, we launch the user interface loading that file directly on start-up by running the following command on the command line:

    python launch-gui.py experiments/HPW_2012_41046_standardized.csv

You may also omit the argument and will then be prompted to select the file using a dialog window.

Since the measured variables have very different scales, it is a bit difficult to see anything in the plot of the time-series.
Thus, we first check the box labeled "Plot pre-processed instead of original data", which will show a normalized version of the time-series,
which is also used by `libmaxdiv`.

Since hurricanes usually last at least 12 hours and are considered two independent storms if they last longer than 3 days, we set the "minimum
interval length" to 12 and the "maximum interval length" to 72.
In order to keep the visualization clean and the analysis easy, we set the "number of detections" to 5.

After hitting the "Detect Anomalous Intervals" button, the time-series will be overlayed with the top 5 detections, shown as red regions.
The first three detections should correspond to the three hurricanes mentioned above.

You can use the buttons next to the visualization for zooming and panning or navigating between the individual detections.