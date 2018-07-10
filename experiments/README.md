Experiments
===========

This directory contains several experiments conducted using the MDI algorithm.

- [`hpw.py`](hpw.py) runs MDI on the meteocean data in `HPW_2012_41046.csv` recorded in 2012
  at a location near the Bahamas and compare the detected intervals with the three
  major hurricanes in that year.  
  This data may also be explored interactively in the GUI by running
  
      python launch-gui.py experiments/HPW_2012_41046_standardized.csv
  
  from the root directory. Make sure to check "Plot pre-processed instead of original data" and set the minimum and maximum length of the intervals to be detected to 12 and 72, respectively.

- [`coastdat`](coastdat/): North Sea Storm detection described in section 4.1 of the paper.

- [`SLP`](SLP/): Detection of low-pressure fields in spatio-temporal data (section 4.2 of the paper).

- [`NLP`](NLP/): Detection of stylistic anomalies in texts of natural language (section 4.3 of the paper).

- [`VideoDetection`](VideoDetection/): Anomaly detection in videos (section 4.4 of the paper).

- [`synthetic`](synthetic/): Quantitative evaluation on synthetic data (section 3 of the paper).