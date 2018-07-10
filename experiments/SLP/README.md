Video Anomaly Detection Experiment
==================================

This directory contains code used to produce the results reported in section 4.2 of the paper.


Detecting Anomalies
-------------------

The python script `detect_slp.py` can be used to detect anomalies in the data provided in `SLP_ATL.mat`.

Using the default configuration and simply running `python detect_slp.py` should result in the following detections:

    #0
    TIMEFRAME: 1996-01-06 - 1996-01-15
    LOCATION:  40.00 N, -52.50 E - 65.00 N, -2.50 E
    SCORE:     5940.691491412244

    #1
    TIMEFRAME: 1990-01-28 - 1990-02-06
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 7.50 E
    SCORE:     5551.022030785701
    IDENT:     Storm Herta (Feb 01-05)

    #2
    TIMEFRAME: 1989-12-22 - 1989-12-31
    LOCATION:  45.00 N, -52.50 E - 65.00 N, -2.50 E
    SCORE:     5198.512847864493

    #3
    TIMEFRAME: 2009-01-18 - 2009-01-27
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4959.828578949225
    IDENT:     Cyclone Joris (Jan 23)

    #4
    TIMEFRAME: 1982-12-14 - 1982-12-23
    LOCATION:  50.00 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4811.574708018882

    #5
    TIMEFRAME: 1990-12-25 - 1991-01-03
    LOCATION:  52.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4703.992643858513
    IDENT:     Storm Undine (Jan 02-09)

    #6
    TIMEFRAME: 1974-01-03 - 1974-01-12
    LOCATION:  47.50 N, -52.50 E - 65.00 N, -5.00 E
    SCORE:     4594.737384981167

    #7
    TIMEFRAME: 1986-12-08 - 1986-12-17
    LOCATION:  47.50 N, -52.50 E - 65.00 N, -2.50 E
    SCORE:     4417.5684694986185
    IDENT:     Storm 1986/Dec (Dec 14-15)

    #8
    TIMEFRAME: 1997-12-30 - 1998-01-08
    LOCATION:  50.00 N, -52.50 E - 65.00 N, 10.00 E
    SCORE:     4377.532127551314
    IDENT:     Cyclone Fanny (Jan 03-05)

    #9
    TIMEFRAME: 1995-01-26 - 1995-02-04
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4376.734832155474

    #10
    TIMEFRAME: 2006-12-03 - 2006-12-12
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4306.92269669387

    #11
    TIMEFRAME: 1997-02-18 - 1997-02-27
    LOCATION:  52.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4249.086730533059

    #12
    TIMEFRAME: 1958-01-04 - 1958-01-13
    LOCATION:  50.00 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4206.593603848399

    #13
    TIMEFRAME: 1978-12-06 - 1978-12-15
    LOCATION:  45.00 N, -52.50 E - 65.00 N, 2.50 E
    SCORE:     4151.843224284061

    #14
    TIMEFRAME: 1976-12-01 - 1976-12-10
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4139.642138954728

    #15
    TIMEFRAME: 1971-01-18 - 1971-01-27
    LOCATION:  45.00 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     4030.4772826771264

    #16
    TIMEFRAME: 1992-11-29 - 1992-12-08
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     3962.119334371546

    #17
    TIMEFRAME: 1994-01-27 - 1994-02-05
    LOCATION:  50.00 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     3933.831766100014
    IDENT:     Cyclone Lore (Jan 27-28)

    #18
    TIMEFRAME: 2007-12-02 - 2007-12-11
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     3931.694410784094
    IDENT:     Cyclone Fridtjof (Dec 02-03)

    #19
    TIMEFRAME: 1959-12-18 - 1959-12-27
    LOCATION:  47.50 N, -52.50 E - 65.00 N, 15.00 E
    SCORE:     3910.998749535182

    MATCHED DETECTIONS:   7/20
    UNIQUE MATCHES:       7/20
    TOP-10 DETECTIONS:    5

The computation can be sped up significantly using interval proposals with
only slight changes of the resulting detections.
Passing the flag `--proposals` should make it complete in less than a minute,


Visualizing the Results
-----------------------

The provided script `animate_detections.py` consumes the output written by
`detect_slp.py` to stdout (not stderr, where progress and debug information are
written) and visualizes the data and the detections on a map.

Without any argument, just the data is plotted.
To plot the detections as well, the path of the file containing the output of
`detect_slp.py` has to be given as first argument.

The detections will be shown in chronological order, not ranked by their score.

If a second argument is passed to `animate_detections.py`, it can be used to
specify a filename where the animation should be stored as video instead of
being shown directly.