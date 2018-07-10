Evaluation on Synthetic Data
============================

This directory contains code used to produce the results on synthetic data reported in section 3 of the paper.

To generate the synthetic dataset, run `python testbench.py large`. This will create the file `testcube.pickle`.

Executing `python run_synthetic_benchmarks.py` will run and evaluate the MDI algorithm with differente divergence
measures and probability density estimators on that synthetic dataset.

`python run_tests.py`, on the other hand, provides more fine-grained control over the algorithm parameters and
visualizes the detected intervals. For example, the following would run the MDI algorithm with the unbiased
KL divergence and the Gaussian distribution model on all test cases of the synthetic dataset:

    python run_tests.py --method gaussian_cov --mode TS --td_dim 3