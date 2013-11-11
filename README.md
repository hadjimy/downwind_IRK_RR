Code to reproduce tests from [Step Sizes for Strong Stability Preservation with
Downwind-biased Operators](http://arxiv.org/abs/1105.5798).

To run these scripts, you should have:

Python 2.5 
Matplotlib
Scipy 0.9
Numpy 1.5

or more recent versions.

Files:
- dwrk_tests.py: Main file containing scripts that produce the figures in the paper.
- burgers_char.py: solves Burgers equation by characteristics in order to plot exact solution.
- recon.py: WENO reconstruction routine.
