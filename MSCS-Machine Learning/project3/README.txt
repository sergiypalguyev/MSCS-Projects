The source code and data can be found at : https://github.gatech.edu/spalguyev3/Fall2020CS7641-A3

The source code in this directory was done in Python 3.7
Analysis was done using the latest version of scikit-learn, a Python machine learning library. 

Installation:
To run the code in this project, make sure you have 
	matplotlib
	numpy
	sklearn
	pandas.
Additionally, multiprocessing, functools, itertools and time libraries are used to measure, and speedup execution with parallelized processing.

Top level folder overview:
analysis.py - contains all the code needed to run analyses/experiments on the datasets and generate all graphs and plots.
spalguyev3-analysis.pdf - contains the written report derived from the performed analyses/experiments.

How to use:
- The code contains commented headers which describe what each section of comments does.
- The code will save graphs in corresponding dataset folders.
- Simply run analysis.py.
- CAUTION: There have been moments when models failed to converge runnign one after another, If this occurs, comment out individual models and run them one at a time.

How to run a Python file:
- on Windows: python "filename".py - with no quotations
- on Mac: python3 "filename".py - with no quotations
