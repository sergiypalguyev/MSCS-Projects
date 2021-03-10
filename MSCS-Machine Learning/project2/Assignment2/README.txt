The source code and data can be found at : https://github.com/sergiypalguyev/Fall2020CS7641-Assignment2

The source code in this directory was done in Python 3.7
Analysis was done using the latest version of scikit-learn, a Python machine learning library. 
This code executed on multiple threads, one for each problem. Threading must be supported by host PC.

Installation:
To run the code in this project, make sure you have 
	matplotlib
	numpy
	sklearn
	pandas
	mlrose
	threading
Additionally, multiprocessing, functools, itertools and time libraries are used to measure, and speedup execution with parallelized processing.

Top level folder overview:
analysis.py - contains all the code needed to run analyses/experiments on the datasets and generate all graphs and plots.
spalguyev3-analysis.pdf - contains the written report derived from the performed analyses/experiments.

How to use:
- The code contains commented headers which describe what each section of comments does, be it KNN analysis looking for k, or AdaBoost pruning
- The code will save graphs in corresponding dataset folders.
- Simply run analysis.py.

How to run a Python file:
- on Windows: python "filename".py - with no quotations
- on Mac: python3 "filename".py - with no quotations
