## Introduction

This is a demo project to show how to do a preliminary exploration of data sets
using sklearn's parameter search function. Two example public data sets are
used, but the code can be adapted to any data set. The only dataset-specific
file is `processing.py`

#### Data Sets
1. http://archive.ics.uci.edu/ml/datasets/Cardiotocography
2. https://archive.ics.uci.edu/ml/datasets/wine+quality


#### Required Packages

Python 3.6.5
 - scikit-learn==0.19.1
 - pandas==0.23.0
 - numpy==1.14.3
 - matplotlib==2.2.2

#### File contents

 - `processing.py`: script to pre-process raw data into feature vectors
   represented as `pandas` dataframes, parsable by `sklearn` models.
 - `evaluate.py`: module that contains methods to split data, plot data distribution,
   and perform model fitting and evaluation
 - `main.py`: main class to run model evaluation pipeline on various models
 - `model_params.py`: configuration file that specifies parameter search
   space for model optimization

#### Execution

For all scripts, modify file path and names as necessary. For reference, my folder structure is
```
project_root
 - src
 - data
 - output
```

0. Modify file paths as necessary. Run `python processing.py` to get transformed data.
1. Modify `model_params` as necessary to change parameter space.
2. Run `python main.py -h` to see what arguments to pass to the script. Example:
 - `python main.py ../data/wine_processed.csv ../output/ wine > stdout_wine.txt`
