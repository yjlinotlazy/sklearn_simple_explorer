### Here two sets of data sets are processed into desired format
import pandas as pd

"""
Data set one
http://archive.ics.uci.edu/ml/datasets/Cardiotocography

Prediction of fetal state based on cardiotocograms.

Manual preprocessing:
1. Export the "raw data" sheet of the xls data file to ../data/ctg.csv
2. Manually remove the 2nd row and the last 3 rows that are not relevant
"""

raw_df = pd.read_csv("../data/ctg.csv")

columns = ['LB', 'AC', 'FM', 'UC',
            'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DS', 'DP', 'DR', 'Width', 'Min',
            'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
            'Tendency', 'NSP']
raw_df = raw_df[columns]

raw_df.to_csv('../data/ctg_processed.csv', index=None)

"""
Data set two

https://archive.ics.uci.edu/ml/datasets/wine+quality

Wine quality assessment.
"""

raw_df2 = pd.read_csv('../data/winequality-white.csv', delimiter=';')
raw_df2.to_csv('../data/wine_processed.csv', index=None)