import pandas as pd
import numpy as np
import seaborn as sns
from node import Node
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

 
#0.03021127139349215

data = pd.read_csv('accuracy_50.csv').astype(float)
group_a = data[data['Method'] == 0]['Score']
group_b = data[data['Method'] == 1]['Score']
group_c = data[data['Method'] == 2]['Score']
group_d = data[data['Method'] == 3]['Score']

stat, p = friedmanchisquare(group_a, group_b, group_c, group_d)

print('Statistics=%.19f, p=%.19f' % (stat, p))

# Stack the data
stacked_data = data.stack().reset_index()
stacked_data.columns = ['id', 'methods', 'scores']

print(stacked_data)

# Perform the Nemenyi Test
nemenyi_results = sp.posthoc_nemenyi_friedman(stacked_data, y_col='scores', block_col='id', group_col='methods')
print(nemenyi_results)