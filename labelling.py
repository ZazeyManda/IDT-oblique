import numpy as np
import pandas as pd
import networkx as nx
import copy
import pickle
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from isotonic import Isotonic
import math

class Labelling:
    """Class for reading functions from workspaces using R programming language"""
    def __init__(self) -> None:
        # Defining the R script and loading the instance in Python
        r = robjects.r
        r['source']('tools.R')

        # Activate translation from R to Python
        pandas2ri.activate()
        importr('MASS')
        importr('parallel')
    
    def give_labels(self, datamatrix:pd.DataFrame, nrep=10, k=2) -> np.array:
        """
        Generate random monotone labelling function
        nrep: amount of monotone labellings
        k: amount of labels (2 if binary)
        returns: list of balanced labellings that are between 40-60% of class balance
        """
        # Make order matrix given data
        make_order = robjects.globalenv['make.order']
        ordermatrix = make_order(robjects.conversion.py2rpy(datamatrix))
        multi_propp_wilson = robjects.globalenv['multi.propp.wilson2']

        # Each row in the labellings matrix is a labelling of the data points, note that our labellings is 0,1 instead of 1,2
        labellings = np.array(multi_propp_wilson(ordermatrix, nrep, k)) - 1
        # return labellings
        # print(np.array(ordermatrix))
        # ordermatrix = np.array(ordermatrix)

        # Return all labellings such that their class ratio is between at most 60%
        # If there aren't any, just return all possible labellings
        arr = np.array([labelling for labelling in labellings[0] if (0.4 <= np.sum(labelling) / len(labelling) <= 0.6)])
        if len(arr) == 0:
            print('Did not find balanced labellings.')
            return labellings[0]
        else:
            return arr
        
        # labellings[0.4 <= np.sum(labelling) / len(labelling) <= 0.6 for labelling in labellings]
        # balanced_val = len(ordermatrix[0]) / 2
        # current_balanced = float('inf')
        # balanced_labelling = ordermatrix[0]
        # for labelling in labellings:
        #     proportion = len(labelling) - np.count_nonzero(labelling)
        #     if abs(balanced_val - proportion) < current_balanced:
        #         current_balanced = proportion
        #         balanced_labelling = labelling

        # Insert the correct labelling
        # datamatrix.insert(loc=len(datamatrix.columns), column='class', value=balanced_labelling.astype(int)) 
        # return datamatrix
    
    def get_ordermatrix(self, X: pd.DataFrame) -> np.array:
        """Returns order matrix given dataset"""
        ordermat = {}
        for idx_i,i in X.iterrows():
            ordermat[idx_i] = []
            for idx_j,j in X.iterrows():
                # Add 1 if and only if i is being dominated by j (i <= j) for all components of the points
                if idx_i != idx_j and all(i <= j):
                    ordermat[idx_i].append(1)
                else:
                    ordermat[idx_i].append(0)
        return np.array(pd.DataFrame.from_dict(ordermat, orient='index', columns=ordermat.keys()))

    def disjoint_comparable_pairs(self, X: pd.DataFrame, y: np.array) -> np.array:
        """Returns the disjoint and comparable pairs given an order matrix, such that the labels of a pair are the same"""
        disjoint_comparable_pairs = []
        ordermatrix = self.get_ordermatrix(X)
        for i, row in enumerate(ordermatrix):
            for j, column in enumerate(row):
                if column == 1:
                    if y[i] == y[j]:
                        overlap = i in (item for sublist in disjoint_comparable_pairs for item in sublist)
                        overlap = overlap or (j in (item for sublist in disjoint_comparable_pairs for item in sublist))
                        if not overlap:
                            disjoint_comparable_pairs.append([i,j])
        return np.array(disjoint_comparable_pairs)
    
    def noise(self, X: pd.DataFrame, y: np.array, percentage) -> pd.DataFrame:
        """
        Introduce noise on a given labelling y
        X: dataset without labelling
        y: labelling of dataset X
        percentage: percentage of requested amount of noise
        returns: labelling with noise, changes are equal to the percentage given (divided by size of dataset), if there are disjoint comparable pairs
        """  
        disjoint_comparable_pairs = self.disjoint_comparable_pairs(X, y)
        # If we want to add more noise than the amount of datapoints, set amount to length of y, if there are disjoint comparable pairs
        if len(disjoint_comparable_pairs) > 0:
            amount = min(min(math.ceil(percentage * len(y)), len(y)), len(disjoint_comparable_pairs))
            print(f'Amount of noise added (in percentage of total): {percentage}')
            for idx in range(amount):
                [i, j] = disjoint_comparable_pairs[idx]
                if y[i] == 0:
                    y[i] = 1
                else: y[j] = 0
        return y
