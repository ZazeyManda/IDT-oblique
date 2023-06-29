import numpy as np
import pandas as pd
from labelling import Labelling
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os 

class Correlation(Enum):
    Positive = 1
    Zero = 2
    Negative = 3

class Generator: 
    @staticmethod
    def dataset_split(dataset: pd.DataFrame, labels:np.array, train_size, validation_size, noise=0.05):
        """Given dataset, generate train, validation and test set, including noise of default 5% of the size of the dataset"""

        # Split in train, validation and test sets
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, stratify=labels,train_size=train_size)
        X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, stratify= y_test,train_size=validation_size)

        # Add noise independently
        labelling = Labelling()
        print('Adding noise to train set...')
        y_train = labelling.noise(X_train, y_train, noise)
        print('Adding noise to validation set...')
        y_validation = labelling.noise(X_validation, y_validation, noise)
        print('Adding noise to test set...')
        y_test = labelling.noise(X_test, y_test, noise)

        return X_train, X_validation, X_test, y_train, y_validation, y_test
    """Class for generating synthetic data, with labelling"""
    @staticmethod
    def generate(N=150, d=2, correlation=Correlation.Positive) -> pd.DataFrame:
        """
            Generate dataset using multivariate normal distribution
            N: number of data points to generate
            d: number of features
            many_comparable_points: whether the data points should be comparable (according to product order)
        """
        # We are uninterested in the mean
        mean_vector = np.zeros(d)

        # Set sigma to 0.9 everywhere if many comparable points are requested
        if correlation == Correlation.Positive:
            sigma_matrix = np.full((d,d), 0.9)
        # If no correlation is requested (randomness)
        elif correlation == Correlation.Zero:
            sigma_matrix = np.full((d,d), 0)
        # Else we create two groups of equal size such that the two groups are highly positive correlated within but highly negative correlated between
        else:
            sigma_matrix = np.full((d,d), 0.9)
            sigma_matrix[0:(d//2), (d//2):len(sigma_matrix)].fill(-0.9)
            sigma_matrix[(d//2): len(sigma_matrix), 0:(d//2)].fill(-0.9)

        np.fill_diagonal(sigma_matrix, 1)
        dataset = np.random.multivariate_normal(mean_vector, sigma_matrix, N)
        
        # Add columns
        dataset = pd.DataFrame(dataset, columns=["X"+str(i) for i in range(1, len(dataset.transpose()) + 1)])

        return dataset
    