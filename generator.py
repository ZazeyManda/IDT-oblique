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
    
    @staticmethod
    def statistics(class_regression:str, correlation=True):
        target = 'class'
        dir = "datasets/" + class_regression
        for filename in sorted(os.listdir(dir)):
            f = os.path.join(dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                if(filename == 'Readme Computer.txt'): 
                    continue
                df = pd.read_csv(f)
                df = df.select_dtypes(exclude=['object'])
                X = df.loc[:, df.columns != target]
                y = df[target]
                reg = LinearRegression().fit(X, y)
                # for idx, col in enumerate(X):
                #     print(f"{col} & {round(reg.coef_[idx], 2)}\\\\")
                    
                # for idx, col in enumerate(X):
                #     # Change monotone decreasing to monotone increasing
                #     if reg.coef_[idx] < 0:
                #         x_min = min(df[col])
                #         x_max = max(df[col])
                #         for i,_ in enumerate(df[col]):
                #             df[col][i] = x_max - df[col][i] + x_min
                # df.to_csv(f, index=0)
                # cardinality = len(df)
                # nof = len(df.columns) - 1
                # target_mean = round(mean(df['class']),2)
                # target_variance = round(variance(df['class']),2)
                # category = file 
                # print(f"{filename} & {cardinality} & {nof} & {target_mean} & {target_variance} & {category}\\\\")

        
        # """Give the correlation matrix of the datasets in the datasets folder"""
        # target = "class"
        # df = pd.read_csv(file)
        # df = df.select_dtypes(exclude=['object'])
        # print(df.corr().to_latex())
        # if correlation:
        #     print(df.drop(target, axis=1).apply(lambda x: x.corr(df[target]))) 
        # else:
        #     # Linear regression
        #     X = df.loc[:, df.columns != target]
        #     X = X.loc[:, X.columns != 'car name']
        #     X = X.loc[:, X.columns != 'origin']
        #     y = df[target]
        #     reg = LinearRegression().fit(X, y)
        #     for idx, col in enumerate(X):
        #         print(f"{col} & {round(reg.coef_[idx], 2)}\\\\")

    
# labelling = Labelling()
# df = Generator.generate(100, 10, Correlation.Positive)
# print('dataset has been created...')
# labellings = labelling.give_labels(df, 10, 2)
# for labelling in labellings:
#     print(sum(labelling) / len(labelling))
# print('labelling is finished...')

# def comp_pairs(X:pd.DataFrame):
#     count = 0
#     for _,row1 in X.iterrows():
#         for _,row2 in X.iterrows():
#             if row1 is not row2 and all(row1 >= row2):
#                 count = count + 1
#     return count


# import matplotlib.pyplot as plt
# positive = Generator.generate(20)
# zero = Generator.generate(20, correlation=Correlation.Zero)
# negative = Generator.generate(20, correlation=Correlation.Negative)

# plt.plot(positive['X1'], positive['X2'], linestyle="", marker='o')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.savefig(f'positive_correlation.png', bbox_inches='tight')
# print(f'positive comp pairs: {comp_pairs(positive)}')

# plt.plot(zero['X1'], zero['X2'], linestyle="", marker='o')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.savefig(f'zero_correlation.png', bbox_inches='tight')
# print(f'zero comp pairs: {comp_pairs(zero)}')

# plt.plot(negative['X1'], negative['X2'], linestyle="", marker='o')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.savefig(f'negative_correlation.png', bbox_inches='tight')
# print(f'negative comp pairs: {comp_pairs(negative)}')
# Generator.statistics('datasets/regression/AutoMPG.csv')

# Generator.statistics('Classification')
# Generator.statistics('Regression')

# df = pd.read_csv('datasets/regression/0.Admission.csv')
# indices = np.array([499,551,191,281,243,534,907,299,74,383,642,1219,157,700,937,5,1111,148,361,56,1283,44,796,944,805,1195,685,360,501,814,709,597,549,732,578,439,1275,564,129,697,1007,1198,612,1174,1121,843,343,546,404,624,109,1077,734,130,339,646,368,433,1199,242,747,461,1164,675,1022,1149,442,867,1117,645,1200,1141,1002,143,476,375,190,1223,849,1082,387,765,1310,1272,781,31,565,1070,520,661,226,357,1299,792,854,285,861,869,941,541,221,636,914,1175,454,73,971,318,1256,390,124,264,15,1020,891,1017,1043,1232,417,384,61,768,580,888,138,458,650,1092,1186,1159,117,50,530,240,1165,625,10,672,180,342,82,1055,517,649,111,717,684,999,1225,1118,1307,1049,1241,1220,471,16,212,1109,884,967,977,668,912,333,903,1024,294,1271,990,575,900,380,412,97,1205,1042,204,813,1197,277,1114,424,665,1034,491,1030,698,388,596,878,1026,254,943,552,189,548,135,766,940,1177,593,852,1000,786,406,1087,758,676,561,806,632,1250,1138,901,1100,17,871,719,870,851,562,686,39,453,951,139,1080,1227,1098,978,218,1115,77,452,826,863,1129,621,644,128,759,472,485,587,270,236,659,883,488,929,57,451,1053,1262,1014,346,892,448,1135,473,716,1258,737,586,727,623,146,1090,389,1296,641,220,155,722,314,262,816,279,435,757,1239,1154,310,991,1076,84,595,953,966,763,465,1243,347,1108,1131,887]) - 1
# # features = df.loc[:, df.columns != 'class']
# # ones = df.loc[df['class'] == 1]
# ones = df.iloc[indices]
# zeroes =  df.loc[df['class'] == 0]
# df2 = pd.concat([ones, zeroes])

# df2 = df2.select_dtypes(exclude=['object'])

# ones = df.loc[df['class'] == 1]
# zeroes =  df.loc[df['class'] == 0]
# zeroes = zeroes.sample(220)
# df = pd.concat([ones, zeroes])

# df = pd.read_csv('datasets/classification/Bankrupt/0.Bankrupt.csv')
# df = pd.read_csv('data.csv')
# from sklearn.linear_model import LinearRegression
# X = df.loc[:, df.columns != 'class']
# y = df['class']

# print('LINEAR REGRESSION ON DATASET')
# reg = LinearRegression().fit(X, y)
# for idx, col in enumerate(X):
#     print(f'{col} & {round(reg.coef_[idx], 2)}\\\\')

# ones = df.loc[df['class'] == 1]
# zeroes =  df.loc[df['class'] == 0]
# zeroes = zeroes.sample(220)
# df = pd.concat([ones, zeroes])

# print(df['class'].mean())
# print(df['class'].var())
# #'After-tax Net Profit Growth Rate' geeft positieve coefficient maar hoort negatief te zijn
# df = df[['class','Debt ratio %', 'Current Liability to Current Assets', 'Cash Flow to Liability', 'Net Income to Total Assets', 'Fixed Assets to Assets']]
# X = df.loc[:, df.columns != 'class']
# X = X.reset_index(drop=True)
# y = df['class']
# y = y.reset_index(drop=True)
# df = df.reset_index(drop=True)

# print('\n\n\n\nLINEAR REGRESSION ON SAMPLE')
# reg = LinearRegression().fit(X, y)
# for idx, col in enumerate(X):
#     print(f'{col} & {round(reg.coef_[idx], 2)}\\\\')
#     # print(f'{col} & {reg.coef_[idx]}\\\\')
                    
# for idx, col in enumerate(X):
#     # Change monotone decreasing to monotone increasing
#     if reg.coef_[idx] < 0:
#         x_min = min(X[col])
#         x_max = max(X[col])
#         for i,_ in enumerate(X[col]):
#             df[col][i] = x_max - df[col][i] + x_min
            
# # df['class'] = round(df['class'], 5)
# df.to_csv('datasets/classification/Bankrupt/0.Bankrupt.csv', index=0)

# df = pd.read_csv('datasets/classification/Water/4.Water.csv')
# df = df.round(3)
# df.to_csv('datasets/classification/Water/4.Water.csv', index=0)
# df2.to_csv('datasets/classification/4.Water.csv', index=0)


# import seaborn as sns
# hm = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# plt.figure(figsize=(16, 6))
# heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
# plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
