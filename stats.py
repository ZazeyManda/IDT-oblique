from node import Node, MAX_VAL
import pandas as pd
import numpy as np
import math
import itertools
from scipy import stats
import copy
from statistics import mean
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
import os 
from generator import Generator
import json
from datetime import datetime

STATISTICAL_TEST = True
if STATISTICAL_TEST:
    from scikit_posthocs import posthoc_nemenyi_friedman

# Number of features in artificial dataset
NUMBER_OF_FEATURES = 10
VALIDATION_SIZE = 0.5
TARGET = 'class'
REGRESSIONSETS = ['Admission', 'AutoMPG', 'Computer', 'Kuiper', 'Wages', 'Windsor']
CLASSIFICATIONSETS = ['Bankrupt', 'Compas', 'Credit', 'Haberman', 'Water']
ARTIFICIALSETS = ['Positive', 'Zero', 'Negative']
STATISTICSET = ['Accuracy', 'F1', 'Auroc', 'Precision', 'Recall']
ALGORITHMLIST = ['local_global', 'not_local_global', 'local_not_global', 'not_local_not_global']
FLIST = ['2', '6', '10']
ALGORITHMCOUNT = len(ALGORITHMLIST)
NMIN = 12
MINLEAF = 6
TRAIN_SIZE = 150
class Stats:
    @staticmethod
    def cross_validation(dataset: pd.DataFrame, global_=True, amount_folds=5, positive=True, RSS=False, types=None) -> Node:
        """Cross-validation procedure to return the best possible decision tree"""
        # Round to nearest int
        fold_size = round(len(dataset) / amount_folds)
        folds = [None] * amount_folds
        folds = np.array(folds)
        for i in range(amount_folds - 1):
            folds[i] = dataset[ (i * fold_size) : ((i+1) * fold_size) ]
        # Last fold gets the remainder (could be that dataset is not perfectly divisible by amount_folds)
        folds[amount_folds - 1] = dataset[ ((amount_folds-1) * fold_size) : ]

        # Grow tree on entire dataset
        X = dataset.loc[:, dataset.columns != TARGET]
        y = dataset[TARGET]
        tree = Node(X, y, positive=positive, RSS = RSS, types=types, nmin=NMIN, minleaf=MINLEAF)
        tree.prune()

        # if global_:
        #     for i in range(len(tree.tree_sequence)):
        #         tree.tree_sequence[i].make_monotone_leaves()
        #         tree.tree_sequence[i].ict_prune()
        
        # Determine representatives per tree in pruning sequence
        betas = [None] * len(tree.g)
        betas[0] = 0
        betas[len(betas) - 1] = MAX_VAL
        for i in range(len(betas) - 2):
            betas[i + 1] = math.sqrt(tree.g[i + 1] * tree.g[i + 2])
        
        indices = np.arange(amount_folds)

        # Initialise the errors for each beta value to 0
        betas_errors_dict = {}
        for beta in betas:
            betas_errors_dict[beta] = 0

        # Train on all folds minus 1 and predict on the remaining, keep track of the errors
        for comb in itertools.combinations(np.arange(amount_folds), amount_folds - 1):
            comb = np.array(comb)
            data_folds = folds[comb]
            data = pd.concat(data_folds)
            unused_fold_index = np.delete(indices, comb)
            unused_fold = folds[unused_fold_index[0]]
            X = data.loc[:, data.columns != TARGET]
            y = data[TARGET]
            reduced_tree = Node(X, y, positive=positive, RSS = RSS, types=types, nmin=NMIN, minleaf=MINLEAF)
            reduced_tree.prune()
            # if global_:
            #     for i in range(len(reduced_tree.tree_sequence)):
            #         reduced_tree.tree_sequence[i].make_monotone_leaves()
            #         reduced_tree.tree_sequence[i].ict_prune()
            for beta in betas:
                pruned_tree_beta = reduced_tree.get_tree_given_beta(beta)
                y_predicted = np.round(unused_fold.loc[:, unused_fold.columns != TARGET].apply(lambda x : pruned_tree_beta.predict(x), axis = 1).to_numpy())
                error = np.sum(y_predicted != unused_fold[TARGET]) 
                betas_errors_dict[beta] = betas_errors_dict[beta] + error
       
        # Determine the best overall beta, with lowest total error
        best_beta = min(betas_errors_dict, key=betas_errors_dict.get)
  
        # Return best tree given this cross-validation procedure, only ICT-prune with classification!
        best_tree = tree.get_tree_given_beta(best_beta)
        if global_:
            best_tree.make_monotone_leaves()
            if not RSS:
                best_tree.ict_prune()
        return best_tree
    
    @staticmethod
    def statistical_test(algo_table:list=None, alpha=0.05, path='classification'):
        """
            Expects table where each row is: algortihm_mode=[statistic_dataset1, statistic_dataset2,...]
            Return whether there is statistical significance and if so, what algorithms are better/worse
        """        
        _, pvalue = stats.friedmanchisquare(algo_table[0], algo_table[1], algo_table[2])
        print(f"pvalue of Friedman test: {pvalue}")
        # Reject null hypothesis that all algorithms are equally good, perform post hoc test
        if pvalue < alpha:
            df = posthoc_nemenyi_friedman(np.array([algo_table[0], algo_table[1], algo_table[2]]).T)
            df.columns = ARTIFICIALSETS
            caption="The $p$-values from the post-hoc Nemenyi test for " + path + "with train set size equal to " + str(TRAIN_SIZE) + "."
            print(df.to_latex(caption=caption, label=f"table:experiments_post_hoc"))

            for ir, row in df.iterrows():
                for ic, column in enumerate(row):
                    if column < alpha and ir <= ic:
                        # This is just a shortcut indication, it could sometimes be wrong!
                        if mean(algo_table[ir]) > mean(algo_table[ic]):
                            caption = caption + f"Algorithm {ARTIFICIALSETS[ir]} is significantly better based on accuracy than {ARTIFICIALSETS[ic]} ({round(mean(algo_table[ir]), 2)} vs. {round(mean(algo_table[ic]), 2)}). "
                        else:
                            caption = caption + f"Algorithm {ARTIFICIALSETS[ic]} is significantly better based on accuracy than {ARTIFICIALSETS[ir]} ({round(mean(algo_table[ic]), 2)} vs. {round(mean(algo_table[ir]),2)}). "
                    else:
                        print("Nemenyi post hoc test could not find significant differences.")
            print(df.to_latex(caption=caption, label=f"table:experiments_{path}_post_hoc_{TRAIN_SIZE}"))
        else:
            print('Cannot reject null hypothesis')


    @staticmethod 
    def train_test_validation(X_trainset: pd.DataFrame, y_trainset: pd.DataFrame,X_validationset: pd.DataFrame, y_validationset: pd.DataFrame,  RSS=False, positive=True,  global_=True, types=None) -> Node:
        """Returns best tree with respect to train and validation set"""
        tree = Node(X_trainset.copy(deep=True), copy.deepcopy(y_trainset), positive=positive, RSS=RSS, types=types, nmin=NMIN, minleaf=MINLEAF) 
        tree.prune()
        # if global_:
        #     for i in range(len(tree.tree_sequence)):
        #         tree.tree_sequence[i].make_monotone_leaves()
        #         if not RSS:
        #             tree.tree_sequence[i].ict_prune()
        
        # Choose best tree according to validation set
        best_tree = tree.tree_sequence[0]
        current_best = 0
        for t in tree.tree_sequence:
            # Determine the predictions on validation set
            y_validation_tree = np.round(X_validationset.apply(lambda x : t.predict(x), axis = 1).to_numpy())
            accuracy = np.sum(y_validation_tree == y_validationset.to_numpy()) / len(y_validationset)
            if accuracy > current_best:
                current_best = accuracy
                best_tree = t
        if global_:
            best_tree.make_monotone_leaves()
            if not RSS:
                best_tree.ict_prune()
        return best_tree
    
    @staticmethod
    def get_tree(X_trainset: pd.DataFrame, y_trainset: pd.DataFrame,X_validationset: pd.DataFrame=None, y_validationset: pd.DataFrame=None, positive=True, global_=True, RSS=False, types=None):
        trainset = pd.concat([X_trainset, y_trainset], axis=1)
        # Real datasets
        if X_validationset is None:
            tree = Stats.cross_validation(trainset, global_=global_, positive=positive, RSS=RSS, types=types)
        
        # Artificial datasets
        else:
            tree = Stats.train_test_validation(X_trainset, y_trainset, X_validationset, y_validationset, positive=positive, global_=global_, RSS=RSS, types=types)

        return tree
    
    @staticmethod 
    def stats(X_train, y_train, X_validationset, y_validationset, positive, global_, RSS, types, X_testset, y_testset, classification) -> np.array:
        """Given tree and true labels of test set, returns array of the form [accuracy, F1, auroc] if classification, and MSE if regression"""
        y_testset = y_testset.to_numpy()
        tree = Stats.get_tree(X_train, y_train, X_validationset, y_validationset, positive, global_, RSS, types)
        tree.print_tree()
        predictions = X_testset.apply(lambda x : tree.predict(x), axis = 1).to_numpy()
        if not RSS:
            predictions = np.round(predictions).astype(int)
        
        # Classification 
        if classification:
            accuracy = np.sum(predictions == y_testset) / len(y_testset)
            F1 = f1_score(y_testset, predictions)
            auroc = roc_auc_score(y_testset, predictions)
            precision = precision_score(y_testset, predictions)
            recall = recall_score(y_testset, predictions)
            ret = [accuracy, F1, auroc, precision, recall]
            return np.array(ret)
        # Regression
        return mean_squared_error(y_testset, predictions)

    @staticmethod
    def all_data(path='classification'):
        """Retrieve the test statistics for classification or regression for real datasets, using all 4 modes of our algorithm"""
        dirs = 'datasets/' + path
        if path == 'classification':
            STATSCOUNT = 5
        else:
            STATSCOUNT = 1

        if path == 'regression':
            DATASETCOUNT = len(REGRESSIONSETS)
        else:
            DATASETCOUNT = len(CLASSIFICATIONSETS)

        table = np.arange(ALGORITHMCOUNT * DATASETCOUNT * STATSCOUNT, dtype="float").reshape((ALGORITHMCOUNT,DATASETCOUNT, STATSCOUNT))
        for dir in sorted(os.listdir(dirs)):
            dir = os.path.join(dirs, dir)
            # checking if it is a file
            for f in reversed(sorted(os.listdir(dir))):
                f = f'{dir}/{f}'
                types = None
                if os.path.isfile(f) and 'Bankrupt' in f:
                    if f.endswith('json'):
                        types = json.load(open(f))
                    else:
                        print(f"File under consideration is: {f}")
                        df = pd.read_csv(f)
                        X = df.loc[:, df.columns != TARGET]
                        y = df[TARGET]
                        X_validationset = None
                        y_validationset = None
                        if path == 'classification':
                            RSS = False
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,train_size=TRAIN_SIZE)
                        elif path == 'artifical':
                            RSS = False 
                            X_train, X_validationset, X_test, y_train, y_validationset, y_test = Generator.dataset_split(dataset=X, labels=y, train_size=TRAIN_SIZE, validation_size=VALIDATION_SIZE)
                        else: # Regression
                            RSS = True
                            X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=TRAIN_SIZE)
                        
                        dataset_number = int(f.split('/')[3][0])

                        print('\nConstructed tree for local_global:\n')
                        print(f'started at {datetime.today()}')
                        local_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, True, True, RSS, types, X_test, y_test, path != 'regression')
                        print('\nConstructed tree for not_local_global:\n')
                        print(f'started at {datetime.today()}')
                        not_local_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, False, True, RSS, types, X_test, y_test, path != 'regression')
                        print('\nConstructed tree for local_not_global:\n')
                        print(f'started at {datetime.today()}')
                        local_not_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, True, False, RSS, types, X_test, y_test, path != 'regression')
                        print('\nConstructed tree for not_local_not_global:\n')
                        print(f'started at {datetime.today()}')
                        not_local_not_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, False, False, RSS, types, X_test, y_test, path != 'regression')
 
                        table[0][dataset_number] = local_global
                        table[1][dataset_number] = not_local_global
                        table[2][dataset_number] = local_not_global
                        table[3][dataset_number] = not_local_not_global

        for idx, sub_table in enumerate(table):
            df = pd.DataFrame(sub_table)
            if path != 'regression':
                df.columns = STATISTICSET
                df.index = CLASSIFICATIONSETS
            else: 
                df.columns = ['MSE']
                df.index = REGRESSIONSETS

            print(df.to_latex(caption=f"Experiments run for algorithm: {ALGORITHMLIST[idx]}.", label=f"table:experiments_{path}_{ALGORITHMLIST[idx]}"))

        # Print the table with all statistics
        print(table)

        # Do statistical testing, F1 in case of classification, MSE in case of regression
        test_table = np.vstack([table[0].T[ int(not RSS) ], table[1].T[ int(not RSS) ], table[2].T[ int(not RSS)], table[3].T[ int(not RSS) ]])
        
        if STATISTICAL_TEST:
            Stats.statistical_test(test_table, path=path)
        else:
            print('This is the table to do statistical significant testing on\n')
            print(test_table)


    @staticmethod
    def artificial_data_results(path, number_of_positive, number_of_zero, number_of_negative, number_of_features):
        """Path is of the form datasets/artificial/2features"""
        STATSCOUNT = 5
        DATASETCOUNT = len(ARTIFICIALSETS)
        table = np.zeros(ALGORITHMCOUNT * DATASETCOUNT * STATSCOUNT, dtype="float").reshape((ALGORITHMCOUNT,DATASETCOUNT, STATSCOUNT))
        RSS = False 
        types = None

        for f in sorted(os.listdir(path)):
            if os.path.isfile(f'{path}/{f}'):
                print(f"File under consideration is: {f}")
                df = pd.read_csv(f'{path}/{f}')
                X = df.loc[:, df.columns != TARGET]
                y = df[TARGET]
                
                X_train, X_validationset, X_test, y_train, y_validationset, y_test = Generator.dataset_split(dataset=X, labels=y.to_numpy(), train_size=TRAIN_SIZE, validation_size=VALIDATION_SIZE)
                y_train = pd.DataFrame(y_train, columns=[TARGET])
                y_train = y_train[TARGET]
                y_validationset = pd.DataFrame(y_validationset, columns=[TARGET])
                y_validationset = y_validationset[TARGET]
                y_test = pd.DataFrame(y_test, columns=[TARGET])
                y_test = y_test[TARGET]
                print('\nConstructed tree for local_global:\n')
                print(f'started at {datetime.today()}')
                local_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, True, True, RSS, types, X_test, y_test, path != 'regression')
                print('\nConstructed tree for not_local_global:\n')
                print(f'started at {datetime.today()}')
                not_local_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, False, True, RSS, types, X_test, y_test, path != 'regression')
                print('\nConstructed tree for local_not_global:\n')
                print(f'started at {datetime.today()}')
                local_not_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, True, False, RSS, types, X_test, y_test, path != 'regression')
                print('\nConstructed tree for not_local_not_global:\n')
                print(f'started at {datetime.today()}')
                not_local_not_global = Stats.stats(X_train, y_train, X_validationset, y_validationset, False, False, RSS, types, X_test, y_test, path != 'regression')

                if "positive" in f:
                    dataset_number = 0
                    n = number_of_positive
                elif "zero" in f:
                    dataset_number = 1
                    n = number_of_zero
                else:
                    dataset_number = 2
                    n = number_of_negative

                table[0][dataset_number] = table[0][dataset_number] + (local_global / n)
                table[1][dataset_number] = table[1][dataset_number] + (not_local_global /n)
                table[2][dataset_number] = table[2][dataset_number] + (local_not_global / n)
                table[3][dataset_number] = table[3][dataset_number] + (not_local_not_global / n)

        print('average of tables:')
        print(table)

        for idx, sub_table in enumerate(table.tolist()):
            df = pd.DataFrame(sub_table)
            df.columns = STATISTICSET 
            df.index = ARTIFICIALSETS
            print(df.to_latex(caption=f"Experiments run for algorithm: {ALGORITHMLIST[idx]}, datasets with {number_of_features} features and train set size: {TRAIN_SIZE}; results are averaged over all labellings of particular correlation type.", label=f"table:experiments_artificial_{ALGORITHMLIST[idx]}_{number_of_features}_features_{TRAIN_SIZE}"))

        # Do statistical testing, F1 in case of classification, MSE in case of regression
        test_table = np.vstack([table[0].T[ int(not RSS) ], table[1].T[ int(not RSS) ], table[2].T[ int(not RSS)], table[3].T[ int(not RSS) ]])
        
        if STATISTICAL_TEST:
            Stats.statistical_test(test_table, path=f'artificial_{number_of_features}_features')
        else:
            print('This is the table to do statistical significant testing on\n')
            print(test_table)

    @staticmethod
    def create_artifical_datasets(file):
        with open(file, 'rb') as f: 
            dataset_positive = pickle.load(f)
            # dataset_negative = pickle.load(f)
            # dataset_zero = pickle.load(f)

            labels_positive = pickle.load(f) 
            # labels_zero = pickle.load(f)
            # labels_negative = pickle.load(f)
   
            Stats.set_dataset(file + '_positive', dataset_positive, labels_positive)
            # Stats.set_dataset(file + '_zero', dataset_zero, labels_zero)
            # Stats.set_dataset(file + '_negative', dataset_negative, labels_negative)
            
    @staticmethod
    def set_dataset(file:str, dataset: pd.DataFrame, labels: np.array):
        for idx, labelling in enumerate(labels):
                print(np.mean(labelling))
                # dataset_name = f"{file}_labelling_{idx}.csv"
                # dataset[TARGET] = labelling
                # dataset[TARGET] = dataset[TARGET].astype('int')
                # dataset.to_csv(dataset_name, index=0)

    @staticmethod
    def printLatexTable(table, type, number_of_features=None):
        if type == 'regression':
            df = pd.DataFrame(np.array(table).reshape(-1).reshape(6,4,order="F"))
            df.columns = ALGORITHMLIST
            df.index = REGRESSIONSETS
            print(df.to_latex(caption=f"MSE for experiments run for {type}, train set size: {TRAIN_SIZE}.", label=f"table:experiments_{type}_{TRAIN_SIZE}"))
        else:
            for i,statistic_name in enumerate(STATISTICSET):
                dict = {}
                for j,algo_table in enumerate(table):
                    statistic = np.transpose(algo_table)[i]
                    dict[ALGORITHMLIST[j]] = statistic
                df = pd.DataFrame.from_dict(dict)
                df.index = ARTIFICIALSETS
                print(df.to_latex(caption=f"Experiments run for artificial data with {number_of_features} features, statistic is {statistic_name.lower()} and train set size is {TRAIN_SIZE}.", label=f"table:experiments_artificial_{number_of_features}_features_{statistic_name}_{TRAIN_SIZE}"))
    


if __name__ == '__main__':
    tab = [
            [
                0.8883,0.8831,0.8875,0.8822,0.9,0.9,0.9,0.9,0.921,0.917,0.921,0.917,0.86,0.74,0.86,0.84,0.873,0.678,0.872,0.858,0.87,0.61,0.87,0.86,0.8385,0.7017,0.8398,0.8125,0.85,0.64,0.85,0.83,0.86,0.58 ,0.86,0.83
            ],
            [
                0.8786,0.8776,0.8786,0.8781,0.892,0.892,0.891,0.891,0.9,0.9,0.9,0.9,0.643,0.56,0.641,0.629,0.655,0.63,0.655,0.657,0.6682,0.6495,0.6678,0.665,0.508,0.498,0.507,0.503,0.5174,0.5058,0.5134,0.5129,0.515,0.509,0.518,0.512
            ],
            [
                0.813,0.828,0.826,0.83,0.8455,0.8445,0.8444,0.8435,0.8417,0.8411,0.8434,0.8427,0.581,0.532,0.583,0.572,0.583,0.55,0.59,0.587,0.587,0.555,0.584,0.577,0.497,0.502,0.494,0.496,0.4945,0.5086,0.4971,0.4999,0.506,0.511,0.51,0.507
            ]
    ]
 

    Stats.statistical_test(tab, 0.05, 'classification')


    

    # if TRAIN_SIZE == 100:
    #     NMIN = 8
    #     MINLEAF = 4
    # elif TRAIN_SIZE == 150:
    #     NMIN = 12
    #     MINLEAF = 8
    # else:
    #     NMIN = 2
    #     MINLEAF = 1
    # if NUMBER_OF_FEATURES == 2:
    #     p = 6
    #     z = 6
    # else:
    #     p = 10
    #     z = 10
    # n = 10
    # Stats.artificial_data_results(f'datasets/artificial/{NUMBER_OF_FEATURES}features', p, z, n, NUMBER_OF_FEATURES)



    # Stats.create_artifical_datasets('datasets/artificial/2features.pickle')
    # Stats.create_artifical_datasets('datasets/artificial/6features.pickle')
    # Stats.create_artifical_datasets('datasets/artificial/6features.pickle')
    # Experiments for real datasets    
    # Stats.all_data('regression')
    # print('\n\n\n\n###########################################################\n\n\n\n\n')
    # NMIN = 2
    # MINLEAF = 1
    # Stats.all_data('classification')
    # df = pd.read_csv('credit.csv')
    # X = df.loc[:, df.columns != 'class']
    # Y = df['class']
    # root = Node(X, Y, oblique=True, positive=False)
    # root.prune()
    # Stats.cross_validation(df, False, 5, False, False, [], None)
