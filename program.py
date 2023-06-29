from generator import Generator, Correlation
from node import Node
import pandas as pd
import numpy as np
import pickle
from tabulate import tabulate
from labelling import Labelling
import copy
from datetime import datetime

# Global constants for data generating/loading
GENERATE = True
FEATURES = 6
DATASET_SIZE = 2000
TRAIN_SET_SIZE = 150
VALIDATION_SIZE = 0.5
FILE_NAME = str(FEATURES) + '_features' + datetime.today().strftime('%d-%m-%Y') + '.pickle'


class Program:
    @staticmethod
    def get_tree_with_prediction_testset(X_trainset: pd.DataFrame, y_trainset: pd.DataFrame, X_testset:pd.DataFrame, y_testset:pd.DataFrame,X_validationset: pd.DataFrame=None, y_validationset: pd.DataFrame=None, categorical_features=[], local_=False, global_=False):

        if not local_:
            tree = Node(X_trainset.copy(deep=True), copy.deepcopy(y_trainset), positive=False, categorical_features=categorical_features)
        else:
            tree = Node(X_trainset.copy(deep=True), copy.deepcopy(y_trainset), categorical_features=categorical_features)
        
        tree.prune()

        if global_:
            for i in range(len(tree.tree_sequence)):
                tree.tree_sequence[i].make_monotone_leaves()
                tree.tree_sequence[i].ict_prune()
        
        # Choose best tree according to validation set
        best_tree = tree.tree_sequence[0]
        if X_validationset is not None:
            current_best = -float('inf')
            for t in tree.tree_sequence:
                # Determine the predictions on validation set
                y_validation_tree = np.round(X_validationset.apply(lambda x : t.predict(x), axis = 1).to_numpy())
                accuracy = np.sum(y_validation_tree == y_validationset) / len(y_validationset)
                if accuracy > current_best:
                    current_best = accuracy
                    best_tree = t
        
        y_test_tree = np.round(X_testset.apply(lambda x : best_tree.predict(x), axis = 1).to_numpy())
        accuracy = np.sum(y_test_tree == y_testset) / len(y_testset)
        return (best_tree, accuracy)

    @staticmethod
    def run(X_trainset: pd.DataFrame, y_trainset: pd.DataFrame, X_testset:pd.DataFrame, y_testset:pd.DataFrame, X_validationset: pd.DataFrame=None, y_validationset: pd.DataFrame=None,categorical_features=[], plot=True, correlation='positive'):
        """"Runs an experiment, given trainset, validationset and testset"""
        (tree_local_not_global, local_not_global) = Program.get_tree_with_prediction_testset(X_trainset, y_trainset, X_testset, y_testset, X_validationset, y_validationset, categorical_features, local_=True)

        (tree_not_local_not_global, not_local_not_global) = Program.get_tree_with_prediction_testset(X_trainset, y_trainset, X_testset, y_testset, X_validationset, y_validationset, categorical_features)

        (tree_local_global, local_global) = Program.get_tree_with_prediction_testset(X_trainset, y_trainset, X_testset, y_testset, X_validationset, y_validationset, categorical_features, local_=True, global_=True)

        (tree_not_local_global, not_local_global) = Program.get_tree_with_prediction_testset(X_trainset, y_trainset, X_testset, y_testset, X_validationset, y_validationset, categorical_features, global_=True)


        table = [["local_global",local_global],["local_not_global",local_not_global], ["not_local_global",not_local_global],["not_local_not_global",not_local_not_global]]
        print(tabulate(table, tablefmt='latex'))

        if plot:
            titles = [f'tree_not_local_not_global, correlation: {correlation}', f'tree_local_not_global, correlation: {correlation}', f'tree_not_local_global, correlation: {correlation}', f'tree_local_global, correlation: {correlation}']
            tree_not_local_not_global.plot_leaves(titles[0])
            tree_local_not_global.plot_leaves(titles[1])
            tree_not_local_global.plot_leaves(titles[2])
            tree_local_global.plot_leaves(titles[3])

            # latex = ""
            # for title in titles:
            #     latex += f"\hspace*{{-1cm}}\includegraphics[width=0.95\\paperwidth]{{plots/{title}.png}}\n"
            # print(latex)

if GENERATE:        
    print("Busy with generating datasets...")
    # Note, datasets are without labels column yet
    dataset_positive = Generator.generate(N=DATASET_SIZE, d=FEATURES)
    # dataset_negative = Generator.generate(N=DATASET_SIZE, d=FEATURES, correlation=Correlation.Negative)
    # dataset_zero = Generator.generate(N=DATASET_SIZE, d=FEATURES, correlation=Correlation.Zero)

    print("Busy with generating labels...")
    labelling = Labelling()
    labels_positive = labelling.give_labels(dataset_positive)
    # print("labelling for positive correlation is done...")
    # labels_negative = labelling.give_labels(dataset_negative)
    # print("labelling for negative correlation is done...")
    # labels_zero = labelling.give_labels(dataset_zero)
    # print("labelling for zero correlation is done.")

    # Positive correlation
    # all_positive_sets = []
    # for labels in labels_positive:
    #     X_train_positive, X_validation_positive, X_test_positive, y_train_positive, y_validation_positive, y_test_positive = Generator.dataset_split(dataset=dataset_positive, labels=labels, train_size=TRAIN_SET_SIZE, validation_size=VALIDATION_SIZE)
    #     all_positive_sets.append( (X_train_positive, X_validation_positive, X_test_positive, y_train_positive, y_validation_positive, y_test_positive) )
    
    # # Zero correlation
    # all_zero_sets = [] 
    # for labels in labels_zero:
    #     X_train_zero, X_validation_zero, X_test_zero, y_train_zero, y_validation_zero, y_test_zero = Generator.dataset_split(dataset=dataset_zero, labels=labels, train_size=TRAIN_SET_SIZE, validation_size=VALIDATION_SIZE)
    #     all_zero_sets.append( (X_train_zero, X_validation_zero, X_test_zero, y_train_zero, y_validation_zero, y_test_zero) )
    
    # # Negative correlation
    # all_negative_sets = []
    # for labels in labels_negative:
    #     X_train_negative, X_validation_negative, X_test_negative, y_train_negative, y_validation_negative, y_test_negative = Generator.dataset_split(dataset=dataset_negative, labels=labels, train_size=TRAIN_SET_SIZE, validation_size=VALIDATION_SIZE)
    #     all_negative_sets.append( (X_train_negative, X_validation_negative, X_test_negative, y_train_negative, y_validation_negative, y_test_negative) )


    with open(FILE_NAME, 'wb') as handle:
        pickle.dump(dataset_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(dataset_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(dataset_zero, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
        pickle.dump(labels_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(labels_zero, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(labels_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    with open(FILE_NAME, 'rb') as f: 
        dataset_positive = pickle.load(f)
        dataset_negative = pickle.load(f)
        dataset_zero = pickle.load(f)
        all_positive_sets = pickle.load(f)
        all_zero_sets = pickle.load(f)
        all_negative_sets = pickle.load(f)
        # X_train_positive = pickle.load(f) 
        # X_validation_positive = pickle.load(f) 
        # X_test_positive = pickle.load(f)
        # y_train_positive = pickle.load(f) 
        # y_validation_positive = pickle.load(f) 
        # y_test_positive = pickle.load(f)
        # X_train_zero = pickle.load(f) 
        # X_validation_zero = pickle.load(f) 
        # X_test_zero = pickle.load(f)
        # y_train_zero = pickle.load(f) 
        # y_validation_zero = pickle.load(f) 
        # y_test_zero = pickle.load(f)
        # X_train_negative = pickle.load(f) 
        # X_validation_negative = pickle.load(f) 
        # X_test_negative = pickle.load(f)
        # y_train_negative = pickle.load(f) 
        # y_validation_negative = pickle.load(f) 
        # y_test_negative = pickle.load(f)
        labels_positive = pickle.load(f) 
        labels_zero = pickle.load(f)
        labels_negative = pickle.load(f)

    print("Busy with performing emperimentations...")
    print("#####################")
    print('N=2000, train_size=150, d=2, nrep=10\n')
    
    #TODO: change so that multiple labellings are used, for now just one split is taken
    X_train_positive, X_validation_positive, X_test_positive, y_train_positive, y_validation_positive, y_test_positive = all_positive_sets[0]

    X_train_zero, X_validation_zero, X_test_zero, y_train_zero, y_validation_zero, y_test_zero = all_zero_sets[0]

    X_train_negative, X_validation_negative, X_test_negative, y_train_negative, y_validation_negative, y_test_negative = all_negative_sets[0]

    print('Positive correlation:\n')
    Program.run(X_train_positive, y_train_positive, X_test_positive, y_test_positive, X_validation_positive, y_validation_positive)
    print("#####################")

    print('Zero correlation:\n')
    Program.run(X_train_zero, y_train_zero, X_test_zero, y_test_zero, X_validation_zero, y_validation_zero)
    print("#####################")

    print('Negative correlation:\n')
    Program.run(X_train_negative, y_train_negative, X_test_negative, y_test_negative, X_validation_negative, y_validation_negative, correlation='negative')
    print("#####################")
