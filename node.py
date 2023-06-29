import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from check import Check
from isotonic import Isotonic
import itertools
import copy
import sys

EPSILON = sys.float_info.epsilon
MAX_VAL = float('inf')
USE_MPL = True
if USE_MPL:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    # Unique identifier for node
    newid = itertools.count()
    
    def __init__(
        self, 
        X: pd.DataFrame,
        Y: pd.DataFrame,
        parent=None,
        # All splitting rules leading to current node, triple of ([feature, coefficient], direction, value) where direction ∈ {'<=', '>'}
        rules=[],
        # Splits are of the form: best_feature <= split_val (or >)
        # Left child node contains all points that conform to the splitting rule
        best_feature=None,
        split_val=None,
        left_child=None,
        right_child=None,
        # Whether we want oblique splitting, with positive or non-constraint (local monotonicity)
        oblique=True,
        positive=True,
        # Whether we are interested in performing regression tree or classification tree
        RSS=False,
        # Features and their types, retrieved via JSON
        types = None,
        # If a node contains less than nmin cases, then it becomes a leaf node
        nmin=2,
        # A split is not allowed if it creates a child node with less then minleaf cases
        minleaf=1
    ):
        # Initialize the node 
        self.Y = Y 
        self.X = X
        self.id = next(self.newid)
        # Default values of node 
        self.best_feature = best_feature
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.rules = rules
        self.tree_sequence = None
        self.pruned_tree = None
        # Categorical features is a list of feature names that are categorical
        if types is not None:
            self.categorical_features = [feature for feature in types if types[feature]['type'] == 'categorical']
        else:
            self.categorical_features = []
        self.oblique = oblique
        self.positive = positive
        self.RSS=RSS
        self.types = types
        self.nmin = nmin 
        self.minleaf = minleaf

        # Prediction is calculated as relative class frequency of label 1, or in case of regression tree as mean of training observations
        # EPSILON is necessary for rounding because otherwise e.g. 0.5 gets rounded to 0
        self.prediction = np.sum(self.Y) / len(self.Y) + EPSILON
        self.tree_grow()

        # After growing tree, determine the non-terminal nodes (the non-leaves)
        self.leaves = self.calc_leaves()
        self.non_leaves = self.non_terminal_nodes()

    def tree_grow(self): 
        """
        Recursive method to create decision tree
        """
        if not self.can_split():
            self.leaf = True
            return 
        
        new_feature_name = ''
        if self.oblique:
            # Add additional linear combination feature (column) to dataframe, if oblique is requested
            X_, self.regression = self.linear_combination_split()
            features_coefs = []
            for idx, feature_name in enumerate(X_):
                if self.regression.coef_[idx] != 0:
                    new_feature_name = new_feature_name + str(round(self.regression.coef_[idx], 4)) + '*' + feature_name 
                    features_coefs.append((feature_name, self.regression.coef_[idx])) 
                    if idx is not len(self.X.columns) - 1:
                        new_feature_name = new_feature_name + '+'
            if new_feature_name.endswith('+'):
                new_feature_name = new_feature_name[:len(new_feature_name)-1]
            new_feature_values = self.regression.predict(X_)
            if new_feature_name not in self.X.columns:
                self.X.insert(loc=len(self.X.columns), column=new_feature_name, value= new_feature_values)      
            
        # Try and find the best split
        split_data = self.best_split()
        
        # If no split was made we are leaf
        if split_data[0] == MAX_VAL:
            self.X = self.X.drop(new_feature_name, axis=1, errors='ignore')
            self.leaf = True
            return 

        # Extract the split_data
        feature = split_data[0]
        splitpoint = split_data[1]
        classes_left = split_data[2]
        classes_right = split_data[3]

        # Remove the added linear combination split column before entering children
        # Retrieve the concrete examples of left and right children after the split
        if feature not in self.categorical_features:
            examples_left = self.X[np.array(self.X[feature]) <= splitpoint].drop(new_feature_name, axis=1, errors='ignore')
            examples_right = self.X[np.array(self.X[feature]) > splitpoint].drop(new_feature_name, axis=1, errors='ignore')
        else:
            examples_left = self.X[ np.array( [x in splitpoint for x in np.array(self.X[feature])] ) ].drop(new_feature_name, axis=1, errors='ignore')
            examples_right = self.X[ np.array( [x not in splitpoint for x in np.array(self.X[feature])] ) ].drop(new_feature_name, axis=1, errors='ignore')

        # Check which feature was the best and update feature-coefficient list accordingly
        if feature != new_feature_name:
            features_coefs = [(feature, 1)]

        # Append to the node the expansion of children
        self.leaf = False
        self.best_feature = feature
        self.split_val = splitpoint
        self.X = self.X.drop(new_feature_name, axis=1, errors='ignore')
        if feature in self.categorical_features:
            self.left_child = Node(copy.deepcopy(examples_left), copy.deepcopy(classes_left), self, self.rules + [(feature, 'in', splitpoint)], oblique=self.oblique, positive=self.positive, RSS=self.RSS, types=self.types, nmin=self.nmin, minleaf=self.minleaf)
            self.right_child = Node(copy.deepcopy(examples_right), copy.deepcopy(classes_right), self, self.rules + [(feature, 'not in', splitpoint)], oblique=self.oblique, positive=self.positive, RSS=self.RSS, types= self.types, nmin=self.nmin, minleaf=self.minleaf)
        else:
            self.left_child = Node(copy.deepcopy(examples_left), copy.deepcopy(classes_left), self, self.rules + [(features_coefs, '<=', splitpoint)], oblique=self.oblique, positive=self.positive, RSS=self.RSS, types= self.types, nmin=self.nmin, minleaf=self.minleaf)
            self.right_child = Node(copy.deepcopy(examples_right), copy.deepcopy(classes_right), self, self.rules + [(features_coefs, '>', splitpoint)], oblique=self.oblique, positive=self.positive, RSS= self.RSS, types= self.types, nmin=self.nmin, minleaf=self.minleaf)
    
    def can_split(self) -> bool:
        """
        Checks whether a split is possible. returns false when all instances have the same class, or all data points are the same, or if node is not allowed to split due to nmin
        """
        if len(set(self.Y)) <= 1 or len(self.X.drop_duplicates()) <= 1 or len(self.X) < self.nmin:
            return False
        else:
            return True
    
    def linear_combination_split(self):
        """
        Given current node, determine the linear combination split via desired least squares method
        """
        # First remove the categorical variable
        X_ = copy.deepcopy(self.X)
        for cat in self.categorical_features:
            X_ = X_.drop(cat, axis=1, errors='ignore')
        regression = LinearRegression(positive=self.positive).fit(X_, self.Y)
        # We are only interested in the coefficients from the linear regression, not the bias
        regression.intercept_ = 0 
        return (X_, regression)

    def best_split(self):
        """
        Goes through each feature to determine split, calculates its quality, and selects the best one
        """
        current_reduction = MAX_VAL
        # Loop over every feature (column)
        for feature in self.X:
            split_data = self.best_split_helper(self.X[feature], feature)
            if split_data[0] < current_reduction:
                current_reduction = split_data[0]
                current_splitpoint = split_data[1]
                current_classes_left = split_data[2]
                current_classes_right = split_data[3]

                # Also remember the feature we used to split on
                current_feature = feature
        
        # If no split was made at all 
        if current_reduction == MAX_VAL:
            return [current_reduction]
        
        # A split was made, return the split data
        return [current_feature, current_splitpoint, current_classes_left, current_classes_right]

    def best_split_helper(self, x: pd.DataFrame, feature: str) -> list:
        """
        Goes through each possible location to split on the given feature to find the best possible one
        x: vector of values of a specific feature
        feature: feature name
        """
        y = np.array(self.Y)
        
        if feature not in self.categorical_features:
            x_sorted = sorted(x.unique())
            # Candidate splits
            x_split_points = np.array([(x+z)/2 for x,z in zip(x_sorted[0:len(x_sorted)-1], x_sorted[1:len(x_sorted)])])
        else:
            x_arr = x.to_numpy()
            vals, counts = np.unique(x_arr, return_counts = True)
            vals_new, prob_counts = np.unique(x_arr[y == 0], return_counts = True)
            dic = dict(zip(vals_new, prob_counts))
            prob_table = []
            for idx,_ in enumerate(vals):
                if vals[idx] in dic:
                    prob_table.append( (vals[idx], dic[vals[idx]] / counts[idx]) )
                else:
                    prob_table.append( (vals[idx], 0) )
            prob_table.sort(key = lambda a: a[1])
            prob_table = [i[0] for i in prob_table]
            x_split_points = []
            for i in range(len(prob_table) - 1):
                x_split_points.append(prob_table[0:i+1])
        
        # Try all splits and remember the best
        impurity_sum_min = MAX_VAL
        for splitpoint in x_split_points:
            # Determine the classes that will go to the left child and right child because of the split
            if feature not in self.categorical_features:
                to_left = y[x <= splitpoint]
                to_right = y[x > splitpoint]
            else:
                arr = x.to_numpy()
                to_left = y[ np.array([el in splitpoint for el in arr]) ]
                to_right = y[ np.array([el not in splitpoint for el in arr]) ]
            
            # No split possible if children are empty
            if len(to_left) < 1 or len(to_right) < 1:
                continue

            # The sum of impurities of children for the current split
            if self.RSS and feature not in self.categorical_features:
                proportion_left = 1
                proportion_right = 1 
            else:
                # Determine the proportion of examples that will go to the left and right child
                proportion_left = len(to_left) / len(x)
                proportion_right = 1 - proportion_left
            impurity_sum = proportion_left * self.impurity(to_left) + proportion_right * self.impurity(to_right)
                            
            # Save the best split, without violating minleaf constraint
            if impurity_sum < impurity_sum_min and len(to_left) >= self.minleaf and len(to_right) >= self.minleaf:
                impurity_sum_min = impurity_sum
                impurity_sum_min_to_left = to_left
                impurity_sum_min_to_right = to_right
                impurity_sum_min_splitpoint = splitpoint

        # If no split was made, return just the negative impurity reduction
        if impurity_sum_min == MAX_VAL:
            return [impurity_sum_min]
        
        # The best split was made so return it
        return [impurity_sum_min, impurity_sum_min_splitpoint, impurity_sum_min_to_left, impurity_sum_min_to_right]
        
    def impurity(self, y:np.array) -> float:
        """
        Calculates the Gini impurity of a child node for the impurity reduction calculation if classification, otherwise calculates RSS
        y: a vector of class labels (0/1) in case of classification. In case of regression it is a vector of numbers to perform RSS on
        Returns: a number containing the impurity of the child node
        """
        if not self.RSS:
            p0 = sum(y) / len(y)
            return p0 * (1 - p0)
        else:
            return self.calc_RSS(y)

    def calc_RSS(self, y:np.array) -> float:
        y_avg = np.mean(y)
        total_RSS = 0
        for y_i in y:
            total_RSS += (y_i - y_avg)**2
        return total_RSS
    
    def calc_leaves(self):
        """
        Returns all leaves that are decendant of a given node, leaves are returned from left to right
        """
        # This is a leaf node
        if not self.left_child and not self.right_child:
            return [self]
        
        # Otherwise, check for leaves left and right
        leaves = []
        if self.left_child:
            leaves = self.left_child.calc_leaves()
        if self.right_child:
            leaves = leaves + self.right_child.calc_leaves()
        return leaves

    def leaf_order(self) -> np.array:
        """
        Determines the leaf order, returns order matrix
        """
        checker = Check(self.categorical_features, self.types)
        ordermat = [None] * len(self.leaves)
        for i1, leaf1 in enumerate(self.leaves):
            ordermat[i1] = [None] * len(self.leaves)
            for i2, leaf2 in enumerate(self.leaves):
                # Add 1 if and only if leaf1 is dominated by leaf2 (leaf1 <= leaf2)
                if leaf1.id != leaf2.id and checker.dominates(leaf2, leaf1):
                    ordermat[i1][i2] = 1
                else:
                    ordermat[i1][i2] = 0
        return np.array(ordermat)

    def make_monotone_leaves(self) -> None:
        """
        Makes the classifier globally monotone via isotonic regression
        """
        ordermat = self.leaf_order()
        print('ordermatrix:')
        print(ordermat)
        y = np.array([self.leaves[i].prediction for i,_ in enumerate(self.leaves)])
        iso = Isotonic(ordermat=ordermat, y=y)
        for i,_ in enumerate(self.leaves):
            self.leaves[i].prediction = iso.result[i] + EPSILON

    def predict(self, x: pd.DataFrame):
        """
        Given data point x, give its prediction.
        Note that leaves are disjoint by construction, monotone leaves are chosen automatically
        """
        for leaf in self.leaves:
            if self.in_leaf(leaf, x) is True:
                return leaf.prediction

    def in_leaf(self, leaf, x: pd.DataFrame):
        """Determine whether data point falls into leaf"""
        return all([self.rule_validator(rule, x) for rule in leaf.rules])

    def rule_validator(self, rule, x: pd.DataFrame) -> bool:
        """Given rule and data point x, retrieve whether data point satisfy the rule"""
        (features_coefficients, direction, value) = rule
        expression = True
        if direction != 'in' and direction != 'not in':
            constraint = 0
            for _, (feature, coefficient) in enumerate(features_coefficients):
                constraint = constraint + coefficient * x[feature]
            if (direction == '<='):
                constraint = constraint <= value
            else:
                constraint = constraint > value
            expression = expression and constraint
        elif direction == 'in':
            # This is for example 'Type' in ['Sedan', 'Coupe']
            feature = features_coefficients
            expression = expression and (x[feature] in value)
        else: # Direction is 'not in'
            feature = features_coefficients
            expression = expression and (x[feature] not in value)
        return expression
    
    def ict_prune(self):
        """
        Prunes tree to common parent if two leaves have same label as parent
        """
        state = False
        tuples = [(a, b) for idx, a in enumerate(self.leaves) for b in self.leaves[idx + 1:]]
        for (leaf1, leaf2) in tuples:
            if leaf1.parent.id == leaf2.parent.id and ((leaf1.prediction >= 0.5 and leaf2.prediction >= 0.5) or (leaf1.prediction < 0.5 and leaf2.prediction < 0.5)):
                leaf1.parent.left_child = None
                leaf1.parent.right_child = None
                leaf1.parent.leaf = True
                state = True
        self.set_leaves_and_non_leaves()

        # Go into recursion if pruning was performed
        if state == True:
            self.make_monotone_leaves()
            self.ict_prune()

    def get_g(self, node, amount_examples, gini):
        """Gets the g-value for pruning, given the node to prune in"""
        return (self.resub_node(node, amount_examples, gini) - self.resub_tree(node, amount_examples, gini)) / (len(node.leaves) - 1)

    def prune(self, gini=False):
        """
        Prune tree and saves tree sequences in terms of their root node
        See http://www.cs.uu.nl/docs/vakken/mdm/Slides/dm-classtrees-2-2021.pdf for more details
        """
        amount_examples = len(self.X)
        self.set_leaves_and_non_leaves()
        tree_sequence = np.array([self])
        k = 0
        g = np.array([])
        g = np.append(g, 0)

        while(len(tree_sequence[k].leaves) > 1):
            tree = copy.deepcopy(tree_sequence[k])
            g = np.append(g, MAX_VAL)
            # Determine new alpha value, for pruning
            for i,_ in enumerate(tree.non_leaves):
                g[k+1] = min(g[k+1], self.get_g(tree.non_leaves[i], amount_examples, gini))

            # Prune in nodes if equal to alpha pruning value
            for i,_ in enumerate(tree.non_leaves):
                if self.get_g(tree.non_leaves[i], amount_examples, gini) == g[k+1]:
                    # Prune tree
                    tree.non_leaves[i].left_child = None
                    tree.non_leaves[i].right_child = None
                    tree.non_leaves[i].leaf = True
            # Re-calculate the leaves and non-leaves of the tree
            tree.set_leaves_and_non_leaves()
            tree_sequence = np.append(tree_sequence, tree)
            k = k + 1
        self.tree_sequence = tree_sequence
        self.g = g
    
    def get_tree_given_beta(self, beta:float):
        for i in range(len(self.g) - 1):
            if beta >= self.g[i] and beta < self.g[i + 1]:
                return self.tree_sequence[i]
        # Beta must fall into last interval
        return self.tree_sequence[len(self.tree_sequence) - 1]

    def set_leaves_and_non_leaves(self):
        """Top-down approach to set the correct leaves and non-leaves of the entire tree, per node"""
        if self.left_child:
            self.left_child.set_leaves_and_non_leaves()
        if self.right_child:
            self.right_child.set_leaves_and_non_leaves()
        self.leaves = self.calc_leaves()
        self.non_leaves = self.non_terminal_nodes()
        
    def resub_node(self, node, total: int, gini) -> float:
        # Classification tree
        if not self.RSS:
            # Classification tree with gini
            if gini:
                fraction = len(node.Y) / total
                g = self.impurity(node.Y)
                return fraction * g
            # Classification tree with resubstitution error
            if node.prediction >= 0.5:
                amount = len(node.Y) - np.count_nonzero(node.Y)
            else:
                amount = np.count_nonzero(node.Y)
            return amount / total
        # Regression tree
        else:
            return self.impurity(node.Y)
            
    
    def resub_tree(self, tree, total: int, gini) -> float:
        leaves = tree.leaves
        acc = 0
        for leaf in leaves:
            acc = acc + self.resub_node(leaf, total, gini)
        return acc

    def non_terminal_nodes(self):
        """Return non-leaf nodes of tree, follows post-order traversal"""
        # Base case is necessary because this call will also be applied on leaf nodes
        if self.leaf is True:
            return []
        
        if self.left_child.leaf is True and self.right_child.leaf is True:
            return [self]
        
        nodes = []
        if not self.left_child.leaf:
            nodes = self.left_child.non_terminal_nodes()
        if not self.right_child.leaf:
            nodes =  nodes + self.right_child.non_terminal_nodes()
        nodes = nodes + [self]
        return nodes
    
    def feasible_region(self, leaf:np.array, amount_points:int):
        dx1 = np.linspace(min(self.X['X1']), max(self.X['X1']), amount_points)
        dx2 = np.linspace(min(self.X['X2']), max(self.X['X2']), amount_points)
        X1, X2 = np.meshgrid(dx1, dx2)
        equations = []
        for rule in leaf.rules:
            features_coefs, direction, value = rule
            if len(features_coefs) == 1:
                [(feature, coef)] = features_coefs
                if feature == 'X1':
                    if direction == '<=':
                        equations.append(X1 <= value / coef)
                    else: equations.append(X1 > value / coef)
                else:
                    if direction == '<=':
                        equations.append(X2 <= value / coef)
                    else: equations.append(X2 > value / coef)
            else:
                print(features_coefs)
                [(feature1, coefficient1), (_, coefficient2)] = features_coefs
                if feature1 == 'X1':
                    if direction == '<=':
                        equations.append(coefficient1 * X1 + coefficient2 * X2 <= value)
                    else: equations.append(coefficient1 * X1 + coefficient2 * X2 > value)
                else:
                    if direction == '<=':
                        equations.append(coefficient2 * X1 + coefficient1 * X2 <= value)
                    else: equations.append(coefficient2 * X1 + coefficient1 * X2 > value)
        return (equations, X1, X2)
    
    def plot_leaves(self, title:str, amount_points=700):
        if not USE_MPL:
            print("Matplotlib disabled, please enable it to plot the leaves")
            return
        # Create blank image
        # This image will be like a histogram, that keeps a count of how many equations have "true" on a given pixel. This approach works since we make a grey-scale image
        final_image = None
        final_image = [0] * amount_points
        final_image = np.array(list(map(lambda _: [0] * amount_points, final_image)))
        all_images = []
        labels = []
        
        plt.figure(figsize = (16,9))

        for leaf in self.leaves:
            equations, X1, X2 = self.feasible_region(leaf, amount_points)
            plt.xlabel(r'$X1$', fontsize=14)
            plt.ylabel(r'$X2$', fontsize=14)
            plt.title(title, fontsize=20, y=1.03)
            # If leaf is the root
            if len(self.leaves) == 1:
                final_image.fill(1)
                labels.append(f'{round(self.prediction)}')
                break
            equation = equations[0]
            for i in range(1, len(equations)):
                equation = equation & equations[i]
            all_images.append(equation)
            labels.append(f'{round(leaf.prediction)}')

        for idx, equation in enumerate(all_images):
            final_image += equation * idx
        
        im = plt.imshow(final_image, extent=(X1.min(),X1.max(),X2.min(),X2.max()), origin='lower', cmap="gist_ncar")
        values = np.unique(final_image)
        colors = [ im.cmap(im.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label='Prediction {l}'.format(l=labels[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
        plt.grid(True)
        # plt.show()
        plt.savefig(f'plots/{title}.png', bbox_inches='tight')
    
    def print_tree(self):
        """Prints tree in simple ACII"""
        print('Tree is traversed via depth-first traversal.')
        print('Node level) splitting rule to current node ; #data points ; impurity (Gini for classification, MSE for regression) ; average y_value (regression) or relative frequency of class 1 (classification) ; star (*) if node is leaf')
        self.traverse(self, 0)
    
    def traverse(self, node, level):
        """Prints: node level) splitting rule to current node ; #data points ; impurity ; average_y_value ; leaf_or_not (* if leaf, nothing if not)"""
        print(f"{level}) ; {(node.parent.best_feature + ' ' + node.rules[len(node.rules)-1][1] + ' ' + str(node.parent.split_val)) if node.rules != [] else 'root'} ; {len(node.X)} ; {round(node.impurity(node.Y), 4)} ; {node.prediction} {'; *' if node.leaf else ''}")
        if node.left_child:
            self.traverse(node.left_child, level+1)
        if node.right_child:
            self.traverse(node.right_child, level+1)

# Example of creating (monotone) oblique classification tree

df = pd.read_csv('datasets/artificial/2features/2features.pickle_negative_labelling_9.csv')
TARGET = 'class'
from generator import Generator

X = df.loc[:, df.columns != TARGET]
y = df[TARGET]
X, _, _, y_train, _, _ = Generator.dataset_split(dataset=X, labels=y.to_numpy(), train_size=50, validation_size=0.5)
y_train = pd.DataFrame(y_train, columns=[TARGET])
y = y_train[TARGET]

not_local_not_global = Node(copy.deepcopy(X), copy.deepcopy(y), oblique=True, positive=False)
not_local_not_global.plot_leaves('Effect of not enforcing any monotonicity constraint')

local_global = Node(copy.deepcopy(X), copy.deepcopy(y), oblique=True, positive=True)
local_global.make_monotone_leaves()
local_global.ict_prune()
local_global.plot_leaves('Effect of enforcing the local but not the global monotonicity constraint')
