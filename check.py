import z3

class Check:
    def __init__(self, categorical_features, types) -> None:
        self.categorical_features = categorical_features
        self.types = types
    
    def dominates(self, leaf1, leaf2) -> bool:
        """
        Determine whether leaf1 contains a point that dominates a point in leaf2
        """
        # Initiate solver and formulate the regular constraints
        s = z3.Solver()
        constraints = []

        # Categorical variables are non-empty modifier
        empty_Z = False
        # First index corresponds to leaf1
        rules = [leaf1.rules, leaf2.rules]
        features = [[], []]

        # Dictionary to map feature names
        # Store all feature names of both leaves
        # Declare all features
        feature_dicts = [{}, {}]
        for i in range(len(feature_dicts)):
            for (features_coefs, direction, _) in rules[i]:
                if direction != 'in' and direction != 'not in':
                    for (feature, _) in features_coefs:
                        features[i].append(feature)
            features[i] = set(features[i])
            for feature in features[i]:
                if self.types and feature in self.types:
                    if self.types[feature]['type'] != 'categorical':
                        if self.types[feature]['type'] == 'real':
                            feature_dicts[i][feature] = z3.Real(f'{feature}' + f'{i + 1}')
                        elif self.types[feature]['type'] == 'int':
                            feature_dicts[i][feature] = z3.Int(f'{feature}' + f'{i + 1}')
                        else: # Binary
                            feature_dicts[i][feature] = z3.Int(f'{feature}' + f'{i + 1}')
                            constraints.append(feature_dicts[i][feature] >= 0)
                            constraints.append(feature_dicts[i][feature] <= 1)
                        if 'min' in self.types[feature]:
                            constraints.append(feature_dicts[i][feature] >= self.types[feature]['min'])
                        if 'max' in self.types[feature]:
                            constraints.append(feature_dicts[i][feature] <= self.types[feature]['max'])
                else:
                    feature_dicts[i][feature] = z3.Real(f'{feature}' + f'{i + 1}')

        # Determine if there exists an empty overlap between some categorical feature
        if self.categorical_features != []:
            categorical_features = [[], []]
            categorical_features_dicts = [{}, {}]
            for i in range(len(categorical_features_dicts)):
                for (feature, direction, _) in rules[i]:
                    if direction == 'in' or direction == 'not in':
                        categorical_features[i].append(feature)
                categorical_features[i] = set(categorical_features[i])
                for feature in categorical_features[i]:
                    categorical_features_dicts[i][feature] = []
            for i in range(len(categorical_features_dicts)):
                for (feature, direction, value) in rules[i]:
                    if direction == 'in' or direction == 'not in':
                        categorical_features_dicts[i][feature].append(f"{value};{direction}")
            for feature1, values1 in categorical_features_dicts[0].items():
                for feature2, values2 in categorical_features_dicts[1].items():
                    if feature1 == feature2:
                        set1 = set(values1) 
                        set2 = set(values2)
                        if len(set1.intersection(set2)) == 0:
                            empty_Z = True
                        if not (any([value1.split(';')[0] != value2.split(';')[0] and value1.split(';')[1] != value2.split(';') for value1 in values1 for value2 in values2]) is True):
                            empty_Z = True

        # Early stopping
        if empty_Z:
            return False
 
        for i in range(len(feature_dicts)):
            for (features_coefs, direction, value) in rules[i]:
                # Excluding categorical variables
                if direction != 'in' and direction != 'not in':
                    constraint = 0
                    for _, (feature, coefficient) in enumerate(features_coefs):
                        constraint = constraint + coefficient * feature_dicts[i][f"{feature}"]
                        # if idx is not len(features_coefs) - 1:
                        #     constraint = constraint + '+'
                    if (direction == '<='):
                        constraint = constraint <= value
                    else:
                        constraint = constraint > value
                    constraints.append(constraint)
        
        # Add constraint that point of leaf1 should be greater than point of leaf2
        intersection = features[0].intersection(features[1])
        for feature in intersection:
            constraints.append(feature_dicts[0][f"{feature}"] >= feature_dicts[1][f"{feature}"])

        # Add constraints to solver
        for constraint in constraints:
            s.add(constraint)

        # No solution returns false
        if s.check() == z3.CheckSatResult(-1):
            return False
        else:
            return True
        