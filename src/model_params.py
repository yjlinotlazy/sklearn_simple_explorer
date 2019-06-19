"""Tune parameters here for parameter search"""

PARAMETERS = {
    'ctg': {
        'tree': {"max_features": [0.05, 0.1, 0.2, 0.3],
                 "min_samples_split": [10, 20, 30],
                 "criterion": ["gini", "entropy"]},
        'boosting': {'base_estimator__criterion': ["gini", "entropy"],
                        'base_estimator__max_depth': [3, 5, 7, 9],
                        'n_estimators': [10, 30, 50, 100]},
        'knn': {'n_neighbors': [5, 10, 20, 30, 40, 50],
                'weights': ['uniform', 'distance']},
        'svm': {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.01, 0.1],
                'max_iter': [10000, 100000, 1000000],
                'gamma': [0.01, 0.001, 0.1]},
        'nn': {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
               'solver': ['sgd', 'adam'],
               'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive']}
    },
    'wine': {
        'tree': {"max_features": [0.05, 0.1, 0.2, 0.3],
                 "min_samples_split": [10, 20, 30],
                 "criterion": ["gini", "entropy"]},
        'boosting': {'base_estimator__criterion': ["gini", "entropy"],
                        'base_estimator__max_depth': [3, 5, 7, 9],
                        'n_estimators': [10, 30, 50, 100]},
        'knn': {'n_neighbors': [10, 20, 30, 40, 50],
                'weights': ['uniform', 'distance']},
        'svm': {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.01, 0.1],
                'max_iter': [10000, 100000, 1000000],
                'gamma': [0.01, 0.001, 0.1]},
        'nn': {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
               'solver': ['sgd', 'adam'],
               'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive']}
    }
}