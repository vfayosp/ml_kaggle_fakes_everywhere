import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_A():
    df=pd.read_csv('../database/train_A_derma.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    #X = df.drop(['Fake/Real', 'Doughnuts consumption', 'Id'], axis=1)
    X = df.drop(['Doughnuts consumption', 'Id'], axis=1)
    return X,Y

def read_A_size():
    df=pd.read_csv('../database/train_A_input_sizes.csv')
    Y = df['Fake/Real']
    X = df.drop(['Fake/Real', 'Mid', 'Small', 'Large'], axis=1)
    return X,Y

def read_A_one_hot():
    df=pd.read_csv('../database/train_A_derma_one_hot_nan.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df
    return X,Y


X,Y = read_A_one_hot()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real', 'Unnamed: 0'], axis=1)
X_test  = X_test.drop(['Fake/Real', 'Unnamed: 0'], axis=1)

def scale_features(X, col_names):
        scaled_features = X.copy()
        features = X[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        X = scaled_features
        return X, scaler

def scale_features_scaler(X, col_names, scaler):
        scaled_features = X.copy()
        features = X[col_names]
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        X = scaled_features
        return X

X_train, scaler = scale_features(X_train, ['Genetic Propensity'])
X_test = scale_features_scaler(X_test, ['Genetic Propensity'], scaler)
X_train, scaler = scale_features(X_train, ['Num NaN'])
X_test = scale_features_scaler(X_test, ['Num NaN'], scaler)

columns = X_train.columns
print("columns: ", columns)

################################# TPOT ##################################

classifier_config_dict = {

    # Classifiers
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': np.arange(100,500,50),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': np.arange(100,500,50),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': np.arange(100,500,50),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': np.arange(100,500,50),
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}

pipeline_optimizer = TPOTClassifier(generations=10, population_size=80, cv=10,
                                    random_state=42, verbosity=2, 
                                    config_dict=classifier_config_dict,
                                    scoring='balanced_accuracy')

pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_A_exported_pipeline.py')
print("Test data: ", pipeline_optimizer.score(X_test, y_test))
