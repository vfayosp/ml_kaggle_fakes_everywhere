import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_B():
    df=pd.read_csv('../database/train_B_text_processed.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0'], axis=1)
    return X,Y

def read_B_distance():
    df=pd.read_csv('../database/train_B_text_processed_distance.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    return X,Y

def read_B_perplexity():
    df=pd.read_csv('../database/train_B_text_processed_perplexity_chat.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    #X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # turn 'Magic bit' into a column with value 1 only when 'Magic bit' is equal to 1
    X['Magic bit'] = df['Magic bit'].apply(lambda x: 1 if x == 1 else 0)

    # turn length column into 1 only of length less then 5 or greater then 12
    X['length'] = df['length'].apply(lambda x: 1 if x < 5 or x > 12 else 0)

    return X,Y

X,Y = read_B_perplexity()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42, stratify=Y)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

def scale_features(X, col_names):
        scaled_features = X.copy()
        features = X[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        features = (features + 1.0) / 2.0
        scaled_features[col_names] = features
        X = scaled_features
        # scale X from -1, 1 to 0, 1
        return X, scaler

def scale_features_scaler(X, col_names, scaler):
        scaled_features = X.copy()
        features = X[col_names]
        features = scaler.transform(features.values)
        features = (features + 1.0) / 2.0
        scaled_features[col_names] = features
        X = scaled_features
        return X


# Logical or column
X_train['has_dots_OR_noun_low'] = X_train['has_dots'] | X_train['has_any_noun_verb_lower']
X_test['has_dots_OR_noun_low'] = X_test['has_dots'] | X_test['has_any_noun_verb_lower']
X_train.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)
X_test.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)


X_train, scaler = scale_features(X_train, ['closest_emb'])
X_test = scale_features_scaler(X_test, ['closest_emb'], scaler)

#X_train, scaler = scale_features(X_train, ['mean_word_len'])
#X_test = scale_features_scaler(X_test, ['mean_word_len'], scaler)

X_train, scaler = scale_features(X_train, ['length'])
X_test = scale_features_scaler(X_test, ['length'], scaler)

X_train, scaler = scale_features(X_train, ['count_quotes'])
X_test = scale_features_scaler(X_test, ['count_quotes'], scaler)

# clip 'perplexity' column to 500
X_train['perplexity'] = X_train['perplexity'].clip(upper=500)
X_test['perplexity'] = X_test['perplexity'].clip(upper=500)
X_train, scaler = scale_features(X_train, ['perplexity'])
X_test = scale_features_scaler(X_test, ['perplexity'], scaler)

for colum in ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT']:
    X_train, scaler = scale_features(X_train, [colum])
    X_test = scale_features_scaler(X_test, [colum], scaler)


'''
# Create a column equal to 'closest_emb'*'heuristic_label_closest'
X_train['closest_emb*1-heuristic_label_closest'] = X_train['closest_emb']*(1.0 - X_train['heuristic_label_closest'])
X_test['closest_emb*1-heuristic_label_closest'] = X_test['closest_emb']*(1.0 - X_test['heuristic_label_closest'])

X_train['1-closest_emb*heuristic_label_closest'] = (1.0 - X_train['closest_emb'])*X_train['heuristic_label_closest']
X_test['1-closest_emb*heuristic_label_closest'] = (1.0 - X_test['closest_emb'])*X_test['heuristic_label_closest']

X_train['closest_emb*perplexity'] = X_train['closest_emb']*X_train['perplexity']
X_test['closest_emb*perplexity'] = X_test['closest_emb']*X_test['perplexity']

X_train['mean_word_len*perplexity'] = X_train['mean_word_len']*X_train['perplexity']
X_test['mean_word_len*perplexity'] = X_test['mean_word_len']*X_test['perplexity']

X_train['1-mean_word_len*perplexity*magic bit*1-closest'] = (1.0 - X_train['mean_word_len'])*X_train['perplexity']*X_train['Magic bit']*(1.0 - X_train['closest_emb'])
X_test['1-mean_word_len*perplexity*magic bit*1-closest'] = (1.0 - X_test['mean_word_len'])*X_test['perplexity']*X_test['Magic bit']*(1.0 - X_test['closest_emb'])
'''

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)
X_test  = X_test.drop(['Fake/Real'], axis=1)


columns = X_train.columns
print("COLUMNS: ", columns)

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
    'xgboost.XGBClassifier': {
        'n_estimators': np.arange(100,500,10),
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'colsample_bytree': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'lambda': np.arange(0.00, 1.01, 0.05),  # Regularization term
    	'alpha': np.arange(0.00, 1.01, 0.05),  # Regularization term
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

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
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

pipeline_optimizer = TPOTClassifier(generations=5, population_size=100, cv=10,
                                    random_state=42, verbosity=2, 
                                    config_dict=classifier_config_dict,
                                    scoring='balanced_accuracy')

pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_B_exported_pipeline.py')
print("Test data: ", pipeline_optimizer.score(X_test, y_test))
