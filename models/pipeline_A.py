import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

########################## Preprocessing ############################

db      = pd.read_csv('../database/train_A_derma_one_hot_nan.csv')
db_test = pd.read_csv('../database/test_A_derma_one_hot_nan.csv')

X_train = db
y_train = db['Fake/Real']
X_test = db_test

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real', 'Unnamed: 0', 'Lession'], axis=1)
X_test = X_test.drop(['Unnamed: 0', 'Lession'], axis=1)

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
print("COLUMNS: ", columns)

training_target = y_train
training_features = X_train
testing_features = X_test

###############################################################################

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.8394995935866378
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=7, max_features=0.25, min_samples_leaf=4, min_samples_split=13, n_estimators=100, subsample=0.35000000000000003)),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=18, min_samples_split=4, n_estimators=300)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

############################ Save output #############################
output = np.where(results == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'
output.to_csv('../output/output_A.csv')
