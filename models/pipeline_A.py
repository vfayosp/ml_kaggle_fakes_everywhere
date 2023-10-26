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

########################## Preprocessing ############################

db      = pd.read_csv('../database/train_A_derma_one_hot_nan.csv')
db_test = pd.read_csv('../database/test_A_derma_one_hot_nan.csv')

X_train = db
y_train = db['Fake/Real']
X_test = db_test

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)

columns = X_train.columns

def scale_features(X, col_names):

    scaled_features = X.copy()
    features = X[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    X = scaled_features

    return X

X_train = scale_features(X_train, ['Genetic Propensity'])
X_test = scale_features(X_test, ['Genetic Propensity'])

training_target = y_train
training_features = X_train
testing_features = X_test

###############################################################################

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.8305865899528241
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.45, min_samples_leaf=2, min_samples_split=20, n_estimators=150)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

############################ Save output #############################
output = np.where(results == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'
output.to_csv('../output/output_A.csv')
