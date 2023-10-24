import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
=======
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
>>>>>>> 2a4f40091fa2e93303b1128bff160b32df9d6422
from sklearn.impute import SimpleImputer

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

<<<<<<< HEAD
# Average CV score on the training set was: 0.8265199161425578
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=1, min_samples_split=16, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)
=======
# Average CV score on the training set was: 0.8284416491963661
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.8, min_samples_leaf=3, min_samples_split=17, n_estimators=150)),
    LinearSVC(C=0.5, dual=True, loss="hinge", penalty="l2", tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)
>>>>>>> 2a4f40091fa2e93303b1128bff160b32df9d6422

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
