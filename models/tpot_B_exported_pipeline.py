import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../database/train_B_text_processed.csv')
features = tpot_data.drop(['Fake/Real', 'Id', 'Title'], axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Fake/Real'], test_size=0.1, random_state=42)

# Average CV score on the training set was: 0.9193109028300521
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.1),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.25, min_samples_leaf=16, min_samples_split=4, n_estimators=100, subsample=0.8500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(accuracy_score(results, testing_target))
