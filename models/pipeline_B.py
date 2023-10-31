import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier

########################## Preprocessing ############################

db      = pd.read_csv('../database/train_B_text_processed_distance.csv')
db_test = pd.read_csv('../database/test_B_text_processed_distance.csv')

X_train = db
y_train = db['Fake/Real']
X_test = db_test

# Precompute duplicate predictions
print(X_train)
all_train = X_train[['Title', 'Fake/Real']].values.tolist()
# turn duplicates into a dictionary title: id
all_train = {k:v for k,v in all_train}
test_duplicates = {}
print(X_test)
for title, id in X_test[['Title', 'Unnamed: 0']].values.tolist():
    if title in all_train:
        test_duplicates[id] = all_train[title]

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)

X_train = X_train.drop(['Title', 'Id','Unnamed: 0', 'Unnamed: 0.1'], axis=1)
for i in range(0,1536):
    X_train = X_train.drop(['emb'+str(i)], axis=1)
X_test  = X_test.drop(['Title','Unnamed: 0', 'Unnamed: 0.1'], axis=1)
for i in range(0,1536):
    X_test = X_test.drop(['emb'+str(i)], axis=1)

columns = X_train.columns
print(columns)

training_target = y_train
training_features = X_train
testing_features = X_test

############################## Train ###############################

# Average CV score on the training set was: 0.950309744126413
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features=0.25, min_samples_leaf=8, min_samples_split=20, n_estimators=350, subsample=0.45)),
    SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

results_train = exported_pipeline.predict(training_features)
print("Train accuracy: ", accuracy_score(results_train, training_target))

############################ Save output #############################
output = np.where(results == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'

# We set the duplicates that we saved earlier
for id, label in test_duplicates.items():
    output.loc[id] = 'fake' if label == 1 else 'real'

output.to_csv('../output/output_B.csv')













