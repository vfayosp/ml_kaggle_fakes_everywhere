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


########################## Preprocessing ############################

db      = pd.read_csv('../database/train_B_text_processed.csv')
db_test = pd.read_csv('../database/test_B_text_processed.csv')

X_train = db
y_train = db['Fake/Real']
X_test = db_test

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)

X_train = X_train.drop(['Title', 'Id','Unnamed: 0'], axis=1)
X_test  = X_test.drop(['Title','Unnamed: 0'], axis=1)

columns = X_train.columns

training_target = y_train
training_features = X_train
testing_features = X_test

############################## Train ###############################

# Average CV score on the training set was: 0.9258736941634386
exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=5, max_features=0.05, min_samples_leaf=13, min_samples_split=6, n_estimators=400, subsample=1.0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
results_train = exported_pipeline.predict(training_features)

print("Train accuracy: ", accuracy_score(results_train, training_target))

############################ Save output #############################
output = np.where(results == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'
output.to_csv('../output/output_B.csv')













