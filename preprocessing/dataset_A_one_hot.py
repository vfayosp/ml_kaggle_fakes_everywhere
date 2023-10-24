import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


DATASET_TRAIN = '../database/train_A_derma.csv'
DATASET_TEST = '../database/test_A_derma.csv'


def preprocess(db):

# Replace real/fake with 0/1, drop Id and Doughnuts consumption
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})
db.drop(['Id', 'Doughnuts consumption'],axis=1, inplace=True)

# Add Num NaN column
db['Num NaN'] = db.isnull().sum(axis=1)

X = db
Y = db['Fake/Real']

# Change column type to category (to avoid warnings)
for column in ['Lession', 'Skin X test', 'Skin color', 'Small size', 
               'Mid size', 'Large size', 'Small', 'Mid', 'Large']:
    X[column] = X[column].astype("category")

# Add one-hot encoded columns for NaN values
for column in X.columns: 
    if column == 'Fake/Real' or column == 'Num NaN': 
        continue
    one_hot_encoded = pd.get_dummies(X[column].isna(), prefix=column+'_isNaN', dtype=int)
    X = pd.concat([X, one_hot_encoded[column+'_isNaN'+'_True']], axis=1) # get only one column


