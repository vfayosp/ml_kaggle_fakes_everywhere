import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

db = pd.read_csv('../database/train_A_derma.csv')

#Preprocessing
db = pd.read_csv('../database/train_A_derma.csv')

db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})
db.drop(['Id', 'Doughnuts consumption'],axis=1, inplace=True)

db['Num NaN'] = db.isnull().sum(axis=1)

Y = db['Fake/Real']
X = db.drop(['Fake/Real'], axis=1)

#x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.85, random_state=44, stratify=Y)

xgb_model = xgb.XGBClassifier()

dtrain = xgb.DMatrix(X, label=Y)

cv = xgb.cv(xgb_model.get_params(), dtrain=dtrain, nfold=5)

#y_train_pred = xgb_model.predict(x_train)
#y_val_pred = xgb_model.predict(x_val)

#print('Accuracy train: ', accuracy_score(y_train, y_train_pred))
#print('Accuracy validation: ', accuracy_score(y_val, y_val_pred))

#cm = confusion_matrix(y_val,y_val_pred)
print(cv)

