from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import xgboost as xgb
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(model, input):
    input = input.split(' ')
    embedding = tf.reduce_mean(model(input), axis=0).numpy().tolist()
    return embedding

db = pd.read_csv('../database/train_B_text.csv')

columns = []
for i in range(0,512):
    columns.append('emb'+str(i))

db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})
values=db['Title'].apply(embed)
emb_db = pd.DataFrame(values.values.tolist(), columns=columns)

X=emb_db
Y=db['Fake/Real']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42, stratify=Y)

ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

cv_params = {
    'metrics': 'error', # logloss, error, etc...
    'nfold': 5,
    'num_boost_round': 1000,  
    'early_stopping_rounds': 50,
}

xgb_parms = {
    'objective': 'binary:logistic',
    'n_estimators': 9,
    'seed': 0,
    'learning_rate': 0.1,  
    'max_depth': 3,  # e.g., 3, 6, 9
    'subsample': 0.9,  # between 0.6 and 1.0
    'colsample_bytree': 0.6,  # between 0.6 and 1.0
    'lambda': 0.1,  # Regularization term
    'alpha': 0.0,  # Regularization term
}

xgb_model = xgb.XGBClassifier(
    **xgb_parms,
    scale_pos_weight=ratio,
    enable_categorical=True,
)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
cv = xgb.cv(
    xgb_model.get_params(), 
    dtrain=dtrain, 
    stratified=True,
    **cv_params)
print(cv)
#cm = confusion_matrix(y_val,y_val_pred)


############################ Train & test #############################

xgb = xgb.XGBClassifier(
    **xgb_parms,
    scale_pos_weight=ratio,
    enable_categorical=True,
)
xgb.fit(X_train, y_train)
y_train_pred = xgb.predict(X_train)
y_test_pred  = xgb.predict(X_test)

print()
print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test accuracy: ", accuracy_score(y_test, y_test_pred))

plt.barh(columns, xgb.feature_importances_)
plt.show()
