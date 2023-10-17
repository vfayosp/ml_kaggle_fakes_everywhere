import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

db = pd.read_csv('../database/train_A_derma_negative_nan.csv')

Y = db['Fake/Real']
X = db.drop(['Id','Fake/Real','Doughnuts consumption'], axis=1)


x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.85, random_state=44, stratify=Y)

rf_model = RandomForestClassifier(n_estimators=200, criterion='log_loss',max_depth=3,n_jobs=-1, verbose=1)

rf_model.fit(x_train, y_train)

y_train_pred = rf_model.predict(x_train)
y_val_pred = rf_model.predict(x_val)

print('Accuracy train: ', accuracy_score(y_train, y_train_pred))
print('Accuracy validation: ', accuracy_score(y_val, y_val_pred))

cm = confusion_matrix(y_val,y_val_pred)
print(cm)

