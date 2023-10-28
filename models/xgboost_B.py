import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


########################## Preprocessing ############################

def read_B():
    df=pd.read_csv('../database/train_B_text_processed.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id'], axis=1)
    return X,Y

X,Y = read_B()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42, stratify=Y)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)
X_test  = X_test.drop(['Fake/Real'], axis=1)
columns = X_train.columns

############################## Cross val ###############################

cv_params = {
    'metrics': 'error', # logloss, error, etc...
    'nfold': 5,
    'num_boost_round': 1000,  
    'early_stopping_rounds': 50,
}

xgb_parms = {
    'objective': 'binary:logistic',
    'n_estimators': 28,
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













