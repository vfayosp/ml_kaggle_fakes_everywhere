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

############################## Cross val ###############################

cv_params = {
    'metrics': 'error', # logloss, error, etc...
    'nfold': 5,
    'num_boost_round': 1000,  
    'early_stopping_rounds': 50,
}

xgb_parms = {
    'objective': 'binary:logistic',
    'num_boost_round': 52,
    'seed': 0,
    'learning_rate': 0.1,  
    'max_depth': 4,  # e.g., 3, 6, 9
    'subsample': 1.0,  # between 0.6 and 1.0
    'colsample_bytree': 0.6,  # between 0.6 and 1.0
    'lambda': 0.9,  # Regularization term
    'alpha': 0.8,  # Regularization term
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

plt.barh(columns, xgb.feature_importances_)
plt.show()

############################ Save output #############################
output = np.where(y_test_pred == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'
output.to_csv('../output/xgboost_A.csv')













