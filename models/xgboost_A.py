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
db = pd.read_csv('../database/train_A_derma.csv')

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

    
# Compute ratio sum(negative instances) / sum(positive instances)
X_train, y_train = X, Y
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)
X_train = X_train.drop(['Fake/Real'], axis=1)

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
    'num_boost_round': 24,
    'seed': 0,
    'learning_rate': 0.3,  
    'max_depth': 3,  # e.g., 3, 6, 9
    'subsample': 1.0,  # between 0.6 and 1.0
    'colsample_bytree': 0.6,  # between 0.6 and 1.0
    'lambda': 0.7,  # Regularization term
    'alpha': 0.0,  # Regularization term
}


def scale_features(X, col_names):

    scaled_features = X.copy()
    features = X[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    X = scaled_features

    return X

X_train = scale_features(X_train, ['Genetic Propensity'])

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

print("\nTrain accuracy: ", accuracy_score(y_train, y_train_pred))

plt.barh(columns, xgb.feature_importances_)
plt.show()















