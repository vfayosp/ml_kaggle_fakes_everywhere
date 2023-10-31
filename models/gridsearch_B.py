import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def read_B():
    df=pd.read_csv('../database/train_B_text_processed.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0'], axis=1)
    return X,Y

def read_B_distance():
    df=pd.read_csv('../database/train_B_text_processed_distance.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0', 'Unnamed: 0.1', 'ADJ', 'ADV', 'NOUN', 'VERB',
       'PROPN', 'PUNCT'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)
    return X,Y

X,Y = read_B_distance()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42, stratify=Y)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

# Drop target column from X
X_train = X_train.drop(['Fake/Real'], axis=1)
X_test  = X_test.drop(['Fake/Real'], axis=1)
columns = X_train.columns

print("COLUMNS: ", columns)

#############################################################################

# Define the XGBoost model
xgb_model = xgb.XGBClassifier()

# Define the hyperparameters grid for the grid search
param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 9],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'lambda': [0.1, 0.2, 0.3, 0.4, 0.5],
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',  # Choose an appropriate scoring metric
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
