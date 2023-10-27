import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df=pd.read_csv('../database/train_A_derma.csv')
Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})

X = df.drop(['Fake/Real', 'Doughnuts consumption', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier()

# Define the hyperparameters grid for the grid search
param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9, 12, 15],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
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
grid_search.fit(X, Y)

# Print the best parameters and the corresponding score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)