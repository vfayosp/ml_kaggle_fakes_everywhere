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
    X = df.drop(['Title', 'Id', 'Unnamed: 0'], axis=1)
    return X,Y

def read_B_distance():
    df=pd.read_csv('../database/train_B_text_processed_distance.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    return X,Y

def read_B_perplexity():
    df=pd.read_csv('../database/train_B_text_processed_perplexity_chat.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Title', 'Id', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # turn 'Magic bit' into a column with value 1 only when 'Magic bit' is equal to 1
    X['Magic bit'] = df['Magic bit'].apply(lambda x: 1 if x == 1 else 0)

    # turn length column into 1 only of length less then 5 or greater then 12
    X['length'] = df['length'].apply(lambda x: 1 if x < 5 or x > 12 else 0)

    return X,Y

X,Y = read_B_perplexity()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42, stratify=Y)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

def scale_features(X, col_names):
        scaled_features = X.copy()
        features = X[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        X = scaled_features
        # scale X from -1, 1 to 0, 1
        #X = (X + 1.0) / 2.0
        return X, scaler

def scale_features_scaler(X, col_names, scaler):
        scaled_features = X.copy()
        features = X[col_names]
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        X = scaled_features
        #X = (X + 1.0) / 2.0
        return X


# Logical or column
X_train['has_dots_OR_noun_low'] = X_train['has_dots'] | X_train['has_any_noun_verb_lower']
X_test['has_dots_OR_noun_low'] = X_test['has_dots'] | X_test['has_any_noun_verb_lower']
X_train.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)
X_test.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)


X_train, scaler = scale_features(X_train, ['closest_emb'])
X_test = scale_features_scaler(X_test, ['closest_emb'], scaler)

X_train, scaler = scale_features(X_train, ['mean_word_len'])
X_test = scale_features_scaler(X_test, ['mean_word_len'], scaler)

#X_train, scaler = scale_features(X_train, ['length'])
#X_test = scale_features_scaler(X_test, ['length'], scaler)

# clip 'perplexity' column to 500
X_train['perplexity'] = X_train['perplexity'].clip(upper=500)
X_test['perplexity'] = X_test['perplexity'].clip(upper=500)
X_train, scaler = scale_features(X_train, ['perplexity'])
X_test = scale_features_scaler(X_test, ['perplexity'], scaler)

#for colum in ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT']:
#    X_train, scaler = scale_features(X_train, [colum])
#    X_test = scale_features_scaler(X_test, [colum], scaler)


'''
# Create a column equal to 'closest_emb'*'heuristic_label_closest'
X_train['closest_emb*1-heuristic_label_closest'] = X_train['closest_emb']*(1.0 - X_train['heuristic_label_closest'])
X_test['closest_emb*1-heuristic_label_closest'] = X_test['closest_emb']*(1.0 - X_test['heuristic_label_closest'])

X_train['1-closest_emb*heuristic_label_closest'] = (1.0 - X_train['closest_emb'])*X_train['heuristic_label_closest']
X_test['1-closest_emb*heuristic_label_closest'] = (1.0 - X_test['closest_emb'])*X_test['heuristic_label_closest']

X_train['closest_emb*perplexity'] = X_train['closest_emb']*X_train['perplexity']
X_test['closest_emb*perplexity'] = X_test['closest_emb']*X_test['perplexity']

X_train['mean_word_len*perplexity'] = X_train['mean_word_len']*X_train['perplexity']
X_test['mean_word_len*perplexity'] = X_test['mean_word_len']*X_test['perplexity']

X_train['1-mean_word_len*perplexity*magic bit*1-closest'] = (1.0 - X_train['mean_word_len'])*X_train['perplexity']*X_train['Magic bit']*(1.0 - X_train['closest_emb'])
X_test['1-mean_word_len*perplexity*magic bit*1-closest'] = (1.0 - X_test['mean_word_len'])*X_test['perplexity']*X_test['Magic bit']*(1.0 - X_test['closest_emb'])


# Drop target column from X
X_train = X_train.drop(['Fake/Real', 'has_apostrophe_s',
       'has_apostrophe', 'has_number', 'has_comma', 'has_colon',
       'has_parenthesis', 'has_hyphen', 'has_and', 'has_percentage',
       'has_interrogation', 'has_exclamation', 'ADJ', 'ADV', 'NOUN', 'VERB',
       'PROPN', 'PUNCT',
       'closest_emb', 'heuristic_label_closest', 'mean_word_len'], axis=1)
X_test  = X_test.drop(['Fake/Real', 'has_apostrophe_s',
       'has_apostrophe', 'has_number', 'has_comma', 'has_colon',
       'has_parenthesis', 'has_hyphen', 'has_and', 'has_percentage',
       'has_interrogation', 'has_exclamation', 'ADJ', 'ADV', 'NOUN', 'VERB',
       'PROPN', 'PUNCT', 
       'closest_emb', 'heuristic_label_closest', 'mean_word_len'], axis=1)
'''

X_train = X_train.drop(['Fake/Real'], axis=1)
X_test  = X_test.drop(['Fake/Real'], axis=1)

columns = X_train.columns
print("COLUMNS: ", columns)

############################## Cross val ###############################

cv_params = {
    'metrics': 'error', # logloss, error, etc...
    'nfold': 5,
    'num_boost_round': 1000,  
    'early_stopping_rounds': 50,
}

xgb_parms = {
    'objective': 'binary:logistic',
    'n_estimators': 100,
    'seed': 69,
    'learning_rate': 0.1,  
    'max_depth': 7,  # e.g., 3, 6, 9
    'min_child_weight': 8,
    'subsample': 0.8,  # between 0.6 and 1.0
    'colsample_bytree': 0.8,  # between 0.6 and 1.0
    'lambda': 0.0,  # Regularization term
    'alpha': 0.0,  # Regularization term
}

xgb_model = xgb.XGBClassifier(
    **xgb_parms,
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













