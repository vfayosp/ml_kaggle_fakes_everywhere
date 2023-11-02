import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
import xgboost as xgb 
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import StackingClassifier

########################## Preprocessing ############################



def read_B_perplexity():
    df=pd.read_csv('../database/train_B_text_processed_perplexity_chat.csv')
    Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})
    X = df.drop(['Id', 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # turn 'Magic bit' into a column with value 1 only when 'Magic bit' is equal to 1
    X['Magic bit'] = df['Magic bit'].apply(lambda x: 1 if x == 1 else 0)

    # turn length column into 1 only of length less then 5 or greater then 12
    X['length'] = df['length'].apply(lambda x: 1 if x < 5 or x > 12 else 0)

    return X,Y
    
def read_B_perplexity_test():
    df=pd.read_csv('../database/test_B_processed_perplexity_chat.csv')
    X = df.drop([ 'Unnamed: 0.1'], axis=1)
    for i in range(0,1536):
        X = X.drop(['emb'+str(i)], axis=1)

    # add column with mean of words lenght of each title
    X['mean_word_len'] = df['Title'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # turn 'Magic bit' into a column with value 1 only when 'Magic bit' is equal to 1
    X['Magic bit'] = df['Magic bit'].apply(lambda x: 1 if x == 1 else 0)

    # turn length column into 1 only of length less then 5 or greater then 12
    X['length'] = df['length'].apply(lambda x: 1 if x < 5 or x > 12 else 0)

    return X

X_train, y_train = read_B_perplexity()
X_test = read_B_perplexity_test()


# Precompute duplicate predictions
print(X_train)
all_train = X_train[['Title', 'Fake/Real']].values.tolist()
# turn duplicates into a dictionary title: id
all_train = {k:v for k,v in all_train}
test_duplicates = {}
print(X_test)
for title, id in X_test[['Title', 'Unnamed: 0']].values.tolist():
    if title in all_train:
        test_duplicates[id] = all_train[title]
      

# Logical or column
X_train['has_dots_OR_noun_low'] = X_train['has_dots'] | X_train['has_any_noun_verb_lower']
X_test['has_dots_OR_noun_low'] = X_test['has_dots'] | X_test['has_any_noun_verb_lower']
X_train.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)
X_test.drop(['has_dots', 'has_any_noun_verb_lower'], axis=1, inplace=True)

# Compute ratio sum(negative instances) / sum(positive instances)
ratio = np.sum(X_train['Fake/Real'] == 0) / np.sum(X_train['Fake/Real'] == 1)

def scale_features(X, col_names):
        scaled_features = X.copy()
        features = X[col_names].astype(float)
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        features = (features + 1.0) / 2.0
        scaled_features[col_names] = features
        X = scaled_features
        return X, scaler

def scale_features_scaler(X, col_names, scaler):
        scaled_features = X.copy()
        features = X[col_names].astype(float)
        features = scaler.transform(features.values)
        features = (features + 1.0) / 2.0
        scaled_features[col_names] = features
        X = scaled_features
        return X


X_train, scaler = scale_features(X_train, ['closest_emb'])
X_test = scale_features_scaler(X_test, ['closest_emb'], scaler)

X_train, scaler = scale_features(X_train, ['mean_word_len'])
X_test = scale_features_scaler(X_test, ['mean_word_len'], scaler)

X_train, scaler = scale_features(X_train, ['length'])
X_test = scale_features_scaler(X_test, ['length'], scaler)

# clip 'perplexity' column to 500
X_train['perplexity'] = X_train['perplexity'].clip(upper=500)
X_test['perplexity'] = X_test['perplexity'].clip(upper=500)
X_train, scaler = scale_features(X_train, ['perplexity'])
X_test = scale_features_scaler(X_test, ['perplexity'], scaler)

for colum in ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT']:
    X_train, scaler = scale_features(X_train, [colum])
    X_test = scale_features_scaler(X_test, [colum], scaler)


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

X_train = X_train.drop(['Fake/Real', 'Title','Unnamed: 0'], axis=1)
X_test  = X_test.drop(['Title', 'Unnamed: 0'], axis=1)

columns = X_train.columns
print("COLUMNS: ", columns)

columns = X_test.columns
print("COLUMNS: ", columns)

##################################

training_target = y_train
training_features = X_train
testing_features = X_test

############################## Train ###############################
# Average CV score on the training set was: 0.5

xgb_parms = {
    'n_estimators': 400,
    'seed': 42,
    'learning_rate': 0.1,  
    'max_depth': 3,  # e.g., 3, 6, 9
    'min_child_weight': 2,
    'subsample': 1.0,  # between 0.6 and 1.0
    'colsample_bytree': 0.8,  # between 0.6 and 1.0
    'lambda': 0.0,  # Regularization term
    'alpha': 0.0,  # Regularization term
}

estimator = [
    ('tree', ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=1.0, min_samples_leaf=7, min_samples_split=11, n_estimators=250)),
    ('xgbclassifier', xgb.XGBClassifier())
]
exported_pipeline = StackingClassifier(estimators=estimator)


exported_pipeline.fit(training_features, training_target)
results_train = exported_pipeline.predict(training_features)
results  = exported_pipeline.predict(testing_features)


print(accuracy_score(results_train, training_target))
############################ Save output ############################

output = np.where(results == 1, 'fake', 'real')
output = pd.DataFrame(output, columns=['Predictions'])
output.index.name = 'Id'

# We set the duplicates that we saved earlier
for id, label in test_duplicates.items():
    output.loc[id] = 'fake' if label == 1 else 'real'

output.to_csv('../output/output_B.csv')

# Print Title from samples that where incorrectly classified
db      = pd.read_csv('../database/train_B_text_processed_distance.csv')
for i in range(len(results_train)):
    if results_train[i] != y_train[i]:
        print("Title: ", db['Title'][i])
        print("Predicted: ", results_train[i])
        print("Real: ", y_train[i])
        print("")













