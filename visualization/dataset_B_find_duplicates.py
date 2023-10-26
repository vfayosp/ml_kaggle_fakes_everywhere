import numpy as np
import pandas as pd
import spacy 


db = pd.read_csv('../database/train_B_text.csv')
db = db.drop(['Id'], axis=1)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})

db_test = pd.read_csv('../database/test_B_text.csv')
db_test = db_test.drop(['Id'], axis=1)

# get the 'Title' column as a list
titles = set(db['Title'].tolist())
titles_test = set(db_test['Title'].tolist())

count = 0
for title in titles_test:
    if title in titles:
        if '"' not in title:
            print(title)
            count += 1
print(count)
