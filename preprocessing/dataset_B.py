import numpy as np
import pandas as pd

db = pd.read_csv('../database/train_B_text.csv')

db = db.drop(['Id'], axis=1)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})

def process_title(title):
    quotes = title.count('"')
    s_quotes = title.count("'")
    title = title.replace('"','').replace("'",'')
    words = title.split()
    
    upper_letters = 0
    for word in words:
        if word[0].isupper():
            upper_letters += 1

    return quotes,s_quotes,upper_letters

db[['quotes', 's_quotes', 'upper_letters']] = db['Title'].apply(process_title).apply(pd.Series)

print(db)
