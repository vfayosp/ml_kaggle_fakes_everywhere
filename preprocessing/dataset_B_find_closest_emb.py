import numpy as np
import pandas as pd
import scipy.spatial
import spacy

'''
Columns:

'count_quotes', 'has_quote_start', 'has_dots', 'has_apostrophe_s', \
    'has_apostrophe', 'has_number', 'has_comma', 'has_colon', 'has_parenthesis', \
    'has_hyphen', 'has_and', 'has_percentage', 'has_interrogation', 'has_exclamation', \
    'has_only_first_upper', 'has_any_noun_verb_lower', 'length', \
    'ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT'
+ 'emb0', 'emb2', ..., 'emb1535'
'''


TRAIN_DATASET = '../database/train_B_text_processed_embeddings_oai.csv'
TEST_DATASET = '../database/test_B_text_processed_embeddings_oai.csv'
OUTPUT_TRAIN_DATASET = '../database/train_B_text_processed_distance.csv'
OUTPUT_TEST_DATASET = '../database/test_B_text_processed_distance.csv'

nlp = spacy.load("en_core_web_sm")

###############################################################################

db = pd.read_csv(TRAIN_DATASET)
#db = db.drop(['Id'], axis=1)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})

db_test = pd.read_csv(TEST_DATASET)
#db_test = db_test.drop(['Id'], axis=1)

# join train and test dataset into a third dataset (for looking for the closest embedding)
db_all = pd.concat([db, db_test], ignore_index=True)

# Get a dictionary of title and label (1 or 0)
titles = db['Title'].tolist()
labels = db['Fake/Real'].tolist()
titles_labels_train = dict(zip(titles, labels))

'''
Get only rows where 'has_quote_start', 'has_dots', \
    'has_apostrophe', 'has_number', 'has_comma', 'has_colon', 'has_parenthesis', \
    'has_hyphen', 'has_and', 'has_percentage', \
    'has_only_first_upper', 'has_any_noun_verb_lower' is 0
'''
db = db[db['has_quote_start'] == 0]
db = db[db['has_dots'] == 0]
#db = db[db['has_number'] == 0]
#db = db[db['has_comma'] == 0]
#db = db[db['has_colon'] == 0]
db = db[db['has_parenthesis'] == 0]
#db = db[db['has_hyphen'] == 0]
#db = db[db['has_and'] == 0]
#db = db[db['has_percentage'] == 0]
#db = db[db['has_only_first_upper'] == 0]
db = db[db['has_any_noun_verb_lower'] == 0]
#db = db[db['length'] >= 6]
db = db[db['length'] <= 12]

print("Uncertain titles in train: , ", len(db))

# Get a list of real and a list of fake titles
real_titles = db[db['Fake/Real'] == 0]['Title'].tolist()
fake_titles = db[db['Fake/Real'] == 1]['Title'].tolist()

for i in range(len(real_titles)):
    print(real_titles[i])
print(".........................................................")
for i in range(len(fake_titles)):
    print(fake_titles[i])

print("num real titles: ", len(real_titles), " num fake titles: ", len(fake_titles))

############################# SORRY FOR REPEATED CODE !!! ################################

db = pd.read_csv(TEST_DATASET)
db = db[db['has_quote_start'] == 0]
db = db[db['has_dots'] == 0]
#db = db[db['has_number'] == 0]
#db = db[db['has_comma'] == 0]
#db = db[db['has_colon'] == 0]
db = db[db['has_parenthesis'] == 0]
#db = db[db['has_hyphen'] == 0]
#db = db[db['has_and'] == 0]
#db = db[db['has_percentage'] == 0]
#db = db[db['has_only_first_upper'] == 0]
db = db[db['has_any_noun_verb_lower'] == 0]
#db = db[db['length'] >= 6]
db = db[db['length'] <= 12]

# get titles as a list
titles_test = db['Title'].tolist()
print("Uncertain titles in test: , ", len(titles_test))

###############################################################################

# Get all the embeddings columns and store it in a numpy array
embs = []
for i in range(len(db_all)):
    emb = db_all.iloc[i][25:25+1536].to_numpy()
    assert len(emb) == 1536
    embs.append(emb)
embs = np.array(embs)


def heuristic_label(title):
    if '"' == title[0]:
        return 1.0
    if "..." in title:
        return 0.0
    lower_noun = False
    for token in nlp(title):
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            if token.text[0].islower():
                lower_noun = True
                break
    if lower_noun:
        return 0.0
    if len(title.split()) > 12:
        return 0.0
    if len(title.split()) < 6:
        return 0.0
    if ":" in title:
        return 0.0
    if "-" in title:
        return 0.0
    if "&" in title or "!" in title or "?" in title:
        return 0.0
    if "(" in title or ")" in title:
        return 0.0
    if "'" in title:
        return 0.0
    if any(char.isdigit() for char in title):
        return 0.0
    if ',' in title:
        return 0.0
    if '%' in title:
        return 0.0  
    words = title.split()
    all_upper = True
    for word in words:
        if not word[0].isupper() and word[0].isalpha():
            all_upper = False
            break
    if not all_upper:
        return 0.0  
    return 1.0



def process_dataset(db_, db_all, titles_labels_train, real_titles, fake_titles, start_column, start_colum_all):
    # Find the closest embedding for each of the remaining titles,
    # using the columns 'emb1', 'emb2', ..., 'emb512'
    # and store the result in a new column 'closest_emb'
    # use scipy.spatial.distance.cosine
    print("Finding closest embeddings...")
    db_['closest_emb'] = None
    db_['heuristic_label_closest'] = None
    for i in range(len(db_)):
        
        title = db_.iloc[i]['Title']
        if title not in real_titles and title not in fake_titles:
            db_.loc[i, 'closest_emb'] = 0.0
            db_.loc[i, 'heuristic_label_closest'] = 0.0
            continue

        print("i: ", i, " out of ", len(db_))
        print("title: ", title)
        title_emb = db_.iloc[i][start_colum_all:(start_colum_all+1536)].to_numpy()
        assert len(title_emb) == 1536
        min_dist = 1
        min_dist_title = ""
        for j in range(len(db_all)):
            title2 = db_all.iloc[j]['Title']
            title2_emb = db_all.iloc[j][start_column:(start_column+1536)].to_numpy()
            dist = scipy.spatial.distance.cosine(title_emb, title2_emb)
            if dist < min_dist and title != title2:
                min_dist = dist
                min_dist_title = title2
        print("    min_dist: ", min_dist)
        print("    min_dist_title: ", min_dist_title)
        db_.loc[i, 'closest_emb'] = min_dist
        if min_dist_title in titles_labels_train:
            print("    Real: ", titles_labels_train[min_dist_title])
            db_.loc[i, 'heuristic_label_closest'] = titles_labels_train[min_dist_title]
        else:
            label = heuristic_label(min_dist_title)
            print("    Heuristic: ", label)
            db_.loc[i, 'heuristic_label_closest'] = label
    return db_

#db_train = pd.read_csv(TRAIN_DATASET)
##db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})
#db_train = process_dataset(db_train, db_all, titles_labels_train, real_titles, fake_titles, 27, 27)
#print(db_train)
#pd.DataFrame.to_csv(db_train, OUTPUT_TRAIN_DATASET)

db_test = pd.read_csv(TEST_DATASET)
db_test = process_dataset(db_test, db_all, titles_labels_train, titles_test, set([]), 27, 25)
print(db_test)
pd.DataFrame.to_csv(db_test, OUTPUT_TEST_DATASET)