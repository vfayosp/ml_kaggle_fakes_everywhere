import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import spacy 


TRAIN_DATASET = '../database/train_B_text.csv'
OUTPUT_TRAIN_DATASET = '../database/train_B_text_processed_embeddings.csv'
TEST_DATASET = '../database/test_B_text.csv'
OUTPUT_TEST_DATASET = '../database/test_B_text_processed_embeddings.csv'


def is_only_first_upper(words):
    all_upper = True
    for word in words:
        # check only when it is an alphabetical character
        if not word[0].isupper() and word[0].isalpha():
            all_upper = False
            break
    return all_upper


def is_any_noun_verb_lower(nlp, title):
    # check if a noun or verb are writte in lowercase
    lower_noun = False
    for token in nlp(title):
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            if token.text[0].islower():
                lower_noun = True
                break
    return lower_noun


def count_spacy_token(spacy_tokens, target_token):
    count = 0
    for token in spacy_tokens:
        if str(token.pos_) == target_token:
            count += 1
    return count


def tokens_count_dict(nlp, title):
    aux = nlp(title.lower()) # NOTICE: we have to lower the title
    return {
        'ADJ': count_spacy_token(aux, 'ADJ'), 
        'ADV': count_spacy_token(aux, 'ADV'), 
        'NOUN': count_spacy_token(aux, 'NOUN'), 
        'VERB': count_spacy_token(aux, 'VERB'), 
        'PROPN': count_spacy_token(aux, 'PROPN'), 
        'PUNCT': count_spacy_token(aux, 'PUNCT')}


def process_title(title):
    count_quotes = title.count('"')
    has_quote_start = 1 if title[0] == '"' else 0
    #title = title.replace('"','').replace("'",'')
    has_dots = 1 if '...' in title else 0 
    has_apostrophe_s = 1 if "'s" in title else 0
    has_apostrophe = 1 if "'" in title else 0
    has_number = 1 if any(char.isdigit() for char in title) else 0
    has_comma = 1 if ',' in title else 0
    has_colon = 1 if ':' in title else 0
    has_parenthesis = 1 if '(' in title or ')' in title else 0
    has_hyphen = 1 if '-' in title else 0
    has_and = 1 if '&' in title else 0
    has_percentage = 1 if '%' in title else 0
    has_interrogation = 1 if '?' in title else 0
    has_exclamation = 1 if '!' in title else 0
    has_only_first_upper = 1 if is_only_first_upper(title.split()) else 0
    has_any_noun_verb_lower = 1 if is_any_noun_verb_lower(nlp, title) else 0
    length = len(title.split())
    tokens_dict = tokens_count_dict(nlp, title)
    

    return count_quotes, has_quote_start, has_dots, \
            has_apostrophe_s, has_apostrophe, has_number, has_comma, \
            has_colon, has_parenthesis, has_hyphen, has_and, \
            has_percentage, has_interrogation, has_exclamation, \
            has_only_first_upper, has_any_noun_verb_lower, length, \
            tokens_dict['ADJ'], tokens_dict['ADV'], \
            tokens_dict['NOUN'], tokens_dict['VERB'], tokens_dict['PROPN'], \
            tokens_dict['PUNCT']

def embed(input):
    input = input.split(' ')
    embedding = tf.reduce_mean(model(input), axis=0).numpy().tolist()
    return embedding

nlp = spacy.load("en_core_web_sm")

######################### Train dataset #########################

db = pd.read_csv(TRAIN_DATASET)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})

db[['count_quotes', 'has_quote_start', 'has_dots', 'has_apostrophe_s', \
    'has_apostrophe', 'has_number', 'has_comma', 'has_colon', 'has_parenthesis', \
    'has_hyphen', 'has_and', 'has_percentage', 'has_interrogation', 'has_exclamation', \
    'has_only_first_upper', 'has_any_noun_verb_lower', 'length', \
    'ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT']] \
    = db['Title'].apply(process_title).apply(pd.Series)
    
   
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)

emb_columns = []
for i in range(0,512):
    emb_columns.append('emb'+str(i))

db[emb_columns] = db['Title'].apply(embed).apply(pd.Series)

print(db)
pd.DataFrame.to_csv(db, OUTPUT_TRAIN_DATASET)

######################### Test dataset #########################

db = pd.read_csv(TEST_DATASET)
db = db.drop(['Id'], axis=1)

db[['count_quotes', 'has_quote_start', 'has_dots', 'has_apostrophe_s', \
    'has_apostrophe', 'has_number', 'has_comma', 'has_colon', 'has_parenthesis', \
    'has_hyphen', 'has_and', 'has_percentage', 'has_interrogation', 'has_exclamation', \
    'has_only_first_upper', 'has_any_noun_verb_lower', 'length', \
    'ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'PUNCT']] \
    = db['Title'].apply(process_title).apply(pd.Series)
    
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)

emb_columns = []
for i in range(0,512):
    emb_columns.append('emb'+str(i))

db[emb_columns] = db['Title'].apply(embed).apply(pd.Series)

print(db)
pd.DataFrame.to_csv(db, OUTPUT_TEST_DATASET)
