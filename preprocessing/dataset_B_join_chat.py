import numpy as np
import pandas as pd
#import tensorflow_hub as hub
#import tensorflow as tf
#import spacy 
import time


TRAIN_DATASET = '../database/train_B_text_perplexity.csv'
TRAIN_DATASET_2 = '../database/train_B_text_chat.csv'
TEST_DATASET = '../database/test_B_text_perplexity.csv'
TEST_DATASET_2 = '../database/test_B_text_chat.csv'
OUTPUT_TEST_DATASET = '../database/test_B_processed_perplexity_chat.csv'
OUTPUT_TRAIN_DATASET = '../database/train_B_text_processed_perplexity_chat.csv'

db = pd.read_csv(TRAIN_DATASET)
db_2 = pd.read_csv(TRAIN_DATASET_2)
db['Magic bit'] = db_2['Magic bit']
print(db_2['Magic bit'])
db.drop(['Unnamed: 0'], axis=1, inplace=True)
print(db)
pd.DataFrame.to_csv(db, OUTPUT_TRAIN_DATASET)

######################### Test dataset #########################

db = pd.read_csv(TEST_DATASET)
db_2 = pd.read_csv(TEST_DATASET_2)
db['Magic bit'] = db_2['Magic bit']
print(db_2['Magic bit'])
db.drop(['Unnamed: 0'], axis=1, inplace=True)
print(db)
pd.DataFrame.to_csv(db, OUTPUT_TEST_DATASET)
