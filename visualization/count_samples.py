import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Train dataset
db_a = pd.read_csv('../database/train_A_derma.csv')
db_a['Fake/Real'] = db_a['Fake/Real'].replace({'real': 0, 'fake': 1})

# count number of possitive and negative samples
print("Train dataset class distribution")
print(db_a['Fake/Real'].value_counts())