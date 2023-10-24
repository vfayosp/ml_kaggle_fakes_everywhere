import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

db = pd.read_csv('../database/train_A_derma.csv')

db.drop(['Id','Doughnuts consumption'], axis=1, inplace=True)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})


print(db.describe())
def populate_nan(row):
    if 1 in row.values:
        return row.fillna(0)
    elif (row == 0).sum() == 2:
        return row.fillna(1)
    else:
        return row
    
def populate_size(row):
    row[:3] = populate_nan(row[:3])
    row[3:] = populate_nan(row[3:])
    
    for i in range(0,len(row)):
        if row[i] == np.nan:
            row[i] == row[(i+3)%6]
    return row
            
print('Preprocessing Size')
db[['Small size','Mid size', 'Large size', 'Small', 'Mid', 'Large']] \
        = db[['Small size','Mid size', 'Large size','Small', 'Mid', 'Large']] \
        .apply(populate_size,axis=1)

print(db.describe())

pd.DataFrame.to_csv(db, '../database/train_A_input_sizes.csv')
