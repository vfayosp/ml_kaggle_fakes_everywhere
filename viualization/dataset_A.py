import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

db_a = pd.read_csv('../database/train_A_derma.csv')

db_a['Fake/Real'] = db_a['Fake/Real'].replace({'real': 0, 'fake': 1})

print(db_a.info())

sns.pairplot(db_a, hue='Fake/Real')

plt.savefig('pairplot_dataset_a.jpg')


