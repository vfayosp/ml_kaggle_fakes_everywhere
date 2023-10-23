import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('../database/train_A_derma.csv')
Y = df['Fake/Real'].replace({'real': 0, 'fake': 1})

X = df.drop(['Fake/Real', 'Doughnuts consumption', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=42)

pipeline_optimizer = TPOTClassifier(generations=20, population_size=40, cv=5,
                                    random_state=42, verbosity=2, config_dict='TPOT light')

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
