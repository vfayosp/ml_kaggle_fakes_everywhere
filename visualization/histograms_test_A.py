import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_DERMA = "../database/test_A_derma.csv"

df = pd.read_csv(TRAIN_DERMA)


for name in ['Genetic Propensity', 'Skin X test', 'Skin color', 'Small size', 'Mid size', 'Large size', 'Mid', 'Small', 'Large', 'Doughnuts consumption']:

    # Print 2 histograms with real and fake data
    fig, ax = plt.subplots()

    # Replace nan with the max value + 1
    df[[name]] = df[[name]].fillna(df[name].max() + 1)

    df.hist(column=name, ax=ax)
    
    fig.suptitle(f"Variable: {name}")
    plt.show()
