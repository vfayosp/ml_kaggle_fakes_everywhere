import pandas as pd

# Read the first CSV file
df1 = pd.read_csv('output_A.csv')

# Read the second CSV file
df2 = pd.read_csv('output_B.csv')

# Find the maximum index in the first CSV
max_index = df1['Id'].max() + 1

# Shift the indexes in the second CSV
df2['Id'] += max_index 

# Concatenate the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('output.csv', index=False)

