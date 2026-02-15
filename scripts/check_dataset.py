import pandas as pd
import sys

df = pd.read_csv('Train_Data.csv')
if 'Legendary' not in df.columns:
    print('ERROR: Legendary column not found')
    sys.exit(2)
print('Legendary value counts:')
print(df['Legendary'].value_counts(dropna=False))
print('\nshape:', df.shape)
print('nulls in Legendary:', df['Legendary'].isnull().sum())
