import pandas as pd
import sys

path = 'data/gestures.csv'
df = pd.read_csv(path)
print('COLUMNS:')
print(df.columns.tolist())
print('\nDTYPES:')
print(df.dtypes)
print('\nHEAD:')
print(df.head().to_string())

print('\nSHAPE:', df.shape)

sys.exit(0)
