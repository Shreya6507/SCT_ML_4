import pandas as pd

p = 'data/gestures.csv'
print('Loading', p)
df = pd.read_csv(p)
# Normalize column names (strip stray spaces) and labels
df.columns = df.columns.str.strip()
labels = df['label'].astype(str).str.strip()
print('Before clean label counts:')
print(labels.value_counts(dropna=False))
# Drop obvious bad labels
mask = ~labels.str.lower().isin(['', 'nan', 'none'])
removed = len(df) - mask.sum()
if removed:
    print(f'Removing {removed} rows with empty/nan labels')
    df = df[mask].copy()

# Append synthetic samples for thumbs_up and close
n_append = 6
thumbs = []
close = []
for i in range(n_append):
    base1 = 0.18 + 0.01 * i
    base2 = 0.6 + 0.01 * i
    thumbs.append([round(base1 + (j % 3) * 0.01, 6) for j in range(63)] + ['thumbs_up'])
    close.append([round(base2 + (j % 3) * 0.01, 6) for j in range(63)] + ['close'])
cols = df.columns.tolist()
add_df = pd.DataFrame(thumbs + close, columns=cols)
df2 = pd.concat([df, add_df], ignore_index=True)
df2.to_csv(p, index=False)
print('After append label counts:')
print(df2['label'].astype(str).value_counts())
print('\nWrote', p)
