#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
import sys
from sklearn import preprocessing

glob = '*20??010[12345]*'

files = Path('ALL').glob(glob)

dfs = []

for file in sorted(files):
    print(file)
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs)

print(df.shape)

idx = { 'normal' : 0, 'attack' : 1 }
df['class'] = df['class'].apply(lambda x: idx[x])

scaler = preprocessing.MinMaxScaler()
scaler.fit(df[df.columns[0:-1]])

files = Path('ALL').iterdir()
for file in sorted(files):
    print(f"Scaling file {file}")
    df = pd.read_csv(file)
    df['class'] = df['class'].apply(lambda x: idx[x])
    df_labels = df[['class']]

    df = df.drop(columns=['class'])

    df[df.columns] = scaler.transform(df[df.columns])

    df = df.copy()

    df['class'] = df_labels['class']

    dst_file = Path('SCALED') / file.name.replace('ALL', 'SCALED_ALL')
    print(f'Scaled to {dst_file}')

    df.to_csv(dst_file, index=False)
