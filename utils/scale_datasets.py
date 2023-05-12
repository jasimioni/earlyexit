#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from sklearn import preprocessing
import sys
import os

scaler = preprocessing.MinMaxScaler()

author = sys.argv[1] if len(sys.argv) > 1 else 'VIEGAS'

def read_files(year, month):
    files = Path(f'../../datasets/balanced/{year}/{author}/{month}').iterdir()

    df = pd.DataFrame()

    for file in files:
        temp = pd.read_csv(file)
        df = pd.concat([df, temp])

    df_labels = df[['class']]
    df = df.drop(columns=['MAWILAB_taxonomy', 'MAWILAB_distance', 'MAWILAB_nbDetectors', 'MAWILAB_label', 'class'])

    return df, df_labels

df, df_labels = read_files('2016', '01')

df[df.columns] = scaler.fit_transform(df[df.columns])
df['class'] = df_labels['class']

directory = f'../../datasets/scaled/{author}/'

try:
    os.makedirs(directory)
except Exception as e:
    print(f'Diretório não criado: {e}')

df.to_csv(os.path.join(directory, 'scale_fit_reference.csv'), index=False)

for year in (2016, 2017, 2018, 2019):
    for month in range(12):
        month += 1
        print(f"Running {author} {year} {month:02d}")
        df, df_labels = read_files(f'{year}', f'{month:02d}')
        df[df.columns] = scaler.transform(df[df.columns])
        df['class'] = df_labels['class']

        directory = f'../../datasets/scaled/{author}/{year}/{month:02d}/'
        try:
            os.makedirs(directory)
        except Exception as e:
            print(f'Diretório não criado: {e}')

        df.to_csv(os.path.join(directory, f'{year}_{month:02d}.csv'), index=False)

