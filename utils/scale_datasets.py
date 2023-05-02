#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

def read_files(year, month):
    files = Path(f'../../datasets/balanced/{year}/VIEGAS/{month}').iterdir()

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

df.to_csv('../../datasets/scaled/2016/01/dataset.csv')

for year in (2016, 2017, 2018, 2019):
    for month in range(12):
        month += 1
        print(f"Running {year} {month:02d}")
        df, df_labels = read_files(f'{year}', f'{month:02d}')
        df[df.columns] = scaler.transform(df[df.columns])
        df['class'] = df_labels['class']

        df.to_csv(f'../../datasets/scaled/{year}/{month:02d}/testing_{year}_{month:02d}.csv')

