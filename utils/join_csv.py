#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
import sys

def validate_and_join_files(dst_file, *files):
    print(f"Validating {files} and joining them to {dst_file}")
    dfs = []
    for i, file in enumerate(files):
        dfs.append(pd.read_csv(file))

    base_class = dfs[0][['class']]

    # Compare class in each df
    for df in dfs[1:]:
        assert (base_class.equals(df[['class']])), f"Could not validate {files} - they are not equal"

    # Drop class on 3 dfs
    for i in range(len(dfs) - 1):
        dfs[i] = dfs[i].drop(['class'], axis=1)

    # Create consolidate df and drop duplicated columnes
    df = pd.concat(dfs, axis=1);

    # Compare duplicated fields in VIEGAS and ORUNADA dfs
    for field in ('averagePacketSize', 'percentageICMPRedirect', 'percentageICMPTimeExceeded', 'percentageICMPUnreacheable', 'percentageICMPOtherTypes'):
        orunada = df[[f'ORUNADA_{field}']].rename(columns={f'ORUNADA_{field}' : field})
        viegas  = df[[f'VIEGAS_{field}']].rename(columns={f'VIEGAS_{field}' : field})
        assert(orunada.equals(viegas)), f"{field} is not equal"

    df = df.drop(['ORUNADA_averagePacketSize',
                  'ORUNADA_percentageICMPRedirect', 
                  'ORUNADA_percentageICMPTimeExceeded', 
                  'ORUNADA_percentageICMPUnreacheable', 
                  'ORUNADA_percentageICMPOtherTypes'], axis=1)

    print(df.shape)

    df.to_csv(dst_file, index=False)

years = Path('.').iterdir()
for year in sorted(years):
    if year.is_dir():
        moore = year / 'MOORE'
        nigel = year / 'NIGEL'
        orunada = year / 'ORUNADA'
        viegas = year / 'VIEGAS'
        files = moore.iterdir()
        for moore_file in sorted(files):
            try:
                nigel_file = nigel / moore_file.name.replace('MOORE', 'NIGEL')
                orunada_file = orunada / moore_file.name.replace('MOORE', 'ORUNADA')
                viegas_file = viegas / moore_file.name.replace('MOORE', 'VIEGAS')

                for file in (moore_file, nigel_file, orunada_file, viegas_file):
                    if not file.is_file():
                        raise Exception(f'{file} is not a valid file')

                dst_file = year / 'ALL'
                if not dst_file.is_dir():
                    dst_file.mkdir()

                file_name = moore_file.name.replace('MOORE', 'ALL')
                dst_file = dst_file / file_name

                validate_and_join_files(dst_file, moore_file, nigel_file, orunada_file, viegas_file)
            except Exception as e:
                print(f"Exception: {e}")
            
