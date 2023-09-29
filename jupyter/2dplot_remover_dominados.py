#!/usr/bin/env/python3

#%%

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import scipy
import sys

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

print(os.getcwd())

network = 'mobilenet'
# network = 'alexnet'

file = f'/home/jasimioni/ppgia/earlyexit/jupyter/{network}_x_f_0.7_2016_23.sav'

with open(file, 'rb') as f:
    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

#%%

results = []

print(f'e1: Accuracy: {100 * accuracy_e1:.2f}% - Acceptance: {100 * acceptance_e1:.2f}% - Cost: {min_time:.2f}us')
print(f'e2: Accuracy: {100 * accuracy_e2:.2f}% - Acceptance: {100 * acceptance_e2:.2f}% - Cost: {max_time:.2f}us\n')

def remove_dominated(results):
    to_delete = []
    for i in range(len(results)):
        for f in range(len(results)):
            if f == i or ( len(to_delete) and to_delete[-1] == i ):
                continue
            if results[f][1] <= results[i][1] and results[f][2] <= results[i][2]:
                to_delete.append(i)

    print(f'to_delete: {to_delete}')
    to_delete.reverse()
    for i in to_delete:
        results.pop(i)
    return len(to_delete)
    
for i in range(len(F)):
    f = F[i]
    x = X[i]
    quality = 100 * (1 - sum(f) / len(f))
    results.append([ quality, *f, *x ])

    print(f'{i:02d}: {quality:.2f} {100 * (1-f[0]):.2f} {100*(1-f[1]):.2f}')

while remove_dominated(results):
    pass

df = pd.DataFrame(F, columns = ['Accuracy', 'Time'])
df['Score'] = ( df['Accuracy'] + df['Time'] ) / 3

df = df.sort_values(by='Score')

# Filter only with acceptance >= 85%
# df = df.loc[df['Acceptance'] <= 0.15]

y = df['Accuracy'].to_numpy()
x = df['Time'].to_numpy()
score = df['Score'].to_numpy()

y1 = 100 * ( 1 - y )
x1 = min_time + x * (max_time - min_time)

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111)

ax.set_xlabel('Accuracy')
ax.set_ylabel('Time')

ax.plot(x1, y1)
ax.scatter(x1, y1, c=score, cmap='viridis')
# surf = ax.plot_trisurf(x1, y1, z1, cmap=cm.coolwarm)
# fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))

fig.tight_layout()

plt.show()

# %%
