#!/usr/bin/env/python3

#%%
#https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

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

file = f'/home/jasimioni/ppgia/earlyexit/jupyter/nsgaresults/{network}_x_f_2016_23.sav'

with open(file, 'rb') as f:
    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

#%%

results = []

print(f'e1: Accuracy: {accuracy_e1:.2f}% - Acceptance: {acceptance_e1}% - Cost: {min_time:.2f}us')
print(f'e2: Accuracy: {accuracy_e2:.2f}% - Acceptance: {acceptance_e2}% - Cost: {max_time:.2f}us\n')

def remove_dominated(results):
    to_delete = []
    for i in range(len(results)):
        for f in range(len(results)):
            if f == i or ( len(to_delete) and to_delete[-1] == i ):
                continue
            if results[f][1] <= results[i][1] and results[f][2] <= results[i][2] and results[f][3] <= results[i][3]:
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

    print(f'{i:02d}: {quality:.2f} {100 * (1-f[0]):.2f} {100*(1-f[1]):.2f} {100*(1-f[2]):.2f}')

while remove_dominated(results):
    pass

df = pd.DataFrame(F, columns = ['Accuracy', 'Acceptance', 'Time'])
df['Score'] = ( df['Accuracy'] + df['Acceptance'] + df['Time'] ) / 3

df = df.sort_values(by='Score')

# Filter only with acceptance >= 85%
# df = df.loc[df['Acceptance'] <= 0.15]

x = df['Accuracy'].to_numpy()
y = df['Acceptance'].to_numpy()
z = df['Time'].to_numpy()
score = df['Score'].to_numpy()

x1 = 100 * ( 1 - x )
y1 = 100 * ( 1 - y )
z1 = min_time + z * (max_time - min_time)

min_accuracy = np.min(x1)
max_accuracy = 100

X = np.arange(min_accuracy, max_accuracy, 1)
Z = np.arange(min_time, max_time, 0.1)
X, Z = np.meshgrid(X, Z)
Y = 80 + 0 * X

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

ax.plot3D(x1, y1, z1)
ax.scatter(x1, y1, z1, c=score, cmap='viridis')
# surf = ax.plot_trisurf(x1, y1, z1, cmap=cm.coolwarm)
# fig.colorbar(surf)

ax.plot_surface(X, Y, Z, alpha=0.5)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()
