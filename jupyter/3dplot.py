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

## Which one to check
# file = 'c:/Users/jasim/Documents/ppgia/earlyexit/jupyter/nsgaresults/alexnet_x_f_2016_23.sav'
# file = 'c:/Users/jasim/Documents/ppgia/earlyexit/jupyter/nsgaresults/mobilenet_x_f_2016_23.sav'
# file = 'c:/Users/jasim/Documents/ppgia/earlyexit/jupyter/nsgaresults/mobilenet_x_f_2016_23_10000.sav'

file = '/home/jasimioni/ppgia/earlyexit/jupyter/nsgaresults/mobilenet_x_f_2016_23.sav'

with open(file, 'rb') as f:
    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

#%%

results = []

print(f'e1: Accuracy: {accuracy_e1:.2f}% - Acceptance: {acceptance_e1}% - Cost: {min_time:.2f}us')
print(f'e2: Accuracy: {accuracy_e2:.2f}% - Acceptance: {acceptance_e2}% - Cost: {max_time:.2f}us\n')

for i in range(len(F)):
    f = F[i]
    x = X[i]
    quality = 100 * (1 - sum(f) / len(f))
    results.append([ quality, *f, *x ])

valid_results = []
best_time = None
best_accuracy = None
best_score = None

for i, r in enumerate(sorted(results, key=lambda x: x[0], reverse=True)):
    # Acceptance at least 85%
    if r[2] > 0.15:
        continue

    if best_score is None:
        best_score = r

    if best_time is None or r[3] < best_time[3]:
        best_time = r

    if best_accuracy is None or r[1] < best_accuracy[1]:
        best_accuracy = r

    print(f'{i:02d}: {r[0]:.2f}% => {100 * (1 - r[1]):.2f}% : {100 * (1 - r[2]):.2f}% : {min_time + (r[3] * (max_time - min_time)):.2f}us', end='')
    print(f'\t{r[4]:.4f} : {r[5]:.4f} : {r[6]:.4f} : {r[7]:.4f}')

print('Score => Accuracy : Acceptance : Cost\tn_e1 : a_e1 : n_e2 : a_e2')

r = best_score
print(f'Melhor score:')
print(f'{i:02d}: {r[0]:.2f}% => {100 * (1 - r[1]):.2f}% : {100 * (1 - r[2]):.2f}% : {min_time + (r[3] * (max_time - min_time)):.2f}us', end='')
print(f'\t{r[4]:.4f} : {r[5]:.4f} : {r[6]:.4f} : {r[7]:.4f}\n')

r = best_accuracy
print(f'Melhor accuracy (com acceptance > 85%):')
print(f'{i:02d}: {r[0]:.2f}% => {100 * (1 - r[1]):.2f}% : {100 * (1 - r[2]):.2f}% : {min_time + (r[3] * (max_time - min_time)):.2f}us', end='')
print(f'\t{r[4]:.4f} : {r[5]:.4f} : {r[6]:.4f} : {r[7]:.4f}\n')

r = best_time
print(f'Melhor tempo (com acceptance > 85%):')
print(f'{i:02d}: {r[0]:.2f}% => {100 * (1 - r[1]):.2f}% : {100 * (1 - r[2]):.2f}% : {min_time + (r[3] * (max_time - min_time)):.2f}us', end='')
print(f'\t{r[4]:.4f} : {r[5]:.4f} : {r[6]:.4f} : {r[7]:.4f}')


#%%

# print(F)

df = pd.DataFrame(F, columns = ['Accuracy', 'Acceptance', 'Time'])
df['Score'] = ( df['Accuracy'] + df['Acceptance'] + df['Time'] ) / 3

df = df.sort_values(by='Score')

# Filter only with acceptance >= 85%
# df = df.loc[df['Acceptance'] <= 0.15]

x = df['Accuracy'].to_numpy()
y = df['Acceptance'].to_numpy()
z = df['Time'].to_numpy()

x1 = 100 * ( 1 - x )
y1 = 100 * ( 1 - y )
z1 = min_time + z * (max_time - min_time)

x = 100 * ( 1 - F[:, 0])
y = 100 * ( 1 - F[:, 1])
z = min_time + F[:, 2] * (max_time - min_time)

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

# ax.plot3D(x1, y1, z1)
ax.scatter(x1, y1, z1, color='blue')
surf = ax.plot_trisurf(x1, y1, z1, cmap=cm.coolwarm)

# ax.plot3D(x, y, z)
# surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
# fig.colorbar(surf)

#ax.scatter(x, y, z, color='green')

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()
