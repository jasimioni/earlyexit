# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import scipy

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

print(os.getcwd())

# network = 'alexnet'
network = 'mobilenet'

with open(f'c:/Users/jasim/Documents/ppgia/earlyexit/jupyter/fix_{network}_x_f_2016_23.sav', 'rb') as f:
    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

#%%
def plot(network, *cnfs):
    directory = directories[network]

    datapoints = {
        'network' : network,
        'labels' : [],
        'accuracy_e1' : [],
        'accuracy_e2' : [],
        'accuracy_model' : [],
        'acceptance_e1' : [],
        'acceptance_e2' : [],
        'acceptance_model' : [],
        'time_e1' : [],
        'time_e2' : [],
        'time_model' : [],
    }

    for year in range(2016, 2020):
        year = f'{year:04d}'
        for month in range(1, 13):
            month = f'{month:02d}'
            glob = f'{year}_{month}'
            csv = os.path.join(directory, f'{glob}.csv')
            df = pd.read_csv(csv)
            
            accuracy_e1, acceptance_e1, time_e1 = get_objectives(df, 0, 0, 1, 1)
            accuracy_e2, acceptance_e2, time_e2 = get_objectives(df, 2, 2, 0, 0)
            accuracy, acceptance, time = get_objectives(df, *cnfs)

            datapoints['labels'].append(glob)
            datapoints['accuracy_e1'].append(accuracy_e1)
            datapoints['accuracy_e2'].append(accuracy_e2)
            datapoints['accuracy_model'].append(accuracy)
            datapoints['acceptance_e1'].append(acceptance_e1)
            datapoints['acceptance_e2'].append(acceptance_e2)
            datapoints['acceptance_model'].append(acceptance)
            datapoints['time_e1'].append(time_e1)
            datapoints['time_e2'].append(time_e2)
            datapoints['time_model'].append(time)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    fig.autofmt_xdate(rotation=90)
    plt.title(f"Accuracy {datapoints['network']} - min cnf: {cnfs}")

    axs[0].plot(datapoints['labels'], datapoints['accuracy_e1'], label='accuracy_e1')
    axs[0].plot(datapoints['accuracy_e2'], label='accuracy_e2')
    axs[0].plot(datapoints['accuracy_model'], label='accuracy_model')
    axs[0].plot(datapoints['acceptance_model'], label='acceptance_model')            
    axs[0].legend()

    axs[1].plot(datapoints['labels'], datapoints['time_e1'], label='time_e1')
    axs[1].plot(datapoints['time_e2'], label='time_e2')
    axs[1].plot(datapoints['time_model'], label='time_model')
    axs[1].legend()
    
    plt.show

#%%

results = []

print(f'e1: Accuracy: {accuracy_e1:.2f}% - Acceptance: {acceptance_e1}% - Cost: {min_time:.2f}us')
print(f'e2: Accuracy: {accuracy_e2:.2f}% - Acceptance: {acceptance_e2}% - Cost: {max_time:.2f}us\n')

for i in range(len(F)):
    f = F[i]
    x = X[i]
    quality = 100 * (1 - sum(f) / len(f))
    results.append([ quality, f[0], f[1], f[2], x[0], x[1], x[2], x[3] ])

valid_results = []
best_time = None
best_accuracy = None
best_score = None

for i, r in enumerate(sorted(results, key=lambda r: r[0], reverse=True)):
    # Acceptance at least 85%
    if r[2] > 0.15:
        continue

    if best_score is None:
        best_score = r

    if best_time is None or r[3] < best_time[3]:
        best_time = r

    if best_accuracy is None or r[1] < best_accuracy[1]:
        best_accuracy = r

    # print(f'{i:02d}: {r[0]:.2f}% => {100 * (1 - r[1]):.2f}% : {100 * (1 - r[2]):.2f}% : {min_time + (r[3] * (max_time - min_time)):.2f}us', end='')
    # print(f'\t{r[4]:.4f} : {r[5]:.4f} : {r[6]:.4f} : {r[7]:.4f}')

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
x = 100 * ( 1 - F[:, 0])
y = 100 * ( 1 - F[:, 1])
z = min_time + F[:, 2] * (max_time - min_time)

'''
# x = F[:, 0]
# y = F[:, 1]
# z = F[:, 2]

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

surf = ax.plot_trisurf(x, y, z, color=(0.5,0.5,0.5,0.5), edgecolor='Gray')
# surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
# fig.colorbar(surf)

ax.scatter(x, y, z)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()
'''
#%%

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

surf = ax.plot3D(x, y, z)
# surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
# fig.colorbar(surf)

ax.scatter(x, y, z)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()
