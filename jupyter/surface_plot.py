# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

with open('jupyter/alexnet_x_f_2016_23.sav', 'rb') as f:
    X, F, min_time, max_time = pickle.load(f)

x = 100 * ( 1 - F[:, 0])
y = 100 * ( 1 - F[:, 1])
z = min_time + F[:, 2] * (max_time - min_time)

# x = F[:, 0]
# y = F[:, 1]
# z = F[:, 2]

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()