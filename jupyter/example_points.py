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

x = [ 1, 2, 3, 1 ]
y = [ 1, 2, 3, 2 ]
z = [ 1, 2, 3, 4 ]

    
fig = plt.figure(figsize = (15, 7))
ax = plt.axes(projection ="3d")
ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")

ax.view_init(45, -45)

# show plot
plt.show()