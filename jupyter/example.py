#%%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


# Make data.
X = np.arange(-5, 5, 1)
Y = np.arange(-5, 5, 1)

#%%
X = [0, 1, 2, 3]
Y = [4, 5, 6]

x, y = np.meshgrid(X, Y)

print(x)
print(y)

Z = x + y

print(Z)



#%%

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()