import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm    

x = [ 1, 2, 3, 1, 1 ]
y = [ 1, 2, 3, 3, 2 ]
z = [ 1, 2, 3, 4, 4 ]

fig = plt.figure(figsize = (15, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Acceptance')
ax.set_zlabel('Time')

#surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

surf = ax.scatter3D(x, y, z, cmap=cm.coolwarm)

# fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()