from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
import matplotlib
import numpy as np                      # NumPy
import seaborn
from numpy import linalg as LA
from collections import Counter
from Functions import xyzimport, Hkay, Onsite, Hop
# import sys
# np.set_printoptions(threshold=sys.maxsize)

# Set hopping potential
Vppi = -1

# Define lattice vectors
shiftx = 32.7862152500
shifty = 8.6934634800

# Retrieve unit cell
xyz = xyzimport('fab_NPG_C.fdf')
# Calculate onsite nearest neighbours
Ham = Onsite(xyz, Vppi)

# Shift unit cell
xyz1 = xyz + np.array([shiftx, 0, 0])
# Calculate offsite nearest neighbours
V1 = Hop(xyz, xyz1, Vppi)

# Shift unit cell
xyz2 = xyz + np.array([0, shifty, 0])
# Calculate offsite nearest neighbours
V2 = Hop(xyz, xyz2, Vppi)

# Shift unit cell
xyz3 = xyz + np.array([shiftx, shifty, 0])
# Calculate offsite nearest neighbours
V3 = Hop(xyz, xyz3, Vppi)


print(np.sum(Ham))
Show = 0
if Show == 1:
    plt.imshow(Ham)
    plt.colorbar()
    plt.show()
    plt.imshow(V1)
    plt.colorbar()
    plt.show()
    plt.imshow(V2)
    plt.colorbar()
    plt.show()
    plt.imshow(V3)
    plt.colorbar()
    plt.show()

Erange = np.linspace(-1, 1, Ham.shape[0])
print(Erange)
Z = np.zeros((Erange.shape[0]), dtype=complex)
Z[:] = Erange[:] + 1j * 0.0000001
print(Z)
Z = np.diag(Z)
print(Z)
OnS = Z - Ham
SelfE = V1 @ LA.inv(OnS) @ np.transpose(V1)
print(SelfE)
