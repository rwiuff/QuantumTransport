from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
import matplotlib
import numpy as np                      # NumPy
import seaborn
from numpy import linalg as LA
from collections import Counter
from Functions import xyzimport, Hkay, Onsite, Hop
import sys
np.set_printoptions(threshold=sys.maxsize)

# Set hopping potential
Vppi = -1

# Define lattice vectors
# shiftx = 32.7862152500
# shifty = 8.6934634800
#
# # Retrieve unit cell
# xyz = xyzimport('TestCell.fdf')
# # Calculate onsite nearest neighbours
# Ham = Onsite(xyz, Vppi)
#
# # Shift unit cell
# xyz1 = xyz + np.array([shiftx, 0, 0])
# # Calculate offsite nearest neighbours
# V1 = Hop(xyz, xyz1, Vppi)
#
# # Shift unit cell
# xyz2 = xyz + np.array([0, shifty, 0])
# # Calculate offsite nearest neighbours
# V2 = Hop(xyz, xyz2, Vppi)
#
# # Shift unit cell
# xyz3 = xyz + np.array([shiftx, shifty, 0])
# # Calculate offsite nearest neighbours
# V3 = Hop(xyz, xyz3, Vppi)

h = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
V = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

h = h * Vppi
V = V * Vppi

# print(np.sum(h))

Show = 0
if Show == 1:
    plt.imshow(h)
    plt.colorbar()
    plt.show()
    plt.imshow(V)
    plt.colorbar()
    plt.show()

jsize = 4

z = np.linspace(-1, 1, jsize)
z = np.diag(z) - 0.001j
es = h
a = np.transpose(V)
b = V
e = h
g = LA.inv(z - e)

Recurs = np.zeros((h.shape[0], ) * 4, dtype=complex)

for j in range(jsize):
    if j == 0:
        Recurs[0, 0, :] = z - es
        Recurs[0, 1, :] = -a
    elif j == jsize - 1:
        Recurs[-1, -1, :] = z - e
        Recurs[-1, -2, :] = -b
    else:
        Recurs[j, j - 1, :] = -b
        Recurs[j, j, :] = z - e
        Recurs[j, j + 1, :] = -a
print(Recurs)
print(np.sum(np.abs(Recurs)))
q = 1
while np.sum(np.abs(a)) != 0:
    for j in range(jsize):
        if j == 0:
            g = LA.inv(z - e)
            es = es + a @ g @ b
            e = e + a @ g @ b + b @ g @ a
            a = a @ g @ a
            b = b @ g @ b
            Recurs[0, 0, :] = z - es
            Recurs[0, 1, :] = -a
        elif j == jsize - 1:
            Recurs[-1, -1, :] = z - e
            Recurs[-1, -2, :] = -b
        else:
            Recurs[j, j - 1, :] = -b
            Recurs[j, j, :] = z - e
            Recurs[j, j + 1, :] = -a
    print(q)
    print(b)
    print(np.sum(np.abs(a)))
    q = q + 1
print(Recurs)
SelfER = es - h
print(SelfER)
SelfEL = e - h - SelfER
print(SelfEL)
G00 = LA.inv(z-es)
X = np.linspace(-1, 1, G00.flatten().shape[0])
Y = np.sort(G00.flatten())
print(G00.flatten())
Y1 = Y.real
Y2 = Y.imag
plt.plot(X, Y1)
plt.plot(X, Y2)
plt.show()
