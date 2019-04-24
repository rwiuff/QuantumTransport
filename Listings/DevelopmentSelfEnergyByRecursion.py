from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib import patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
import matplotlib
import numpy as np                      # NumPy
# import seaborn
from numpy import linalg as LA
# from collections import Counter
# from Functions import xyzimport, Hkay, Onsite, Hop
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

En = np.linspace(-1, 1, 100)
En = np.linspace(-1, 1, 3)


def RecursionRoutine(En, h, V):
    ns = h.shape[0]
    z = np.identity(ns) * (En - 1e-21j)
    a0 = np.transpose(V)
    b0 = V
    es0 = h
    e0 = h
    g0 = LA.inv(z - e0)
    q = 1
    while np.max(np.abs(a0)) > 0.000000001:
        ag = a0 @ g0
        a1 = ag @ a0
        bg = b0 @ g0
        b1 = bg @ b0
        e1 = e0 + ag @ b0 + bg @ a0
        es1 = es0 + ag @ b0
        g1 = LA.inv(z - e1)

        a0 = a1
        b0 = b1
        e0 = e1
        es0 = es1
        g0 = g1
        q = q + 1
    print(q)
    e, es = e0, es0
    SelfER = es - h
    SelfEL = e - h - SelfER
    G00 = LA.inv(z - es)
    return G00


G00 = np.zeros((En.shape[0]), dtype=complex)
for i in range(En.shape[0]):
    G = RecursionRoutine(En[i], h, V)
    G = np.diag(G)
    G00[i] = G[0]
print(G00)
Y = G00
X = En
Y1 = Y.real
Y2 = Y.imag
Y1 = np.sort(Y1)
Y2 = np.sort(Y2)
print(Y1, Y2)
real, = plt.plot(X, Y1, label='real')
imag, = plt.plot(X, Y2, label='imag')
plt.axis('equal')
plt.grid(which='major', axis='both')
plt.legend(handles=[imag, real])
plt.show()
