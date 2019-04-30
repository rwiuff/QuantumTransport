# -------------------------------------------------------------------- #
#                                                                      #
#            Python script containing various routines                 #
#        for calculating Green's functions using Tight-binding         #
#                                                                      #
#                                                                      #
#                             Written by                               #
#                                                                      #
#               Christoffer Sørensen (chves@dtu.dk)                    #
#                       Rasmus Wiuff (rwiuff@gmail.com)                #
#                                                                      #
# -------------------------------------------------------------------- #


from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
from scipy.sparse import dia_matrix
import numpy as np                      # NumPy
from numpy import linalg as LA
from collections import Counter
import sisl as si
from sisl import Atom


def xyzimport(path):
    fdf = si.io.siesta.fdfSileSiesta(path, mode='r', base=None)
    geom = fdf.read_geometry(output=False)
    xyz = geom.xyz
    return xyz


def Onsite(xyz, Vppi):
    h = np.zeros((xyz.shape[0], xyz.shape[0]))
    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[0]):
            h[i, j] = LA.norm(np.subtract(xyz[i], xyz[j]))
    h = np.where(h < 1.6, Vppi, 0)
    h = np.subtract(h, Vppi * np.identity(xyz.shape[0]))
    return h


def Hop(xyz, xyz1, Vppi):
    hop = np.zeros((xyz.shape[0], xyz.shape[0]))
    for i in range(xyz.shape[0]):
        for j in range(xyz1.shape[0]):
            hop[i, j] = LA.norm(np.subtract(xyz[i], xyz1[j]))
    hop = np.where(hop < 1.6, Vppi, 0)
    return hop


def Hkay(Ham, V1, V2, V3, x, y):
    Ham = Ham + (V1 * np.exp(-1.0j * x)
                 + np.transpose(V1) * np.exp(1.0j * x)
                 + V2 * np.exp(-1.0j * y)
                 + np.transpose(V2) * np.exp(1.0j * y)
                 + V3 * np.exp(-1.0j * x) * np.exp(-1.0j * y)
                 + np.transpose(V3) * np.exp(1.0j * x) * np.exp(1.0j * y))
    e = LA.eigh(Ham)[0]
    v = LA.eigh(Ham)[1]
    return e, v


def RecursionRoutine(En, h, V):
    z = np.identity(h.shape[0]) * (En - 1e-6j)
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
    e, es = e0, es0
    SelfER = es - h
    SelfEL = e - h - SelfER
    G00 = LA.inv(z - es)
    # print(q)
    return G00, SelfER, SelfEL


def GrapheneSheet(nx, ny):
    Graphene = si.Geometry([[0.62, 3.55, 0],
                            [0.62, 0.71, 0],
                            [1.85, 2.84, 0],
                            [1.85, 1.42, 0]], [Atom('C')], [2.46, 4.26, 0])
    Graphene = Graphene.tile(nx, 0).tile(ny, 1)
    return Graphene
