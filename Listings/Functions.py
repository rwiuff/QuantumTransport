from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
from scipy.sparse import dia_matrix
import numpy as np                      # NumPy
from numpy import linalg as LA
from collections import Counter
import sisl as si


def xyzimport(path):
    fdf = si.io.siesta.fdfSileSiesta(path, mode='r', base=None)
    geom = fdf.read_geometry(output=False)
    xyz = geom.xyz
    return xyz


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
