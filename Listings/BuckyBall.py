from ase import Atoms                   # Used to extract molecular coordinates
from ase.build import molecule          # Constructs molecules
from matplotlib import pyplot as plt    # Pyplot for nice graphs
from sympy import I, simplify           # Imaginary unit and simplify
import math                             # Maths
import sympy as sym                     # SymPy
import numpy as np                      # NumPy
Vppi = -1

np.set_printoptions(threshold=np.inf)

BB = molecule('C60')
xyz = BB.get_positions()
Ham = np.subtract.outer(xyz[0, :], xyz[:, 0])
print(xyz)
print(Ham)
print(Ham.shape)
for i in range(Ham.shape[0]):
    for j in range(Ham.shape[1]):
        if abs(Ham[i, j]) > 1.6:
            Ham[i, j] = 0
        else:
            Ham[i, j] = Vppi
print(Ham)
print(Ham.shape)
