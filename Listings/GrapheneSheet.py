from matplotlib import pyplot as plt     # Pyplot for nice graphs
import matplotlib
import numpy as np                      # NumPy
from numpy import linalg as LA
from Functions import GrapheneSheet
import sisl as si
import sys
np.set_printoptions(threshold=sys.maxsize)

# Set hopping potential
Vppi = -1

Graphene = GrapheneSheet(10, 1)
print(Graphene)
plt.scatter(Graphene.xyz[:, 0], Graphene.xyz[:, 1])
plt.axis('equal')
plt.show()
plt.plot(Graphene, atom_indices=True)
