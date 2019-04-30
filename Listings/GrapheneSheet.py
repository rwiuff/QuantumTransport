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

Graphene = GrapheneSheet(10, 10)
print(Graphene)
plt.plot(Graphene)
plt.show()
coord = Graphene.xyz[0, :]
temp = Graphene.remove(0)
plt.plot(temp)
plt.scatter(temp.xyz[:, 0], temp.xyz[:, 1])
plt.scatter(coord[0], coord[1], c='r', s=200)
plt.show()
