from matplotlib import pyplot as plt     # Pyplot for nice graphs
import numpy as np                      # NumPy
from Functions import GrapheneSheet
import sisl as si
import sys
np.set_printoptions(threshold=sys.maxsize)

Graphene = GrapheneSheet(1, 1)

print(Graphene)
print(Graphene.xyz)

plt.scatter(Graphene.xyz[:, 0], Graphene.xyz[:, 1])
plt.axis('equal')
for i in range(Graphene.xyz[:, 0].shape[0]):
    s = i
    xy = (Graphene.xyz[i, 0], Graphene.xyz[i, 1])
    plt.annotate(s, xy)
plt.show()

filename = input('Enter filename: ')
filename = filename + '.fdf'

Graphene.write(filename)
