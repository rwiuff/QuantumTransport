from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import Import, NPGElectrode
from Functions import EnergyRecursion, Transmission, PeriodicHamiltonian, Hkay, Hop, Onsite, ImportSystem 
import sys
from fractions import Fraction
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(threshold=sys.maxsize)

Vppi=-1

# Retrieve unit cell
xyz, shiftx, shifty, filename = ImportSystem(1)
# Calculate onsite nearest neighbours
Ham = Onsite(xyz, Vppi)

plt.imshow(Ham)
plt.colorbar()
plt.show()
# Shift unit cell
xyz1 = xyz + np.array([shiftx, 0, 0])
# Calculate offsite nearest neighbours
V1 = Hop(xyz, xyz1, Vppi)

plt.imshow(V1)
plt.colorbar()
plt.show()
plt.imshow(np.transpose(V1))
plt.colorbar()
plt.show()
# Shift unit cell
xyz2 = xyz + np.array([0, shifty, 0])
# Calculate offsite nearest neighbours
V2 = Hop(xyz, xyz2, Vppi)

plt.imshow(V2)
plt.colorbar()
plt.show()
plt.imshow(np.transpose(V2))
plt.colorbar()
plt.show()
# Shift unit cell
xyz3 = xyz + np.array([shiftx, shifty, 0])
# Calculate offsite nearest neighbours
V3 = Hop(xyz, xyz3, Vppi)

plt.imshow(V3)
plt.colorbar()
plt.show()
plt.imshow(np.transpose(V3))
plt.colorbar()
plt.show()
eta = 1e-6j

# Define k-space range
k = np.linspace(0, np.pi, 1000)
# Array for X-bands
X = np.zeros((Ham.shape[0], k.size))
# Array for Z-bands
Z = np.zeros((Ham.shape[0], k.size))
# Get bands from gamma to X and Z
for i in range(k.shape[0]):
    X[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=-k[i], y=0)[0]*2.7
    Z[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=k[i])[0]*2.7
# Get energies at k(0,0)
zero = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)[0]
# Renormalise distances according to lattice vectors
Xspace = np.linspace(0, 1 / shifty, 1000)
Zspace = np.linspace(0, 1 / shiftx, 1000)
# Plot Bandstructures
ax = plt.figure(figsize=(1, 6))
for i in range(X.shape[0]):
    plt.plot(np.flip(-Zspace, axis=0),
             np.flip(X[i, :], axis=0), 'k', linewidth=1)
    plt.plot(Xspace, Z[i, :], 'k', linewidth=1)
xtick = np.array([-1 / shiftx, 0, 1 / shifty])
plt.xticks(xtick, ('X', r'$\Gamma$', 'Z'))
plt.axvline(x=0, linewidth=1, color='k', linestyle='--')
filename = filename.replace('.fdf', '')
plt.title(filename)
plt.ylim(-1.5, 1.5)
plt.show()
# savename = filename + 'Bandstructures.eps'
# plt.savefig(savename, bbox_inches='tight')