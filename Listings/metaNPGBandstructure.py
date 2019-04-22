from matplotlib import pyplot as plt     # Pyplot for nice graphs
import numpy as np                      # NumPy
from Functions import xyzimport, Hkay, Onsite, Hop

# Set hopping potential
Vppi = -1

# Define lattice vectors
shiftx = 40.2591751900
shifty = 8.6935000000

# Retrieve unit cell
xyz = xyzimport('meta_C.fdf')
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

# Define k-space range
k = np.linspace(0, np.pi, 1000)
# Array for X-bands
X = np.zeros((Ham.shape[0], k.size))
# Array for Z-bands
Z = np.zeros((Ham.shape[0], k.size))
# Get bands from gamma to X and Z
for i in range(k.shape[0]):
    X[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=-k[i], y=0)[0]
    Z[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=k[i])[0]
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
plt.title('NPG-meta')
plt.ylim(-1, 1)
# plt.show()
plt.savefig('metaNPGBS.eps', bbox_inches='tight')
