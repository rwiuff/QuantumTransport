from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import ImportSystem, Onsite, Hop, DefineDevice
from Functions import EnergyRecursion, Transmission
import sys

np.set_printoptions(threshold=sys.maxsize)

nx = 1
ny = 1
shiftx = 2.46

xyz, UX, UY, filename = ImportSystem(nx)

RestL, RestR, L, R, C = DefineDevice(xyz)

HD = Onsite(xyz=xyz, Vppi=-1, f=1)[0]

HL = HD[0:L.shape[0], 0:L.shape[0]]
HR = HD[-R.shape[0]:, -R.shape[0]:]
Lxyz = xyz[L]
Rxyz = xyz[R]

Lxyz1 = Lxyz - np.array([shiftx, 0, 0])
Rxyz1 = Rxyz + np.array([shiftx, 0, 0])
VL = Hop(xyz=Lxyz1, xyz1=Lxyz, Vppi=-1)
VR = Hop(xyz=Rxyz, xyz1=Rxyz1, Vppi=-1)
gs = GridSpec(2, 2, width_ratios=[1, 2])
plt.figure(figsize=(7, 4))
ax1 = plt.subplot(gs[:, 1])
plt.imshow(HD)
ax2 = plt.subplot(gs[0, 0])
plt.imshow(HL)
ax3 = plt.subplot(gs[1, 0])
plt.imshow(HR)
plt.show()
plt.figure(figsize=(7, 4))
plt.subplot(121)
plt.imshow(VL)
plt.subplot(122)
plt.imshow(VR)
plt.show()

En = np.linspace(-3, 3, 1000)
eta = 1e-6j

GD, GammaL, GammaR = EnergyRecursion(HD, HL, HR, VL, VR, En, eta)
site = int(input('Choose site: '))
G = np.zeros((En.shape[0]), dtype=complex)
bar = Bar('Retrieving Greens function ', max=En.shape[0])
for i in range(En.shape[0]):
    G[i] = GD["GD{:d}".format(i)].diagonal()[site]
    bar.next()
bar.finish()

Y = G
X = En
Y1 = Y.real
Y2 = Y.imag
real, = plt.plot(X, Y1, label='real')
imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
plt.ylim((-4, 4))
plt.grid(which='both', axis='both')
plt.legend(handles=[imag, real])
if site == 1:
    plt.title('Greens function at {:d}st site'.format(site))
if site == 2:
    plt.title('Greens function at {:d}nd site'.format(site))
if site == 3:
    plt.title('Greens function at {:d}rd site'.format(site))
else:
    plt.title('Greens function at {:d}th site'.format(site))
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel('Re[G(E)]/Im[G(E)]')

plt.show()

T = Transmission(GammaL=GammaL, GammaR=GammaR, GD=GD, En=En)

Y = T.real
X = En
plt.plot(X, Y)
# plt.ylim((0, 1))
plt.grid(which='both', axis='both')
plt.title('Transmission')
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel(r'T(E)')
#savename = filename.replace('.fdf', 'TE.eps')
#plt.savefig(savename, bbox_inches='tight')
plt.show()
