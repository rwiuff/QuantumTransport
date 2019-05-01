from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib import path
import matplotlib.patches as patches
import numpy as np                      # NumPy
from numpy import linalg as LA
from Functions import xyzimport, Onsite, Hop, Hkay, RecursionRoutine
import sisl as si
import sys
np.set_printoptions(threshold=sys.maxsize)

filename = input('Enter filename: ')
filename = filename + '.fdf'

xyz = xyzimport(filename)

plt.scatter(xyz[:, 0], xyz[:, 1])
plt.axis('equal')
for i in range(xyz[:, 0].shape[0]):
    s = i
    xy = (xyz[i, 0], xyz[i, 1])
    plt.annotate(s, xy)
plt.grid(b=True, which='both', axis='both')
plt.show()

# LX = input('(x1, x2): ')
# LY = input('(y1, y2): ')
# RX = input('(x1, x2): ')
# RY = input('(y1, y2): ')
LX = '0, 2.46'
LX = np.fromstring(LX, dtype=float, sep=',')
LY = '0, 4.26'
LY = np.fromstring(LY, dtype=float, sep=',')
RX = '4.92, 7.38'
RX = np.fromstring(RX, dtype=float, sep=',')
RY = '0, 4.26'
RY = np.fromstring(RY, dtype=float, sep=',')
Lverts = [(LX[0], LY[0]),
          (LX[1], LY[0]),
          (LX[1], LY[1]),
          (LX[0], LY[1]),
          (LX[0], LY[0])]

Rverts = [(RX[0], RY[0]),
          (RX[1], RY[0]),
          (RX[1], RY[1]),
          (RX[0], RY[1]),
          (RX[0], RY[0])]

Lpath = path.Path(Lverts)
Rpath = path.Path(Rverts)

Lcell = np.array([])
Rcell = np.array([])
xy = np.delete(xyz, 2, 1)
Lxyz = np.array([[0, 0, 0]])
print(xy)
Test = Lpath.contains_points(xy)
for i in range(Test.size):
    if Test[i] == True:
        Lcell = np.append([Lcell], [i])
        Lxyz = np.append(Lxyz, [xyz[i, :]], axis=0)
Lxyz = np.delete(Lxyz, 0, 0)

Rxyz = np.array([[0, 0, 0]])
Test = Rpath.contains_points(xy)
for i in range(Test.size):
    if Test[i] == True:
        Rcell = np.append([Rcell], [i])
        Rxyz = np.append(Rxyz, [xyz[i, :]], axis=0)
Rxyz = np.delete(Rxyz, 0, 0)

fig = plt.figure()
ax = fig.add_subplot(111)
LPatch = patches.PathPatch(Lpath, facecolor='orange', lw=2)
ax.add_patch(LPatch)
RPatch = patches.PathPatch(Rpath, facecolor='orange', lw=2)
ax.add_patch(RPatch)
plt.grid(b=True, which='both', axis='both')
plt.scatter(xyz[:, 0], xyz[:, 1])
plt.axis('equal')
for i in range(xyz[:, 0].shape[0]):
    s = i
    xy = (xyz[i, 0], xyz[i, 1])
    plt.annotate(s, xy)
plt.show()

HD = Onsite(xyz=xyz, Vppi=-1)
HL = Onsite(xyz=Lxyz, Vppi=-1)
HR = Onsite(xyz=Rxyz, Vppi=-1)
shiftx = 2.46
Lxyz1 = Lxyz - np.array([shiftx, 0, 0])
Rxyz1 = Rxyz + np.array([shiftx, 0, 0])
VL = Hop(xyz=Lxyz, xyz1=Lxyz1, Vppi=-1)
VR = Hop(xyz=Rxyz, xyz1=Rxyz1, Vppi=-1)
# plt.imshow(HD)
# plt.colorbar()
# plt.show()
# plt.imshow(HL)
# plt.colorbar()
# plt.show()
# plt.imshow(HR)
# plt.colorbar()
# plt.show()
# plt.imshow(VL)
# plt.colorbar()
# plt.show()
# plt.imshow(VR)
# plt.colorbar()
# plt.show()

En = np.linspace(-3, 3, 1000)
GD = np.zeros([En.shape[0], HD.shape[0], HD.shape[1]], dtype=complex)
GammaL = np.zeros([En.shape[0], HD.shape[0], HD.shape[1]], dtype=complex)
GammaR = np.zeros([En.shape[0], HD.shape[0], HD.shape[1]], dtype=complex)
q = 0
eta = 1e-6j
for i in En:
    gl, crap, SEL = RecursionRoutine(i, HL, VL, eta=eta)
    gr, SER, crap = RecursionRoutine(i, HR, VR, eta=eta)
    SS = SEL.shape[0]
    Matrix = np.zeros((HD.shape), dtype=complex)
    Matrix[0:SS, 0:SS] = SEL
    SEL = Matrix

    SS = SER.shape[0]
    Matrix = np.zeros((HD.shape), dtype=complex)
    Matrix[-SS:, -SS:] = SER
    SER = Matrix

    GD[q] = LA.inv(np.identity(HD.shape[0]) *
                   (i + eta) - HD - SEL - SER)
    GammaL[q] = 1j * (SEL - SEL.conj().T)
    GammaR[q] = 1j * (SER - SER.conj().T)
    q = q + 1

G00 = np.zeros((En.shape[0]), dtype=complex)
for i in range(En.shape[0]):
    G = np.diag(GD[i])
    G00[i] = G[0]

Y = G00
X = En
Y1 = Y.real
Y2 = Y.imag
# Y1 = np.sort(Y1)
# Y2 = np.sort(Y2)
print(Y1, Y2)
real, = plt.plot(X, Y1, label='real')
# imag, = plt.plot(X, Y2, label='imag')
imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
plt.ylim((-10, 20))
# plt.axis('equal')
plt.grid(which='both', axis='both')
plt.legend(handles=[imag, real])
plt.title('Greens function of a simple four-atom unit cell')
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
savename = filename.replace('.fdf', 'imrealTE.eps')
plt.savefig(savename, bbox_inches='tight')
plt.show()

T = np.zeros(En.shape[0], dtype=complex)
for i in range(En.shape[0]):
    T[i] = np.trace(GammaR[i] @ GD[i] @ GammaL[i] @ GD[i].conj().T)
print(T)
Y = T
X = En
plt.plot(X, Y)
plt.ylim((0, 1))
# plt.axis('equal')
plt.grid(which='both', axis='both')
plt.title('Transmission for simple system')
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel(r'T(E)')
savename = filename.replace('.fdf', 'TE.eps')
plt.savefig(savename, bbox_inches='tight')
plt.show()
