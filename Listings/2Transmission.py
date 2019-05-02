from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib import path
import matplotlib.patches as patches
import numpy as np                      # NumPy
from numpy import linalg as LA
from Functions import ImportSystem, Onsite, Hop, Hkay, RecursionRoutine
import sisl as si
import sys
np.set_printoptions(threshold=sys.maxsize)

filename = input('Enter filename: ')
filename = filename + '.fdf'

nx = 1
ny = 1

xyz, UX, UY = ImportSystem(filename, nx, ny)

print('Unit Cell x: {}'.format(UX))
print('Unit Cell y: {}'.format(UY))

plt.scatter(xyz[:, 0], xyz[:, 1])
plt.axis('equal')
for i in range(xyz[:, 0].shape[0]):
    s = i
    xy = (xyz[i, 0], xyz[i, 1])
    plt.annotate(s, xy)
plt.grid(b=True, which='both', axis='both')
plt.show()

L = np.fromstring(
    input('Left contant atomic indices (#-#): '), dtype=int, sep='-')
R = np.fromstring(
    input('Right contant atomic indices (#-#): '), dtype=int, sep='-')
L = np.arange(L[0], L[1]+1, 1, dtype=int)
R = np.arange(R[0], R[1]+1, 1, dtype=int)
# LX = '0, 2.46'
# LX = np.fromstring(LX, dtype=float, sep=',')
# LY = '0, 4.26'
# LY = np.fromstring(LY, dtype=float, sep=',')
# RX = '4.92, 7.38'
# RX = np.fromstring(RX, dtype=float, sep=',')
# RY = '0, 4.26'
# RY = np.fromstring(RY, dtype=float, sep=',')

Lxyz = xyz[L]
Rxyz = xyz[R]
RmArray = np.append(L, R).astype(int)
Cxyz = np.delete(xyz, RmArray, 0)

plt.scatter(Lxyz[:, 0], Lxyz[:, 1], c='red', label='L')
plt.scatter(Cxyz[:, 0], Cxyz[:, 1], c='orange', label='C')
plt.scatter(Rxyz[:, 0], Rxyz[:, 1], c='blue', label='R')
plt.legend()
plt.axis('equal')
for i in range(xyz[:, 0].shape[0]):
    s = i
    xy = (xyz[i, 0], xyz[i, 1])
    plt.annotate(s, xy)
plt.grid(b=True, which='both', axis='both')
plt.show()

HD = Onsite(xyz=xyz, Vppi=-1)
HL = HD[0:L.shape[0], 0:L.shape[0]]
# HL = Onsite(xyz=Lxyz, Vppi=-1)
HR = HD[-L.shape[0]:, -L.shape[0]:]
HR = Onsite(xyz=Rxyz, Vppi=-1)
shiftx = 2.46
Lxyz1 = Lxyz - np.array([shiftx, 0, 0])
Rxyz1 = Rxyz + np.array([shiftx, 0, 0])
VL = Hop(xyz=Lxyz1, xyz1=Lxyz, Vppi=-1)
VR = Hop(xyz=Rxyz, xyz1=Rxyz1, Vppi=-1)
plt.imshow(HD)
plt.colorbar()
plt.show()
plt.imshow(HL)
plt.colorbar()
plt.show()
plt.imshow(HR)
plt.colorbar()
plt.show()
plt.imshow(VL)
plt.colorbar()
plt.show()
plt.imshow(VR)
plt.colorbar()
plt.show()

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

    GD[q] = LA.inv(np.identity(HD.shape[0])
                   * (i + eta) - HD - SEL - SER)
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
plt.title('Greens function at 0th site')
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
savename = filename.replace('.fdf', 'imrealTE.eps')
plt.savefig(savename, bbox_inches='tight')
plt.show()

T = np.zeros(En.shape[0], dtype=complex)
for i in range(En.shape[0]):
    T[i] = np.trace(GammaR[i] @ GD[i] @ GammaL[i] @ GD[i].conj().T)
print(T)
Y = T.real
print(Y)
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
