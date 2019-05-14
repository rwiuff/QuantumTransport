from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import ImportSystem, Onsite, Hop, DefineDevice
from Functions import EnergyRecursion, Transmission, PeriodicHamiltonian
import sys

np.set_printoptions(threshold=sys.maxsize)

nx = 1
ny = 2
shiftx = 2.46
shifty = 4.26
En = np.linspace(-3, 3, 1000)
eta = 1e-4j
kP = np.linspace(-np.pi, np.pi, ny)

xyz, UX, UY, filename = ImportSystem(nx)

RestL, L, R, C, RestR = DefineDevice(xyz)

TT = np.zeros((kP.shape[0], En.shape[0]))
GG = np.zeros((kP.shape[0], En.shape[0]), dtype=complex)
q = 0
for i in kP:
    Ham = PeriodicHamiltonian(xyz, UY, i)
    HL = Ham[L]
    HL = HL[:, L]
    HR = Ham[R]
    HR = HR[:, R]
    VL = Ham[L]
    VL = VL[:, RestL]
    VR = Ham[RestR]
    VR = VR[:, R]
    gs = GridSpec(2, 2, width_ratios=[1, 2])
    a = plt.figure(figsize=(7, 4))
    ax1 = plt.subplot(gs[:, 1])
    plt.imshow(Ham.real)
    plt.colorbar()
    ax2 = plt.subplot(gs[0, 0])
    plt.imshow(HL.real)
    ax3 = plt.subplot(gs[1, 0])
    plt.imshow(HR.real)
    a.show()
    b = plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plt.imshow(VL.real)
    plt.subplot(122)
    plt.imshow(VR.real)
    plt.colorbar()
    b.show()

    input('Press any key to continue')

    GD, GammaL, GammaR = EnergyRecursion(Ham, HL, HR, VL, VR, En, eta)

    G = np.zeros((En.shape[0]), dtype=complex)
    bar = Bar('Retrieving Greens function ', max=En.shape[0])
    for i in range(En.shape[0]):
        G[i] = GD["GD{:d}".format(i)].diagonal()[0]
        bar.next()
    bar.finish()

    # Y = G00
    # X = En
    # Y1 = Y.real
    # Y2 = Y.imag
    # real, = plt.plot(X, Y1, label='real')
    # imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
    # plt.ylim((-10, 20))
    # plt.grid(which='both', axis='both')
    # plt.legend(handles=[imag, real])
    # plt.title('Greens function at 0th site')
    # plt.xlabel('Energy E arb. unit')
    # plt.ylabel('Re[G00(E)]/Im[G00(E)]')
    # savename = filename.replace('.fdf', 'imrealTE.eps')
    # plt.savefig(savename, bbox_inches='tight')
    # plt.show()

    T = Transmission(GammaL=GammaL, GammaR=GammaR, GD=GD, En=En)

    # Y = T.real
    # X = En
    # plt.plot(X, Y)
    # plt.ylim((0, 1))
    # plt.grid(which='both', axis='both')
    # plt.title('Transmission')
    # plt.xlabel(r'$E(V_{pp\pi})$')
    # plt.ylabel(r'T(E)')
    # savename = filename.replace('.fdf', 'TE.eps')
    # plt.savefig(savename, bbox_inches='tight')
    # plt.show()
    GG[q, :] = G
    TT[q, :] = T.real
    q = q + 1

plt.subplot(131)
Y = GG[0, :]
X = En
Y1 = Y.real
Y2 = Y.imag
real, = plt.plot(X, Y1, label='real')
imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
plt.ylim((-10, 20))
plt.grid(which='both', axis='both')
plt.legend(handles=[imag, real])
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
plt.subplot(132)
Y = GG[1, :]
X = En
Y1 = Y.real
Y2 = Y.imag
real, = plt.plot(X, Y1, label='real')
imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
plt.ylim((-10, 20))
plt.grid(which='both', axis='both')
plt.legend(handles=[imag, real])
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
plt.subplot(133)
G = np.average(GG, axis=0)
Y = G
X = En
Y1 = Y.real
Y2 = Y.imag
real, = plt.plot(X, Y1, label='real')
imag, = plt.fill(X, Y2, c='orange', alpha=0.8, label='imag')
plt.ylim((-10, 20))
plt.grid(which='both', axis='both')
plt.legend(handles=[imag, real])
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
plt.title('Greens function at 0th site')
savename = filename.replace('.fdf', 'AverageimrealTE.eps')
plt.savefig(savename, bbox_inches='tight')
plt.show()

plt.subplot(131)
T = TT[0]
Y = T.real
X = En
plt.plot(X, Y)
# plt.ylim((0, 1))
plt.grid(which='both', axis='both')
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel(r'T(E)')
plt.subplot(132)
T = TT[1]
Y = T.real
X = En
plt.plot(X, Y)
# plt.ylim((0, 1))
plt.grid(which='both', axis='both')
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel(r'T(E)')
plt.subplot(133)
T = np.average(TT, axis=0)
Y = T.real
X = En
plt.plot(X, Y)
# plt.ylim((0, 1))
plt.grid(which='both', axis='both')
plt.xlabel(r'$E(V_{pp\pi})$')
plt.ylabel(r'T(E)')
plt.title('Average Transmission')
savename = filename.replace('.fdf', 'AverageTE.eps')
plt.savefig(savename, bbox_inches='tight')
plt.show()
