from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import Import, NPGElectrode
from Functions import EnergyRecursion, Transmission, PeriodicHamiltonian
import sys

np.set_printoptions(threshold=sys.maxsize)

nx = 1
ny = 2
contactrep = 1
shiftx = 2.46
ev1 = -1.5
ev2 = 1.5
numkP = 5
En = np.linspace(ev1 / 2.7, ev2 / 2.7, 200)
En = np.delete(En, np.where(En == 1))
En = np.delete(En, np.where(En == -1))

eta = 1e-6j
kP = np.linspace(-np.pi, np.pi, numkP)

xyz, UX, UY, filename, dgeom, cellsize = Import(nx, contactrep)

RestL, L, R, C, RestR = NPGElectrode(xyz, dgeom, cellsize, nx)

TT = np.zeros((kP.shape[0], En.shape[0]))
GG = np.zeros((kP.shape[0], En.shape[0]), dtype=complex)
q = 0
for i in kP:
    print('----------------------------------------------------------------------')
    print('Calculating for k-point:    {}'.format(i))
    Ham = PeriodicHamiltonian(xyz, UY, i)
    HL = Ham[L]
    HL = HL[:, L]
    HR = Ham[R]
    HR = HR[:, R]
    VL = Ham[L]
    VL = VL[:, RestL]
    VR = Ham[RestR]
    VR = VR[:, R]
    # gs = GridSpec(2, 2, width_ratios=[1, 2])
    # a = plt.figure(figsize=(7, 4))
    # ax1 = plt.subplot(gs[:, 1])
    # plt.imshow(Ham.real)
    # ax2 = plt.subplot(gs[0, 0])
    # plt.imshow(HL.real)
    # ax3 = plt.subplot(gs[1, 0])
    # plt.imshow(HR.real)
    # a.show()
    # b = plt.figure(figsize=(7, 4))
    # plt.subplot(121)
    # plt.imshow(VL.real)
    # plt.subplot(122)
    # plt.imshow(VR.real)
    # b.show()
    # input('Press any key to continue')

    GD, GammaL, GammaR = EnergyRecursion(Ham, HL, HR, VL, VR, En, eta)

    G = np.zeros((En.shape[0]), dtype=complex)
    bar = Bar('Retrieving Greens function ', max=En.shape[0])
    for i in range(En.shape[0]):
        G[i] = GD["GD{:d}".format(i)].diagonal()[0]
        bar.next()
    bar.finish()

    T = Transmission(GammaL=GammaL, GammaR=GammaR, GD=GD, En=En)

    GG[q, :] = G
    TT[q, :] = T.real
    q = q + 1
X = En * 2.7
Y0 = 0
numplot = numkP + 1
if numplot % 2 == 0:
    if numplot % 3 == 0:
        ncol = 3
    else:
        ncol = 2
else:
    if numplot % 3 == 0:
        ncol = 3
nrow = numplot / ncol
numplot = int(numplot)
nrow = int(nrow)
ncol = int(ncol)
q = int(0)
numplot = int(numplot)
print('Plotting Greens functions')
for i in range(nrow):
    for j in range(ncol):
        if q + 1 == numplot:
            subplotn = "{}{}{}".format(nrow, ncol, numplot)
            plt.subplot(int(subplotn))
            G = np.average(GG, axis=0)
            Y = G
            Y1 = Y.real
            Y2 = Y.imag
            real, = plt.plot(X, Y1, label='real')
            imag, = plt.fill(X, Y2, color='orange', alpha=0.8, label='imag')
            plt.grid(which='both', axis='both')
            plt.legend(handles=[imag, real])
            plt.xlabel('Energy E arb. unit')
            plt.ylabel('Re[G00(E)]/Im[G00(E)]')
            plt.title('Greens function at 0th site')
        else:
            subplotn = "{}{}{}".format(nrow, ncol, int(q + 1))
            plt.subplot(int(subplotn))
            Y = GG[q, :]
            Y1 = Y.real
            Y2 = Y.imag
            real, = plt.plot(X, Y1, label='real')
            imag, = plt.fill(X, Y2, color='orange', alpha=0.8, label='imag')
            plt.grid(which='both', axis='both')
            plt.legend(handles=[imag, real])
            plt.xlabel('Energy E arb. unit')
            plt.ylabel('Re[G00(E)]/Im[G00(E)]')
            q = q + int(1)
plt.show()

q = int(0)
print('Plotting Transmission')
for i in range(nrow):
    for j in range(ncol):
        if q + 1 == numplot:
            subplotn = "{}{}{}".format(nrow, ncol, numplot)
            plt.subplot(int(subplotn))
            T = np.average(TT, axis=0)
            Y = T.real
            plt.plot(X, Y)
            plt.grid(which='both', axis='both')
            plt.xlabel(r'$E(V_{pp\pi})$')
            plt.ylabel(r'T(E)')
            plt.title('Average Transmission')
        else:
            subplotn = "{}{}{}".format(nrow, ncol, int(q + 1))
            plt.subplot(int(subplotn))
            T = TT[q]
            Y = T.real
            plt.plot(X, Y)
            plt.grid(which='both', axis='both')
            plt.xlabel(r'$E(V_{pp\pi})$')
            plt.ylabel(r'T(E)')
            q = q + int(1)
plt.show()

input("Press any key to quit")
quit()
