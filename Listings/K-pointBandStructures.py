from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import Import, NPGElectrode
from Functions import EnergyRecursion, Transmission, PeriodicHamiltonian, Hkay
import sys
from fractions import Fraction
from matplotlib.ticker import FormatStrFormatter

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
    # Define k-space range
    k = np.linspace(0, np.pi, 1000)
    # Array for X-bands
    X = np.zeros((HL.shape[0], k.size))
    # Array for Z-bands
    Z = np.zeros((HL.shape[0], k.size))
    # Get bands from gamma to X and Z
    for i in range(k.shape[0]):
        X[:, i] = Hkay(Ham=HL, V1=VL, V2=0, V3=0, x=-k[i], y=0)[0]
        Z[:, i] = Hkay(Ham=HL, V1=VL, V2=0, V3=0, x=k[i], y=0)[0]
    # Get energies at k(0,0)
    zero = Hkay(Ham=HL, V1=VL, V2=0, V3=0, x=0, y=0)[0]
    # Renormalise distances according to lattice vectors
    Xspace = np.linspace(0, 1 / UX, 1000)
    Zspace = np.linspace(0, 1 / UY, 1000)
    # Plot Bandstructures
    ax = plt.figure(figsize=(1, 6))
    for i in range(X.shape[0]):
        plt.plot(np.flip(-Zspace, axis=0),
                 np.flip(X[i, :], axis=0), 'k', linewidth=1)
        plt.plot(Xspace, Z[i, :], 'k', linewidth=1)
    xtick = np.array([-1 / UY, 0, 1 / UX])
    plt.xticks(xtick, ('X', r'$\Gamma$', 'Z'))
    plt.axvline(x=0, linewidth=1, color='k', linestyle='--')
    filename = filename.replace('.fdf', '')
    plt.title(filename)
    plt.ylim(-1.5, 1.5)
    plt.show()
