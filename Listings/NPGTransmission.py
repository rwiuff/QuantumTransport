from matplotlib import pyplot as plt     # Pyplot for nice graphs
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
import numpy as np                      # NumPy
from Functions import Import, NPGElectrode
from Functions import EnergyRecursion, Transmission, PeriodicHamiltonian
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
axnames = ''
print('Plotting Greens functions')
for i in range(numplot):
    a = 'ax{},'.format(i + 1)
    axnames = axnames + a
fig, (axnames) = plt.subplots(nrow, ncol, sharex=True)
for i in range(nrow):
    for j in range(ncol):
        if q + 1 == numplot:
            G = np.average(GG, axis=0)
            Y = G
            Y1 = Y.real
            Y2 = Y.imag
            fig.axes[numplot - 1].plot(X, Y1, label='real')
            fig.axes[numplot - 1].fill_between(X, 0, Y2, color='orange', alpha=0.8, label='imag')
            fig.axes[numplot - 1].grid(which='both', axis='both')
            fig.axes[numplot - 1].legend(loc="upper right")
            fig.axes[numplot - 1].set_title('Average over k-points')
            fig.axes[numplot - 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            Y = GG[q, :]
            Y1 = Y.real
            Y2 = Y.imag
            fig.axes[q].plot(X, Y1, label='real')
            fig.axes[q].fill_between(X, 0, Y2, color='orange', alpha=0.8, label='imag')
            fig.axes[q].grid(which='both', axis='both')
            fig.axes[q].legend(loc="upper right")
            frac = Fraction(kP[q]*(1/np.pi))
            pi = r'$\ \pi$'
            fig.axes[q].set_title('{}'.format(frac)+pi)
            fig.axes[q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            q = q + int(1)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Energy E arb. unit')
plt.ylabel('Re[G00(E)]/Im[G00(E)]')
plt.show()

q = int(0)
axnames = ''
print('Plotting Transmission')
for i in range(numplot):
    a = 'ax{},'.format(i + 1)
    axnames = axnames + a
fig, (axnames) = plt.subplots(nrow, ncol, sharex=True)
for i in range(nrow):
    for j in range(ncol):
        if q + 1 == numplot:
            T = np.average(TT, axis=0)
            Y = T.real
            fig.axes[numplot - 1].plot(X, Y)
            fig.axes[numplot - 1].grid(which='both', axis='both')
            fig.axes[numplot - 1].set_title('Average over k-points')
            fig.axes[numplot - 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            T = TT[q]
            Y = T.real
            fig.axes[q].plot(X, Y)
            fig.axes[q].grid(which='both', axis='both')
            frac = Fraction(kP[q]*(1/np.pi))
            pi = r'$\ \pi$'
            fig.axes[q].set_title('{}'.format(frac)+pi)
            fig.axes[q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            q = q + int(1)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('E[eV]')
plt.ylabel('T(E)')
plt.show()

input("Press any key to quit")
quit()
