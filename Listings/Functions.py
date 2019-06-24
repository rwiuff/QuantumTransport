# -------------------------------------------------------------------- #
#                                                                      #
#            Python script containing various routines                 #
#        for calculating Green's functions using Tight-binding         #
#                                                                      #
#                                                                      #
#                             Written by                               #
#                                                                      #
#               Christoffer Sørensen (chves@dtu.dk)                    #
#                       Rasmus Wiuff (rwiuff@gmail.com)                #
#                                                                      #
# -------------------------------------------------------------------- #


from matplotlib import pyplot as plt  # Pyplot for nice graphs
import numpy as np                    # NumPy
from numpy import linalg as LA        # Linalg module
import sisl as si                     # SISL for importing geometries
from sisl import Atom                 # The atom class for recreating geoms.
from progress.bar import Bar          # Progress bars in CLI
import time                           # Insight in execution times
import scipy.sparse as scp            # SciPy's sparse matrices


def xyzimport(path):
    fdf = si.io.siesta.fdfSileSiesta(path, mode='r', base=None)
    geom = fdf.read_geometry(output=False)
    xyz = geom.xyz
    return xyz


def Onsite(xyz, Vppi, f):
    h = np.zeros((xyz.shape[0], xyz.shape[0]))  # Empty matrix
    for i in range(xyz.shape[0]):  # Take an atomic coordinate
        for j in range(xyz.shape[0]):  # Take another atomic coordinate
            h[i, j] = LA.norm(np.subtract(xyz[i], xyz[j]))  # Measure distances
    h = np.where(h < 1.6, Vppi, 0)  # Replace distances under 1.6 angstrom with Vppi
    h = np.subtract(h, Vppi * np.identity(xyz.shape[0]))  # Remove the diagonal
    if f == 1:
        o = 'n'
    else:
        o = input('Change onsite? (y/n) ')
    if o == 'n':
        p = 0
    if o == 'y':
        charr = np.array(['PS4O', 'PS4OH', 'MS2O', 'MS2OH', 'MA2O', 'MA2OH'])
        for i in range(charr.shape[0]):
            print('{}: {}'.format(i + 1, charr[i]))
        struct = int(input('Choose structure: '))
        if struct >= 3:
            p = float(input('Potential: '))
            h[-1, -1] = p
            h[-2, -2] = p
        if struct <= 2:
            p = float(input('Potential: '))
            h[-1, -1] = p
            h[-2, -2] = p
            h[-3, -3] = p
            h[-4, -4] = p
    return h, p


def Hop(xyz, xyz1, Vppi):
    hop = np.zeros((xyz1.shape[0], xyz.shape[0]))
    for i in range(xyz1.shape[0]):
        for j in range(xyz.shape[0]):
            hop[i, j] = LA.norm(np.subtract(xyz1[i], xyz[j]))
    hop = np.where(hop < 1.6, Vppi, 0)
    return hop


def Hkay(Ham, V1, V2, V3, x, y):
    Ham = Ham + (V1 * np.exp(-1.0j * x)  # Onsite hops and hops in x
                 + np.transpose(V1) * np.exp(1.0j * x)  # Hops in -x
                 + V2 * np.exp(-1.0j * y)  # Hops in y
                 + np.transpose(V2) * np.exp(1.0j * y)  # Hops in -y
                 + V3 * np.exp(-1.0j * x) * np.exp(-1.0j * y)  # Hops in both x and y
                 + np.transpose(V3) * np.exp(1.0j * x) * np.exp(1.0j * y))  # Hops in both -x and -y
    e = LA.eigh(Ham)[0]  # Eigen energies from the Hamiltonian
    v = LA.eigh(Ham)[1]  # Eigen vectors from the Hamiltonian
    return e, v


def RecursionRoutine(En, h, V, eta):
    z = np.identity(h.shape[0]) * (En - eta)
    a0 = V.conj().transpose()
    b0 = V
    es0 = h
    e0 = h
    g0 = LA.inv(z - e0)
    q = 1
    while np.max(np.abs(a0)) > 1e-6:  # Loop with hop matrix as threshold
        ag = a0 @ g0  # Product defined here and used multiple times
        a1 = ag @ a0  # New hop matrix (transposed)
        bg = b0 @ g0  # Product defined here and used multiple times
        b1 = bg @ b0  # New hop matrix
        e1 = e0 + ag @ b0 + bg @ a0  # New onsite for "other cells"
        es1 = es0 + ag @ b0  # New onsite
        g1 = LA.inv(z - e1)  # New Green's function
        a0 = a1  # Overwrite old variable
        b0 = b1  # Overwrite old variable
        e0 = e1  # Overwrite old variable
        es0 = es1  # Overwrite old variable
        g0 = g1  # Overwrite old variable
        q = q + 1  # Counter (for diagnostic purposes)
    e, es = e0, es0  # Define the onsite Hamiltonians
    SelfER = es - h  # Self-energy from the right
    SelfEL = e - h - SelfER  # Self-energy from the left
    G00 = LA.inv(z - es)  # Green's functions
    # print(q)
    return G00, SelfER, SelfEL


def GrapheneSheet(nx, ny):
    Graphene = si.Geometry([[0.62, 3.55, 25],
                            [0.62, 0.71, 25],
                            [1.85, 2.84, 25],
                            [1.85, 1.42, 25]], [Atom('C')], [2.46, 4.26, 0])
    Graphene = Graphene.tile(nx, 0).tile(ny, 1)
    # Graphene = Graphene.sort(axes=(1, 0, 2))
    return Graphene


def ImportSystem(nx):
    filename = input('Enter filename: ')
    filename = filename + '.fdf'
    fdf = si.get_sile(filename)
    geom = fdf.read_geometry(output=False)
    C_list = (geom.atoms.Z == 6).nonzero()[0]
    O_list = (geom.atoms.Z == 8).nonzero()[0]
    C_O_list = np.concatenate((C_list, O_list))
    Subbed = geom.sub(C_O_list)
    Subbed.reduce()
    geom = Subbed
    geom = geom.tile(nx, 0)
    # xyz = geom.xyz
    # xyz = np.round(xyz, decimals=2)
    # geom = si.Geometry(xyz, [Atom('C')], [2.46, 4.26, 0])
    # geom = geom.sort(axes=(2, 1, 0))
    xyz = geom.xyz
    LatticeVectors = fdf.get('LatticeVectors')
    UX = np.fromstring(LatticeVectors[0], dtype=float, sep=' ')[0]
    UY = np.fromstring(LatticeVectors[1], dtype=float, sep=' ')[1]
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
    return xyz, UX, UY, filename


def DefineDevice(xyz):
    L = np.fromstring(
        input('Left contact atomic indices (#-#): '), dtype=int, sep='-')
    R = np.fromstring(
        input('Right contact atomic indices (#-#): '), dtype=int, sep='-')
    L = np.arange(L[0], L[1] + 1, 1, dtype=int)
    R = np.arange(R[0], R[1] + 1, 1, dtype=int)
    C = np.arange(L[1] + 3, R[0], 1, dtype=int)
    RestL = np.arange(0, L[0], dtype=int)
    RestR = np.arange(R[1] + 3, xyz.shape[0], 1, dtype=int)
    print(RestL)
    print(L)
    print(C)
    print(R)
    print(RestR)

    Lxyz = xyz[L]
    Rxyz = xyz[R]
    Cxyz = xyz[C]
    RmArray = np.append(L, C).astype(int)
    RmArray = np.append(RmArray, R).astype(int)
    Restxyz = np.delete(xyz, RmArray, 0)

    plt.scatter(Lxyz[:, 0], Lxyz[:, 1], c='red', label='L')
    plt.scatter(Cxyz[:, 0], Cxyz[:, 1], c='orange', label='C')
    plt.scatter(Rxyz[:, 0], Rxyz[:, 1], c='blue', label='R')
    plt.scatter(Restxyz[:, 0], Restxyz[:, 1], c='k')
    plt.legend()
    plt.title('Point plot of the simple system')
    plt.axis('equal')
    for i in range(xyz[:, 0].shape[0]):
        s = i
        xy = (xyz[i, 0], xyz[i, 1])
        plt.annotate(s, xy)
    plt.grid(b=True, which='both', axis='both')
#    savename = 'pointplot.eps'
#    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    return RestL, RestR, L, R, C


def EnergyRecursion(HD, HL, HR, VL, VR, En, eta):
    HD = scp.csr_matrix(HD)
    HL = scp.csr_matrix(HL)
    HR = scp.csr_matrix(HR)  # Matrices converted to Compressed Sparse Row Matrix
    VL = scp.csr_matrix(VL)
    VR = scp.csr_matrix(VR)
    start = time.time()
    GD = {}
    GammaL = {}
    GammaR = {}
    bar = Bar('Running Recursion          ', max=En.shape[0])
    q = 0
    for i in En:  # Iteration over each energy
        gl, scrap, SEL = RecursionRoutine(i, HL, VL, eta=eta)  # From the left
        gr, SER, scrap = RecursionRoutine(i, HR, VR, eta=eta)  # From the right
        SS = SEL.shape[0]
        Matrix = np.zeros((HD.shape), dtype=complex)
        Matrix[0:SS, 0:SS] = SEL
        SEL = Matrix

        SS = SER.shape[0]
        Matrix = np.zeros((HD.shape), dtype=complex)
        Matrix[-SS:, -SS:] = SER
        SER = Matrix

        SEL = scp.csr_matrix(SEL)
        SER = scp.csr_matrix(SER)
        GD["GD{:d}".format(q)] = scp.linalg.inv(  # Device Green's functions are saved
            scp.identity(HD.shape[0]) * (i + eta) - HD - SEL - SER)  # in a dictionary.
        GammaL["GammaL{:d}".format(q)] = 1j * (SEL - SEL.conj().transpose())  # Rate matrices are
        GammaR["GammaR{:d}".format(q)] = 1j * (SER - SER.conj().transpose())  # likewise saved.
        q = q + 1
        bar.next()
    bar.finish()
    end = time.time()
    print('Recursion Execution Time:   {} s'.format(end - start))
    return GD, GammaL, GammaR


def Transmission(GammaL, GammaR, GD, En):
    T = np.zeros(En.shape[0], dtype=complex)
    bar = Bar('Calculating Transmission   ', max=En.shape[0])
    for i in range(En.shape[0]):  # Iteration for every energy point.
        T[i] = np.trace((GammaR["GammaR{:d}".format(i)] @ GD["GD{:d}".format(
            i)] @ GammaL["GammaL{:d}".format(i)] @ GD["GD{:d}".format(i)].conj(
        ).transpose()).todense())  # Calculate transmission (equation V.9).
        bar.next()
    bar.finish()
    return T


def PeriodicHamiltonian(xyz, UY, i):
    h, p = Onsite(xyz=xyz, Vppi=-1, f=1)  # Calculate onsite hops
    V = Hop(xyz=xyz, xyz1=xyz + np.array([0, UY, 0]), Vppi=-1)  # Calculate hops
    print('Number of hopping elements: {}'.format(np.sum(np.abs(V))))  # Number of hops
    Ham = h + V * np.exp(1j * i) + np.transpose(V) * np.exp(-1j * i)  # Hamiltonian for ith kp
    return Ham


def Import(nx, contactrep):  # Function takes in number of device and contact repitition
    filename = input('Enter filename: ')  # Filename of fdf structure
    filename = filename + '.fdf'  # Add file ending
    fdf = si.get_sile(filename)  # Import the system
    geom = fdf.read_geometry(output=False)  # Load geometry from fdf file
    C_list = (geom.atoms.Z == 6).nonzero()[0]  # List of carbon atoms
    O_list = (geom.atoms.Z == 8).nonzero()[0]  # List of oxygen atoms
    C_O_list = np.concatenate((C_list, O_list))  # Add lists
    Subbed = geom.sub(C_O_list)  # Keep only C or O atoms in geometry
    Subbed.reduce()  # Remove "empty" atom places
    geom = Subbed
    cellsize = geom.xyz.shape[0]  # Determine number of atoms
    geom = geom.tile(nx + 4 * contactrep, 1)  # Repeat structure
    geom = geom.rotate(270, v=[0, 0, 1], origo=geom.center(
        what='xyz'))  # Rotate structure
    xyz = geom.xyz
    xyz = np.round(xyz, decimals=1)  # Round off coordinates

    # Create geometry based on coordinates
    geom = si.Geometry(xyz, [Atom('C')], [2.46, 4.26, 0])
    # geom = geom.sort(axes=(2,1,0))
    LatticeVectors = fdf.get('LatticeVectors')  # Gather lattice vectors
    UY = np.fromstring(LatticeVectors[0], dtype=float, sep=' ')[0]  # ... in Y
    UX = np.fromstring(LatticeVectors[1], dtype=float, sep=' ')[1]  # ... in X
    print('Unit Cell x: {}'.format(UX))
    print('Unit Cell y: {}'.format(UY))
    xyz = geom.xyz
    dgeom = geom
    plt.scatter(xyz[:, 0], xyz[:, 1])
    plt.axis('equal')
    for i in range(xyz[:, 0].shape[0]):
        s = i
        xy = (xyz[i, 0], xyz[i, 1])
        plt.annotate(s, xy)
    plt.grid(b=True, which='both', axis='both')
    plt.show()
    return xyz, UX, UY, filename, dgeom, cellsize


def NPGElectrode(xyz, dgeom, cellsize, nx):
    csize = cellsize * nx
    esize = cellsize
    device = dgeom
    # device = dgeom.sort(axes=(2, 1, 0))
    xyz = device.xyz
    print(xyz.shape)
    RestL = np.arange(0, esize)
    L = np.arange(esize, esize * 2)
    C = np.arange(esize * 2, esize * 2 + csize)
    R = np.arange(esize * 2 + csize, esize * 2 + csize + esize)
    RestR = np.arange(esize * 2 + csize + esize, xyz.shape[0])
    print(RestL)
    print(L)
    print(C)
    print(R)
    print(RestR)
    Lxyz = xyz[L]
    Cxyz = xyz[C]
    Rxyz = xyz[R]
    RmArray = np.append(L, C).astype(int)
    RmArray = np.append(RmArray, R).astype(int)
    Restxyz = np.delete(xyz, RmArray, 0)

    plt.scatter(Lxyz[:, 0], Lxyz[:, 1], c='red', label='L')
    plt.scatter(Cxyz[:, 0], Cxyz[:, 1], c='orange', label='C')
    plt.scatter(Rxyz[:, 0], Rxyz[:, 1], c='blue', label='R')
    plt.scatter(Restxyz[:, 0], Restxyz[:, 1], c='k')
    plt.legend()
    plt.axis('equal')
    for i in range(xyz[:, 0].shape[0]):
        s = i
        xy = (xyz[i, 0], xyz[i, 1])
        plt.annotate(s, xy)
    plt.grid(b=True, which='both', axis='both')
    plt.show()
    return RestL, L, R, C, RestR
