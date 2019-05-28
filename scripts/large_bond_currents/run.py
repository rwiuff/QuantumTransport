import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylab import *
import sisl as si
import numpy as np

###########################
# Input TSHS file
H_dft = si.get_sile('../single_point15_51/RUN.fdf').read_hamiltonian()

# Atoms whose orbitals will be extracted from DFT
C_list = (H_dft.atoms.Z == 6).nonzero()[0]
O_list = (H_dft.atoms.Z == 8).nonzero()[0]
#print(C_list)
# Here we consider both C and O atoms
C_O_list = np.concatenate((C_list, O_list)) ; #C_O_list = np.sort(C_O_list)
#He = H_dft.sub(C_list); He.reduce()

# Purging C orbitals --> only taking Pz
H_h_o_cpz = H_dft.sub_orbital(H_dft.geometry.atoms[C_list[0]], orbital=[2]) 
# Purging O orbitals --> only taking Pz
H_h_opz_cpz = H_h_o_cpz.sub_orbital(H_h_o_cpz.geometry.atoms[O_list[0]], orbital=[2])
# Removing unnecessary H atoms
H_TB = H_h_opz_cpz.sub(C_O_list); H_TB.reduce()

print(H_TB)

# Save electrode
H_TB.write('He.nc')
H_TB.geom.write('He.xyz')

# Tile it to save device
H = H_TB.repeat(10,0).tile(100,1)
#H.set_nsc([1,1,1])
H.geom.write('inside_HS_DEV.xyz')
H.write('inside_HS_DEV.nc')

# Create CAP
from lib_dft2tb import CAP
dH_CAP = CAP(H.geom, 'left+right', dz_CAP=50, write_xyz=True, zaxis=2)
dH_CAP_sile = si.get_sile('CAP.delta.nc', 'w')
dH_CAP_sile.write_delta(dH_CAP)

############################
############################
############################
# Define atoms where Gamma exists
a_tip = [3704]
# Check
tmp = H.geom.copy(); tmp.atom[a_tip] = si.Atom(16, R=[1.44]); tmp.write('a_tip.xyz')
# Setup TBTGF for tip injection
# It is vital that you also write an electrode Hamiltonian,
# i.e. the Hamiltonian object passed as "Htip", has to be written:
# Build HS for tip
HStip = H.sub(a_tip)

HStip.write('TBTGF_H.nc')
# Now generate a TBTGF file
GF = si.io.TBTGFSileTBtrans('Gamma.TBTGF')

# Below we load whatever TBT file from which we can take the energies and kpoints
# In the next TBTrans run we MUST use the same energy points and kpoints we load here
#TBT = si.get_sile("elist.TBT.nc")
# Energy contour
eta = 0.001
#eta = TBT.eta(0)
print('Eta read from external TBT: {}'.format(eta))
# Energy contour
ne = 7
#Ens = np.array([0.3, 0.8, 0.9, 1.0])
Ens = np.linspace(-0.6, 0.0, ne)
tbl = si.io.table.tableSile('contour.IN', 'w')
tbl.write_data(Ens, np.zeros(ne), np.ones(ne), fmt='.8f')

E = Ens + 1j*eta
# Brillouin zone (should it be one kpoint or should we read them from TBT?)
## You should definitely read k-points from TBT, otherwise TBT would crash,
## but then you would know! ;)
BZ = si.BrillouinZone(HStip); BZ._k = np.array([[0.,0.,0.]]); BZ._wk = np.array([1.0])
#BZ = si.BrillouinZone(Htip); BZ._k = TBT.kpt; BZ._w = TBT.wkpt
GF.write_header(BZ, E, obj=HStip) # Htip HAS to be a Hamiltonian object, E has to be complex (WITH eta)

###########################
# Define Gamma
Gamma = np.eye(HStip.shape[0])  # onsite Gamma of 1 eV
print('Injecting using Gamma = {} eV'.format(Gamma))

###########################
# Writing an energy independent Gamma as TBTGF
print('Computing and storing Gamma in TBTGF format...')
ZERO = np.zeros(Gamma.shape, dtype=np.complex128)
for i, (ispin, HS4GF, _, e) in enumerate(GF):
    # One must write the quantity S*e - H - SE
    if HS4GF:
        GF.write_hamiltonian(ZERO, ZERO)
    GF.write_self_energy(1j * Gamma)
