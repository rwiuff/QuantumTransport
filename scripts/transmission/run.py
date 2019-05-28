import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylab import *
import sisl as si
import numpy as np

###########################
# Input TSHS file
H_dft = si.get_sile('../../single_point15_51/RUN.fdf').read_hamiltonian()

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

#print(H_TB)

# Save electrode
H_TB.write('He.nc')
H_TB.geom.write('He.xyz')

# Tile it to save device
H_TB_dev = H_TB.tile(3,0)
#H.set_nsc([1,1,1])
H_TB_dev.geom.write('H.xyz')
H_TB_dev.write('H.nc')

print(H_TB_dev)
