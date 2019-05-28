import numpy as np
import sisl as si
import matplotlib.pyplot as plt 
import os.path

plt.subplot(121)

###########################
# Input TSHS file
H_dft = si.get_sile('../../single_point15_51/RUN.fdf').read_hamiltonian()
print(H_dft)

# Atoms whose orbitals will be extracted from DFT
C_list = (H_dft.atoms.Z == 6).nonzero()[0]
O_list = (H_dft.atoms.Z == 8).nonzero()[0]
# Here we consider both C and O atoms
C_O_list = np.concatenate((C_list, O_list)) ; #C_O_list = np.sort(C_O_list)
#print(C_list)
#print(O_list)
#print(C_O_list)

#########################################
# Plot bandstructure from DFT
band_dft = si.BandStructure(H_dft, [[0.5, 0, 0], [0, 0, 0], [0, 0.5, 0]],
                        100, [r'$X$', r'$\Gamma$', r'$Y$'])
lk, kt, kl = band_dft.lineark(True)

if(os.path.isfile('old_eigh.dat')):
    tbl = si.io.table.tableSile('old_eigh.dat', 'r')
    bs_dft = tbl.read_data()
else:
    # Calculate all eigenvalues
    bs_dft = band_dft.eigh(spin=0, eta=True)
    tbl = si.io.table.tableSile('old_eigh.dat', 'w')
    tbl.write_data(bs_dft)

for bk in bs_dft.T:
    plt.plot(lk, bk, 'r', label=r'SZP')

plt.legend(loc=0, frameon=False, fontsize=11)
plt.xticks(kt, kl)
plt.axvline(kt[1], c='k', ls=':', lw=0.5)
plt.axhline(0, c='k', ls='--', lw=0.5)
plt.xlim(0, lk[-1])
plt.ylim([-1.3, 1.8])
plt.ylabel(r'$E-E_{\rm F}\, (e{\rm V})$')
plt.title(r'$w_A = w_B$ (DFT)')



plt.subplot(122)

plotTB = True
##############
if plotTB:
    # Plot bandstructure from parametrized TB
    # Purging C orbitals --> only taking Pz
    H_h_o_cpz = H_dft.sub_orbital(H_dft.geometry.atoms[C_list[0]], orbital=[2])
    # Purging O orbitals --> only taking Pz
    H_h_opz_cpz = H_h_o_cpz.sub_orbital(H_h_o_cpz.geometry.atoms[O_list[0]], orbital=[2])
    # Removing unnecessary H atoms
    H_TB = H_h_opz_cpz.sub(C_O_list); H_TB.reduce()
    print(H_TB)
    band = si.BandStructure(H_TB, [[0.5, 0, 0], [0, 0, 0], [0, 0.5, 0]],
                            100, [r'$X$', r'$\Gamma$', r'$Y$'])
    
    if(os.path.isfile('old_eigh_pz.dat')):
        tbl = si.io.table.tableSile('old_eigh_pz.dat', 'r')
        bs = tbl.read_data()
    else:
        # Calculate all eigenvalues
        bs = band.eigh(spin=0, eta=True)
        tbl = si.io.table.tableSile('old_eigh_pz.dat', 'w')
        tbl.write_data(bs)

    for bk in bs.T:
        plt.plot(lk, bk, 'r--', label=r'$p_z$TB')

    #plt.legend(loc='center left', frameon=False, bbox_to_anchor=(0.48,0.15), fontsize=11)
    plt.legend(loc=0, frameon=False, fontsize=11)
##############

plt.xticks(kt, kl)
plt.axvline(kt[1], c='k', ls=':', lw=0.5)
plt.axhline(0, c='k', ls='--', lw=0.5)
plt.xlim(0, lk[-1])
plt.ylim([-1.3, 1.8])
plt.ylabel(r'$E-E_{\rm F}\, (e{\rm V})$')
plt.title(r'$w_A = w_B$ (TB_c&o)')

plt.tight_layout()
plt.savefig('bands.png')

asdfadsf


plt.subplot(122)
###########################
# Input TSHS file
tshs_0 = si.get_sile('/work3/isrov/GNRs_Gaetano/alternating_widths/para_15/b_opt/RUN.fdf').read_hamiltonian()
print(tshs_0)

# Atoms whose orbitals will be extracted from DFT
C_list = (tshs_0.atoms.Z == 6).nonzero()[0]

#########################################
# Plot bandstructure from DFT
band_dft = si.BandStructure(tshs_0, [[0.5, 0, 0], [0, 0, 0], [0, 0.5, 0]],
                        100, [r'$X$', r'$\Gamma$', r'$Y$'])
lk, kt, kl = band_dft.lineark(True)

if(os.path.isfile('nw_eigh.dat')):
    tbl = si.io.table.tableSile('nw_eigh.dat', 'r')
    bs_dft = tbl.read_data()
else:
    # Calculate all eigenvalues
    bs_dft = band_dft.eigh(spin=0, eta=True)
    tbl = si.io.table.tableSile('nw_eigh.dat', 'w')
    tbl.write_data(bs_dft)

for bk in bs_dft.T:
    plt.plot(lk, bk, 'b', label=r'SZP')

plotTB = True
##############
if plotTB:
    # Plot bandstructure from parametrized TB
    H = tshs_0.sub(C_list); H.reduce()
    H = H.sub(H.atoms[0], orb_index=[2])
    print(H)
    band = si.BandStructure(H, [[0.5, 0, 0], [0, 0, 0], [0, 0.5, 0]],
                            100, [r'$X$', r'$\Gamma$', r'$Y$'])
    
    if(os.path.isfile('nw_eigh_pz.dat')):
        tbl = si.io.table.tableSile('nw_eigh_pz.dat', 'r')
        bs = tbl.read_data()
    else:
        # Calculate all eigenvalues
        bs = band.eigh(spin=0, eta=True)
        tbl = si.io.table.tableSile('nw_eigh_pz.dat', 'w')
        tbl.write_data(bs)

    for bk in bs.T:
        plt.plot(lk, bk, 'b--', label=r'$p_z$TB')

    #plt.legend(loc='center left', frameon=False, bbox_to_anchor=(0.48,0.15), fontsize=11)
    plt.legend(loc=0, frameon=False, fontsize=11)
##############

plt.xticks(kt, kl)
plt.axvline(kt[1], c='k', ls=':', lw=0.5)
plt.axhline(0, c='k', ls='--', lw=0.5)
plt.xlim(0, lk[-1])
plt.ylim([-0.2, 1.8])
#plt.ylabel(r'$E-E_{\rm F}\, (e{\rm V})')
plt.tick_params(labelleft=False)
plt.title(r'$w_A < w_B = 1.5\,{\rm nm}$ (para)')




plt.tight_layout()
plt.savefig('bands.pdf')

