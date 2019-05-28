from __future__ import print_function, division
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import scipy as sp
from operator import truediv
import math, time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import groupby
import sisl as si
from numbers import Integral

# I don't know why, but the lines below were 
# fucking up my routine "makeTB_FrameOutside", on the "contruct" command
#try:
#    from itertools import izip as zip
#except:
#    pass

def dagger(M):
    return np.conjugate(np.transpose(M))

def displaySparse(m, filename, dpi=300):
    if not isinstance(m, sp.sparse.coo_matrix):
        m = sp.sparse.coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=10)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename, facecolor='black', edgecolor='black', dpi=dpi)
    return ax

def get_potential(TSHS, iio, atoms):
    """
    iio:    index (0-based) of orbital in basis set (i.e., pz in SZP: iio = 2)
    """
    orbs = TSHS.a2o(atoms)+iio
    on = TSHS.Hk(dtype=np.float64, format='array')[orbs, orbs]
    return on

def check_Dirac(ts, mp, displacement=[0,0,0]):
    mp = si.MonkhorstPack(ts, mp, displacement=displacement)
    print('Check that Dirac is in here: ')
    print(mp.k)
    print('Check that this is in *.KP file : {}'.format(mp.tocartesian([0., 1./3, 0]) * si.unit.siesta.unit_convert('Bohr', 'Ang')))
    i_dirac = (np.logical_and(mp.k[:,1] == 1./3, mp.k[:,0] == 0.)).nonzero()[0]
    if len(i_dirac) != 1:
        print('Dirac point is not in the grid')
        exit(1)
    else: 
        print('Dirac point is at kindex: {}'.format(i_dirac[0]))

def get_Dirac(hs, mp, displacement=[0,0,0]):
    #check_Dirac(hs.geom, mp, displacement)
    ens_dirac = hs.eigh(k=[0., 1./3, 0])
    i_dirac = hs.na * 2 - 1
    return np.average(ens_dirac[i_dirac:i_dirac+2])
    
def plot_PotDiff(TSHS, TSHS_0, ia, axis, iio, o_dev, o_inner):  # include option for frame!
    on, yy, atoms = get_potential(TSHS, ia, axis, iio)
    on0 = get_potential(TSHS_0, ia, axis, iio)[0]
    on0 = np.array([np.mean(on0)]*len(on))
    # Check
    print('y (Ang)\t\tPot (eV)\tPot0 (eV)\tPot-Pot0 (eV)')
    a_dev = TSHS.o2a(o_dev, unique=True)
    a_inner = TSHS.o2a(o_inner, unique=True)
    for iia, y, o, o0 in zip(atoms, yy, on, on0):
        if iia in a_inner:
            print('{:7.4f}\t\t{:7.4f}\t\t{:7.4f}\t\t{:7.4f}\t\t(inner)'.format(y,o,o0,o-o0))    
        else:
            print('{:7.4f}\t\t{:7.4f}\t\t{:7.4f}\t\t{:7.4f}'.format(y,o,o0,o-o0))    
    # Subtract pristine potential
    PotDiff = on-on0
    # Write to file
    with open('PotDiff.dat', 'w') as pf:
        for yc, pd in zip(yy, PotDiff):
            pf.write('{}\t\t{}\n'.format(yc, pd))
    # Plot
    figure()
    plot(yy, PotDiff, 'b')
    md, Md = np.amin(TSHS.xyz[a_dev, axis]), np.amax(TSHS.xyz[a_dev, axis]) 
    axvline(md, color='k', linestyle='dashed', linewidth=2)
    axvline(Md, color='k', linestyle='dashed', linewidth=2)
    tmp_dev = TSHS.geom.sub(a_dev); tmp_inner = tmp_dev.sub(a_inner)
    mi, Mi = np.amin(tmp_inner.xyz[a_inner, axis]), np.amax(tmp_inner.xyz[a_inner, axis])
    axvspan(mi, Mi, alpha=0.3, facecolor='blue', edgecolor='none')
    ylabel(r'$H_{p_z}-H^0_{p_z}\, (e{\rm V})$', fontsize=20)
    xlabel(r'$y\, (\AA)$', fontsize=20)
    xlim(0, TSHS.cell[axis, axis])
    #xlim(TSHS.center(what='cell')[1], TSHS.cell[1,1])
    legend(loc=0); savefig('PotDiff.pdf', bbox_inches='tight')

def get_potential_profile(TSHS, ia, axis, iio):
    """
    ia:     atom crossed by the line
    axis:   direction of the line
    iio:    index (0-based) of orbital in basis set (i.e., pz in SZP: iio = 2)
    """
    # Find atoms in line passing by center of
    xyz0, xyz = TSHS.xyz[ia, axis%1], TSHS.xyz[:, axis%1]
    atoms = np.where(np.logical_and(xyz0-1.43 < xyz, xyz < xyz0+1.43))[0]
    v = TSHS.geom.copy(); v.atom[atoms] = si.Atom(8, R=[1.43]); v.write('checkPot.xyz')
    orbs = TSHS.a2o(atoms)+iio
    on = TSHS.Hk(dtype=np.float64, format='array')[orbs, orbs]
    ylist = TSHS.xyz[atoms, axis]
    idxs = np.argsort(ylist)
    on, ylist = on[idxs], ylist[idxs]
    return on, ylist, atoms

def xyz2polar(tbt, origin=0):
    na = tbt.na
    # radii from origin
    if isinstance(origin, Integral):
        origin = tbt.xyz[origin]
    _, r = tbt.geom.close_sc(origin, R=np.inf, ret_rij=True)

    # angles from origin
    transl = tbt.geom.translate(-origin)
    y = transl.xyz[:,1]
    i_ypos = np.where(y >= 0)[0]
    i_yneg = np.setdiff1d(np.arange(na), i_ypos)
    t = np.zeros(na)
    t[i_ypos] = transl.angle(i_ypos, dir=(1., 0, 0), rad=True)
    t[i_yneg] = transl.angle(i_yneg, dir=(-1., 0, 0), rad=True) +np.pi
    return r, t

def radial_T_from_bc(tbt, elec, E=None, kavg=True, 
    origin=0, thetamin=0., thetamax=2*np.pi, ntheta=360, 
    Rmin=5., Rmax=999999999, dr=40., 
    input=None, save='radial_T_from_bc.txt', saveinput='rt.txt'):
    
    if E:
        Eidx = tbt.Eindex(E)
        en = tbt.E[Eidx]
    else:
        en = tbt.E[0]
    print('Using E = {} eV'.format(en))

    na = tbt.na
    if isinstance(origin, Integral):
        origin = tbt.xyz[origin]

    # (x, y) ----> (r, t)
    if input:
        r, t = np.loadtxt(input, delimiter='\t', usecols=(1, 2), unpack=True, skiprows=1)
    else:
        r, t = xyz2polar(tbt, origin=origin)
        f = open(saveinput, 'w')
        f.write('ia\tr (Angstrom)\tangle (radians; center {})\n'.format(origin))
        for ia, rr, tt in zip(np.arange(na), r, t):
            f.write('{}\t{}\t{}\n'.format(ia, rr, tt))
        f.close()
    print('(x,y) ---> (r,t):  DONE')
    
    # theta bins
    thetas = np.linspace(thetamin, thetamax, ntheta, endpoint=False)
    dtheta = thetas[1]-thetas[0]
    print(len(thetas), dtheta, thetas)
    # Digitize t into thetas
    inds = np.digitize(t, thetas) -1  # First bin is associated to 0.0 rad
    print('Digitize theta:  DONE')

    # radii[i] is the radius of the interface between 2 crowns centered at the position of the tip
    newRmax = np.amin(np.absolute(np.array([origin[0], origin[1], 
                            (origin-tbt.cell[0]-tbt.cell[1])[0], (origin-tbt.cell[0]-tbt.cell[1])[1]])))
    radii = np.arange(np.amax([Rmin, dr]), np.amin([Rmax, newRmax])+2*dr, dr)
    nradii = len(radii)
    print(nradii, dr, radii)

    # indices of atom within the various shells
    # atoms in list ishell[i] belong to [radii[i], radii[i+1]]    
    ishell = tbt.geom.close_sc(origin, R=radii, idx=tbt.a_dev)
    print('Close: DONE')

    # Read bond-current
    bc = tbt.bond_current(0, en, kavg=kavg, only='all', uc=True)
    print('bc: DONE')

    Tavg = np.zeros(ntheta*nradii)
    thetas_toplot = Tavg.copy()
    radii_toplot = Tavg.copy()
    j=0
    for id in np.arange(ntheta):  # Loop over unique angles
        print('  Doing theta #{} of {} ({} rad)'.format(id+1, ntheta, thetas[id]))
        idx_intheta = np.where(inds == id)[0]  # find indices of atoms whose t is in sector theta
        for id_r in np.arange(1,nradii-1):   # Loop over unique radii
            print('    Doing radius #{} of {} ({} Ang)'.format(id_r, nradii, radii[id_r]))
            idx_1_indr = ishell[id_r]     # Indices of atoms within internal shell
            mask = np.in1d(idx_1_indr, idx_intheta)  
            idx_1 = idx_1_indr[mask]  # Indices of atoms in internal shell AND sector theta
            idx_2 = ishell[id_r+1]   # # Indices of atoms within external shell
            Tavg[j] = bc[idx_1.reshape(-1, 1), idx_2.reshape(1, -1)].sum()
            thetas_toplot[j] = thetas[id]
            radii_toplot[j] = radii[id_r]
            #print('    ({} Ang, {} rad) --> {}'.format(radii_toplot[j], thetas_toplot[j], Tavg[j]))
            j+=1 
    
    # Write
    f = open(save, 'w')
    f.write('center {}\n'.format(origin))
    f.write('radius (Ang), \t theta (rad), \tT from radial bond current\n')
    for rr, theta, ttt in zip(radii_toplot, thetas_toplot, Tavg):
        f.write('{}\t{}\t{}\n'.format(rr, theta, ttt))
    f.close()

    return radii_toplot, thetas_toplot, Tavg 

def atom_current_radial(tbt, elec, E, kavg=True, activity=True, 
    origin=0, thetamin=0., thetamax=2*np.pi, ntheta=360, 
    Rmin=5., Rmax=999999999, dr=40., 
    input=None, save='atom_current_radial.txt', saveinput='ac_input.txt'):
    
    if E:
        Eidx = tbt.Eindex(E)
        en = tbt.E[Eidx]
    else:
        en = tbt.E[0]
    print('Using E = {} eV'.format(en))

    na = tbt.na
    if isinstance(origin, Integral):
        origin = tbt.xyz[origin]
    # (x, y) ----> (r, t)
    if input:
        r, t, ac = np.loadtxt(input, delimiter='\t', usecols=(1, 2, 3), unpack=True, skiprows=1)
    else:
        r, t = xyz2polar(tbt, origin=origin)
        print('start extraction of atom_current...')
        ac = tbt.atom_current(elec, E, kavg, activity)
        print('...end extraction of atom_current')
        f = open(saveinput, 'w')
        f.write('ia\tr (Ang)\tangle (rad; center {})\tatom current\n'.format(origin))
        for ia, rr, tt, a in zip(np.arange(na), r, t, ac):
            f.write('{}\t{}\t{}\t{}\n'.format(ia, rr, tt, a))
        f.close()
    print('(x,y) ---> (r,t):  DONE')

    # theta bins
    thetas = np.linspace(thetamin, thetamax, ntheta, endpoint=False)
    dtheta = thetas[1]-thetas[0]
    print('Thetas entries:')
    print(len(thetas), dtheta, thetas)
    # Digitize t into thetas
    inds = np.digitize(t, thetas) -1  # First bin is associated to 0.0 rad
    print('Digitize theta:  DONE')

    # radii[i] is the radius of the interface between 2 crowns centered at the position of the tip
    newRmax = np.amin(np.absolute(np.array([origin[0], origin[1], 
                            (origin-tbt.cell[0]-tbt.cell[1])[0], (origin-tbt.cell[0]-tbt.cell[1])[1]])))    
    radii = np.arange(np.amax([Rmin, dr]), np.amin([Rmax, newRmax])+dr, dr)
    nradii = len(radii)
    print('Radii entries:')
    print(nradii, dr, radii)

    # indices of atom within the various shells
    # atoms in list ishell[i] belong to [radii[i], radii[i+1]]    
    #ishell = tbt.geom.close_sc(origin, R=radii, idx=tbt.a_dev)
    #print('Close: DONE')

    current_r = np.zeros((nradii, ntheta))
    for ir, rr in enumerate(radii):   # Loop over unique radii
        current_t = np.zeros(ntheta)
        counts_t = current_t.copy()
        inR = np.where(r < rr)[0]
        for id, a in zip(inds[inR], ac[inR]):
            current_t[id] += a
            counts_t[id] += 1
        current_r[ir, :] = np.divide(current_t, counts_t)

    # Write
    np.savetxt(save, np.transpose(np.vstack([thetas, current_r])), delimiter='\t', 
        newline='\n', comments='', header=', '.join(str(e) for e in radii))

    return radii, thetas, current_r

def plot_LDOS(geom, LDOS, figname='figure.png', 
    vmin=None, vmax=None):
    
    import matplotlib.collections as collections
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x, y = geom.xyz[:,0], geom.xyz[:,1]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    vmin, vmax = vmin, vmax
    if vmin is None:
        vmin = np.min(LDOS)
    if vmax is None:
        vmax = np.max(LDOS)
    colors = LDOS
    area = 15
    image = ax.scatter(x, y, c=colors, s=area, marker='o', edgecolors='None', cmap='viridis')
    image.set_clim(vmin, vmax)
    image.set_array(LDOS)

    ax.autoscale()
    ax.margins(0.1)
    plt.xlabel('$x (\AA)$')
    plt.ylabel('$y (\AA)$')
    plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axcb = plt.colorbar(image, cax=cax, format='%1.2f', ticks=[vmin, vmax])

    plt.savefig(figname, bbox_inches='tight', transparent=True, dpi=300)
    print('Successfully plotted to "{}"'.format(figname))


def CAP(geometry, side, dz_CAP=30, write_xyz=True, zaxis=2):
    # Determine orientation
    if zaxis == 2:
        xaxis, yaxis = 0, 1
    elif zaxis == 0:
        xaxis, yaxis = 1, 2
    elif zaxis == 1:
        xaxis, yaxis = 0, 2
    # Natural units (see "http://superstringtheory.com/unitsa.html")
    hbar = 1 
    m = 0.511e6 # eV
    c = 2.62
    print('\nSetting up CAP regions: {}'.format(side))
    print('Width of absorbing walls = {} Angstrom'.format(dz_CAP))
    Wmax = 100
    dH_CAP = si.Hamiltonian(geometry, dtype='complex128')
    CAP_list = []
    ### EDGES
    if 'right' in side:
        print('Setting at right')
        z, y = geometry.xyz[:, xaxis], geometry.xyz[:, yaxis]
        z2 = np.max(geometry.xyz[:, xaxis]) + 1.
        z1 = z2 - dz_CAP
        idx = np.where(np.logical_and(z1 <= z, z < z2))[0]
        fz = (4/(c**2)) * ((dz_CAP/(z2-2*z1+z[idx]))**2 + (dz_CAP/(z2-z[idx]))**2 - 2 )
        Wz = ((hbar**2)/(2*m)) * (2*np.pi/(dz_CAP/2000))**2 * fz
        orbs = dH_CAP.geom.a2o(idx) # if you have just 1 orb per atom, then orb = ia
        for orb,wz in zip(orbs, Wz):
            dH_CAP[orb, orb] = complex(0, -wz)
        CAP_list.append(idx)
        #print(list2range_TBTblock(idx))

    if 'left' in side:
        print('Setting at left')
        z, y = geometry.xyz[:, xaxis], geometry.xyz[:, yaxis]
        z2 = np.min(geometry.xyz[:, xaxis]) - 1.
        z1 = z2 + dz_CAP
        idx = np.where(np.logical_and(z2 < z, z <= z1))[0]
        fz = (4/(c**2)) * ((dz_CAP/(z2-2*z1+z[idx]))**2 + (dz_CAP/(z2-z[idx]))**2 - 2 )
        Wz = ((hbar**2)/(2*m)) * (2*np.pi/(dz_CAP/2000))**2 * fz
        orbs = dH_CAP.geom.a2o(idx) # if you have just 1 orb per atom, then orb = ia
        for orb,wz in zip(orbs, Wz):
            dH_CAP[orb, orb] = complex(0, -wz)
        CAP_list.append(idx)
        #print(list2range_TBTblock(idx))

    if 'top' in side:
        print('Setting at top')
        z, y = geometry.xyz[:, xaxis], geometry.xyz[:, yaxis]
        y2 = np.max(geometry.xyz[:, yaxis]) + 1.
        y1 = y2 - dz_CAP
        idx = np.where(np.logical_and(y1 <= y, y < y2))[0]
        fz = (4/(c**2)) * ( (dz_CAP/(y2-2*y1+y[idx]))**2 + (dz_CAP/(y2-y[idx]))**2 - 2 )
        Wz = ((hbar**2)/(2*m)) * (2*np.pi/(dz_CAP/2000))**2 * fz
        orbs = dH_CAP.geom.a2o(idx) # if you have just 1 orb per atom, then orb = ia
        for orb,wz in zip(orbs, Wz):
            dH_CAP[orb, orb] = complex(0, -wz)
        CAP_list.append(idx)
        #print(list2range_TBTblock(idx))

    if 'bottom' in side:
        print('Setting at bottom')
        z, y = geometry.xyz[:, xaxis], geometry.xyz[:, yaxis]
        y2 = np.min(geometry.xyz[:, yaxis]) - 1.
        y1 = y2 + dz_CAP
        idx = np.where(np.logical_and(y2 < y, y <= y1))[0]
        fz = (4/(c**2)) * ( (dz_CAP/(y2-2*y1+y[idx]))**2 + (dz_CAP/(y2-y[idx]))**2 - 2 )
        Wz = ((hbar**2)/(2*m)) * (2*np.pi/(dz_CAP/2000))**2 * fz
        orbs = dH_CAP.geom.a2o(idx) # if you have just 1 orb per atom, then orb = ia
        for orb,wz in zip(orbs, Wz):
            dH_CAP[orb, orb] = complex(0, -wz)
        CAP_list.append(idx)
        #print(list2range_TBTblock(idx))

    CAP_list = np.concatenate(CAP_list).ravel().tolist()
    if write_xyz:
        # visualize CAP regions
        visualize = geometry.copy()
        visualize.atom[CAP_list] = si.Atom(8, R=[1.44])
        visualize.write('CAP.xyz')

    return dH_CAP

def read_fullTSHS(HSfilename, geomFDFfilename):
    """ Read Hamiltonian and Geometry objects
    and update Atoms properties of 'TSHS' from 'FDF' """
    if isinstance(HSfilename, str):
        HSfile = si.get_sile(HSfilename).read_hamiltonian()
    else:
        HSfile = HSfilename.copy()
    if isinstance(geomFDFfilename, str):
        geomFDF = si.get_sile(geomFDFfilename).read_geometry(True)
    else:
        geomFDF = geomFDFfilename.copy()
    # Update species
    for ia, (a, afdf) in enumerate(zip(HSfile.atom, geomFDF.atom)):
        A = si.Atom(afdf.Z, a.orbital, afdf.mass, afdf.tag)
        HSfile.atom[ia] = A
    HSfile.reduce()
    return HSfile

def T_from_bc(tbt, elec, idx_1, idx_2, E=None, kavg=True, write_xyz=None):
    if write_xyz:   # visualize regions
        visualize = tbt.geom.copy()
        visualize.atom[idx_1] = si.Atom(8, R=[1.44])
        visualize.atom[idx_2] = si.Atom(9, R=[1.44])
        visualize.write('{}.xyz'.format(write_xyz))
    if E:
        Eidx = tbt.Eindex(E)
        energies = np.array([tbt.E[Eidx]])
    else:
        energies = tbt.E
    T = np.zeros(len(energies))
    for ie,e in enumerate(energies):
        print('Doing E # {} of {}  ({} eV)'.format(ie+1, len(energies), e)) 
        bc = tbt.bond_current(elec, e, kavg=kavg, only='all', uc=True)
        T[ie] += bc[idx_1.reshape(-1, 1), idx_2.reshape(1, -1)].sum()
    return T

def T_from_bc_from_orbital(tbt, elec, o_idx, idx_1, idx_2, E=None, 
    kavg=True, write_xyz=None):
    if write_xyz:   # visualize regions
        visualize = tbt.geom.copy()
        visualize.atom[idx_1] = si.Atom(8, R=[1.44])
        visualize.atom[idx_2] = si.Atom(9, R=[1.44])
        visualize.write('{}.xyz'.format(write_xyz))
    if E:
        Eidx = tbt.Eindex(E)
        energies = np.array([tbt.E[Eidx]])
    else:
        energies = tbt.E
    T = np.zeros(len(energies))
    for ie,e in enumerate(energies):
        print('Doing E # {} of {}  ({} eV)'.format(ie+1, len(energies), e)) 
        Jij = tbt.orbital_current(elec, e, kavg=kavg)
        orbs_1 = tbt.geom.a2o(idx_1) + o_idx
        orbs_2 = tbt.geom.a2o(idx_2) + o_idx
        T[ie] = Jij[orbs_1.reshape(-1, 1), orbs_2.reshape(1, -1)].sum()
        #bc = tbt.bond_current(elec, e, kavg=kavg, only='all', uc=True)
    return T

def list2range_TBTblock(lst):
    """ Convert a list of elements into a string of ranges

    Examples
    --------
    >>> list2range([2, 4, 5, 6])
    2, 4-6
    >>> list2range([2, 4, 5, 6, 8, 9])
    2, 4-6, 8-9
    """
    lst = [el+1 for el in lst]
    lst.sort()
    # Create positions
    pos = [j - i for i, j in enumerate(lst)]
    t = 0
    rng = ''
    for _, els in groupby(pos):
        ln = len(list(els))
        el = lst[t]
        if t > 0:
            rng += '\n'
        t += ln
        if ln == 1:
            rng += '  atom ['+str(el)+']'
        else:
            rng += '  atom [{} -- {}]'.format(el, el+ln-1)
    return rng

def create_kpath(Nk):
    G2K = (0.4444444444444444 + 0.1111111111111111) ** 0.5
    K2M = ((0.6666666666666666 - 0.5) ** 2 + (0.3333333333333333 - 0.5) ** 2) ** 0.5
    M2G = (0.25 + 0.25) ** 0.5
    Kdist = G2K + K2M + M2G
    NG2K = int(Nk / Kdist * G2K)
    NK2M = int(Nk / Kdist * K2M)
    NM2G = int(Nk / Kdist * M2G)

    def from_to(N, f, t):
        full = np.empty([N, 3])
        ls = np.linspace(0, 1, N, endpoint=False)
        for i in range(3):
            full[:, i] = f[i] + (t[i] - f[i]) * ls

        return full

    kG2K = from_to(NG2K, [0.0, 0.0, 0.0], [0.6666666666666666, 0.3333333333333333, 0])
    kK2M = from_to(NK2M, [0.6666666666666666, 0.3333333333333333, 0], [0.5, 0.5, 0.0])
    kM2G = from_to(NM2G, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
    xtick = [0, NG2K - 1, NG2K + NK2M - 1, NG2K + NK2M + NM2G - 1]
    label = ['G', 'K', 'M', 'G']
    return ([xtick, label], np.vstack((kG2K, kK2M, kM2G)))

def plot_bandstructure(H, Nk, ymin=None, ymax=None, style='.', 
    color='k', label=None):
    if type(H) is str:
        H = si.get_sile(H).read_hamiltonian()
    ticks, k = create_kpath(Nk)
    eigs = np.empty([len(k), H.no], np.float64)
    for ik, k in enumerate(k):
        print('{} / {}'.format(ik+1, Nk), end='\r')
        eigs[ik, :] = H.eigh(k=k, eigvals_only=True)

    ax = plt.gca()
    for n in range(H.no):
        print('{} / {}'.format(n+1, H.no), end='\r')
        ax.plot(eigs[:, n], style, color=color, label=label if n == 0 else "")

    ax.xaxis.set_ticks(ticks[0])
    ax.set_xticklabels(ticks[1])
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(ymin, ymax)

    for tick in ticks[0]:
        ax.plot([tick, tick], [ymin, ymax], 'k')
    return ax

def list2colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))



def get_dft_param(tshs, ia, iio, jjo, unique=False, onlynnz=False, idx=None):
    """ Read Hamiltonian and get coupling constants between 
    'iio'-th orbital of atom 'ia' and 'jjo'-th orbital of all other atoms
    """
    # Read Hamiltonian
    if isinstance(tshs, str):
        tshs = si.get_sile(tshs).read_hamiltonian()
    HS = tshs.copy()
    # Index of iio-th orbital of ia-th atom
    io = HS.a2o(ia) + iio

    # Coupling elements (all orbitals)
    edges = HS.edges(orbital=io, exclude=-1)
    # Remove non-jjo connections
    # convert to atoms (only unique values)
    edges = HS.o2a(edges, unique=True)
    if idx is not None:
        mask = np.in1d(edges, idx)
        edges = edges[mask]
    # backconvert to the jjo'th orbital on the connecting atoms
    edges = HS.a2o(edges) + jjo
    r = HS.orij(io, edges)
    couplings = HS[io, edges]

    # Sort according to r
    idx_sorted = np.argsort(r)
    r = r[idx_sorted]
    couplings = couplings[idx_sorted, :]

    if unique:
        idx_uniq, cnt_uniq = np.unique(r.round(decimals=2), return_index=True, return_counts=True)[1:]
        r = r[idx_uniq]
        couplings = np.array([np.average(couplings[iu:(iu+cu), :], axis=0) for iu,cu in zip(idx_uniq, cnt_uniq)]) 
    return r, couplings


def get_R_hop(tshs, tbt, xyz_tip, pzidx, nn, z_gr=None, return_S=False):
    a_dev = tbt.a_dev
    tshs_dev = tshs.sub(a_dev)
    if z_gr == None:
        z_gr = tshs_dev.xyz[0, 2]
    C_list = (tshs_dev.xyz[:, 2] == z_gr).nonzero()[0]
    # Check that we have selected only carbon atoms
    for ia, a in zip(C_list, tshs_dev.atom[C_list]):
        if a.Z != 6:
            print('WARNING: Some atoms are not carbons in the graphene plane: {} {}'.format(ia, tshs_dev.xyz[ia]))
    # Get distances of all C atoms from tip (x,y) position 
    # (notice that tshs_dev.xyz = tshs.xyz, so we need to use xyz_tip wrt full geom)
    #xyz_tip_dev = xyz_tip - tshs_dev.xyz[0]
    #xyz_tip_dev[2] = tshs_dev.xyz[0, 2]
    _, distance = tshs_dev.geom.close_sc(xyz_tip, R=np.inf, idx=C_list, ret_rij=True)
    # Get onsite and couplings for each of the atoms, up to the 3rd nn 
    hoppings = np.empty((len(distance), nn+1))
    if return_S:
        overlaps = np.empty((len(distance), nn+1))
    for ia in C_list:
        # Extracting only pz-projected parameters from TSHS of graphene with tip
        _, tmp = get_dft_param(tshs_dev, ia, pzidx, pzidx, unique=True, onlynnz=True, idx=C_list)
        for i in range(nn+1):
            hoppings[ia, i] = tmp[i][0]
            if return_S:
                overlaps[ia, i] = tmp[i][1]

    # Write sorted data for future usage
    isort = np.argsort(distance)
    si.io.TableSile('couplings.txt', 'w').write_data(distance[isort], *hoppings[isort].T)
    if return_S:
        return distance[isort], hoppings[isort].T, overlaps[isort].T
    return distance[isort], hoppings[isort].T


def plot_couplings_dft2tb(tshs_pristine, tshs, tbt, xyz_tip, pzidx=2, figname='dH.pdf'):
    """
    Compare onsite and couplings of pristine graphene with those of a 
    dirty graphene system.
    Plots both raw data and relative difference.
    #
    # param0[i][j]
    #   i=0: on-site
    #   i=1: 1nn coupling
    #   i=2: 2nn coupling
    #   i=3: 3nn coupling
    #       j=0 : Hamiltonian matrix
    #       j=1 : Overlap matrix

    Example:
    import sisl as si
    from tbtncTools import plot_couplings_dft2tb
    tshs_pristine = si.get_sile('../../pristine_300kpt/GR.TSHS').read_hamiltonian()
    tshs = si.get_sile('../../tip_atop_szp/z1.8/GR.TSHS').read_hamiltonian()
    tbt = si.get_sile('../../tip_atop_szp/z1.8/siesta.TBT.nc')
    xyz_tip =  tshs.xyz[-1, :]
    plot_couplings_dft2tb(tshs_pristine, tshs, tbt, xyz_tip, pzidx=2, figname='dH.pdf')
    """
    
    # Plot reference lines for well converged pristine graphene system
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Extracting only pz-projected parameters from TSHS of perfect graphene
    _, param0 = get_dft_param(tshs_pristine, 0, pzidx, pzidx, unique=True, onlynnz=True)
    # Plot
    ax.axhline(y=param0[0][0], label='On-site', c='k', ls='-')
    ax.axhline(y=param0[1][0], label='1nn coupling', c='g', ls='-')
    ax.axhline(y=param0[2][0], label='2nn coupling', c='r', ls='-')
    ax.axhline(y=param0[3][0], label='3nn coupling', c='b', ls='-')

    # Plot onsite and couplings for well converged "dirty" graphene system
    distance, param = get_R_hop(tshs, tbt, xyz_tip, pzidx)
    # Plot
    ax.scatter(distance, param[0], label='On-site (tip)', c='k')#, ls='--')
    ax.scatter(distance, param[1], label='1nn coupling (tip)', c='g')#, ls='--')
    ax.scatter(distance, param[2], label='2nn coupling (tip)', c='r')#, ls='--')
    ax.scatter(distance, param[3], label='3nn coupling (tip)', c='b')#, ls='--')
    
    # Mark the distance between the tip (x,y) and the closest distance from outmost frame atoms   
    rM01 = np.absolute(np.amax(tshs.xyz[:, 0]) - xyz_tip[0])
    rM02 = np.absolute(np.amin(tshs.xyz[:, 0]) - xyz_tip[0])
    rM11 = np.absolute(np.amax(tshs.xyz[:, 1]) - xyz_tip[1])
    rM12 = np.absolute(np.amin(tshs.xyz[:, 1]) - xyz_tip[1])
    rM = np.amin([rM01, rM02, rM11, rM12])
    ax.axvline(x=rM, c='k', ls='--')
    
    # General plot settings
    plt.xlim(0., np.amax(distance))
    ax.set_xlabel('$r-r_{\mathrm{tip}}\,(\AA)$')
    ax.set_ylabel('E (eV)')
    plt.legend(loc=4, fontsize=10, ncol=2)
    plt.tight_layout()
    for o in fig.findobj():
        o.set_clip_on(False)
    plt.savefig(figname)

    # Plot relative difference
    f, axes = plt.subplots(4, sharex=True)
    f.subplots_adjust(hspace=0)
    axes[0].scatter(distance, param[0]-np.full(len(distance), param0[0][0]), 
                label='On-site', c='k')
    axes[1].scatter(distance, param[1]-np.full(len(distance), param0[1][0]), 
                label='1nn coupling', c='g')
    axes[2].scatter(distance, param[2]-np.full(len(distance), param0[2][0]), 
                label='2nn coupling', c='r')
    axes[3].scatter(distance, param[3]-np.full(len(distance), param0[3][0]), 
                label='3nn coupling', c='b')
    # Mark the distance between the tip (x,y) and the closest distance from outmost frame atoms   
    for a in axes:
        a.axhline(y=0., c='lightgrey', ls='-')
        a.axvline(x=rM, c='k', ls='--')
        #a.autoscale()       
        a.set_xlim(0., np.amax(distance))
        a.set_ylim(a.get_ylim()[0], 0.)
        a.yaxis.set_major_locator(plt.MaxNLocator(3))
    # General plot settings
    axes[-1].set_xlabel('$r-r_{\mathrm{tip}}\,(\AA)$')
    f.text(0.025, 0.5, '$\Delta E $ (eV)', ha="center", va="center", rotation=90)
    #for o in f.findobj():
    #    o.set_clip_on(False) 
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('diff_'+figname)    

def sc_xyz_shift(geom, axis): 
    return (geom.cell[axis,axis] - (np.amax(geom.xyz[:,axis]) - np.amin(geom.xyz[:,axis])))/2



#def Delta(TSHS, HS_TB, shape='Cuboid', z_graphene=None, ext_offset=None, center=None, 
def Delta(TSHS, shape='Cuboid', z_graphene=None, ext_offset=None, center=None, 
    thickness=None, zaxis=2, atoms=None, segment_dir=None):
    # z coordinate of graphene plane 
    if z_graphene is None:
        print('\n\nPlease provide a value for z_graphene in Delta routine')
        exit(1)
    # Center of shape in TSHS 
    if center is None:
        center = TSHS.center(atom=(TSHS.xyz[:,zaxis] == z_graphene).nonzero()[0])
    center = np.asarray(center)
    # Thickness in Ang
    if thickness is None:
        thickness = 6. # Ang
        #thickness = HS_TB.maxR()+0.01
    thickness = np.asarray(thickness, np.float64)
    # Cuboid or Ellissoid?
    if zaxis == 2:
        size = .5*np.diagonal(TSHS.cell) + [0,0,300] # default radius is half the cell size
    elif zaxis == 0:
        size = .5*np.diagonal(TSHS.cell) + [300,0,0] # default radius is half the cell size
    elif zaxis == 1:
        size = .5*np.diagonal(TSHS.cell) + [0,300,0] # default radius is half the cell size

    if shape == 'Ellipsoid' or shape == 'Sphere':
        mkshape = si.shape.Ellipsoid
    elif shape == 'Cuboid' or shape == 'Cube':
        mkshape = si.shape.Cuboid
        # In this case it's the full perimeter so we double
        size *= 2
        thickness *= 2
        if ext_offset is not None:
            ext_offset = np.asarray(ext_offset, np.float64).copy()
            ext_offset *= 2
    elif shape == 'Segment':
        mkshape = si.shape.Cuboid
        # In this case it's the full perimeter so we double
        size *= 2
        area_tot = mkshape(size, center=TSHS.center(atom=(TSHS.xyz[:,zaxis] == z_graphene).nonzero()[0]))
        size[segment_dir] = thickness
        if ext_offset is not None:
            ext_offset = np.asarray(ext_offset, np.float64).copy()
    else:
        print('\n shape = "{}" is not implemented...'.format(shape))
        exit(1)

    if shape == 'Segment':  # ADD COMPLEMENTARY AREA...
        # Areas
        Delta = mkshape(size, center=center)
        # Atoms within Delta and complementary area
        a_Delta = Delta.within_index(TSHS.xyz)
        if atoms is not None:
            a_Delta = a_Delta[np.in1d(a_Delta, atoms)]
        # Check
        v = TSHS.geom.copy(); v.atom[a_Delta] = si.Atom(8, R=[1.43]); v.write('a_Delta.xyz')
        return a_Delta, Delta
    else:
        # External boundary
        area_ext = mkshape(size, center=center)
        # Adjust with ext_offset if necessary
        if ext_offset is not None:
            ext_offset = np.asarray(ext_offset, np.float64)
            area_ext = area_ext.expand(-ext_offset)
            # Force it to be Cube or Sphere (side = ext_offset) if necessary
            if shape == 'Sphere' or shape == 'Cube':
                if len(ext_offset.nonzero()[0]) > 1:
                    print('Offset is in both axes. Please set "shape" to Cuboid or Ellipsoid')
                    exit(1)
                axis = ext_offset.nonzero()[0][0]
                print('Offset is non-zero along axis: {}...complementary is {}'.format(axis, int(axis<1)))
                new_ext_offset = np.zeros(3); new_ext_offset[int(axis<1)] = ext_offset[axis]
                area_ext = area_ext.expand(-new_ext_offset)
        #a_ext = area_ext.within_index(TSHS.xyz)
        
        # Internal boundary
        area_int = area_ext.expand(-thickness)
        # Disjuction composite shape
        Delta = area_ext - area_int
        # Atoms within Delta and internal boundary
        a_Delta = Delta.within_index(TSHS.xyz)
        a_int = area_int.within_index(TSHS.xyz)
        if atoms is not None:
            a_Delta = a_Delta[np.in1d(a_Delta, atoms)]
        # Check
        v = TSHS.geom.copy(); v.atom[a_Delta] = si.Atom(8, R=[1.43]); v.write('a_Delta.xyz')
        return a_Delta, a_int, Delta, area_ext, area_int


def makeTB(TSHS_0, pzidx, nn, WW, LL, elec=None, save=True, return_bands=False):
    """
    TSHS_0:         tbtncSile object from "pristine graphene" reference calculation
    pzidx:          index of pz orbitals in the basis set used to create 'TSHS_0'
    nn:             no. of neighbours to be used in the TB model
    W:              width of TB geometry (Angstrom) - transverse direction: 0 -
    L:              length of TB geometry (Angstrom) - transport direction: 1 -
    elec:      tbtncSile object from electrode calculation
    """
    
    ########################## From PERFECT graphene reference TSHS
    dR = 0.005

    # Check
    for a in TSHS_0.atom.atom:
        if a.Z != 6:
            print('ERROR: cannot build TB model because the provided geometry \
                is not a pristine graphene')
            exit(1)

    # Extracting only pz-projected parameters from TSHS of perfect graphene
    r, param = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)
    print('\nEffective no. of neighbors per atom from TSHS_0: {}'.format(len(r)-1))
    print('\nr ({}; Angstrom)\t param ({}; eV):'.format(len(r), len(param)))
    for ri, ci in zip(r, param):
        print('{:.5f} \t '.format(ri), ci)

    def get_graphene_H(radii, param, dR=dR):
        # In order to get the correct radii of the orbitals it is
        # best to define them explicitly.
        # This enables one to "optimize" the number of supercells
        # subsequently.
        # Define the radii of the orbital to be the maximum
        C = si.Atom(6, R=radii[-1] + dR)

        # Define graphene
        g = si.geom.graphene(radii[1], C, orthogonal=True)
        g.optimize_nsc()

        # Now create Hamiltonian
        H = si.Hamiltonian(g, orthogonal=False)

        # Define primitive also for check of bandstructure
        g_s = si.geom.graphene(radii[1], C)
        g_s.optimize_nsc()
        H_s = si.Hamiltonian(g_s, orthogonal=False)

        if len(param.shape) == 1:
            # Create a new fake parameter
            # with overlap elements
            new_param = np.zeros([len(param), 2], dtype=np.float64)
            new_param[:, 0] = param
            new_param[0, 1] = 1. # on-site, everything else, zero
            param = new_param

        H.construct((radii+dR, param))
        H_s.construct((radii+dR, param))
        return H, H_s

    # Setup the Hamiltonian building block
    if nn is 'all':
        print('WARNING: you are retaining ALL interactions from DFT model')
        H0, H0_s = get_graphene_H(r, param)
    else:
        print('WARNING: you are retaining only interactions up to {} neighbours'.format(nn))
        H0, H0_s = get_graphene_H(r[:nn+1], param[:nn+1])
    print('\nBuilding block for TB model:\n', H0)

    # Setup TB model 
    W, L = int(round(WW/H0.cell[0,0])), int(round(LL/H0.cell[1,1]))
    # ELECTRODE
    if elec is not None:
        n_el = int(round(elec.cell[1,1]/H0.cell[1,1]))
    else:
        n_el = 2
    HS_elec = H0.tile(W, 0).tile(n_el, 1)
    HS_elec.write('HS_ELEC.nc')
    HS_elec.geom.write('HS_ELEC.fdf')
    HS_elec.geom.write('HS_ELEC.xyz')
    # DEVICE + ELECTRODES (to be written ONLY after selection and rearranging of GF/dSE area)
    HS_dev = H0.tile(W, 0).tile(L, 1)
    if save:
        HS_dev.write('HS_DEV_0.nc')
        HS_dev.geom.write('HS_DEV_0.fdf')
        HS_dev.geom.write('HS_DEV_0.xyz')


    # Check bands with primitive cell
    if return_bands:
        # Open figure outside and bands will automatically be added to the plot
        plot_bandstructure(H0_s, 400, ymin=-3, ymax=3, 
            style='-', color='k', label='Pristine $p_z$ parameters')

    return HS_dev


def makeTB_FrameOutside(tshs, tbt, center, TSHS_0, pzidx, nn, WW, LL, 
    elec=None, save=True, return_bands=False, z_graphene=None):
    """
    tshs:           TSHS object from "dirty graphene" calculation
    tbt:            tbtncSile object from tbtrans calculation with HS: "tshs"
    TSHS_0:         TSHS object from "pristine graphene" reference calculation
    pzidx:          index of pz orbitals in the basis set used to create 'TSHS_0'
    nn:             no. of neighbours to be used in the TB model
    WW:             width of TB geometry (Angstrom) - transverse direction: 0 -
    LL:             length of TB geometry (Angstrom) - transport direction: 1 -
    TSHS_elec:      tbtncSile object from electrode calculation
    save:           True will store device region netcdf files for usage in tbtrans
    """
    
    ########################## From PERFECT graphene reference TSHS
    dR = 0.005

    # Check that TSHS_0 has only carbon atoms
    for a in TSHS_0.atom.atom:
        if a.Z != 6:
            print('ERROR: cannot build TB model because the provided geometry\n\tis not a pristine graphene')
            exit(1)


    # Extracting only pz-projected parameters from TSHS of perfect graphene
    r, param = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)
    print('\nEffective no. of neighbors per atom from TSHS_0: {}'.format(len(r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(r), len(param)))
    for ri, ci in zip(r, param):
        print('{:.5f} \t '.format(ri), ci)

    # Setup the Hamiltonian building block
    if nn is 'all':
        nn = len(r)-1

    # The reference values we wish to target (pristine graphene)
    ref_r, ref_hop, ref_over = r[:nn+1], param[:nn+1, 0], param[:nn+1, 1]
    print('Targeted no. of neighbors per atom from TSHS_0: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(ref_r), len(ref_hop)))
    for ri, ci, oi in zip(ref_r, ref_hop, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)

    # R and hopping from tshs, center is the coordinates of the tip apex
    # This works Only if the frame is the outmost atoms in tbt.a_dev
    # Maybe it's better to define a shape here!
    if z_graphene is None:
        print('\n\nPlease provide a value for z_graphene')
        exit(1)
    if center is None:
        center = tshs.center(atom=(tshs.xyz[:,2] == z_graphene).nonzero()[0])
        print('makeTB: you are considering this as center: {}'.format(center))

    distances, hop = get_R_hop(tshs, tbt, center, pzidx, nn, z_gr=z_graphene)
    hop_atframe = [np.average(hop[i, np.arange(-10, 0)]) for i in range(nn+1)]
    # r's to plot
    r2plot = np.linspace(0, np.amax(distances), 1000)
    f, ax = plt.subplots(nn+1, sharex=True)
    for i in range(nn+1):
        ax[i].scatter(distances, hop[i, :])
        # Plot lines
        ax[i].plot([r2plot.min(), r2plot.max()], [ref_hop[i], ref_hop[i]], '--')
        ymin = np.amin([ref_hop[i], hop_atframe[i]]) - 0.1
        ymax = np.amax([ref_hop[i], hop_atframe[i]]) + 0.1
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_xlim(r2plot.min(), r2plot.max())
    f.savefig('shifting_data.pdf')
    plt.close(f)

    ###### Create device Hamiltonian 
    bond = ref_r[1] # to make it fit in a smaller unit-cell
    C = si.Atom(6, R=ref_r[-1] + dR)
    g0 = si.geom.graphene(bond, C, orthogonal=True)
    g0.optimize_nsc()
    H0 = si.Hamiltonian(g0, orthogonal=False)

    print('\nNo. of neighbors per atom: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t Final parameters from frame ({}; eV):'.format(len(ref_r), len(hop_atframe)))
    for ri, ci, oi in zip(ref_r, hop_atframe, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)

    # Construct TB. onsite is the same as tip tshs, while couplings are the same as pristine
    H0.construct((ref_r+dR, zip(hop_atframe, ref_over)), eta=True)

    # DEVICE + ELECTRODES geometry 
    # Width and length of device
    W, L = int(round(WW/g0.cell[0,0])), int(round(LL/g0.cell[1,1]))
    print('Device is {} x {} supercell of the unit orthogonal cell'.format(W, L))
    # (nc files should be written ONLY after selection and rearranging of GF/dSE area)
    HS_dev = H0.tile(W, 0).tile(L, 1)
    if save:
        HS_dev.write('HS_DEV.nc')
        HS_dev.geom.write('HS_DEV.fdf')
        HS_dev.geom.write('HS_DEV.xyz')

    # ELECTRODE
    if elec is not None:
        n_el = int(round(elec.cell[1,1]/H0.cell[1,1]))
    else:
        n_el = 2
    HS_elec = H0.tile(W, 0).tile(n_el, 1)
    HS_elec.write('HS_ELEC.nc')
    HS_elec.geom.write('HS_ELEC.fdf')
    HS_elec.geom.write('HS_ELEC.xyz')



    # Check bands with primitive cell
    if return_bands:
        g0_s = si.geom.graphene(bond, C)
        g0_s.optimize_nsc()
        H0_s = si.Hamiltonian(g0_s, orthogonal=False)
        H0_s.construct((ref_r+dR, zip(hop_atframe, ref_over)))
        # Open figure outside and bands will automatically be added to the plot
        plot_bandstructure(H0_s, 400, ymin=-3, ymax=3, 
            style='--', color='r', label='Pristine w/ tip $p_z$ onsite')

    return HS_dev

def interp1d(x, y, y0, y1):
    """ Create an interpolation function from x, y.

    The resulting function has these properties:

    x < x.min():
       f(x) = y0
    x.min() < x < x.max():
       f(x) = y
    x.max() < x:
       f(x) = y1
    """
    return sp.interpolate.interp1d(x, y, bounds_error=False,
                                   fill_value=(y0, y1))

def func_smooth_fermi(x, y, first_x, second_x, y1, delta=8):
    """ Return an interpolation function with the following properties:

    x < first_x:
       f(x) = y(first_x)
    first_x < x < second_x:
       f(x) = y
    second_x < x
       f(x) = y1

    `delta` determines the amount of the smearing width that is between `first_x` and
    `second_x`.

    Parameters
    ----------
    x, y : numpy.ndarray
       x/y-data points
    first_x : float
       the point of cut-off for the x-values. In this approximation we assume
       the `y` data-points has a plateau in the neighbourhood of `first_x`
    second_x : float
       above this `x` value all values will be `y1`.
    y1 : float
       second boundary value
    delta : float, optional
       amount of smearing parameter in between `first_x` and `second_x` (should not be below 6!).
    """
        
    # First we will find the first flat plateau
    # We do this considering values -3 : +3 Ang
    if first_x < np.amax(x):
        raise ValueError("first_x has to be larger than maximum, interpolation x value")
    # First we will find the first flat plateau
    # We do this considering values -3 : r_max Ang
    idx = (np.amax(x) - x > -3.).nonzero()[0]
    y0 = np.average(y[idx])
    
    # We already have the second plateau.
    # So all we have to do is calculate the smearing
    # to capture the smoothing range
    mid_x = (first_x + second_x) / 2
    sigma = (second_x - first_x) / delta
    if y0 < y1:
        sigma = - sigma
        b = y0
    else:
        b = y1

    # Now we can create the function
    dd = delta / 2. + 1.
    ## Now calculate function parameters used for interpolation
    #x = np.arange(first_x - dd , second_x + dd, 0.01) # 0.01 Ang precision
    #y = abs(y1 - y0) / (np.exp((x - mid_x) / sigma) + 1) + b
    #return interp1d(x, y, y0, y1)

    # Now we can create the function
    dd = delta / 2. + 1.
    # Now calculate function parameters used for interpolation
    xff = np.arange(first_x, second_x + 2 * dd, 0.01) # 0.01 Ang precision
    yff = abs(y1 - y0) / (np.exp((x - mid_x) / sigma) + 1) + b
    return interp1d(np.append(x, xff), np.append(y, yff), y[0], y1)

def func_smooth_linear(x, y):
    return sp.interpolate.interp1d(x, y, kind='cubic', fill_value=(y[0], y[-1]), bounds_error=False)

def func_smooth(x, y, first_x=None, second_x=None, y1=None, delta=8, what='linear'):
    if what is None:
        what = 'linear'
    if what == 'fermi':
        return func_smooth_fermi(x, y, first_x, second_x, y1, delta)
    elif what == 'linear':
        return func_smooth_linear(x, y)

def makeTB_InterpFrame(tshs, tbt, xyz_tip, TSHS_0, pzidx, nn, WW, LL, 
    elec=None, save=True, return_bands=False, avg=False):
    """
    tshs:           TSHS object from "dirty graphene" calculation
    tbt:            tbtncSile object from tbtrans calculation with HS: "tshs"
    TSHS_0:         TSHS object from "pristine graphene" reference calculation
    pzidx:          index of pz orbitals in the basis set used to create 'TSHS_0'
    nn:             no. of neighbours to be used in the TB model
    WW:             width of TB geometry (Angstrom) - transverse direction: 0 -
    LL:             length of TB geometry (Angstrom) - transport direction: 1 -
    TSHS_elec:      tbtncSile object from electrode calculation
    save:           True will store device region netcdf files for usage in tbtrans
    """
    
    ########################## From PERFECT graphene reference TSHS
    dR = 0.005

    # Check that TSHS_0 has only carbon atoms
    for a in TSHS_0.atom.atom:
        if a.Z != 6:
            print('ERROR: cannot build TB model because the provided geometry\n\tis not a pristine graphene')
            exit(1)

    # Extracting only pz-projected parameters from TSHS of perfect graphene
    r, param = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)
    print('\nEffective no. of neighbors per atom from TSHS_0: {}'.format(len(r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(r), len(param)))
    for ri, ci in zip(r, param):
        print('{:.5f} \t '.format(ri), ci)

    # Setup the Hamiltonian building block
    if nn is 'all':
        nn = len(r)-1

    # The reference values we wish to target (pristine graphene)
    ref_r, ref_hop, ref_over = r[:nn+1], param[:nn+1, 0], param[:nn+1, 1]
    print('Targeted no. of neighbors per atom from TSHS_0: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(ref_r), len(ref_hop)))
    for ri, ci, oi in zip(ref_r, ref_hop, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)


    # Get distance from tip and relative hoppings, sorted 
    distances, hop = get_R_hop(tshs, tbt, xyz_tip, pzidx, nn)
    if avg:
        hop_atframe = [np.average(hop[i, np.arange(-10, 0)]) for i in range(nn+1)]
    else:
        fit = [func_smooth(distances, hop[i, :]) for i in range(nn+1)]
        
        # r's to plot
        r2plot = np.linspace(0, 1.2*distances[-1], 1000)
        f, ax = plt.subplots(nn+1, sharex=True)
        for i in range(nn+1):
            ax[i].scatter(distances, hop[i, :])
            ax[i].plot(r2plot, fit[i](r2plot))
            # Plot lines
            #ax[i].plot([r2plot.min(), r2plot.max()], [ref_hop[i], ref_hop[i]], '--')
            #ymin = np.amin([ref_hop[i], fit[i](distances[-1])]) - 0.1
            #ymax = np.amax([ref_hop[i], fit[i](distances[-1])]) + 0.1
            #ax[i].plot([distances[-1], distances[-1]], [ymin, ymax], '--')
            #ax[i].set_ylim(ymin, ymax)
            ax[i].set_xlim(r2plot.min(), r2plot.max())
        f.savefig('fit_data.pdf')
        plt.close(f)
        ftyifti

        ###### Create device Hamiltonian using the correct parameters
        bond = ref_r[1] # to make it fit in a smaller unit-cell
        C = si.Atom(6, R=ref_r[-1] + dR)
        g0 = si.geom.graphene(bond, C, orthogonal=True)
        g0.optimize_nsc()
        # Width and length of device
        W, L = int(round(WW/g0.cell[0,0])), int(round(LL/g0.cell[1,1]))
        
        # DEVICE + ELECTRODES geometry (without PBC!!!)
        # (nc files should be written ONLY after selection and rearranging of GF/dSE area)
        g = g0.tile(W, 0).tile(L, 1)
        g.set_nsc([1] *3)
        HS_dev = si.Hamiltonian(g, orthogonal=False)
        # Create the connectivity values
        Hc = [np.empty(len(g)) for i in range(nn+1)]

        # # Get tip (x,y) position in large TB
        # frameOrigin_xyz = g.xyz[frameOrigin-TSHS_elec.na]
        # print('Frame reference (x, y, z=z_graphene) coordinates (low-left) in large TB geometry are:\n\t{}'.format(frameOrigin_xyz))
        # c_xyz = frameOrigin_xyz + xyz_tip
        # c_xyz[2] = frameOrigin_xyz[2]
        # print('Tip (x, y, z=z_graphene) coordinates in large TB geometry are:\n\t{}'.format(c_xyz))
        # c_xyz = c_xyz.reshape(1, 3)
        
        # Now loop and construct the Hamiltonian
        def func(self, ia, idxs, idxs_xyz=None):
            idx_a, xyz_a = self.geom.close(ia, R=ref_r+dR, idx=idxs, 
                idx_xyz=idxs_xyz, ret_xyz=True)

            # Calculate distance to center
            # on-site does not need averaging
            rr = np.sqrt(np.square(xyz_a[0] - c_xyz).sum(1))
            f = fit[0](rr)
            self[ia, idx_a[0], 0] = f
            self[ia, idx_a[0], 1] = ref_over[0]
            Hc[0][ia] = np.average(f)

            xyz = g.xyz[ia, :].reshape(1, 3)
            for i in range(1, len(idx_a)):
                rr = np.sqrt(np.square((xyz_a[i] + xyz)/2 - c_xyz).sum(1))
                f = fit[i](rr)
                self[ia, idx_a[i], 0] = f
                self[ia, idx_a[i], 1] = ref_over[i]
                Hc[i][ia] = np.average(f)

        HS_dev.construct(func, eta=True)

        # Extract at Gamma for plot
        Hk = HS_dev.tocsr(0)
        # Check for Hermiticity
        if np.abs(Hk - Hk.T).max() != 0.:
            print('ERROR: Hamitonian is NOT HERMITIAN!')
            exit(0)

        # Plot onsite and coupling maps
        cm = plt.cm.get_cmap('RdYlBu')
        x = HS_dev.xyz[:, 0]
        y = HS_dev.xyz[:, 1]
        for i in range(nn+1):
            plt.figure()
            z = Hc[i]
            sc = plt.scatter(x, y, c=abs(z), edgecolor='none', cmap=cm)
            plt.colorbar(sc)
            plt.savefig('fermifit_{}.png'.format(i), dpi=300)
        
        if save:
            HS_dev.write('HS_DEV.nc')
            HS_dev.geom.write('HS_DEV.fdf')
            HS_dev.geom.write('HS_DEV.xyz')
        
        # ELECTRODE
        n_el = int(round(TSHS_elec.cell[1,1]/g0.cell[1,1]))
        H0 = si.Hamiltonian(g0, orthogonal=False)
        H0.construct((ref_r+dR, zip(ref_hop, ref_over)))
        HS_elec = H0.tile(W, 0).tile(n_el, 1)
        HS_elec.write('HS_ELEC.nc')
        HS_elec.geom.write('HS_ELEC.fdf')
        HS_elec.geom.write('HS_ELEC.xyz')

        # Check bands with primitive cell
        if return_bands:
            g0_s = si.geom.graphene(bond, C)
            g0_s.optimize_nsc()
            H0_s = si.Hamiltonian(g0_s, orthogonal=False)
            H0_s.construct((ref_r+dR, zip(ref_hop, ref_over)))
            # Open figure outside and bands will automatically be added to the plot
            plot_bandstructure(H0_s, 400, ymin=-3, ymax=3, 
                style='-.', color='b', label='After Fermi fit')

    return HS_dev



### TO FIX
def makeTB_fermi(tshs, tbt, xyz_tip, frameOrigin, TSHS_0, pzidx, nn, 
    WW, LL, elec, save=True, cut_R=None, smooth_R=15., return_bands=False):
    """
    tshs:           TSHS object from "dirty graphene" calculation
    tbt:            tbtncSile object from tbtrans calculation with HS: "tshs"
    xyz_tip:        coordinates of tip apex atom in tshs, after setting z=z_graphene 
    TSHS_0:         TSHS object from "pristine graphene" reference calculation
    pzidx:          index of pz orbitals in the basis set used to create 'TSHS_0'
    nn:             no. of neighbours to be used in the TB model
    WW:             width of TB geometry (Angstrom) - transverse direction: 0 -
    LL:             length of TB geometry (Angstrom) - transport direction: 1 -
    elec:      tbtncSile object from electrode calculation
    save:           True will store device region netcdf files for usage in tbtrans
    smooth_R:       The length over which we will smooth the function (Angstrom)
    """
    
    ########################## From PERFECT graphene reference TSHS
    dR = 0.005

    # Check that TSHS_0 has only carbon atoms
    for a in TSHS_0.atom.atom:
        if a.Z != 6:
            print('ERROR: cannot build TB model because the provided geometry\n\tis not a pristine graphene')
            exit(1)

    # Extracting only pz-projected parameters from TSHS of perfect graphene
    r, param = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)
    print('Effective no. of neighbors per atom from TSHS_0: {}'.format(len(r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(r), len(param)))
    for ri, ci in zip(r, param):
        print('{:.5f} \t '.format(ri), ci)

    # Setup the Hamiltonian building block
    if nn is 'all':
        nn = len(r)-1

    # The reference values we wish to target (pristine graphene)
    ref_r, ref_hop, ref_over = r[:nn+1], param[:nn+1, 0], param[:nn+1, 1]
    print('Targeted no. of neighbors per atom from TSHS_0: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(ref_r), len(ref_hop)))
    for ri, ci, oi in zip(ref_r, ref_hop, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)


    # R and hopping from tshs, xyz_tip is the coordinates of the tip apex
    # This works Only if the frame is the outmost atoms in tbt.a_dev
    # Maybe it's better to define a shape here!
    distances, hop = get_R_hop(tshs, tbt, xyz_tip, pzidx, nn)
    # Create Fermi-like function to smooth hop towards ref_hop
    print(np.amax(distances))
    print(cut_R)
    if cut_R is None:
        cut_R = np.amax(distances)
    print('\nCutoff radius in TSHS: {} Ang'.format(cut_R))
    fermi_fit = [func_smooth(distances, hop[i, :], cut_R, cut_R + smooth_R, ref_hop[i]) for i in range(nn+1)]
    # r's to plot
    r2plot = np.linspace(0, cut_R+1.2*smooth_R, 1000)
    f, ax = plt.subplots(nn+1, sharex=True)
    for i in range(nn+1):
        ax[i].scatter(distances, hop[i, :])
        ax[i].plot(r2plot, fermi_fit[i](r2plot))
        # Plot lines
        ax[i].plot([r2plot.min(), r2plot.max()], [ref_hop[i], ref_hop[i]], '--')
        ymin = np.amin([ref_hop[i], fermi_fit[i](cut_R)]) - 0.1
        ymax = np.amax([ref_hop[i], fermi_fit[i](cut_R)]) + 0.1
        ax[i].plot([cut_R, cut_R], [ymin, ymax], '--')
        ax[i].plot([cut_R+smooth_R, cut_R+smooth_R], [ymin, ymax], '--')
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_xlim(r2plot.min(), r2plot.max())
    f.savefig('fermifit_data.pdf')
    plt.close(f)
    fuifguyi

    ###### Create device Hamiltonian using the correct parameters
    bond = ref_r[1] # to make it fit in a smaller unit-cell
    C = si.Atom(6, R=ref_r[-1] + dR)
    g0 = si.geom.graphene(bond, C, orthogonal=True)
    g0.optimize_nsc()
    # Width and length of device
    W, L = int(round(WW/g0.cell[0,0])), int(round(LL/g0.cell[1,1]))
    # DEVICE + ELECTRODES geometry 
    # (nc files should be written ONLY after selection and rearranging of GF/dSE area)
    # MAYBE NEED TO STORE THIS ONLY AFTERWORDS!!!! OR MAYBE NOT...
    g = g0.tile(W, 0).tile(L, 1)
    g.set_nsc([1] *3)
    HS_dev = si.Hamiltonian(g, orthogonal=False)
    # Create the connectivity values
    Hc = [np.empty(len(g)) for i in range(nn+1)]

    # Get tip (x,y) position in large TB
    frameOrigin_xyz = g.xyz[frameOrigin-elec.na]
    print('Frame reference (x, y, z=z_graphene) coordinates (low-left) in large TB geometry are:\n\t{}'.format(frameOrigin_xyz))
    c_xyz = frameOrigin_xyz + xyz_tip
    c_xyz[2] = frameOrigin_xyz[2]
    print('Tip (x, y, z=z_graphene) coordinates in large TB geometry are:\n\t{}'.format(c_xyz))
    c_xyz = c_xyz.reshape(1, 3)
    
    # Now loop and construct the Hamiltonian
    def func(self, ia, idxs, idxs_xyz=None):
        xyz = g.xyz[ia, :].reshape(1, 3)
        idx_a, xyz_a = self.geom.close(ia, R=ref_r+dR, idx=idxs, idx_xyz=idxs_xyz, ret_xyz=True)

        # Calculate distance to center
        # on-site does not need averaging
        rr = np.sqrt(np.square(xyz_a[0] - c_xyz).sum(1))
        f = fermi_fit[0](rr)
        self[ia, idx_a[0], 0] = f
        self[ia, idx_a[0], 1] = ref_over[0]
        Hc[0][ia] = np.average(f)

        for i in range(1, len(idx_a)):
            rr = np.sqrt(np.square((xyz_a[i] + xyz)/2 - c_xyz).sum(1))
            f = fermi_fit[i](rr)
            self[ia, idx_a[i], 0] = f
            self[ia, idx_a[i], 1] = ref_over[i]
            Hc[i][ia] = np.average(f)

    HS_dev.construct(func, eta=True)

    # Extract at Gamma for plot
    Hk = HS_dev.tocsr(0)
    # Check for Hermiticity
    if np.abs(Hk - Hk.T).max() != 0.:
        print('ERROR: Hamitonian is NOT HERMITIAN!')
        exit(0)

    # Plot onsite and coupling maps
    cm = plt.cm.get_cmap('RdYlBu')
    x = HS_dev.xyz[:, 0]
    y = HS_dev.xyz[:, 1]
    for i in range(nn+1):
        plt.figure()
        z = Hc[i]
        sc = plt.scatter(x, y, c=abs(z), edgecolor='none', cmap=cm)
        plt.colorbar(sc)
        plt.savefig('fermifit_{}.png'.format(i), dpi=300)
    
    if save:
        HS_dev.write('HS_DEV.nc')
        HS_dev.geom.write('HS_DEV.fdf')
        HS_dev.geom.write('HS_DEV.xyz')
    
    # ELECTRODE
    n_el = int(round(elec.cell[1,1]/g0.cell[1,1]))
    H0 = si.Hamiltonian(g0, orthogonal=False)
    H0.construct((ref_r+dR, zip(ref_hop, ref_over)))
    HS_elec = H0.tile(W, 0).tile(n_el, 1)
    HS_elec.write('HS_ELEC.nc')
    HS_elec.geom.write('HS_ELEC.fdf')
    HS_elec.geom.write('HS_ELEC.xyz')

    # Check bands with primitive cell
    if return_bands:
        g0_s = si.geom.graphene(bond, C)
        g0_s.optimize_nsc()
        H0_s = si.Hamiltonian(g0_s, orthogonal=False)
        H0_s.construct((ref_r+dR, zip(ref_hop, ref_over)))
        # Open figure outside and bands will automatically be added to the plot
        plot_bandstructure(H0_s, 400, ymin=-3, ymax=3, 
            style='-.', color='b', label='After Fermi fit')

    return HS_dev

### NOT REALLY USEFUL
def makeTB_shifted(tshs, tbt, xyz_tip, TSHS_0, pzidx, nn, WW, LL, TSHS_elec, 
    save=True, shifted=True, return_bands=False):
    """
    tshs:           TSHS object from "dirty graphene" calculation
    tbt:            tbtncSile object from tbtrans calculation with HS: "tshs"
    TSHS_0:         TSHS object from "pristine graphene" reference calculation
    pzidx:          index of pz orbitals in the basis set used to create 'TSHS_0'
    nn:             no. of neighbours to be used in the TB model
    WW:             width of TB geometry (Angstrom) - transverse direction: 0 -
    LL:             length of TB geometry (Angstrom) - transport direction: 1 -
    TSHS_elec:      tbtncSile object from electrode calculation
    save:           True will store device region netcdf files for usage in tbtrans
    """
    
    ########################## From PERFECT graphene reference TSHS
    dR = 0.005

    # Check that TSHS_0 has only carbon atoms
    for a in TSHS_0.atom.atom:
        if a.Z != 6:
            print('ERROR: cannot build TB model because the provided geometry\n\tis not a pristine graphene')
            exit(1)

    # Extracting only pz-projected parameters from TSHS of perfect graphene
    r, param = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)
    print('\nEffective no. of neighbors per atom from TSHS_0: {}'.format(len(r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(r), len(param)))
    for ri, ci in zip(r, param):
        print('{:.5f} \t '.format(ri), ci)

    # Setup the Hamiltonian building block
    if nn is 'all':
        nn = len(r)-1

    # The reference values we wish to target (pristine graphene)
    ref_r, ref_hop, ref_over = r[:nn+1], param[:nn+1, 0], param[:nn+1, 1]
    print('Targeted no. of neighbors per atom from TSHS_0: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t param ({}; eV):'.format(len(ref_r), len(ref_hop)))
    for ri, ci, oi in zip(ref_r, ref_hop, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)

    # R and hopping from tshs, xyz_tip is the coordinates of the tip apex
    # This works Only if the frame is the outmost atoms in tbt.a_dev
    # Maybe it's better to define a shape here!
    distances, hop = get_R_hop(tshs, tbt, xyz_tip, pzidx, nn)
    hop_atframe = [np.average(hop[i, np.arange(-10, 0)]) for i in range(nn+1)]
    # r's to plot
    r2plot = np.linspace(0, np.amax(distances), 1000)
    f, ax = plt.subplots(nn+1, sharex=True)
    for i in range(nn+1):
        ax[i].scatter(distances, hop[i, :])
        # Plot lines
        ax[i].plot([r2plot.min(), r2plot.max()], [ref_hop[i], ref_hop[i]], '--')
        ymin = np.amin([ref_hop[i], hop_atframe[i]]) - 0.1
        ymax = np.amax([ref_hop[i], hop_atframe[i]]) + 0.1
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_xlim(r2plot.min(), r2plot.max())
    f.savefig('shifting_data.pdf')
    plt.close(f)

    ###### Create device Hamiltonian using shifted on-site energy
    bond = ref_r[1] # to make it fit in a smaller unit-cell
    C = si.Atom(6, R=ref_r[-1] + dR)
    g0 = si.geom.graphene(bond, C, orthogonal=True)
    g0.optimize_nsc()
    H0 = si.Hamiltonian(g0, orthogonal=False)
    
    ref_hop_onshifted = ref_hop.copy()
    if shifted:
        ref_hop_onshifted[0] = hop_atframe[0]

    print('\nFinal no. of neighbors per atom retained from TSHS_0: {}'.format(len(ref_r)-1))
    print('r ({}; Angstrom)\t Final parameters ({}; eV):'.format(len(ref_r), len(ref_hop_onshifted)))
    for ri, ci, oi in zip(ref_r, ref_hop_onshifted, ref_over):
        print('{:.5f} \t '.format(ri), ci, oi)

    # Construct TB. onsite is the same as tip tshs, while couplings are the same as pristine
    H0.construct((ref_r+dR, zip(ref_hop_onshifted, ref_over)))
    
    # DEVICE + ELECTRODES geometry 
    # Width and length of device
    W, L = int(round(WW/g0.cell[0,0])), int(round(LL/g0.cell[1,1]))
    # (nc files should be written ONLY after selection and rearranging of GF/dSE area)
    HS_dev = H0.tile(W, 0).tile(L, 1)
    if save:
        HS_dev.write('HS_DEV.nc')
        HS_dev.geom.write('HS_DEV.fdf')
        HS_dev.geom.write('HS_DEV.xyz')
    
    # ELECTRODE
    n_el = int(round(TSHS_elec.cell[1,1]/g0.cell[1,1]))
    HS_elec = H0.tile(W, 0).tile(n_el, 1)
    HS_elec.write('HS_ELEC.nc')
    HS_elec.geom.write('HS_ELEC.fdf')
    HS_elec.geom.write('HS_ELEC.xyz')



    # Check bands with primitive cell
    if return_bands:
        g0_s = si.geom.graphene(bond, C)
        g0_s.optimize_nsc()
        H0_s = si.Hamiltonian(g0_s, orthogonal=False)
        H0_s.construct((ref_r+dR, zip(ref_hop_onshifted, ref_over)))
        # Open figure outside and bands will automatically be added to the plot
        plot_bandstructure(H0_s, 400, ymin=-3, ymax=3, 
            style='--', color='r', label='Pristine w/ tip $p_z$ onsite')

    return HS_dev


def plot_transmission(H, iE1, iE2, ymin=None, ymax=None, style='-', color='k', label=None, 
    xshift=0, yshift=0, plus=None, plot=True, lw=1):
    print('Plotting transmission from elec {} to elec {} in: {}'.format(iE1, iE2, H))
    H = si.get_sile(H)
    tr = H.transmission(H.elecs[iE1], H.elecs[iE2])
    
    ax = plt.gca()

    if not plot:
        return ax, tr

    if plus is not None:
        ax.plot(H.E+xshift, tr+plus+yshift, style, color=color, label=label, linewidth=lw)
    else:
        ax.plot(H.E+xshift, tr+yshift, style, color=color, label=label, linewidth=lw)
    
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('Transmission')
    ax.set_xlabel('$\mathrm{E-E_F}$ $(e\mathrm{V})$')

    if plus is not None:
        return ax, tr+plus+yshift
    else:
        return ax, tr+yshift

def plot_transmission_bulk(H, iE, ymin=None, ymax=None, style='-', color='k', label=None, xshift=0, yshift=0):
    print('Plotting bulk transmission from elec {} in: {}'.format(iE, H))
    H = si.get_sile(H)
    tr = H.transmission_bulk(H.elecs[iE])

    ax = plt.gca()

    ax.plot(H.E+xshift, tr+yshift, style, color=color, label=label)

    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('Transmission')
    ax.set_xlabel('$\mathrm{E-E_F}$ $(e\mathrm{V})$')

    return ax, tr

def read_bondcurrents(f, idx_elec, only='+', E=0.0, k='avg'):#, atoms=None):
    """Read bond currents from tbtrans output

    Parameters
    ----------
    f : string
        TBT.nc file
    idx_elec : int
        the electrode of originating electrons
    only : {'+', '-', 'all'}
        If "+" is supplied only the positive orbital currents are used, for "-", 
        only the negative orbital currents are used, else return the sum of both. 
    E : float or int, 
        A float for energy in eV, int for explicit energy index 
    k : bool, int or array_like
        whether the returned bond current is k-averaged, 
        an explicit k-point or a selection of k-points

    Returns
    -------
    bc, nc.E[idx_E], geom
    bc : bond currents
    nc.E[idx_E] : energy
    geom : geometry

    """
    print('Reading: {}'.format(f))
    nc = si.get_sile(f)
    na, na_dev = nc.na, nc.na_dev
    print('Total number of atoms: {}'.format(na))
    print('Number of atoms in the device region: {}'.format(na_dev))
    geom = nc.geom

    elec = nc.elecs[idx_elec]
    print('Bond-currents from electrode: {}'.format(elec))

    # Check 'k' argument
    if k == 'avg':
        avg = True
    elif k == 'Gamma':
        kpts = nc.kpt
        idx_gamma = np.where(np.sum(np.abs(kpts), axis=1) == 0.)[0]
        if (kpts[idx_gamma] != np.zeros((1, 3))).any(axis=1):
            print('\nThe selected k-point is not Gamma!\n')
            exit(0)
        else:
            print('You have selected the Gamma point!')
        avg = idx_gamma # Index of Gamma point in nc.kpt
    else:
        print('\nInvalid `k` argument: please keep the default `avg` or use `Gamma`!\n')
        exit(0)

    idx_E = nc.Eindex(E)
    print('Extracting bond-currents at energy: {} eV'.format(nc.E[idx_E]))
    bc = nc.bond_current(elec, kavg=avg, isc=[0,0,0], only=only, E=idx_E, uc=True)

    return bc, nc.E[idx_E], geom

    # bc_coo = nc.bond_current(elec, kavg=avg, isc=[0,0,0], only=only, E=idx_E, uc=True).tocoo()
    # i_list = bc_coo.row
    # j_list = bc_coo.col
    # bc_list = bc_coo.data
    # #for i, j, bc in zip(i_list, j_list, bc_list):
    # #    print('{}\t{}\t{}'.format(i, j, bc))
    # print('Number of bond-current entries: {}'.format(np.shape(bc_list)))

    # if atoms is not None:    
    #     i_list_new, j_list_new, bc_list_new = [], [], []
    #     for i, j, bc in zip(i_list, j_list, bc_list):
    #         if i in atoms and j in atoms:
    #             i_list_new.append(i)
    #             j_list_new.append(j)
    #             bc_list_new.append(bc)
    #     i_list = np.array(i_list_new)
    #     j_list = np.array(j_list_new)
    #     bc_list = np.array(bc_list_new)
    
    # #print('i\tj\tBond-current')
    # #for i, j, bc in zip(i_list, j_list, bc_list):
    # #    print('{}\t{}\t{}'.format(i, j, bc))

    # print('MIN bc (from file) = {}'.format(np.min(bc_list)))
    # print('MAX bc (from file) = {}'.format(np.max(bc_list)))

    # return (geom, i_list, j_list, bc_list, nc.E[idx_E])

def bc_sub(bc, atoms):
    """
    bc:     bondcurrents object directly from "read_bondcurrent"
    atoms:  list of selected atoms
    """
    # Get data
    i_list, j_list, bc_list = bc.tocoo().row, bc.tocoo().col, bc.tocoo().data
    # Filter only selected atoms
    print('Reading bond-currents among atoms (1-based!!!):')
    print(list2range_TBTblock(atoms))  # print 0-based idx as 1-based idx
    i_list_new, j_list_new, bc_list_new = [], [], []
    for i, j, bc in zip(i_list, j_list, bc_list):
        if i in atoms and j in atoms:
            i_list_new.append(i)
            j_list_new.append(j)
            bc_list_new.append(bc)
    return np.array(i_list_new), np.array(j_list_new), np.array(bc_list_new)

class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result

def plot_bondcurrents(f, idx_elec, only='+', E=0.0,  k='avg', zaxis=2, avg=True, scale='raw', xyz_origin=None,
    vmin=None, vmax=None, lw=5, log=False, adosmap=False, ADOSmin=None, ADOSmax=None, arrows=False, 
    lattice=False, ps=20, ados=False, atoms=None, out=None, ymin=None, ymax=None, xmin=None, xmax=None, 
    spsite=None, dpi=180, units='angstrom'):   
    """ Read bond currents from tbtrans output and plot them 
    
    Parameters
    ----------
    f : string
        TBT.nc file
    idx_elec : int
        the electrode of originating electrons
    only : {'+', '-', 'all'}
        If "+" is supplied only the positive orbital currents are used, for "-", 
        only the negative orbital currents are used, else return the sum of both. 
    E : float or int, 
        A float for energy in eV, int for explicit energy index 
    k : bool, int or array_like
        whether the returned bond current is k-averaged, 
        an explicit k-point or a selection of k-points
    zaxis : int
        index of out-of plane direction
    avg :  bool
        if "True", then it averages all currents coming from each atom and plots 
        them in a homogeneous map
        if "False" it plots ALL bond currents as lines originating from each atom
    scale : {'%' or 'raw'}
        wheter values are percent. Change vmin and vmax accordingly between 0% and 100%
    vmin : float
        min value in colormap. All data greater than this will be blue 
    vmax : float
        max value in colormap. All data greater than this will be yellow 
    lattice : bool
        whether you want xy coord of atoms plotted as black dots in the figure 
    ps : float
        size of these dots
    spsite : list of int
        special atoms in the lattice that you want to plot as red dots instead
    atoms : np.array or list
        list of atoms for which reading and plotting bondcurrents
    out : string
        name of final png figure 

    .....


    Returns
    -------
    bc, nc.E[idx_E], geom
    bc : bond currents
    nc.E[idx_E] : energy
    geom : geometry

    Notes
    -----
    - atoms must be 0-based
    - Be sure that atoms belong to a single plane (say, only graphene, no tip)
    """
    t = time.time()
    print('\n***** BOND-CURRENTS (2D map) *****\n')    
    nc = si.get_sile(f)
    elec = nc.elecs[idx_elec]
    
    # Read bond currents from TBT.nc file
    bc, energy, geom = read_bondcurrents(f, idx_elec, only, E, k)

    # If needed, select only selected atoms from bc_bg.
    bc_coo = bc.tocoo()
    i_list, j_list, bc_list = bc_coo.row, bc_coo.col, bc_coo.data
    if atoms is None:
        print('Reading bond-currents among all atoms in device region')
        atoms = nc.a_dev
        del bc_coo
    else:
        # Only choose atoms with positive indices
        atoms = atoms[atoms >= 0]
        select = np.logical_and(np.in1d(i_list, atoms), np.in1d(j_list, atoms))
        i_list, j_list, bc_list = i_list[select], j_list[select], bc_list[select]
        del bc_coo, select

    print('Number of bond-current entries: {}'.format(np.shape(bc_list)))
    print('MIN bc among selected atoms (from file) = {}'.format(np.min(bc_list)))
    print('MAX bc among selected atoms (from file) = {}'.format(np.max(bc_list)))
    #print('i\tj\tBond-current')
    #for i, j, bc in zip(i_list, j_list, bc_list):
    #    print('{}\t{}\t{}'.format(i, j, bc))

    # Plot
    import matplotlib.collections as collections
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cmap = cm.viridis

    if out is None:
        figname = 'BondCurrents_{}_E{}.png'.format(elec, energy)
    else:
        figname = '{}_{}_E{}.png'.format(out, elec, energy)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    if log:
        bc_list = np.log(bc_list+1)
        norm = LogNorm()
    else:
        norm=None

    if zaxis == 2:
        xaxis, yaxis = 0, 1
    elif zaxis == 0:
        xaxis, yaxis = 1, 2
    elif zaxis == 1:
        xaxis, yaxis = 0, 2

    if avg:
        # Plot bond currents as avg 2D map
        atoms_sort = np.sort(atoms)
        bc_avg = bc.sum(1).A.ravel()[atoms_sort]

        if scale is 'radial':
            _, r = geom.close_sc(xyz_origin, R=np.inf, idx=atoms_sort, ret_rij=True)
            bc_avg = np.multiply(bc_avg, r)

        if units == 'angstrom':
            unitstr = '$\AA$'
            x, y = geom.xyz[atoms_sort, xaxis], geom.xyz[atoms_sort, yaxis]
            a_mask = 1.54
        elif units == 'nm':
            unitstr = 'nm'
            x, y = .1*geom.xyz[atoms_sort, xaxis], .1*geom.xyz[atoms_sort, yaxis]
            a_mask = .1*1.54

        if scale is '%':
            if vmin is None:
                vmin = np.amin(bc_avg)*100/np.amax(bc_avg) 
            if vmax is None:
                vmax = 100
            vmin = vmin*np.amax(bc_avg)/100
            vmax = vmax*np.amax(bc_avg)/100
        else:
            if vmin is None:
                vmin = np.amin(bc_avg) 
            if vmax is None:
                vmax = np.amax(bc_avg)

        coords = np.column_stack((x, y))

        img, min, max = mask_interpolate(coords, bc_avg, oversampling=30, a=a_mask)
        # Note that we tell imshow to show the array created by mask_interpolate
        # faithfully and not to interpolate by itself another time.
        image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                          origin='lower', interpolation='none', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    else:
        if vmin is None:
            vmin = np.min(bc_list) 
        if vmax is None:
            vmax = np.max(bc_list)
        # Plot bond currents as half-segments
        start_list = zip(geom.xyz[i_list, xaxis], geom.xyz[i_list, yaxis])
        half_end_list = zip(.5*(geom.xyz[i_list, xaxis]+geom.xyz[j_list, xaxis]), 
            .5*(geom.xyz[i_list, yaxis]+geom.xyz[j_list, yaxis]))
        line_list = list(map(list, zip(start_list, half_end_list)))     # segments length = 1/2 bonds length
        linewidths = lw * bc_list / np.max(bc_list)     
        lattice_bonds = collections.LineCollection(line_list, cmap=cmap, linewidths=linewidths, norm=norm)
        lattice_bonds.set_array(bc_list/np.amax(bc_list))
        lattice_bonds.set_clim(vmin/np.amax(bc_list), vmax/np.amax(bc_list))
        ax.add_collection(lattice_bonds)
        image = lattice_bonds
    
    if lattice:
        if units == 'angstrom':
            x, y = geom.xyz[atoms, xaxis], geom.xyz[atoms, yaxis]
        if units == 'nm':
            x, y = .1*geom.xyz[atoms, xaxis], .1*geom.xyz[atoms, yaxis]
        ax.scatter(x, y, s=ps*2, marker='o', facecolors='None', linewidth=0.8, edgecolors='k')

    if spsite is not None:
        if units == 'angstrom':
            xs, ys = geom.xyz[spsite, xaxis], geom.xyz[spsite, yaxis]
        if units == 'nm':
            xs, ys = .1*geom.xyz[spsite, xaxis], .1*geom.xyz[spsite, yaxis]
        ax.scatter(xs, ys, s=ps*2, marker='x', color='red')

    ax.autoscale()
    ax.margins(0.)
    #ax.margins(0.05)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel('x ({})'.format(unitstr))
    plt.ylabel('y ({})'.format(unitstr))
    plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if avg:
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[vmin, vmax])
        if vmin == 0.:
            axcb.ax.set_yticklabels(['0', '$\geq$ {:.3e}'.format(vmax)])
        else:
            axcb.ax.set_yticklabels(['$\leq$ {:.3e}'.format(vmin), '$\geq$ {:.3e}'.format(vmax)])
        print('MIN bc among selected atoms (in final plot) = {}'.format(vmin))
        print('MAX bc among selected atoms (in final plot) = {}'.format(vmax))
    else:
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[vmin/np.amax(bc_list), vmax/np.amax(bc_list)])
        if scale is '%':
            vmin, vmax = vmin*100/max_newbc_bg, vmax*100/max_newbc_bg
            axcb.ax.set_yticklabels(['{:.1f} %'.format(vmin), '{:.1f} %'.format(vmax)])
            print('MIN bc among selected atoms (in final plot) = {:.1f} %'.format(vmin))
            print('MAX bc among selected atoms (in final plot) = {:.1f} %'.format(vmax))
        else:
            axcb.ax.set_yticklabels(['{:.3e}'.format(vmin), '{:.3e}'.format(vmax)])
            print('MIN bc among selected atoms (in final plot) = {}'.format(vmin))
            print('MAX bc among selected atoms (in final plot) = {}'.format(vmax))
    

    plt.savefig(figname, bbox_inches='tight', transparent=True, dpi=dpi)
    print('Successfully plotted to "{}"'.format(figname))
    print('Done in {} sec'.format(time.time() - t))

    return bc_list, vmin, vmax, i_list, j_list


def plot_bondcurrents_old(f, idx_elec, sum='+', E=0.0,  k='avg', f_bg=None, percent_bg=False, 
    vmin=None, vmax=None, lw=5, log=False, adosmap=False, ADOSmin=None, ADOSmax=None, arrows=False, 
    lattice=False, ps=20, ados=False, atoms=None, out=None, ymin=None, ymax=None, dpi=180):   
    """ 
    atoms must be 0-based
    """
    t = time.time()
    print('\n***** BOND-CURRENTS (2D map) *****\n')    
    nc = si.get_sile(f)
    elec = nc.elecs[idx_elec]
    
    # Read bond currents from TBT.nc file
    bc, energy, geom = read_bondcurrents(f, idx_elec, sum, E, k)

    # Read and subtract extra bc, if necessary
    if f_bg:
        #geom must be the same!!!
        print('\n - Subtracting bondcurrents from {}'.format(f_bg))
        bc_bg = read_bondcurrents(f_bg, idx_elec, sum, E, k)[0]
        if percent_bg:
            # If needed, select only selected atoms from bc_bg.
            # Then get max bc value to be used later
            if atoms is None:
                newbc_bg = bc_bg.tocoo().data
            else:
                if atoms[0] < 0:
                    # if atoms is a list of negative numbers, use all atoms except them
                    atoms = list(set(nc.a_dev).difference(set(-np.asarray(atoms))))
                newbc_bg = bc_sub(bc_bg, atoms)[2]
            max_newbc_bg = np.amax(newbc_bg)
        bc -= bc_bg
        bc.eliminate_zeros()

    # If needed, select only selected atoms from bc_bg.
    if atoms is None:
        print('Reading bond-currents among all atoms in device region')
        atoms = nc.a_dev
        i_list, j_list, bc_list = bc.tocoo().row, bc.tocoo().col, bc.tocoo().data
    else:
        if atoms[0] < 0:
            # if atoms is a list of negative numbers, use all atoms except them
            atoms = list(set(nc.a_dev).difference(set(-np.asarray(atoms))))
        i_list, j_list, bc_list = bc_sub(bc, atoms)

    print('Number of bond-current entries: {}'.format(np.shape(bc_list)))
    print('MIN bc among selected atoms (from file) = {}'.format(np.min(bc_list)))
    print('MAX bc among selected atoms (from file) = {}'.format(np.max(bc_list)))
    #print('i\tj\tBond-current')
    #for i, j, bc in zip(i_list, j_list, bc_list):
    #    print('{}\t{}\t{}'.format(i, j, bc))


    # Plot
    import matplotlib.collections as collections
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cmap = cm.viridis

    if out is None:
        figname = 'BondCurrents_{}_E{}.png'.format(elec, energy)
    else:
        figname = '{}_{}_E{}.png'.format(out, elec, energy)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot bond currents as half segments starting from the atoms
    start_list = zip(geom.xyz[i_list, 0], geom.xyz[i_list, 1])
    half_end_list = zip(.5*(geom.xyz[i_list, 0]+geom.xyz[j_list, 0]), 
        .5*(geom.xyz[i_list, 1]+geom.xyz[j_list, 1]))
    line_list = list(map(list, zip(start_list, half_end_list)))     # segments length = 1/2 bonds length
    #end_list = zip(geom.xyz[j_list, 0], geom.xyz[j_list, 1])
    #line_list = list(map(list, zip(start_list, end_list)))     # segments length = bonds length
    
    if log:
        bc_list = np.log(bc_list+1)
        norm = LogNorm()
    else:
        norm=None

    if ados:
        # Plot ADOS
        ADOS = read_ADOS(f, idx_elec, E, k, atoms)[2]
        x, y = geom.xyz[atoms, 0], geom.xyz[atoms, 1]
        if ADOSmin is None:
            ADOSmin = np.min(ADOS)
        if ADOSmax is None:
            ADOSmax = np.max(ADOS)

        if adosmap:
            coords = np.column_stack((x, y))
            values = np.array(ADOS)
            img, min, max = mask_interpolate(coords, values, oversampling=15)
            # Note that we tell imshow to show the array created by mask_interpolate
            # faithfully and not to interpolate by itself another time.
            image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                              origin='lower', interpolation='none', cmap='viridis',
                              vmin=ADOSmin, vmax=ADOSmax)
        else:
            colors = ADOS
            area = 300 # * ADOS / np.max(ADOS)
            image = ax.scatter(x, y, c=colors, s=area, marker='o', edgecolors='None', 
                cmap=cmap, norm=norm)
            image.set_clim(ADOSmin, ADOSmax)
            image.set_array(ADOS)

        # Plot bond-currents        
        if arrows:  # NOT WORKING
            lattice_bonds = ax.quiver(np.array(start_list[0]), np.array(start_list[1]), 
                np.subtract(np.array(half_end_list[0]), np.array(start_list[0])), 
                np.subtract(np.array(half_end_list[1]), np.array(start_list[1])),
                angles='xy', scale_units='xy', scale=1) 
        else:
            if vmin is None:
                vmin = np.min(bc_list)/ np.max(bc_list) 
            if vmax is None:
                vmax = np.max(bc_list)/ np.max(bc_list)
            linewidths = lw * bc_list / np.max(bc_list)     
            idx_lwmax = np.where(vmax < bc_list / np.max(bc_list))[0]
            linewidths[idx_lwmax] = lw * np.max(bc_list) / np.max(bc_list)
            idx_lwmin = np.where(bc_list / np.max(bc_list) < vmin)[0]
            linewidths[idx_lwmin] = lw * np.min(bc_list) / np.max(bc_list)
            
            lattice_bonds = collections.LineCollection(line_list, colors='k', 
                linewidths=linewidths)
            ax.add_collection(lattice_bonds)
    else:
        if vmin is None:
            vmin = np.min(bc_list) #/ np.amax(bc_list) 
        if vmax is None:
            vmax = np.max(bc_list) #/ np.amax(bc_list)
        linewidths = lw * bc_list / np.max(bc_list)     
        #linewidths = 4     
        #idx_lwmax = np.where(vmax < bc_list / np.amax(bc_list))[0]
        #linewidths[idx_lwmax] = 5 * np.amax(bc_list) / np.amax(bc_list)
        #idx_lwmin = np.where(bc_list / np.max(bc_list) < vmin)[0]
        #linewidths[idx_lwmin] = 5 * np.amin(bc_list) / np.amax(bc_list)       
        
        #colors = list2colors(bc_list/np.amax(bc_list), cmap, vmin/np.amax(bc_list), vmax/np.amax(bc_list))
        #colors = list2colors(bc_list/np.amax(bc_list), cmap, vmin, vmax)
        
        #lattice_bonds = collections.LineCollection(line_list, colors=colors, 
        #    cmap=cmap, linewidths=linewidths, norm=norm)
        lattice_bonds = collections.LineCollection(line_list, 
            cmap=cmap, linewidths=linewidths, norm=norm)
        lattice_bonds.set_array(bc_list/np.amax(bc_list))
        lattice_bonds.set_clim(vmin/np.amax(bc_list), vmax/np.amax(bc_list))
        ax.add_collection(lattice_bonds)
        image = lattice_bonds
    
    if lattice:
        #xl, yl = geom.xyz[:, 0], geom.xyz[:, 1]
        #ax.scatter(xl, yl, s=ps, c='w', marker='o', edgecolors='k')
        x, y = geom.xyz[atoms, 0], geom.xyz[atoms, 1]
        ax.scatter(x, y, s=ps*2, marker='o', facecolors='None', linewidth=0.8, edgecolors='k')

    ax.autoscale()
    ax.margins(0.05)
    plt.ylim(ymin, ymax)
    plt.xlabel('$x (\AA)$')
    plt.ylabel('$y (\AA)$')
    plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if ados:
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[ADOSmin, ADOSmax])
    else:
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[vmin/np.amax(bc_list), vmax/np.amax(bc_list)])
        if percent_bg:
            vmin, vmax = vmin*100/max_newbc_bg, vmax*100/max_newbc_bg
            axcb.ax.set_yticklabels(['{:.1f} %'.format(vmin), '{:.1f} %'.format(vmax)])
            print('MIN bc among selected atoms (in final plot) = {:.1f} %'.format(vmin))
            print('MAX bc among selected atoms (in final plot) = {:.1f} %'.format(vmax))
        else:
            axcb.ax.set_yticklabels(['{:.3e}'.format(vmin), '{:.3e}'.format(vmax)])
            print('MIN bc among selected atoms (in final plot) = {}'.format(vmin))
            print('MAX bc among selected atoms (in final plot) = {}'.format(vmax))
    

    plt.savefig(figname, bbox_inches='tight', dpi=dpi)
    print('Successfully plotted to "{}"'.format(figname))
    print('Done in {} sec'.format(time.time() - t))

    return bc_list, vmin, vmax, i_list, j_list



def read_ADOS(f, idx_elec, E=0.0, k='avg', atoms=None, sum=True):
    
    print('\nReading: {}'.format(f))
    nc = si.get_sile(f)
    na, na_dev = nc.na, nc.na_dev
    print('Total number of atoms: {}'.format(na))
    print('Number of atoms in the device region: {}'.format(na_dev))
    geom = nc.geom

    # if atoms is a list of negative numbers, use all atoms except them

    if atoms and (atoms[0] < 0):
        atoms = list(set(nc.a_dev).difference(set(-np.asarray(atoms))))     # this is 0-based

    if atoms is None:
        print('Reading ADOS for all atoms in device region')
    else:
        print('Reading ADOS for atoms (1-based):')
        print(list2range_TBTblock(atoms))  # print 0-based idx as 1-based idx

    elec = nc.elecs[idx_elec]
    print('ADOS from electrode: {}'.format(elec))

    # Check 'k' argument
    if k == 'avg':
        avg = True
    elif k == 'Gamma':
        kpts = nc.kpt
        idx_gamma = np.where(np.sum(np.abs(kpts), axis=1) == 0.)[0]
        if (kpts[idx_gamma] != np.zeros((1, 3))).any(axis=1):
            print('\nThe selected k-point is not Gamma!\n')
            exit(0)
        else:
            print('You have selected the Gamma point!')
        avg = idx_gamma # Index of Gamma point in nc.kpt
    else:
        print('\nInvalid `k` argument: please keep the default `avg` or use `Gamma`!\n')
        exit(0)

    idx_E = nc.Eindex(E)
    print('Extracting ADOS at energy: {} eV'.format(nc.E[idx_E]))
    ADOS_list = nc.ADOS(elec=elec, E=idx_E, kavg=0, atom=atoms, sum=sum).T
    print('Shape of ADOS: {}'.format(np.shape(ADOS_list)))
    
    if atoms is None:
        atoms = nc.a_dev

    # for a,ados in zip(atoms, ADOS_list):
    #     print(a, ados)

    return (geom, atoms, ADOS_list, nc.E[idx_E])


def read_BDOS(f, idx_elec, E=0.0, k='avg'):
    
    print('\nReading: {}'.format(f))
    nc = si.get_sile(f)

    elec = nc.elecs[idx_elec]
    print('First electrode is: {}'.format(elec))

    # Check 'k' argument
    if k == 'avg':
        avg = True
    elif k == 'Gamma':
        kpts = nc.kpt
        idx_gamma = np.where(np.sum(np.abs(kpts), axis=1) == 0.)[0]
        if (kpts[idx_gamma] != np.zeros((1, 3))).any(axis=1):
            print('\nThe selected k-point is not Gamma!\n')
            exit(0)
        else:
            print('You have selected the Gamma point!')
        avg = idx_gamma # Index of Gamma point in nc.kpt
    else:
        print('\nInvalid `k` argument: please keep the default `avg` or use `Gamma`!\n')
        exit(0)

    idx_E = nc.Eindex(E)
    print('Extracting BDOS at energy point: {} eV'.format(nc.E[idx_E]))
    BDOS = nc.BDOS(elec, idx_E, avg).T   # len(rows) = nc.na_dev, len(columns) = 1 (or nc.nE if E flag is not specified) 
    print('Shape of BDOS: {}'.format(np.shape(BDOS)))

    return (BDOS, nc.E[idx_E])



# Adapted from KWANT
def mask_interpolate(coords, values, a=None, method='nearest', oversampling=300):
    """Interpolate a scalar function in vicinity of given points.

    Create a masked array corresponding to interpolated values of the function
    at points lying not further than a certain distance from the original
    data points provided.

    Parameters
    ----------
    coords : np.ndarray
        An array with site coordinates.
    values : np.ndarray
        An array with the values from which the interpolation should be built.
    a : float, optional
        Reference length.  If not given, it is determined as a typical
        nearest neighbor distance.
    method : string, optional
        Passed to ``scipy.interpolate.griddata``: "nearest" (default), "linear",
        or "cubic"
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.

    Returns
    -------
    array : 2d NumPy array
        The interpolated values.
    min, max : vectors
        The real-space coordinates of the two extreme ([0, 0] and [-1, -1])
        points of ``array``.

    Notes
    -----
    - `min` and `max` are chosen such that when plotting a system on a square
      lattice and `oversampling` is set to an odd integer, each site will lie
      exactly at the center of a pixel of the output array.

    - When plotting a system on a square lattice and `method` is "nearest", it
      makes sense to set `oversampling` to ``1``.  Then, each site will
      correspond to exactly one pixel in the resulting array.
    """

    from scipy import spatial, interpolate
    import warnings

    # Build the bounding box.
    cmin, cmax = coords.min(0), coords.max(0)

    tree = spatial.cKDTree(coords)

    points = coords[np.random.randint(len(coords), size=10)]
    min_dist = np.min(tree.query(points, 2)[0][:, 1])
    if min_dist < 1e-6 * np.linalg.norm(cmax - cmin):
        warnings.warn("Some sites have nearly coinciding positions, "
                      "interpolation may be confusing.",
                      RuntimeWarning)

    if coords.shape[1] != 2:
        print('Only 2D systems can be plotted this way.')
        exit()

    if a is None:
        a = min_dist

    if a < 1e-6 * np.linalg.norm(cmax - cmin):
        print("The reference distance a is too small.")
        exit()

    if len(coords) != len(values):
        print("The number of sites doesn't match the number of"
        "provided values.")
        exit()

    shape = (((cmax - cmin) / a + 1) * oversampling).round()
    delta = 0.5 * (oversampling - 1) * a / oversampling
    cmin -= delta
    cmax += delta
    dims = tuple(slice(cmin[i], cmax[i], 1j * shape[i]) for i in range(len(cmin)))
    grid = tuple(np.ogrid[dims])
    img = interpolate.griddata(coords, values, grid, method)
    mask = np.mgrid[dims].reshape(len(cmin), -1).T
    # The numerical values in the following line are optimized for the common
    # case of a square lattice:
    # * 0.99 makes sure that non-masked pixels and sites correspond 1-by-1 to
    #   each other when oversampling == 1.
    # * 0.4 (which is just below sqrt(2) - 1) makes tree.query() exact.
    mask = tree.query(mask, eps=0.4)[0] > 0.99 * a

    return np.ma.masked_array(img, mask), cmin, cmax



def plot_ADOS(f, idx_elec, E=0.0, k='avg', sum=False, vmin=None, vmax=None, log=False, map=False, 
    lattice=False, ps=20, atoms=None, out=None, zaxis=2, spsite=None, scale='raw', dpi=180):

    t = time.time()
    print('\n***** ADOS (2D map) *****\n')    

    if zaxis == 2:
        xaxis, yaxis = 0, 1
    elif zaxis == 0:
        xaxis, yaxis = 1, 2
    elif zaxis == 1:
        xaxis, yaxis = 0, 2

    nc = si.get_sile(f)
    elec = nc.elecs[idx_elec]

    # Read ADOS from TBT.nc file
    geom, ai_list, ADOS, energy = read_ADOS(f, idx_elec, E, k, atoms, sum=sum)

        
    # Plot
    import matplotlib.collections as collections
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if out is None:
        figname = 'ados_{}_E{}.png'.format(elec, energy)
    else:
        figname = '{}_{}_E{}.png'.format(out, elec, energy)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    x, y = geom.xyz[ai_list, xaxis], geom.xyz[ai_list, yaxis]

    if log:
        ADOS = np.log(ADOS+1)

    if scale is '%':
        if vmin is None:
            vmin = np.amin(ADOS)*100/np.amax(ADOS) 
        if vmax is None:
            vmax = 100
        vmin = vmin*np.amax(ADOS)/100
        vmax = vmax*np.amax(ADOS)/100
    else:
        if vmin is None:
            vmin = np.amin(ADOS) 
        if vmax is None:
            vmax = np.amax(ADOS)

    if map:
        coords = np.column_stack((x, y))
        values = np.array(ADOS)
        img, min, max = mask_interpolate(coords, values, oversampling=30)
        # Note that we tell imshow to show the array created by mask_interpolate
        # faithfully and not to interpolate by itself another time.
        image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                          origin='lower', interpolation='none', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    else:
        colors = ADOS
        area = 300 # * ADOS / np.max(ADOS)
        if log:
            image = ax.scatter(x, y, c=colors, s=area, marker='o', edgecolors='None', cmap='viridis', norm=LogNorm())
        else:
            image = ax.scatter(x, y, c=colors, s=area, marker='o', edgecolors='None', cmap='viridis')
        image.set_clim(vmin, vmax)
        image.set_array(ADOS)

    if lattice:
        xl, yl = geom.xyz[atoms, xaxis], geom.xyz[atoms, yaxis]
        ax.scatter(xl, yl, s=ps*2, c='w', marker='o', edgecolors='k')
        ax.scatter(x, y, s=ps*2, c='k', marker='o', edgecolors='None')

    if spsite is not None:
        xs, ys = geom.xyz[spsite, xaxis], geom.xyz[spsite, yaxis]
        ax.scatter(xs, ys, s=ps*2, marker='x', color='red')

    ax.autoscale()
    ax.margins(0.)
    plt.xlabel('$x (\AA)$')
    plt.ylabel('$y (\AA)$')
    plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if scale is '%':
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[vmin/ADOS.max(), vmax/ADOS.max()])
        vmin, vmax = vmin*100/ADOS.max(), vmax*100/ADOS.max()
        axcb.ax.set_yticklabels(['{:.1f} %'.format(vmin), '{:.1f} %'.format(vmax)])
        print('MIN bc among selected atoms (in final plot) = {:.1f} %'.format(vmin))
        print('MAX bc among selected atoms (in final plot) = {:.1f} %'.format(vmax))
    else:
        axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[vmin, vmax])
        axcb.ax.set_yticklabels(['{:.3e}'.format(vmin), '{:.3e}'.format(vmax)])
        print('MIN bc among selected atoms (in final plot) = {}'.format(vmin))
        print('MAX bc among selected atoms (in final plot) = {}'.format(vmax))
    plt.savefig(figname, bbox_inches='tight', transparent=True, dpi=dpi)
    print('Successfully plotted to "{}"'.format(figname))
    print('Done in {} sec'.format(time.time() - t))




def plot_ADOS_stripe(geom, ai_list, ADOS, x, y, i_list, figname='ADOS_x.png', E=0.0, 
    vmin=None, vmax=None, dpi=180):
    
    import matplotlib.collections as collections
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print('Plotting...')

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    vmin, vmax = vmin, vmax
    if vmin is None:
        vmin = np.min(ADOS)
    if vmax is None:
        vmax = np.max(ADOS)
    colors = ADOS
    area = 15
    image = ax.scatter(x, y, c=colors, s=area, marker='o', edgecolors='None', cmap='viridis')
    # Uncomment below to highlight specific atoms in lattice
    ax.scatter(x[i_list], y[i_list], c='w', marker='o', s=8)
    image.set_clim(vmin, vmax)
    image.set_array(ADOS)

    ax.autoscale()
    ax.margins(0.1)
    plt.xlabel('$x (\AA)$')
    plt.ylabel('$y (\AA)$')
    plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axcb = plt.colorbar(image, cax=cax, format='%1.2f', ticks=[vmin, vmax])

    plt.savefig(figname, bbox_inches='tight', transparent=True, dpi=dpi)
    print('Successfully plotted to "{}"'.format(figname))


def plot_collimation_factor(X, Y, const=None, const2=None, E=0.0):

    colors = ['k', 'r', 'b', 'm', 'g', 'y', 'c']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 'v', 's', '^', 'p', 'd', 'x']
    labels = [r'$\frac{\sum_{W-src}{ADOS}}{\sum_{W-gr}{ADOS}}$', r'$\frac{\sum_{W-src}{ADOS}}{\sum_{W-src}{BDOS}}$']

    plt.figure()
    
    if not isinstance(Y, (tuple, list)):
        Y = [Y]
    
    for i, y in enumerate(Y):
        plt.plot(X,y,'%s%s%s' %(markers[i], linestyles[i+3], colors[i+1]), markersize=6, label=labels[i])
    
    plt.ylim(0, 1.09)
    #plt.ylabel('$B/A$')
    plt.xlabel('$<y> (\AA)$')
    plt.legend(loc=1, fancybox=True)

    plt.grid(True, color='0.75')

    plt.axhline(const, linestyle='--', color=colors[1])
    plt.annotate('isotropic limit', color=colors[1], xy=(plt.xlim()[0]+0.3, const+0.01), xytext=(plt.xlim()[0]+0.3, const+0.01))

    plt.axhline(const2, linestyle='--', color=colors[2])
    plt.annotate('backscattering limit', color=colors[2], xy=(plt.xlim()[0]+0.3, const2-0.5), xytext=(plt.xlim()[0]+0.3, const2-0.5))


    plt.gcf()
    plt.savefig('BoA_E{:.2f}.png'.format(E), bbox_inches='tight', transparent=True, dpi=dpi)




def plot_ADOS_x(f, idx_elec, E=0.0, k='avg', vmin=None, vmax=None, log=False):

    t = time.time()
    # Read ADOS from TBT.nc file
    geom, ai_list, ADOS, energy = read_ADOS(f, idx_elec, E, k)
    x, y = geom.xyz[ai_list, 0], geom.xyz[ai_list, 1]
    #for ii,xx,yy in zip(range(len(y)),x,y):
    #    print(ii, xx, yy)

    C = si.Atom(6, R=[1.43])
    g = si.geom.graphene(1.42, atom=C, orthogonal=True)

    # # Number of line cuts along transport direction in the device
    # n_lines = np.int(geom.cell[1,1]/g.cell[1,1])
    # print('Number of lines: {}'.format(n_lines))
    # xmin = np.min(x[np.where((y+0.001)/g.cell[1,1] < 1.8)[0]]) -0.001
    # xmax = np.max(x[np.where((y+0.001)/g.cell[1,1] < 1.8)[0]]) +0.001
    # print('B_xmin = {}, B_xmax = {}'.format(xmin, xmax))
    # # Define line cuts along transport direction in the device (lists of atom indices)
    # i_list_A = [[] for _ in range(n_lines)]
    # i_list_B = [[] for _ in range(n_lines)]
    # ym_list = []
    # for i_line in range(n_lines):
    #     ymin, ymax = (i_line-0.01)*g.cell[1,1], (i_line+.8)*g.cell[1,1]
    #     ym_list.append(.5*(ymin + ymax))
    #     for i in range(len(y)):
    #         if ymin < y[i] and y[i] < ymax:
    #             i_list_A[i_line].append(i)
    #             if xmin < x[i] and x[i] < xmax:
    #                 i_list_B[i_line].append(i)
    #     print('\nLine #{}:\t{} < y < {}\tna = {}'.format(i_line, i_line*g.cell[1,1], (i_line+.8)*g.cell[1,1], len(i_list_A[i_line])))
    #     print('i_list_A:\n{}'.format(i_list_A[i_line]))    
    #     print('i_list_B:\n{}'.format(i_list_B[i_line]))    

    xmin = np.min(x[np.where((y+0.001)/g.cell[1,1] < 1.8)[0]]) -0.001
    xmax = np.max(x[np.where((y+0.001)/g.cell[1,1] < 1.8)[0]]) +0.001
    print('B_xmin = {}, B_xmax = {}'.format(xmin, xmax))

    line_idx_array = np.floor((y+0.001)/g.cell[1,1]).astype(int) -1   # -1 is to keep line_idx 0-based
    idx_sort = np.argsort(line_idx_array)
    sorted_line_idx_array = line_idx_array[idx_sort]
    lines, idx_start = np.unique(sorted_line_idx_array, return_index=True)
    i_list_A = np.split(idx_sort, idx_start[1:])

    i_list_B = [[]] * len(i_list_A)
    idx_B = np.where(np.logical_and(xmin < x, x < xmax))[0]
    for i, list_A in enumerate(i_list_A):
        mask = np.in1d(list_A, idx_B)
        i_list_B[i] = list_A[mask]

    y_list_A = np.split(y[idx_sort], idx_start[1:])
    y_vals = (lines+.5) * g.cell[1,1]
    for line, ym, A, B in zip(lines, y_vals, i_list_A, i_list_B):
        print('line {} (<y> = {:.2f} Ang) --> {} atoms'.format(line, ym, len(A)))
        #print('A (len={}) = {}'.format(len(A), A))
        #print('B (len={}) = {}'.format(len(B), B))

    # Check atoms selected for a certain stripe
    #plot_ADOS_stripe(geom, ai_list, ADOS, x, y, i_list_A[10], 'ADOS_x_E{}_A.png'.format(E), E, vmin=vmin, vmax=vmax)
    #plot_ADOS_stripe(geom, ai_list, ADOS, x, y, i_list_B[10], 'ADOS_x_E{}_B.png'.format(E), E, vmin=vmin, vmax=vmax)

    # Calculate collimation factor
    A = [np.sum(ADOS[els]) for els in i_list_A]
    B = [np.sum(ADOS[els]) for els in i_list_B]
    print('\nA = sum(ADOS)_w-gr =\n{}'.format(A))
    print('\nB = sum(ADOS)_w-src =\n{}'.format(B))
    BoA = map(truediv, B, A)
    print('\nB/A = sum(ADOS)_w-src / sum(ADOS)_w-gr =\n{}'.format(BoA))

    # Calculate isotropic limit
    BoA_iso = len(i_list_B[-5])/len(i_list_A[-5])
    print('\nIsotropic limit = N_src / N_gr = {}'.format(BoA_iso))

    # Calculate the damping factor f
    A_BDOS = read_BDOS(f, idx_elec, E, k)[0].sum()
    BoA_f = np.multiply(BoA, A)/A_BDOS
    print('\nDamped collimation factor = (sum(ADOS)_w-gr / sum(BDOS)_src) * (B/A) =\n{}'.format(BoA_f))

    # Calculate backscattering limit for f*B/A
    nc = si.get_sile(f)
    idx_E = nc.Eindex(energy)
    tr = nc.transmission(nc.elecs[0], nc.elecs[1])[idx_E]
    tr_source = nc.transmission_bulk(nc.elecs[0])[idx_E]
    BoA_back = BoA_iso * tr/tr_source
    print('\nBackscattering limit = Isotropic limit * (T({0} eV) / T_src({0} eV)) = {1}'.format(energy, BoA_back))

    # Plot collimation factor
    plot_collimation_factor(y_vals, [BoA, BoA_f], BoA_iso, BoA_back, energy)
    print('Done in {} sec'.format(time.time() - t))

