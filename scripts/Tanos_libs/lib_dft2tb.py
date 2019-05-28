from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import groupby

import sisl as si

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

def makearea(TSHS, shape='Cuboid', z_area=None, ext_offset=None, center=None, 
    thickness=None, zaxis=2, atoms=None, segment_dir=None):
    """
    In frame cases, we are going to define an outer area 'area_ext' 
    and an inner area 'area_R2', and we are going to subtract them.
    The resulting area is called 'Delta'
    TSHS:           Hamiltonian or geometry object
    shape:          shape (check sisl doc)
                    ['Cube', 'Cuboid', 'Ellipsoid', 'Sphere', 'Segment']
    z_area:         coordinate at which the area should be defined [Angstrom] 
                    Default is along direction 2. Otherwise, change z_axis
    ext_offset:     [3x1 array] Use this to set an offset along any direction in area_ext
    center:         center of area plane in TSHS
    thickness:      thickness of area [Angstrom]
    zaxis:          [0,1,2] direction perpendicular to area plane
    atoms:          list of atomic indices to filter the atoms inside Delta
    segment_dir:    [0,1,2] direction along the smallest side of the segment 
    
    Note:
    - it works for orthogonal cells! Needs some adjustments to be general... 
    """
    # z coordinate of area plane 
    if z_area is None:
        print('\n\nPlease provide a value for z_area in makearea routine')
        exit(1)
    # Center of area plane in TSHS 
    cellcenter = TSHS.center(atom=(TSHS.xyz[:,zaxis] == z_area).nonzero()[0])
    if center is None:
        center = cellcenter
    center = np.asarray(center)   # make sure it's an array
    # Thickness in Ang
    if thickness is None:
        thickness = 6. # Ang
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
        
        # Internal boundary
        area_R2 = area_ext.expand(-thickness)
        # Disjuction composite shape
        Delta = area_ext - area_R2
        # Atoms within Delta and internal boundary
        a_Delta = Delta.within_index(TSHS.xyz)
        a_int = area_R2.within_index(TSHS.xyz)
        if atoms is not None:
            a_Delta = a_Delta[np.in1d(a_Delta, atoms)]
        # Check
        v = TSHS.geom.copy(); v.atom[a_Delta] = si.Atom(8, R=[1.43]); v.write('a_Delta.xyz')
        return a_Delta, a_int, Delta, area_ext, area_R2


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


def dagger(M):
    return np.conjugate(np.transpose(M))


def couplingMat(M, iosel1, iosel2, format='array'):
    iosel1.shape = (-1, 1)
    iosel2.shape = (1, -1)
    Mp = M[iosel1, iosel2]
    if format == 'csr':
        return sp.sparse.csr_matrix(Mp)
    elif format == 'array':
        return Mp


def pruneMat(M, iosel, format='array'):
    """
    M:      sparse matrix in csr format
    iosel:  list of orbitals on which M has to be pruned
    """    
    n_s = M.shape[1] // M.shape[0]
    iosel.shape = (-1, 1)
    if n_s != 1:
        iosel2 = np.arange(n_s) * M.shape[0]
        a = np.repeat(iosel.ravel(), n_s)
        iosel2 = (a.reshape(-1, n_s) + iosel2.reshape(1, -1)).ravel()
        iosel2 = np.sort(iosel2).reshape(1, -1)
    else:
        iosel2 = iosel.reshape(1, -1)
    Mp = M[iosel, iosel2]
    if format == 'csr':
        return sp.sparse.csr_matrix(Mp)
    elif format == 'array':
        return Mp


def rearrange(HS, list, where='end'):
    if 'end' in where:
        # Remove selection from the current TB geom and append it at the end of its atoms list
        # The selected atoms have new indices from -len(a_dSE_host) to -1
        full = np.arange(HS.na)
        full = np.delete(full, list)
        full = np.concatenate((full, list))
        # New HS
        HS_new = HS.sub(full) 
        # List indices in new HS
        list_new = np.arange(HS.na-len(list), HS.na)

    return list_new, HS_new


def map_xyz(A, B, area_R_A, a_R_A=None, center_B=None, pos_B=None, area_for_buffer=None, 
    tol=None):
    ### FRAME
    print('\nMapping from geometry A to geometry B')
    if a_R_A is None:
        # Recover atoms in R_A region of model A
        a_R_A = area_R_A.within_index(A.xyz)
    
    # Find the set R_B of unique corresponding atoms in model B 
    area_R_B = area_R_A.copy()

    if pos_B is not None:
        vector = pos_B
        B_translated = B.translate(-vector) 
    else: 
        if center_B is None: 
            center_B = B.center(what='xyz')
        vector = center_B - area_R_B.center
        B_translated = B.translate(-vector) 
    a_R_B = area_R_B.within_index(B_translated.xyz)

    R_A = A.sub(a_R_A)
    R_B = B.sub(a_R_B)
    v1, v2 = np.amin(R_A.xyz, axis=0), np.amin(R_B.xyz, axis=0)
    xyz_B_shifted = R_B.xyz - v2[None,:]
    xyz_A_shifted = R_A.xyz - v1[None,:]

    if tol is None:
        tol = [0.01, 0.01, 0.01]
    tol = np.asarray(tol)
    print('Tolerance along x,y,z is set to {}'.format(tol))
    

    # Check if a_R_A is included in a_R_B and if yes, try to solve the problem
    # Following `https://stackoverflow.com/questions/33513204/finding-intersection-of-two-matrices-in-python-within-a-tolerance`
    # Get absolute differences between xyz_B_shifted and xyz_A_shifted keeping their columns aligned
    diffs = np.abs(xyz_B_shifted.reshape(-1, 1, 3) - xyz_A_shifted.reshape(1, -1, 3))
    # Compare each row with the triplet from `tol`.
    # Get mask of all matching rows and finally get the matching indices
    x1, x2 = np.nonzero((diffs < tol.reshape(1, 1, -1)).all(2))
    idx_swap = np.argsort(x2[:len(x1)])
    x1_reorder = x1[idx_swap]

    # CHECK
    if len(x1) == len(a_R_A):
        if len(a_R_B) > len(a_R_A):
            print('\nWARNING: len(a_R_A) = {} is not equal to len(a_R_B) = {}'.format(len(a_R_A), len(a_R_B)))
            print('But since a_R_A is entirely contained in a_R_B, I will fix it by removing the extra atoms')
        print('\n OK! The coordinates of the mapped atoms in the two geometries match \
within the desired tolerance! ;)')
    else:
        print('\nWARNING: len(a_R_A) = {} is not equal to len(a_R_B) = {}'.format(len(a_R_A), len(a_R_B)))
        print('\n STOOOOOP: not all elements of a_R_A are in a_R_B')
        print('   Check `a_R_B_not_matching.xyz` vs `a_R_A.xyz` and \
try to set `pos_B` to `B.center(what=\'xyz\') + [dx,dy,dz]`. Or increase the tolerance')
        v = B.geom.copy(); v.atom[a_R_B] = si.Atom(8, R=[1.44]); v.write('a_R_B_not_matching.xyz')
        exit(1)

    a_R_B, a_R_A = a_R_B[x1_reorder], a_R_A[x2]

    # Further CHECK, just to be sure
    if not np.allclose(xyz_B_shifted[x1], xyz_A_shifted[x2], rtol=np.amax(tol), atol=np.amax(tol)):
        print('\n STOOOOOP: The coordinates of the mapped atoms in the two geometries don\'t match \
within the tolerance!!!!')
        print('   Check `a_R_B_not_matching.xyz` vs `a_R_A.xyz` and \
try to set `pos_B` to `B.center(what=\'xyz\') + [dx,dy,dz]`. Or increase the tolerance')
        v = B.geom.copy(); v.atom[a_R_B] = si.Atom(8, R=[1.44]); v.write('a_R_B_not_matching.xyz')
        exit(1)

    print(' Max deviation (Ang) =', np.amax(xyz_A_shifted[x2]-xyz_B_shifted[x1]))

    # WARNING: we are about to rearrange the atoms in the host geometry!!!
    a_R_B_rearranged, new_B = rearrange(B, a_R_B, where='end')
    print("\nSelected atoms mapped into host geometry, after rearrangement\n\
at the end of the coordinates list (1-based): {}\n{}".format(len(a_R_B_rearranged), list2range_TBTblock(a_R_B_rearranged)))

    # Find and print buffer atoms
    if area_for_buffer is not None:
        # NB: that cuboids are always independent from the sorting in the host geometry
        area_B = area_for_buffer.copy()
        if center_B is not None:
            area_B.set_center(center_B)
        almostbuffer = area_B.within_index(new_B.xyz)
        buffer_i = np.in1d(almostbuffer, a_R_B_rearranged, assume_unique=True, invert=True)
        buffer = almostbuffer[buffer_i]
        return a_R_B_rearranged, new_B, buffer
    else:
        return a_R_B_rearranged, new_B


def in2out_frame_PBCoff(TSHS, a_R1, eta_value, energies, TBT, 
    HS_host, orb_idx=None, pos_dSE=None, area_R1=None, area_R2=None, area_for_buffer=None,  
    TBTSE=None, useCAP=None, spin=0, tol=None, EfromTBT=True):
    """
    TSHS:                   TSHS from perturbed DFT system
    a_R1:                idx atoms in sub-region A of perturbed DFT system (e.g. frame)
                            \Sigma will live on these atoms
    eta_value:              imaginary part in Green's function
    energies:               energy in eV for which \Sigma should be computed (closest E in TBT will be used )
    TBT:                    *.TBT.nc (or *.TBT.SE.nc) from a TBtrans calc. where TBT.HS = TSHS 
    HS_host:                host (H, S) model (e.g. a large TB model of unperturbed system). 
                            Coordinates of atoms "a_R1" in TSHS will be mapped into this new model.
                            Atomic order will be adjusted so that mapped atoms will be consecutive and at the end of the list   
    orb_idx (=None):          idx of orbital per atom to be extracted from TSHS, in case HS_host has a reduced basis size
    pos_dSE (=0):           center of region where \Sigma atoms should be placed in HS_host 
    area_R1 (=None):     si.shape.Cuboid object used to select "a_R1" atoms in TSHS
    area_R2 (=None):              internal si.shape.Cuboid object used to construct area_R1 in TSHS
    area_for_buffer (=None):       external si.shape.Cuboid object used to used to construct area_R1 in TSHS
    TBTSE (=None):          *TBT.SE.nc file of self-energy enclosed by the atoms "a_R1" in TSHS (e.g., tip) 
    useCAP (=None):         use 'left+right+top+bottom' to set complex absorbing potential in all in-plane directions
    
    Important output files:
    "HS_DEV.nc":        HS file for TBtrans (to be used with "TBT.HS" flag)
                        this Hamiltonian is identical to HS_host, but it has no PBC 
                        and \Sigma projected atoms are moved to the end of the atom list   
    "SE_i.delta.nc":    \Delta \Sigma file for TBtrans (to be used as "TBT.dSE" flag)
                        it will contain \Sigma from k-averaged Green's function from TSHS,
                        projected on the atoms "a_R1" equivalent atoms of HS_host
    "SE_i.TBTGF":       Green's function file for usage as electrode in TBtrans 
                        (to be used with "GF" flag in the electrode block for \Sigma)
                        it will contain S^{noPBC}*e - H^{noPBC} - \Sigma from TSHS k-averaged Green's function,
                        projected on the atoms "a_R1" equivalent atoms of HS_host    
    "HS_SE_i.nc":       electrode HS file for usage of TBTGF as electrode in TBtrans
                        (to be used with "HS" flag in the electrode block for \Sigma)
    
    NOTES:
    - works for 2D carbon systems (as of now)
    """

    """ 
    Let's first find the orbitals inside R1 and R2
    """ 

    # Indices of atoms in device region
    a_dev = TBT.a_dev
    # a_dev from *TBT.nc and *TBT.SE.nc is not sorted correctly in older versions of tbtrans!!! 
    # a_dev = np.sort(TBT.a_dev)

    # Indices of orbitals in device region 
    o_dev = TSHS.a2o(a_dev, all=True)

    # Check it's only carbon in R1
    for ia in a_R1:
        if TSHS.atom[ia].Z != 6:
            print('\nERROR: please select C atoms in region R1.')
            print('Atoms {} are not carbon \n'.format((TSHS.atoms.Z != 6).nonzero()[0]))
            exit(1)

    # Define R1 region (indices are w.r.t. device device region! )
    if orb_idx is not None:
        # Selected 'orb_idx' orbitals inside R1 region   
        print('WARNING: you are selecting only orbital index \'{}\' in R1 region'.format(orb_idx))
        o_R1 = TSHS.a2o(a_R1) + orb_idx  # IMPORTANT: these are pz indices in the full L+D+R geometry 
    else:
        # If no particular orbital is specified, then consider ALL orbitals inside R1 region
        o_R1 = TSHS.a2o(a_R1, all=True)  # With this  we will basically calculate a Sigma DFT-->DFT instead of DFT-->TB
    # Now we find their indeces with respect to the device region
    o_R1 = np.in1d(o_dev, o_R1).nonzero()[0]    # np.in1d returns true/false. Nonzero turns it into the actual indices

    # In region 2 we will consider ALL orbitals of ALL atoms
    vv = TSHS.geom.sub(a_dev)
    o_R2_tmp = area_R2.within_index(vv.xyz)
    o_R2 = vv.a2o(o_R2_tmp, all=True)  # these are ALL orbitals indices in region 2

    # Check
    v = TSHS.geom.copy()
    v.atom[v.o2a(o_dev, unique=True)] = si.Atom(8, R=[1.44])
    v.write('o_dev.xyz')
    # Check
    vv = TSHS.geom.sub(a_dev)
    vv.atom[vv.o2a(o_R1, unique=True)] = si.Atom(8, R=[1.44])
    vv.write('o_R1.xyz')
    # Check
    vv = TSHS.geom.sub(a_dev)
    vv.atom[vv.o2a(o_R2, unique=True)] = si.Atom(8, R=[1.44])
    vv.write('o_R2.xyz')
    

    """ 
    ### Map a_R1 into host geometry (which includes electrodes!)
    We will now rearrange the atoms in the host geometry
    putting the mapped ones at the end of the coordinates list
    """

    # sometimes this is useful to fix the mapping  
    if area_for_buffer is None:
        area_for_buffer = area_R2.copy()
        print("WARNING: You didn't provide 'area_for_buffer'. \n We are setting it to 'area_R2'. Please check that it is completely correct by comparing 'a_dSE_host.xyz' and 'buffer.xyz'")
    a_dSE_host, new_HS_host, a_buffer_host = map_xyz(A=TSHS, 
        B=HS_host, 
        center_B=pos_dSE, 
        area_R_A=area_R1, 
        a_R_A=a_R1, 
        area_for_buffer=area_for_buffer, 
        tol=tol)
    
    # Write dSE xyz
    v = new_HS_host.geom.copy(); v.atom[a_dSE_host] = si.Atom(8, R=[1.44]); v.write('a_dSE_host.xyz')
    
    # Write buffer atoms fdf block
    v = new_HS_host.geom.copy(); v.atom[a_buffer_host] = si.Atom(8, R=[1.44]); v.write('buffer.xyz')
    with open('block_buffer.fdf', 'w') as fb:
        fb.write("%block TBT.Atoms.Buffer\n")
        fb.write(list2range_TBTblock(a_buffer_host))    
        fb.write("\n%endblock\n")
    
    # Write final host large model, ready to be served to tbtrans
    new_HS_host.geom.write('HS_DEV.xyz')
    new_HS_host.geom.write('HS_DEV.fdf')
    new_HS_host.write('HS_DEV.nc')

    # Set CAP (optional)    
    if useCAP:
        # Create dH | CAP
        dH_CAP = CAP(new_HS_host.geom, useCAP, dz_CAP=50, write_xyz=True)
        dH_CAP_sile = si.get_sile('CAP.delta.nc', 'w')
        dH_CAP_sile.write_delta(dH_CAP)  # TBT.dH



    #############################
    """
    # Now we calculate and store the self-energy
    """

    ### Initialize dSE  (flag for tbtrans is --> TBT.dSE)
    print('Initializing dSE file...')
    o_dSE_host = new_HS_host.a2o(a_dSE_host, all=True).reshape(-1, 1)  # this has to be wrt L+D+R host geometry
    dSE = si.get_sile('SE_i.delta.nc', 'w')


    ### Initialize TBTGF (flag for tbtrans is --> GF inside an electrode block)
    # For this we need a complex energy contour + the sub H, S and geom of R1 in the new_HS_host (large TB)

    # Energy grid
    if EfromTBT:
        Eindices = [TBT.Eindex(en) for en in energies]
        E = TBT.E[Eindices] + 1j*eta_value
    else:
        print('WARNING: energies will not be taken from TBT. Make sure you know what you are doing.')
        E = np.asarray(energies) + 1j*eta_value
    tbl = si.io.table.tableSile('contour.IN', 'w')
    tbl.write_data(E.real, np.zeros(len(E)), np.ones(len(E)), fmt='.8f')

    # Remove periodic boundary conditions from TSHS!!!
    TSHS_n = TSHS.copy()
    print('Removing periodic boundary conditions')
    TSHS_n.set_nsc([1]*3)   # this is how you do it in sisl. Super easy. Removes all phase factors
        
    # Now we extract submatrices of this, pruning into o_R1
    print('Initializing TBTGF files...')
    if TSHS_n.spin.is_polarized:
        H_tbtgf = TSHS_n.Hk(dtype=np.float64, spin=spin)
    else:
        H_tbtgf = TSHS_n.Hk(dtype=np.float64)
    S_tbtgf = TSHS_n.Sk(dtype=np.float64)
    print(' Hk and Sk: DONE')
    
    # Prune to dev region  (again, this is because o_R1 is w.r.t. device)
    H_tbtgf_d = pruneMat(H_tbtgf, o_dev)
    S_tbtgf_d = pruneMat(S_tbtgf, o_dev)
    # Prune now these to o_R1
    H_tbtgf_R1 = pruneMat(H_tbtgf_d, o_R1)
    S_tbtgf_R1 = pruneMat(S_tbtgf_d, o_R1)

    # Finally we need the geometry of R1 in new_HS_host
    geom_R1 = new_HS_host.geom.sub(a_dSE_host)
    
    # It is vital that you also write an electrode Hamiltonian
    Semi = si.Hamiltonian.fromsp(geom_R1, H_tbtgf_R1, S_tbtgf_R1)
    Semi.write('HS_SE_i.nc')   # this is used as HS flag inside the TBTGF electrode block
    
    # We also need a formal Brillouin zone. 
    # "Formal" because we will always use a Gamma-only TBTGF 
    # (there is no periodicity once we plug DFT into TB!)
    BZ = si.BrillouinZone(TSHS_n)
    BZ._k = np.array([[0.,0.,0.]])
    BZ._w = np.array([1.0])
    
    # Now finally we initialize a TBTGF file. We
    # We will fill it further below with the matrix for the DFT-TB self-energy
    GF = si.io.TBTGFSileTBtrans('SE_i.TBTGF')
    GF.write_header(BZ, E, obj=Semi) # Semi HAS to be a Hamiltonian object, E has to be complex (WITH eta)
    ###############

    # If there is a self energy enclosed by the frame (e.g. a semi-infinite tip), 
    # we will need to add it later to H_R2 and S_R2 to generate G_R2
    # For that we will need to know the indices of the orbitals of the device region (in_device=True) 
    # on which the self-energy has been down-folded during the tbtrans BTD process (see tbtrans manual).
    # - When read with BTTSE.pivot, they will be sorted as after the BTD process
    # we can sort them back to the original indices by using sort=True
    if TBTSE:
        pv = TBTSE.pivot('tip', in_device=True, sort=True).reshape(-1, 1)
        pv_R2 = np.in1d(o_R2, pv.reshape(-1, )).nonzero()[0].reshape(-1, 1)

    """
    Now we loop over requested energies and create the DFT-TB self-energy matrices 
    at those energies. FINALLY some physics ;-)
    """
    print('Computing and storing Sigma in TBTGF and dSE format...')
    for i, (ispin, HS4GF, _, e) in enumerate(GF):  # sisl specific. THe _ would be k. But we just have Gamma, so who cares
        print('Doing E # {} of {}  ({} eV)'.format(i+1, len(E), e.real))  
        print('Doing E # {} of {}  ({} eV)'.format(i+1, len(E), e.real), file=open('log', 'a+'))  # Also log while running 

        """ 
        Calculate G_R2
        """
        # Read H and S from full TSHS (L+D+R) - no self-energies here!
        if TSHS_n.spin.is_polarized:
            Hfullk = TSHS_n.Hk(format='array', spin=spin)
        else:
            Hfullk = TSHS_n.Hk(format='array')
        Sfullk = TSHS_n.Sk(format='array')

        # Prune H, S to device region
        H_d = pruneMat(Hfullk, o_dev)
        S_d = pruneMat(Sfullk, o_dev)

        # Prune H, S matrices from device region to region 2
        H_R2 = pruneMat(H_d, o_R2)
        S_R2 = pruneMat(S_d, o_R2)

        # Build inverse of G_R2 (no self-energies such as tip yet!)
        invG_R2 = S_R2*e - H_R2 

        # if there's a self energy enclosed by the frame
        if TBTSE:
            # Read in the correct format
            SE_ext = TBTSE.self_energy('tip', E=e.real, k=[0.,0.,0.], sort=True)
            # Subtract from invG_R2 at the correct rows and columns (given by pv_R2)
            invG_R2[pv_R2, pv_R2.T] -= SE_ext

        # Now invert
        G_R2 = np.linalg.inv(invG_R2)
        

        """ 
        Extract V_21 elements from H_d
        """
        # Coupling matrix from R1 to R2 (len(o_R2) x len(o_R1))
        V_21 = couplingMat(H_d, o_R2, o_R1)
        # CHECK: V_21 OR V_21 - z*S_21
        S_21 = couplingMat(S_d, o_R2, o_R1)


        """ 
        Compute the final self-energy
        """
        # Self-energy is projected (lives) in R1, connecting R1 to R2 (len(o_R1) x len(o_R1))
        SE_R1 = np.dot(np.dot(dagger(V_21), G_R2), V_21)
        #SE_R1 = np.dot(np.dot(dagger(V_21-e*S_21), G_R2), V_21-e*S_21)
        
        
        """
        Now that we have it, we save it!
        """
        # Write Sigma as a dSE file
        Sigma_in_HS_host = sp.sparse.csr_matrix((len(new_HS_host), len(new_HS_host)), dtype=np.complex128)
        Sigma_in_HS_host[o_dSE_host, o_dSE_host.T] = SE_R1
        delta_Sigma = si.Hamiltonian.fromsp(new_HS_host.geom, Sigma_in_HS_host)
        dSE.write_delta(delta_Sigma, E=e.real)

        # Write Sigma as TBTGF
        # tbtrans wants you to write the quantity S_R1*e - H_R1 - SE_R1
        # So, prune H, S matrices from device region to region 1
        H_R1 = pruneMat(H_d, o_R1)
        S_R1 = pruneMat(S_d, o_R1)
        if HS4GF:
            GF.write_hamiltonian(H_R1, S_R1)
        GF.write_self_energy(S_R1*e - H_R1 - SE_R1) 

        # Now run tbrans! :-D


        

def rearrange_H(TSHS, a_list, HS_host, pos_dSE=None, 
    area=None):

    #a_list = np.sort(a_list)
    # Check it's carbon atoms in R1
    for ia in a_list:
        if TSHS.atom[ia].Z != 6:
            print('\nERROR: please select C atoms in R1 region \n')
            exit(1)

    # Map a_list into host geometry (which includes electrodes!)
    # WARNING: we will now rearrange the atoms in the host geometry
    # putting the mapped ones at the end of the coordinates list
    a_dSE_host, new_HS_host = map_xyz(A=TSHS, B=HS_host, center_B=pos_dSE, 
        area_R_A=area, a_R_A=a_list, area_for_buffer=None, tol=None)
    
    return new_HS_host, a_dSE_host


def construct_modular(H0, TSHS, modules, positions):
    """
    H0:         Host TB model to be rearranged. Modules will be progressively appended
                at the end of atomic list
    TSHS:       list of DFT Hamiltonians for each module; needed to recover the input coordinates to map
    modules:    list of tuples (a_Delta, Delta, area_ext, area_R2) as those obtained 
                from tbtncTools.Delta provide one tuple for each module
    positions:  list of xyz object (ndarray with shape=(1,3)) in H0 corresponding to
                the center of mass of each module provide one xyz for each module

    EXAMPLE OF USAGE:
        from tbtncTools import Delta
        a_Delta, _, Delta, area_ext, area_R2 = Delta(TSHS, shape='Cuboid', z_area=TSHS.xyz[0, 2], 
            thickness=10., ext_offset=tshs_0.cell[1,:].copy(), zaxis=2, atoms=C_list)
        frame_tip = (a_Delta, Delta, area_ext, area_R2)
        xyz_tsource = H0.center(what='xyz') +(0.4*H0.cell[1,:]-[0,5.31,0])
        xyz_tdrain = H0.center(what='xyz') -(0.4*H0.cell[1,:]-[0,5.31,0]) -0.5*TSHS.cell[0,:]

        Hfinal, l_al, l_buf = construct_modular(H0=H0,
            TSHS=[TSHS, TSHS] 
            modules=[frame_tip, frame_tip], 
            positions=[xyz_tsource, xyz_tdrain])

    WARNING: maybe in some situations it might overlap some buffer and module atoms.
        So, ALWAYS CHECK THAT FINAL XYZ IS WHAT YOU ACTUALLY EXPECT!!! 
    """
    Htmp = H0.copy()
    l_nal = []

    for i, hs, mod, xyz in zip(range(len(TSHS)), TSHS, modules, positions):
        print('\nciaoooooooooo\n')
        H, al = rearrange_H(hs, mod[0], Htmp, 
            pos_dSE=xyz, area=mod[1])
        nal = len(al)
        print(nal)
        l_nal.insert(0, nal)
        Htmp = H.copy()

    l_nal = np.asarray(l_nal)

    # Find atoms in each frame, write xyz and info about modules
    l_al = []
    for i in range(len(l_nal)):
        first = len(H) - l_nal[:len(l_nal)-i].sum()
        last = len(H) - l_nal[:len(l_nal)-(i+1)].sum()

        al = np.arange(first, last)
        l_al.append(al)

        from tbtncTools import list2range_TBTblock 
        print('After reordering: \n{}'.format(list2range_TBTblock(al)))
        
        v = H.geom.copy()
        v.atom[al] = si.Atom(8, R=[1.44])
        #v.atom[l_buf[i]] = si.Atom(10, R=[1.44])
        v.write('module_{}.xyz'.format(i+1))
        # Print elec-pos end for use in tbtrans
        print("< module_{}.xyz > \n   elec-pos end {} (or {})".format(i+1, 
            al[-1]+1, -1 - (  l_nal[:len(l_nal)-(i+1)].sum()  )))
    
    l_al = np.asarray(l_al)

    # Find buffer
    # NB: that cuboids are always independent from the sorting in the host geometry
    l_buf = []
    for mod, xyz, al in zip(modules, positions, l_al):
        area_B = mod[2].copy()
        area_B.set_center(xyz)
        almostbuffer = area_B.within_index(H.xyz)
        buffer_i = np.in1d(almostbuffer, al, assume_unique=True, invert=True)
        buf = almostbuffer[buffer_i]
        l_buf.append(buf)
        
    # Check final frames and buffers
    all_al = np.concatenate(l_al[:])
    all_buf = np.concatenate(l_buf[:])
    v = H.geom.copy()
    v.atom[all_al] = si.Atom(8, R=[1.44])
    v.atom[all_buf] = si.Atom(10, R=[1.44]); v.write('framesbuffer.xyz')
    
    # Write buffer xyz and fdf block
    from tbtncTools import list2range_TBTblock
    with open('block_buffer.fdf', 'w') as fb:
        fb.write("%block TBT.Atoms.Buffer\n")
        fb.write(list2range_TBTblock(all_buf))    
        fb.write("\n%endblock\n")

    return H, l_al, l_buf