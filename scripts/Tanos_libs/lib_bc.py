from __future__ import print_function, division
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sisl as si

def read_bondcurrents(f, idx_elec, only='+', E=0.0, k='avg'):#, atoms=None):
    """Read bond currents from tbtrans output

    Parameters
    ----------
    f : string
        TBT.nc file
    idx_elec : int
        the electrode of originating electrons
    only : {''+', '-', 'all'}
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
