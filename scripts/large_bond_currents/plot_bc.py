import matplotlib
matplotlib.use('Agg')
import sisl as si
import pylab as plt
import numpy as np
from lib_bc import plot_bondcurrents
import sys

site = int(sys.argv[1])

tbtref = si.get_sile('./tip{}/siesta.TBT.nc'.format(site))
f = './tip{}/siesta.TBT.nc'.format(site)
#for ie, en in enumerate(tbtref.E):

for userenergy in np.linspace(-0.6, 0.0, 7):
    iee = tbtref.Eindex(userenergy)
    en = tbtref.E[iee]
    # If you want all energy points, comment the 3 lines above and shift left the following
    #for en in tbtref.E:
    print('**** Energy = {} eV'.format(en))
    plot_bondcurrents(f, idx_elec=0, only='+',  zaxis=2, k='avg', E=en, avg=True, scale='%',
    vmin=0, vmax=5, ps=2, lw=15, log=False, adosmap=False, arrows=False, 
    lattice=False, ados=False, atoms=None, spsite=site, out='bc_{}'.format(site), units='nm')
