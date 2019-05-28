from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
from matplotlib import *
from pylab import *
import sisl as si
import numpy as np
from netCDF4 import Dataset
import time, sys
import scipy.linalg as sli
import scipy as sp
from scipy import mgrid
import os,sys
from itertools import groupby
from tbtncTools import CAP, mask_interpolate, get_potential
#from PIL import Image
import matplotlib.collections as collections
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 20})

TSHS = si.get_sile("../../single_point15_51/RUN.fdf").read_hamiltonian()  # "../../single_point15_51/RUN.fdf"
fdf = TSHS.geometry
#tbt = si.get_sile('../x_tb/siesta.TBT.nc')

#######################
print(TSHS)

#Hlist = (fdf.atoms.Z == 1).nonzero()[0]
Clist = (fdf.atoms.Z == 6).nonzero()[0]
Flist = (fdf.atoms.Z == 9).nonzero()[0]
all_list = np.concatenate((Clist, Flist))
print(Clist)
print(all_list)
all_list.sort()
print(all_list)
print(size(all_list))

#mask_C = np.in1d(tbt.a_dev, Clist)
#carbons = tbt.a_dev[mask_C]
#onlyCgeom = fdf.sub(carbons)
#onlyCgeom.write('onlyCarbon.xyz')

x, y, z = TSHS.xyz[Clist, 0], TSHS.xyz[Clist, 1], TSHS.xyz[Clist, 2]
iio = 2

on = get_potential(TSHS, iio, Clist)#np.arange(TSHS.na))
print(on)
print(size(on))
on -= on[0]
print(on)
# Write to txt
np.savetxt('pot_constriction.txt', np.transpose([np.arange(len(on)), x, y, z, on]),
    header='index, x, y, z, potential')

# TSHS0 = si.get_sile("transiesta0.TSHS").read_hamiltonian()
# on0 = get_potential(TSHS0, 2, carbons)
#
# #on0=np.append(on0,0.0)
#
# print(len(on), len(on0))
#
# print(x)
#
# on=on-on0
# n=180
# on=np.delete(on,np.arange(180,len(on)))
# x=x[:180]
# y=y[:180]
# z=z[:180]
#
# print(len(on), len(x), len(z))
#
# # Write to txt
# np.savetxt('pot_constriction_adatom.txt', np.transpose([np.arange(len(on)), x, y, z, on]),
#     header='index, x, y, z, potential')



# Scatter plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
image = ax.scatter(x, y, c=on, s=50, marker='o', edgecolors='None', cmap='viridis')
image.set_clim(np.amin(on), np.amax(on))
image.set_array(on)
ax.autoscale()
ax.margins(0.4)
xlabel('$x (\AA)$')
ylabel('$y (\AA)$')
gcf()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[np.amin(on), np.amax(on)])
savefig('pot_scatter_iio{}.png'.format(iio), bbox_inches='tight', dpi=300)
clf()
