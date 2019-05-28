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

TSHS = si.get_sile("../x_tb/He.nc").read_hamiltonian()
fdf = TSHS.geometry
tbt = si.get_sile('../x_tb/siesta.TBT.nc')


#######################
print(TSHS)
print(fdf)

Clist = (fdf.atoms.Z == 6).nonzero()[0]
mask_C = np.in1d(tbt.a_dev, Clist)
carbons = tbt.a_dev[mask_C]

onlyCgeom = fdf.sub(carbons)
onlyCgeom.write('onlyCarbon.xyz')
x, y, z = TSHS.xyz[Clist, 0], TSHS.xyz[Clist, 1], TSHS.xyz[Clist, 2]
iio = 2
on = get_potential(TSHS, iio, Clist)#np.arange(TSHS.na))

on -= on[0]

# Write to txt
np.savetxt('pot_constriction.txt', np.transpose([np.arange(len(on)), x, y, z, on]), 
    header='index, x, y, z, potential')



# TSHS0 = si.get_sile("transiesta0.TSHS").read_hamiltonian()
# on0 = get_potential(TSHS0, 2, carbons)

# #on0=np.append(on0,0.0)

# print(len(on), len(on0))

# print(x)

# on=on-on0
# n=180
# on=np.delete(on,np.arange(180,len(on)))
# x=x[:180]
# y=y[:180]
# z=z[:180]

# print(len(on), len(x), len(z))

# # Write to txt
# np.savetxt('pot_constriction_adatom.txt', np.transpose([np.arange(len(on)), x, y, z, on]), 
#     header='index, x, y, z, potential')



# Scatter plot
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# image = ax.scatter(x, y, c=on, s=50, marker='o', edgecolors='None', cmap='viridis')
# image.set_clim(np.amin(on), np.amax(on))
# image.set_array(on)
# ax.autoscale()
# ax.margins(0.1)
# xlabel('$x (\AA)$')
# ylabel('$y (\AA)$')
# gcf()
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# axcb = plt.colorbar(image, cax=cax, format='%f', ticks=[np.amin(on), np.amax(on)])
# savefig('pot_scatter_iio{}.png'.format(iio), bbox_inches='tight', dpi=300)
# clf()

# Interpolated
fig, ax = plt.subplots()
ax.set_aspect('equal')
#xtip, ytip = TSHS.xyz[1248, 0], TSHS.xyz[1248, 1]
coords = np.column_stack((x, y))
values = np.array(on)
img, min, max = mask_interpolate(coords, values, oversampling=50, a=1.6)
# Note that we tell imshow to show the array created by mask_interpolate
# faithfully and not to interpolate by itself another time.
image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                  origin='lower', interpolation='none', cmap='viridis',
                  vmin=np.amin(on), vmax=np.amax(on))
ax.autoscale()
ax.margins(0.1)
xlabel('$x (\AA)$')
ylabel('$y (\AA)$')
gcf()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
axcb = plt.colorbar(image, cax=cax, format='%.1f', ticks=[np.amin(on), 0, np.amax(on)])
axcb.set_label(r'E$_{p_z}$ - E$^0_{p_z}$ (eV)', rotation=270, labelpad=30, fontsize=20)
savefig('pot_map_iio{}.png'.format(iio), bbox_inches='tight', dpi=300)
