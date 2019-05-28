import matplotlib
matplotlib.use('Agg')
import sisl as si
import pylab as plt
import numpy as np
from tbtncTools import plot_transmission, plot_transmission_bulk

matplotlib.rcParams.update({'font.size': 18})

#plt.figure(figsize=(12,5))
plt.figure()

#plt.subplot(121)

subdir = ['../y_tb/', '../x_tb/']
#subdir = ['../TB_PARA_y/']
file = 'siesta.TBT.nc'
f = [str(subdir[i])+file for i in range(len(subdir))]
plot_transmission(f[0], 0, 1, ymin=None, ymax=None, style=':', lw=2, color='g', label='Along y')
plot_transmission(f[1], 0, 1, ymin=None, ymax=None, style='-', lw=2, color='g', label='Along x')

plt.ylim(0, 4.2)
plt.xlim(-1.6, 1.6)
plt.margins(0.)
plt.legend(loc=0, frameon=False)
plt.tight_layout()
#plt.savefig('Txy_para.pdf')
plt.savefig('Txy.png', dpi=500)


