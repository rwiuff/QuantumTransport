from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
import numpy as np                      # NumPy
from numpy import linalg as LA
from collections import Counter
from Functions import xyzimport, Hkay
Vppi = -1

shiftx = 32.7862152500
shifty = 8.6934634800
xyz = xyzimport('fab_NPG_C.fdf')
Ham = np.zeros((xyz.shape[0], xyz.shape[0]))
for i in range(xyz.shape[0]):
    for j in range(xyz.shape[0]):
        Ham[i, j] = LA.norm(np.subtract(xyz[i], xyz[j]))
Ham = np.where(Ham < 1.6, Vppi, 0)
Ham = np.subtract(Ham, Vppi * np.identity(xyz.shape[0]))

xyz1 = xyz + np.array([shiftx, 0, 0])
V1 = np.zeros((xyz.shape[0], xyz.shape[0]))

for i in range(xyz.shape[0]):
    for j in range(xyz1.shape[0]):
        V1[i, j] = LA.norm(np.subtract(xyz[i], xyz1[j]))
V1 = np.where(V1 < 1.6, Vppi, 0)

xyz2 = xyz + np.array([0, shifty, 0])
V2 = np.zeros((xyz.shape[0], xyz.shape[0]))

for i in range(xyz.shape[0]):
    for j in range(xyz2.shape[0]):
        V2[i, j] = LA.norm(np.subtract(xyz[i], xyz2[j]))
V2 = np.where(V2 < 1.6, Vppi, 0)


xyz3 = xyz + np.array([shiftx, shifty, 0])
V3 = np.zeros((xyz.shape[0], xyz.shape[0]))

for i in range(xyz.shape[0]):
    for j in range(xyz3.shape[0]):
        V3[i, j] = LA.norm(np.subtract(xyz[i], xyz3[j]))
V3 = np.where(V3 < 1.6, Vppi, 0)

print(np.sum(Ham))
# plt.imshow(Ham)
# plt.colorbar()
# plt.show()
# plt.imshow(V1)
# plt.colorbar()
# plt.show()
# plt.imshow(V2)
# plt.colorbar()
# plt.show()
# plt.imshow(V3)
# plt.colorbar()
# plt.show()


k = np.linspace(0, np.pi, 1000)
X = np.zeros((Ham.shape[0], k.size))
Z = np.zeros((Ham.shape[0], k.size))
for i in range(k.shape[0]):
    X[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=-k[i], y=0)[0]
    Z[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=k[i])[0]
zero = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)[0]
Xspace = np.linspace(0, 1 / shifty, 1000)
Zspace = np.linspace(0, 1 / shiftx, 1000)
ax = plt.figure(figsize=(1, 6))
for i in range(X.shape[0]):
    plt.plot(np.flip(-Zspace, axis=0), np.flip(X[i, :], axis=0))
for i in range(X.shape[0]):
    plt.plot(Xspace, Z[i, :])
xtick = np.array([-1 / shiftx, 0, 1 / shifty])
plt.xticks(xtick, ('X', 'G', 'Z'))
plt.axvline(x=0, linewidth=1, color='k', linestyle='--')
plt.title('NPG-normal')
plt.ylim(-1, 1)
plt.show()


e, v = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)
e = np.round(e, decimals=3)
w = e.real
c = Counter(w)
y = np.array([p for k, p in sorted(c.items())])
x = np.asarray(sorted([*c]))
# fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
# ax.vlines(x=x, ymin=0, ymax=y,
#           color='firebrick', alpha=0.7, linewidth=2)
# ax.scatter(x=x, y=y, s=75, color='firebrick', alpha=0.7)
#
# ax.set_title('Energy degeneracy', fontdict={'size': 22})
# ax.set_ylabel('Degeneracy')
# ax.set_xlabel('Energy')
# ax.set_ylim(0, 10)
# ax.tick_params(axis='both', which='both')
# ax.spines['left'].set_position('center')
# plt.grid(which='both')
# for i in range(x.size):
#     ax.text(x[i], y[i] + .5, s=x[i], horizontalalignment='center',
#             verticalalignment='bottom', fontsize=14)
# plt.show()

xlin = np.array([[0, 0]])
ylin = np.array([[0, 0]])
zlin = np.array([[0, 0]])

for i in range(xyz.shape[0]):
    for j in range(xyz.shape[0]):
        if LA.norm(np.subtract(xyz[i], xyz[j])) < 1.6:
            TmpArr = np.array([[xyz[i, 0], xyz[j, 0]]])
            xlin = np.append(xlin, TmpArr, axis=0)
            TmpArr = np.array([[xyz[i, 1], xyz[j, 1]]])
            ylin = np.append(ylin, TmpArr, axis=0)
            TmpArr = np.array([[xyz[i, 2], xyz[j, 2]]])
            zlin = np.append(zlin, TmpArr, axis=0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(xlin.shape[0]):
    ax.plot(xlin[i], ylin[i], zlin[i])

ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], zdir='z', s=300)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

val = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(xlin.shape[0]):
    ax.plot(xlin[i], ylin[i], zlin[i])
s = np.zeros(v.shape[0])
c = np.zeros(v.shape[0])
val = 1
s = np.absolute(v[:, val - 1])
s = s * 900
c = np.where(v[:, val - 1] > 0, 0, 1)
Stateplot = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], zdir='z', s=s)
Stateplot.set_cmap("bwr")
plt.subplots_adjust(bottom=0.25)
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
state = Slider(axfreq, 'State', 1, v.shape[0], valinit=1, valstep=1)


def update(val):
    val = state.val
    val = int(val)
    s = np.absolute(v[:, val - 1])
    s = s * 900
    print(s)
    c = np.where(v[:, val - 1] > 0, 0, 1)
    print(c)
    Stateplot._sizes = s
    Stateplot.set_array(c)
    fig.canvas.draw_idle()


resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    state.reset()


button.on_clicked(reset)
state.on_changed(update)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
