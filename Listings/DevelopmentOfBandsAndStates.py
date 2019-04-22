from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
import matplotlib
import numpy as np                      # NumPy
import seaborn
from numpy import linalg as LA
from collections import Counter
from Functions import xyzimport, Hkay, Onsite, Hop

# Set hopping potential
Vppi = -1

# Define lattice vectors
shiftx = 32.7862152500
shifty = 8.6934634800

# Retrieve unit cell
xyz = xyzimport('fab_NPG_C.fdf')
# Calculate onsite nearest neighbours
Ham = Onsite(xyz, Vppi)

# Shift unit cell
xyz1 = xyz + np.array([shiftx, 0, 0])
# Calculate offsite nearest neighbours
V1 = Hop(xyz, xyz1, Vppi)

# Shift unit cell
xyz2 = xyz + np.array([0, shifty, 0])
# Calculate offsite nearest neighbours
V2 = Hop(xyz, xyz2, Vppi)

# Shift unit cell
xyz3 = xyz + np.array([shiftx, shifty, 0])
# Calculate offsite nearest neighbours
V3 = Hop(xyz, xyz3, Vppi)


print(np.sum(Ham))
Show = 0
if Show == 1:
    plt.imshow(Ham)
    plt.colorbar()
    plt.show()
    plt.imshow(V1)
    plt.colorbar()
    plt.show()
    plt.imshow(V2)
    plt.colorbar()
    plt.show()
    plt.imshow(V3)
    plt.colorbar()
    plt.show()

# Define k-space range
k = np.linspace(0, np.pi, 1000)
# Array for X-bands
X = np.zeros((Ham.shape[0], k.size))
# Array for Z-bands
Z = np.zeros((Ham.shape[0], k.size))
# Get bands from gamma to X and Z
for i in range(k.shape[0]):
    X[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=-k[i], y=0)[0]
    Z[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=k[i])[0]
# Get energies at k(0,0)
zero = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)[0]
# Renormalise distances according to lattice vectors
Xspace = np.linspace(0, 1 / shifty, 1000)
Zspace = np.linspace(0, 1 / shiftx, 1000)
# Plot Bandstructures
ax = plt.figure(figsize=(1, 6))
for i in range(X.shape[0]):
    plt.plot(np.flip(-Zspace, axis=0),
             np.flip(X[i, :], axis=0), 'k', linewidth=1)
    plt.plot(Xspace, Z[i, :], 'k', linewidth=1)
xtick = np.array([-1 / shiftx, 0, 1 / shifty])
plt.xticks(xtick, ('X', r'$\Gamma$', 'Z'))
plt.axvline(x=0, linewidth=1, color='k', linestyle='--')
plt.title('NPG-normal')
plt.ylim(-1, 1)
plt.show()

# Get eigenvalues/vectors for degeneracy and stateplot
e, v = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)
e = np.round(e, decimals=3)
w = e.real
c = Counter(w)
y = np.array([p for k, p in sorted(c.items())])
x = np.asarray(sorted([*c]))
fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
ax.vlines(x=x, ymin=0, ymax=y,
          color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=x, y=y, s=75, color='firebrick', alpha=0.7)

ax.set_title('Energy degeneracy', fontdict={'size': 22})
ax.set_ylabel('Degeneracy')
ax.set_xlabel('Energy')
ax.set_ylim(0, 10)
ax.tick_params(axis='both', which='both')
ax.spines['left'].set_position('center')
plt.grid(which='both')
for i in range(x.size):
    ax.text(x[i], y[i] + .5, s=x[i], horizontalalignment='center',
            verticalalignment='bottom', fontsize=14)
seaborn.despine(left=True, bottom=True, right=True)
plt.show()

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
max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max()
                      - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][0].flatten() + 0.5 * (xyz[:, 0].max()
                                                               + xyz[:, 0].min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][1].flatten() + 0.5 * (xyz[:, 1].max()
                                                               + xyz[:, 1].min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][2].flatten() + 0.5 * (xyz[:, 2].max()
                                                               + xyz[:, 2].min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('X: [Å]')
ax.set_ylabel('Y: [Å]')
ax.set_zlabel('Z: [Å]')
plt.show()

fig = plt.figure()
for i in range(xlin.shape[0]):
    plt.plot(xlin[i], ylin[i])
plt.scatter(xyz[:, 0], xyz[:, 1], s=300)
plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('[Å]')
plt.xlabel('[Å]')
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
s = s * 300
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
    s = s * 300
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
max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max()
                      - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][0].flatten() + 0.5 * (xyz[:, 0].max()
                                                               + xyz[:, 0].min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][1].flatten() + 0.5 * (xyz[:, 1].max()
                                                               + xyz[:, 1].min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][2].flatten() + 0.5 * (xyz[:, 2].max()
                                                               + xyz[:, 2].min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('X: [Å]')
ax.set_ylabel('Y: [Å]')
ax.set_zlabel('Z: [Å]')
plt.show()


e, v = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=np.pi, y=0)
val = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(xlin.shape[0]):
    ax.plot(xlin[i], ylin[i], zlin[i])
s = np.zeros(v.shape[0])
c = np.zeros(v.shape[0])
colors = np.zeros((v.shape[0], 4))
val = 1
s = np.absolute(v[:, val - 1])
s = s * 300
cmap = matplotlib.cm.get_cmap('hsv')
c[:] = np.angle(v[:, val - 1])
vmin = np.min(c)
vmax = np.max(c)
normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# print(np.angle(v[:, val - 1]))
# print(normalize(np.angle(v[:, val - 1])))
# print(cmap(normalize(np.angle(v[:, val - 1]))))

colors[:] = cmap(normalize(np.angle(v[:, val - 1])))
print(colors)
# c = np.where(v[:, val - 1] > 0, 0, 1)
Stateplot = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], zdir='z', s=s)
plt.subplots_adjust(bottom=0.25)
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
state = Slider(axfreq, 'State', 1, v.shape[0], valinit=1, valstep=1)


def update(val):
    val = state.val
    val = int(val)
    s = np.absolute(v[:, val - 1])
    s = s * 300
    print(s)
    colors = np.zeros((v.shape[0], 4))
    c[:] = np.angle(v[:, val - 1])
    vmin = np.min(c)
    vmax = np.max(c)
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors[:] = cmap(normalize(np.angle(v[:, val - 1])))
    print(colors)
    # c = np.where(v[:, val - 1] > 0, 0, 1)
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
max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max()
                      - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][0].flatten() + 0.5 * (xyz[:, 0].max()
                                                               + xyz[:, 0].min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][1].flatten() + 0.5 * (xyz[:, 1].max()
                                                               + xyz[:, 1].min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][2].flatten() + 0.5 * (xyz[:, 2].max()
                                                               + xyz[:, 2].min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('X: [Å]')
ax.set_ylabel('Y: [Å]')
ax.set_zlabel('Z: [Å]')
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
plt.show()

e, v = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=np.pi)
val = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(xlin.shape[0]):
    ax.plot(xlin[i], ylin[i], zlin[i])
s = np.zeros(v.shape[0])
c = np.zeros(v.shape[0])
colors = np.zeros((v.shape[0], 4))
val = 1
s = np.absolute(v[:, val - 1])
s = s * 300
cmap = matplotlib.cm.get_cmap('hsv')
c[:] = normalize(np.angle(v[:, val - 1]))
print(c)
print(np.angle(v[:, val - 1]))
vmin = np.min(c)
vmax = np.max(c)
normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# print(np.angle(v[:, val - 1]))
# print(normalize(np.angle(v[:, val - 1])))
# print(cmap(normalize(np.angle(v[:, val - 1]))))

colors[:] = cmap(normalize(np.angle(v[:, val - 1])))
print(colors)
# c = np.where(v[:, val - 1] > 0, 0, 1)
Stateplot = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], zdir='z', s=s)
plt.subplots_adjust(bottom=0.25)
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
state = Slider(axfreq, 'State', 1, v.shape[0], valinit=1, valstep=1)


def update(val):
    val = state.val
    val = int(val)
    s = np.absolute(v[:, val - 1])
    s = s * 300
    print(s)
    colors = np.zeros((v.shape[0], 4))
    c[:] = np.angle(v[:, val - 1])
    vmin = np.min(c)
    vmax = np.max(c)
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors[:] = cmap(normalize(np.angle(v[:, val - 1])))
    print(colors)
    # c = np.where(v[:, val - 1] > 0, 0, 1)
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
max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max()
                      - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][0].flatten() + 0.5 * (xyz[:, 0].max()
                                                               + xyz[:, 0].min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][1].flatten() + 0.5 * (xyz[:, 1].max()
                                                               + xyz[:, 1].min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                - 1:2:2][2].flatten() + 0.5 * (xyz[:, 2].max()
                                                               + xyz[:, 2].min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('X: [Å]')
ax.set_ylabel('Y: [Å]')
ax.set_zlabel('Z: [Å]')
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
plt.show()
