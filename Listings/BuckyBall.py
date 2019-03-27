# from ase import Atoms                   # Used to extract coordinates
# from ase.build import molecule          # Constructs molecules
from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
# from sympy import I, simplify           # Imaginary unit and simplify
# import math                             # Maths
# import sympy as sym                     # SymPy
import numpy as np                      # NumPy
from numpy import linalg as LA
from collections import Counter
Vppi = -1

# np.set_printoptions(threshold=np.inf)

# BB = molecule('C60')
# xyz = BB.get_positions()
xyz = np.array([[1.3624, 1.5632, 2.8359], [2.0435, 0.36748, 2.7818],
                [1.6002, 2.5246, 1.8519], [0.0036388, 1.2996, 3.3319],
                [1.2172, -0.64172, 3.2237], [2.9886, 0.13386, 1.8164],
                [0.50174, 3.3131, 1.2672], [2.5073, 2.2423, 0.85514],
                [-1.1397, 2.0362, 2.6753], [-0.086852, -0.055936, 3.5613],
                [1.3122, -1.9012, 2.6354], [3.0831, -1.0979, 1.2391],
                [3.2202, 1.0708, 0.8538], [-0.90772, 2.9856, 1.7068],
                [0.78701, 3.4713, -0.071127], [2.0706, 2.8055, -0.32213],
                [-2.2925, 1.2502, 2.225], [-1.3338, -0.83053, 3.1472],
                [2.2289, -2.0986, 1.6273], [0.10933, -2.6948, 2.338],
                [3.3729, -0.9212, -0.082145], [3.4595, 0.4197, -0.32075],
                [-1.9189, 2.7734, 0.66243], [-0.30423, 3.3175, -1.1239],
                [2.3151, 2.1454, -1.5248], [-2.718, 1.7289, 1.0219],
                [-2.4072, -0.1101, 2.4492], [-1.2414, -2.0783, 2.5771],
                [1.6915, -2.9709, 0.70985], [0.34387, -3.3471, 1.1603],
                [2.7975, -1.7395, -1.0186], [2.9824, 0.94083, -1.4955],
                [-1.6529, 2.9328, -0.68622], [-0.061038, 2.6748, -2.3153],
                [1.2982, 2.0899, -2.5875], [-3.3109, 0.91875, 0.095886],
                [-3.0017, -0.92892, 1.5037], [-2.3116, -2.2045, 1.5437],
                [1.9754, -2.7766, -0.63964], [-0.75087, -3.4335, 0.13085],
                [2.3593, -1.2416, -2.2239], [2.4601, 0.1258, -2.4726],
                [-2.2474, 2.1044, -1.6233], [-1.2886, 1.912, -2.6947],
                [1.3859, 0.85338, -3.1625], [-3.5067, -0.40969, 0.32408],
                [-3.1274, 1.1072, -1.2394], [-2.0814, -2.8689, 0.37769],
                [0.92735, -2.9321, -1.6567], [-0.48135, -3.2351, -1.1932],
                [1.1636, -1.9938, -2.6284], [-1.1972, 0.6892, -3.2868],
                [0.12809, 0.10609, -3.5141], [-3.4109, -1.1172, -0.94606],
                [-3.1772, -0.1844, -1.9062], [-2.6065, -2.3553, -0.91036],
                [-1.6415, -2.5559, -1.8293], [0.018087, -1.2314, -3.2618],
                [-2.1215, -0.40907, -2.9139], [-1.3879, -1.5381, -2.8789]])

Ham = np.zeros((xyz.shape[0], xyz.shape[0]))
for i in range(xyz.shape[0]):
    for j in range(xyz.shape[0]):
        Ham[i, j] = LA.norm(np.subtract(xyz[i], xyz[j]))
Ham = np.where(Ham < 1.6, Vppi, 0)
Ham = np.subtract(Ham, Vppi * np.identity(xyz.shape[0]))

print(Ham.shape)
print(np.sum(Ham))
plt.imshow(Ham)
plt.colorbar()
plt.show()
v, e = LA.eig(Ham)
v = np.round(v, decimals=3)
w = v.real
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

ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

val = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(xlin.shape[0]):
    ax.plot(xlin[i], ylin[i], zlin[i])
s = np.zeros(v.size)
for i in range(v.size):
    s[i] = np.absolute(v[i])
s = s * val
print(s)
print(s.shape)
Stateplot = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], zdir='z', s=s)
plt.subplots_adjust(bottom=0.25)
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
state = Slider(axfreq, 'State', 1, 30, valinit=1, valstep=1)


def update(val):
    val = state.val
    for i in range(v.size):
        s[i] = np.absolute(v[i])
    Stateplot._sizes = s*val
    fig.canvas.draw_idle()


resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    state.reset()


button.on_clicked(reset)
state.on_changed(update)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
