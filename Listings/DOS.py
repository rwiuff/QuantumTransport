from matplotlib import pyplot as plt     # Pyplot for nice graphs
from mpl_toolkits.mplot3d import Axes3D  # Used for 3D plots
from matplotlib.widgets import Slider, Button
from scipy.sparse import dia_matrix
import numpy as np                      # NumPy
from numpy import linalg as LA
from collections import Counter

Vppi = -1

xyz = np.array([[23.90793679, 2.22952265, 25],
                [22.69174785, 2.9254843, 25],
                [21.43499043, 0.81545284, 25],
                [21.4270729, 2.25439967, 25],
                [20.14828205, 0.07915835, 25],
                [20.14549736, 2.98710564, 25],
                [20.10526471, 7.36221729, 25],
                [20.10549035, 4.39827542, 25],
                [18.89574135, 0.80908737, 25],
                [18.89401509, 2.25596926, 25],
                [18.89453465, 6.61459871, 25],
                [18.894852, 5.145233, 25],
                [17.64677401, 0.08448949, 25],
                [17.64609748, 2.98053173, 25],
                [17.64212264, 7.33022083, 25],
                [17.64217445, 4.4286262, 25],
                [16.39220985, 0.81275674, 25],
                [16.39229985, 6.60059583, 25],
                [16.39209633, 5.15771721, 25],
                [16.39207166, 2.25200344, 25],
                [15.14251537, 7.32972058, 25],
                [15.14199354, 4.42827567, 25],
                [15.1378122, 0.08402222, 25],
                [15.1379438, 2.9802162, 25],
                [13.89050081, 6.61335694, 25],
                [13.8891081, 5.14411856, 25],
                [13.8888101, 0.80819469, 25],
                [13.89003156, 2.25520477, 25],
                [12.67998471, 7.36029302, 25],
                [12.6783065, 4.39662313, 25],
                [12.63706334, 0.077109, 25],
                [12.63800736, 2.98534282, 25],
                [22.68991755, 0.1331247, 25],
                [11.35040007, 0.81192077, 25],
                [11.35708131, 2.25083971, 25],
                [10.09665776, 0.1271326, 25],
                [10.09100846, 2.91931954, 25],
                [23.94400858, 0.795587, 25],
                [8.84190472, 0.78738436, 25],
                [8.876017, 2.22126636, 25],
                [25.27004622, 7.29112936, 25],
                [25.23582795, 0.03150833, 25],
                [7.5159079, 7.28183394, 25],
                [7.55082585, 0.02232133, 25],
                [26.48446938, 6.59203709, 25],
                [26.49095476, 0.69132805, 25],
                [27.75084679, 7.25984582, 25],
                [27.7443377, 0.0052947, 25],
                [6.30012241, 6.58512312, 25],
                [6.29649336, 0.68363397, 25],
                [5.03469085, 7.25500171, 25],
                [5.04207523, 0.00038864, 25],
                [29.03160377, 6.52492492, 25],
                [29.03133595, 0.73945532, 25],
                [29.07166223, 5.113541, 25],
                [29.07429528, 2.1497284, 25],
                [30.28382443, 7.25471437, 25],
                [30.28308304, 0.00826512, 25],
                [30.28275125, 4.36606705, 25],
                [30.28477024, 2.89668114, 25],
                [31.53170522, 6.52957639, 25],
                [31.53222162, 0.73232608, 25],
                [31.53584504, 5.08154342, 25],
                [31.53685362, 2.18012548, 25],
                [3.75328694, 6.52194155, 25],
                [3.75589214, 0.73668266, 25],
                [3.71307475, 5.11078092, 25],
                [3.7130842, 2.14715817, 25],
                [2.50176554, 7.2530426, 25],
                [2.50352467, 0.00660836, 25],
                [2.50230702, 4.3638956, 25],
                [2.50229931, 2.89462418, 25],
                [1.2538052, 6.52844788, 25],
                [1.25467023, 0.73135149, 25],
                [1.24966309, 5.08043993, 25],
                [1.25002679, 2.17908081, 25],
                [32.78611882, 7.25737235, 25],
                [32.78587833, 4.35161031, 25],
                [0.00038664, 2.90887936, 25],
                [0.00016475, 0.00316796, 25]])

Ham = np.zeros((xyz.shape[0], xyz.shape[0]))
for i in range(xyz.shape[0]):
    for j in range(xyz.shape[0]):
        Ham[i, j] = LA.norm(np.subtract(xyz[i], xyz[j]))
Ham = np.where(Ham < 1.6, Vppi, 0)
Ham = np.subtract(Ham, Vppi * np.identity(xyz.shape[0]))

shiftx = 32.7862152500
xyz1 = xyz + np.array([shiftx, 0, 0])
V1 = np.zeros((xyz.shape[0], xyz.shape[0]))

for i in range(xyz.shape[0]):
    for j in range(xyz1.shape[0]):
        V1[i, j] = LA.norm(np.subtract(xyz[i], xyz1[j]))
V1 = np.where(V1 < 1.6, Vppi, 0)

shifty = 8.6934634800
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

# print(np.sum(Ham))
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


def Hkay(Ham, V1, V2, V3, x, y):
    Ham = Ham + (V1 * np.exp(-1.0j * x) + np.transpose(V1) * np.exp(1.0j * x)
                 + V2 * np.exp(-1.0j * y) + np.transpose(V2) * np.exp(1.0j * y)
                 + V3 * np.exp(-1.0j * x) * np.exp(-1.0j * y) + np.transpose(V3)
                 * np.exp(1.0j * x) * np.exp(1.0j * y))
    e = LA.eigh(Ham)[0]
    return e


k = np.linspace(0, np.pi, 1000)
X = np.zeros((Ham.shape[0], k.size))
Z = np.zeros((Ham.shape[0], k.size))
for i in range(k.shape[0]):
    X[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=-k[i], y=0)
    Z[:, i] = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=k[i])
zero = Hkay(Ham=Ham, V1=V1, V2=V2, V3=V3, x=0, y=0)
plt.scatter(np.zeros((zero.shape[0])), zero)
plt.show()
Xspace = np.linspace(0, 1/shifty, 1000)
Zspace = np.linspace(0, 1/shiftx, 1000)
ax = plt.figure(figsize=(1, 6))
for i in range(X.shape[0]):
    plt.plot(np.flip(-Zspace, axis=0), np.flip(X[i, :], axis=0))
    plt.plot(Xspace, Z[i, :])
# plt.xlim(0, 1)
xtick = np.array([-1/shiftx, 0, 1/shifty])
plt.xticks(xtick, ('X','G','Z'))
plt.ylim(-1, 3)
plt.show()

# print(band[:, 0, 0])
# print(band[0, :, 0])
# print(band[0, 0, :])
# fig = plt.figure()
# plt.subplot(221)
# for i in range(band.shape[0]):
#     plt.plot(kx, band[0, i, :])
# plt.subplot(222)
# for i in range(band.shape[0]):
#     plt.plot(kx, band[0, :, i])
# plt.subplot(223)
# for i in range(band.shape[0]):
#     plt.plot(kx, band[:, 0, i])
# plt.subplot(224)
# for i in range(band.shape[0]):
#     plt.plot(kx, band[i, 0, :])
# plt.show()
