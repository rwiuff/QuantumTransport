from matplotlib import pyplot as plt     # Pyplot for nice graphs
import numpy as np                      # NumPy
from numpy import linalg as LA
from Functions import ImportSystem
from progress.bar import Bar

# Retrieve unit cell
xyz, shiftx, shifty, filename = ImportSystem(1)

repx = int(input('Repetition in x? '))
repy = int(input('Repetition in y? '))
xyztemp = xyz
for i in range(repx):
    shiftarr = xyz + np.array([shiftx*(i+1), 0, 0])
    xyztemp = np.append(xyz, shiftarr, axis=0)
    print(xyz.shape)
xyz = xyztemp
xyztemp = xyz
for i in range(repy):
    shiftarr = xyz + np.array([0, shifty*(i+1), 0])
    xyztemp = np.append(xyz, shiftarr, axis=0)
    print(xyz.shape)
xyz = xyztemp
xlin = np.array([[0, 0]])
ylin = np.array([[0, 0]])
zlin = np.array([[0, 0]])

# bar = Bar('Gathering connections    ', max=xyz.shape[0]+xyz.shape[0])
for i in range(xyz.shape[0]):
    for j in range(xyz.shape[0]):
        if LA.norm(np.subtract(xyz[i], xyz[j])) < 1.6:
            TmpArr = np.array([[xyz[i, 0], xyz[j, 0]]])
            xlin = np.append(xlin, TmpArr, axis=0)
            TmpArr = np.array([[xyz[i, 1], xyz[j, 1]]])
            ylin = np.append(ylin, TmpArr, axis=0)
            TmpArr = np.array([[xyz[i, 2], xyz[j, 2]]])
            zlin = np.append(zlin, TmpArr, axis=0)
            # bar.next()
# bar.finish()
fig = plt.figure(figsize=(15,15))
for i in range(xlin.shape[0]):
    plt.plot(xlin[i], ylin[i])
plt.scatter(xyz[:, 0], xyz[:, 1], s=300)
plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('[Å]')
plt.xlabel('[Å]')
plt.show()
