import numpy as np
import math
from matplotlib import pyplot as plt

k = np.linspace(-math.pi, math.pi, 100)
e = np.array([k, k, k, k])
for i in range(np.size(k)):
    e[0, i] = (1 / 2) + math.sqrt(8 * math.cos(k[i]) + 9) / 2
for i in range(np.size(k)):
    e[1, i] = (1 / 2) - math.sqrt(8 * math.cos(k[i]) + 9) / 2
for i in range(np.size(k)):
    e[2, i] = -(1 / 2) + math.sqrt(8 * math.cos(k[i]) + 9) / 2
for i in range(np.size(k)):
    e[3, i] = -(1 / 2) - math.sqrt(8 * math.cos(k[i]) + 9) / 2
plt.plot(k, e[0], '-', label=r'$\epsilon_1$')
plt.plot(k, e[1], '-', label=r'$\epsilon_2$')
plt.plot(k, e[2], '-', label=r'$\epsilon_3$')
plt.plot(k, e[3], '-', label=r'$\epsilon_4$')
plt.xlabel('k')
plt.ylabel('Energy')
plt.grid(True)
plt.legend(ncol=4)
plt.show()
