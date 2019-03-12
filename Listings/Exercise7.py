import numpy as np
import math
from sympy import *
from matplotlib import pyplot as plt


h = Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
V = Matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
VT = V.T
k = symbols('k')
e = symbols('e')
hVVT = h + V * exp(-I * k) + VT * exp(-I * k)
bcp = hVVT - Matrix([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]]) * e
cp = bcp.det()
print(cp)
# e = hVVT.eigenvals()
# e = list(e.keys())
# print(e[0].rewrite(exp).simplify())
# for j in range(len(roots)):
#     for k in range(np.size(r)):
#         energy[0, i] = roots(k).evalf
#
# plt.plot(k, energy[0], '-', label=r'$\epsilon_1$')
# plt.plot(k, energy[1], '-', label=r'$\epsilon_2$')
# plt.plot(k, energy[2], '-', label=r'$\epsilon_3$')
# plt.plot(k, energy[3], '-', label=r'$\epsilon_4$')
# plt.xlabel('k')
# plt.ylabel('Energy')
# plt.grid(True)
# plt.legend(ncol=4)
# plt.show()
