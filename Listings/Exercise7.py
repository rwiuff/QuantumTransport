import numpy as np
import sympy as sym
import math
from sympy import I, pi, simplify, cos, sin, init_printing
from matplotlib import pyplot as plt
init_printing(use_unicode=True)

h = sym.Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
V = sym.Matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
VT = V.T
k = sym.symbols('k', real=True)
lamda = sym.symbols('lamda', real=True)
hVVT = h + V * sym.exp(-I * k) + VT * sym.exp(-I * k)
m = hVVT - sym.eye(hVVT.shape[0]) * lamda
char = sym.det(m)
char = (char.as_real_imag())[0]
char = sym.collect(simplify(char), k)
print(char)
eigenvals = sym.solve(char, lamda)
print(eigenvals)

# kpoint = np.linspace(-math.pi, math.pi, 100)
# energy = np.zeros((len(eigenvals), 100))
# f = np.array([])
# for i in range(len(eigenvals)):
#     f = np.append(f, sym.lambdify(k, eigenvals[i], "numpy"))
#     for j in range(100):
#         energy[i, j] = f[i](kpoint[j])
#
# # print(energy)
#
# plt.plot(kpoint, energy[0], '-', label=r'$\epsilon_1$')
# plt.plot(kpoint, energy[1], '-', label=r'$\epsilon_2$')
# plt.plot(kpoint, energy[2], '-', label=r'$\epsilon_3$')
# plt.plot(kpoint, energy[3], '-', label=r'$\epsilon_4$')
# plt.xlabel('k')
# plt.ylabel('Energy')
# plt.grid(True)
# plt.legend(ncol=4)
# plt.show()
