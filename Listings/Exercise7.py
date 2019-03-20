import numpy as np
import sympy as sym
import math
from sympy import I, simplify
from matplotlib import pyplot as plt

h = sym.Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
V = sym.Matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
VT = V.T
k = sym.symbols('k', real=True)
lamda = sym.symbols('lamda', real=True)
hVVT = h + V * sym.exp(-I * k) + VT * sym.exp(I * k)
m = hVVT - sym.eye(hVVT.shape[0]) * lamda
char = sym.det(m)
char = (char.as_real_imag())[0]
char = sym.collect(simplify(char), k)
eigenvals = sym.solve(char, lamda)

kpoint = np.linspace(-math.pi, math.pi, 100)
energy = np.zeros((len(eigenvals), 100))
f = np.array([])
for i in range(len(eigenvals)):
    f = np.append(f, sym.lambdify(k, eigenvals[i], "numpy"))
    for j in range(100):
        energy[i, j] = f[i](kpoint[j])

for i in range(len(eigenvals)):
    str = r'$\epsilon_'
    str = str + '{}$'.format(i)
    plt.plot(kpoint, energy[i], '-', label=str)

plt.xlabel('k')
plt.ylabel('Energy')
plt.grid(True)
plt.legend(ncol=len(eigenvals))
plt.show()
