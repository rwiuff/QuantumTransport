# -------------------------------------------------------------------- #
#                                                                      #
#     Python script for calculating the bandstructure of Benzene       #
#                                                                      #
#   This script utilises sympy to write up and finding the deter-      #
#   minant of the tightbinding Hamiltonian for benzene.                #
#   It then solves the characteristical polynomium and plots the       #
#   Eigenenergies as functions of k.                                   #
#                                                                      #
#              Written by Rasmus Wiuff (rwiuff@gmail.com)              #
#                                                                      #
# -------------------------------------------------------------------- #

# --------------------------Import Libraries-------------------------- #
import numpy as np                      # NumPy
import sympy as sym                     # SymPy
import math                             # Maths
from sympy import I, simplify           # Imaginary unit and simplify
from matplotlib import pyplot as plt    # Pyplot for nice graphs
# -------------------------------------------------------------------- #

# --------------------------Define variables-------------------------- #
k = sym.symbols('k', real=True)          # Creates symbolic variable 'k'
lamda = sym.symbols('lamda', real=True)  # Creates symbolic variable 'λ'
# -------------------------------------------------------------------- #

# -----------------------Define the Hamiltonian----------------------- #
h = sym.Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
# ^- On site hopping potential
V = sym.Matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
# ^- Hopping potetial to the right hand site
VT = V.T
# ^- Hopping potetial to the left hand site
hVVT = h + V * sym.exp(-I * k) + VT * sym.exp(I * k)
# ^- Kombined Hamiltonian
# -------------------------------------------------------------------- #

# --------------------Determinant and Eigenvalues--------------------- #
m = hVVT - sym.eye(hVVT.shape[0]) * lamda  # Subtract λ * identity matrix
char = sym.det(m)                          # Determinant
char = (char.as_real_imag())[0]            # Exp to Trig functions
char = sym.collect(simplify(char), k)      # Collect and simplify cos
eigenvals = sym.solve(char, lamda)         # Solve characteristical poly.
# -------------------------------------------------------------------- #

# ---------------------Plotting the bandstructure--------------------- #
kpoint = np.linspace(-math.pi, math.pi, 100)  # Create 100 points ]-π;π[
energy = np.zeros((len(eigenvals), 100))      # Create empty array for energies
f = np.array([])                              # Empty array for functions
for i in range(len(eigenvals)):               # For each eigenvalue
    f = np.append(f, sym.lambdify(k, eigenvals[i], "numpy"))  # SymPy -> NumPy
    for j in range(100):                                      # For each k
        energy[i, j] = f[i](kpoint[j])        # Calculate the energy

for i in range(len(eigenvals)):                   # For each eigenvalue
    str = r'$\epsilon_'                           # Create a raw string
    str = str + '{}$'.format(i)                   # Adds index to the string
    plt.plot(kpoint, energy[i], '-', label=str)   # Plots the band with legend

plt.xlabel('k')                     # 'k' on the x-axis
plt.ylabel('Energy')                # 'Energy' on the y-axis
plt.grid(True)                      # Grid on plot
plt.legend(ncol=len(eigenvals))     # Columnisation of legend
plt.show()                          # Show the plot
