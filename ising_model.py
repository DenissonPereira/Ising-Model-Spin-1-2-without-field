import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


J = 1
k_B = 1
T = sp.Symbol('T', real=True)
partition_function = 0


sigma_i = sp.Matrix([[1, 0], [0, -1]])
sigma_j = sp.Matrix([[1, 0], [0, -1]])

product_spin = sp.kronecker_product(sigma_i, sigma_j)
hamiltonian = -J * product_spin
eigenvalues = hamiltonian.eigenvals()

for value in eigenvalues:
    partition_function += sp.exp(-(1/(k_B * T)) * value)

Z = sp.simplify(partition_function)

F = -T * sp.ln(Z)
U = sp.simplify((k_B * T**2) * sp.diff(sp.ln(Z), T))
C = sp.simplify(sp.diff(U, T)).subs({1/sp.cosh(1/T): sp.sech(1/T)})
S = sp.simplify((U - F) / T)

T_values = np.linspace(0, 6, 100)
F_values = [float(F.subs(T, value)) for value in T_values]
U_values = [float(U.subs(T, value)) for value in T_values]
C_values = [float(C.subs(T, value)) for value in T_values]
S_values = [float(S.subs(T, value)) for value in T_values]

print("Thermodynamic quantities per number of particles are given by: \nf(T) = {}; \nu(T) = {}; \nc(T) = {}; \ns(T) = {}.".format(F, U, C, S))

plt.plot(T_values, F_values, label='Helmholtz free energy')
plt.plot(T_values, U_values, label='Energy')
plt.plot(T_values, C_values, label='Heat Capacity')
plt.plot(T_values, S_values, label='Entropy')
plt.xlabel('Temperature')
plt.ylabel('Free energy(f), Energy(u), Heat Capacity(c), Entropy(s)')
plt.legend(['f', 'u', 'c', 's'])
plt.grid(True)
plt.title('The thermodynamic Quantities for the Ising Model Under Zero Field', color='green')
plt.show()