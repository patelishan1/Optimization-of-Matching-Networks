import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.optimize import differential_evolution, minimize

# Given parameters
Z0 = 50  # Characteristic impedance in ohms
ZL = 75 + 25j  # Load impedance in ohms
freq = 2e9  # Frequency in Hz
omega = 2 * pi * freq
epsilon_r = 4.4  # Relative permittivity for the microstrip line

# Function to calculate reflection coefficient using lumped elements
def reflection_coefficient_lumped(Z0, ZL, L, C, omega):
    ZL_prime = ZL + 1j * omega * L
    Z_in = 1 / (1 / ZL_prime + 1j * omega * C)
    gamma = (Z_in - Z0) / (Z_in + Z0)
    return np.abs(gamma)

# Function to calculate reflection coefficient using microstrip line
def reflection_coefficient_microstrip(Z0, ZL, l, omega, epsilon_r):
    beta = omega / (3e8 / np.sqrt(epsilon_r))
    Z_in = Z0 * ((ZL + 1j * Z0 * np.tan(beta * l)) / (Z0 + 1j * ZL * np.tan(beta * l)))
    gamma = (Z_in - Z0) / (Z_in + Z0)
    return np.abs(gamma)

# Objective function to minimize for lumped elements
def objective_lumped(params):
    L, C = params
    return reflection_coefficient_lumped(Z0, ZL, L, C, omega)

# Objective function to minimize for microstrip line
def objective_microstrip(params):
    l = params[0]
    return reflection_coefficient_microstrip(Z0, ZL, l, omega, epsilon_r)

# Bounds for L and C to ensure physical realizability
bounds_lumped = [(1e-9, 1e-6), (1e-15, 1e-9)]  # (L_bounds, C_bounds)
bounds_microstrip = [(1e-3, 1e-1)]  # (l_bounds)

# Perform the optimization using Differential Evolution with stricter bounds
result_lumped = differential_evolution(objective_lumped, bounds_lumped, strategy='best1bin', maxiter=20000, tol=1e-12)
result_microstrip = differential_evolution(objective_microstrip, bounds_microstrip, strategy='best1bin', maxiter=20000, tol=1e-12)

# Extract the optimal values from Differential Evolution
L_opt, C_opt = result_lumped.x
l_opt = result_microstrip.x[0]

# Refine the solution using the Nelder-Mead method with stricter bounds
local_result_lumped = minimize(objective_lumped, [L_opt, C_opt], method='Nelder-Mead', tol=1e-12, bounds=bounds_lumped)
local_result_microstrip = minimize(objective_microstrip, [l_opt], method='Nelder-Mead', tol=1e-12, bounds=bounds_microstrip)

# Extract the refined optimal values of L, C, and l
L_opt, C_opt = local_result_lumped.x
l_opt = local_result_microstrip.x[0]

# Calculate the optimal reflection coefficients
gamma_opt_lumped = reflection_coefficient_lumped(Z0, ZL, L_opt, C_opt, omega)
gamma_opt_microstrip = reflection_coefficient_microstrip(Z0, ZL, l_opt, omega, epsilon_r)

# Output optimized values
print(f"Optimal Inductor (L): {L_opt:.6e} H")
print(f"Optimal Capacitor (C): {C_opt:.6e} F")
print(f"Optimal Microstrip Line Length (l): {l_opt:.6e} meters")
print(f"Optimal Reflection Coefficient for Lumped Elements (|Gamma|): {gamma_opt_lumped:.6f}")
print(f"Optimal Reflection Coefficient for Microstrip Line (|Gamma|): {gamma_opt_microstrip:.6f}")

# Function to plot Smith Chart (example using matplotlib)
def plot_smith_chart(Z0, ZL, L, C, l, omega, epsilon_r):
    ZL_prime = ZL + 1j * omega * L
    Z_in_lumped = 1 / (1 / ZL_prime + 1j * omega * C)
    beta = omega / (3e8 / np.sqrt(epsilon_r))
    Z_in_microstrip = Z0 * ((ZL + 1j * Z0 * np.tan(beta * l)) / (Z0 + 1j * ZL * np.tan(beta * l)))

    fig, ax = plt.subplots()
    circle_load = plt.Circle((ZL.real, ZL.imag), np.abs(ZL), fill=False, linestyle='-', color='r', label='Load Impedance')
    circle_after_inductor = plt.Circle((ZL_prime.real, ZL_prime.imag), np.abs(ZL_prime), fill=False, linestyle='--', color='g', label='After Inductor')
    circle_after_capacitor = plt.Circle((Z_in_lumped.real, Z_in_lumped.imag), np.abs(Z_in_lumped), fill=False, linestyle=':', color='b', label='After Capacitor')
    circle_microstrip = plt.Circle((Z_in_microstrip.real, Z_in_microstrip.imag), np.abs(Z_in_microstrip), fill=False, linestyle='-.', color='m', label='After Microstrip Line')

    ax.add_artist(circle_load)
    ax.add_artist(circle_after_inductor)
    ax.add_artist(circle_after_capacitor)
    ax.add_artist(circle_microstrip)

    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Smith Chart Representation')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Plot Smith Chart using optimized lumped elements and microstrip line
plot_smith_chart(Z0, ZL, L_opt, C_opt, l_opt, omega, epsilon_r)

# Function to calculate and plot SWR and reflection coefficient vs. frequency
def plot_swr_and_reflection_vs_frequency(Z0, ZL, L, C, freq_range):
    freqs = np.linspace(freq_range[0], freq_range[1], 500)
    omegas = 2 * pi * freqs
    gamma_lumped = []
    swr_lumped = []

    for omega in omegas:
        gamma_l = reflection_coefficient_lumped(Z0, ZL, L, C, omega)
        gamma_lumped.append(gamma_l)
        swr_lumped.append((1 + gamma_l) / (1 - gamma_l))

    plt.figure()
    plt.plot(freqs / 1e9, swr_lumped, label='SWR', linestyle='--')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('SWR')
    plt.title('SWR vs. Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(freqs / 1e9, gamma_lumped, label='Reflection Coefficient', linestyle='--')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Reflection Coefficient (|Gamma|)')
    plt.title('Reflection Coefficient vs. Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define frequency range for the plots
freq_range = (1e9, 3e9)  # 1 GHz to 3 GHz

# Plot SWR and reflection coefficient vs. frequency
plot_swr_and_reflection_vs_frequency(Z0, ZL, L_opt, C_opt, freq_range)
