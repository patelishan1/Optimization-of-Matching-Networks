# Ishan Patel and David Meseonznik

import numpy as np

import matplotlib.pyplot as plt

from scipy.constants import pi
from deap import base, creator, tools, algorithms

# Here we are using scipy, numpy and deap

# Define the objective function to minimize the reflection coefficient

def reflection_coefficient(Z0, ZL, elements, freq):
    """
    Calculate reflection coefficient for an L-section matching network.
    Z0: Characteristic impedance
    ZL: Load impedance
    elements: List of component values [L, C]
    freq: Operating frequency
    """
    L, C = elements
    ZL_prime = ZL + 1j * 2 * pi * freq * L  # Impedance after inductor
    Z_in = 1 / (1 / ZL_prime + 1j * 2 * pi * freq * C)  # Input impedance after capacitor
    gamma = (Z_in - Z0) / (Z_in + Z0)  # Reflection coefficient
    return np.abs(gamma)


# Genetic algorithm setup

def evaluate(individual, Z0, ZL, freq):

    L, C = individual  # Both are individials
    return reflection_coefficient(Z0, ZL, [L, C], freq),


def optimize_matching_network(Z0, ZL, freq):

    # Create fitness function and individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Genetic algorithm toolbox generated

    toolbox = base.Toolbox()

    toolbox.register("attr_float", np.random.uniform, 1e-9, 1e-6)  # Component value range

    toolbox.register("individual", tools.initRepeat,  creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, Z0=Z0, ZL=ZL, freq=freq)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=1e-9, up=1e-6, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=300)
    ngen, cxpb, mutpb = 50, 0.5, 0.2

    # Statistics to keep track of the evolution process
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Running the genetic algorithm

    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

    # Extract the best individual
    best_ind = tools.selBest(population, 1)[0]
    return best_ind, log


# Simulation parameters
Z0 = 50  # Characteristic impedance in ohms
ZL = 100 + 50j  # Load impedance in ohms
freq = 1e9  # Frequency in Hz

# Optimize matching network
best_elements, log = optimize_matching_network(Z0, ZL, freq)
L_opt, C_opt = best_elements

# Display results
print(f"Optimal Inductor (L): {L_opt:.6e} H")
print(f"Optimal Capacitor (C): {C_opt:.6e} F")

# Plot convergence
gen = log.select("gen")
min_fits = log.select("min")
avg_fits = log.select("avg")

fig, ax1 = plt.subplots()

line1 = ax1.plot(gen, min_fits, "b-", label="Minimum Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Minimum Fitness", color="b")

for tl in ax1.get_yticklabels():

    tl.set_color("b")

ax2 = ax1.twinx()

line2 = ax2.plot(gen, avg_fits, "r-", label="Average Fitness")
ax2.set_ylabel("Average Fitness", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

plt.title("Convergence of Genetic Algorithm")

plt.show()


# Function to plot impedance on Smith chart
def plot_smith_chart(Z0, ZL, L, C, freq):
    ZL_prime = ZL + 1j * 2 * pi * freq * L
    Z_in = 1 / (1 / ZL_prime + 1j * 2 * pi * freq * C)
    gamma_initial = (ZL - Z0) / (ZL + Z0)
    gamma_final = (Z_in - Z0) / (Z_in + Z0)

    plt.figure(figsize=(8, 8))

    plt.plot(np.real(gamma_initial), np.imag(gamma_initial), 'ro', label='Initial Reflection Coefficient')

    plt.plot(np.real(gamma_final), np.imag(gamma_final), 'bo', label='Final Reflection Coefficient')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')

    plt.title('Smith Chart Representation')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_smith_chart(Z0, ZL, L_opt, C_opt, freq)


# Function to plot impedance transformation

def plot_impedance_transformation(Z0, ZL, L, C, freq):

    ZL_prime = ZL + 1j * 2 * pi * freq * L
    Z_in = 1 / (1 / ZL_prime + 1j * 2 * pi * freq * C)

    gamma_initial = (ZL - Z0) / (ZL + Z0)
    gamma_final = (Z_in - Z0) / (Z_in + Z0)

    plt.figure()
    plt.plot([Z0.real, ZL.real], [Z0.imag, ZL.imag], 'ro-', label='Initial Impedance Path')

    plt.plot([ZL.real, ZL_prime.real], [ZL.imag, ZL_prime.imag], 'bo-', label='After Inductor')
    plt.plot([ZL_prime.real, Z_in.real], [ZL_prime.imag, Z_in.imag], 'go-', label='After Capacitor')
    plt.xlabel('Real Part of Impedance')

    plt.ylabel('Imaginary Part of Impedance')
    plt.title('Impedance Transformation')

    plt.legend()
    plt.grid(True)
    plt.show()


plot_impedance_transformation(Z0, ZL, L_opt, C_opt, freq)
