import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from deap import base, creator, tools, algorithms

# Define objective function to minimize reflection coefficient for lumped elements
def reflection_coefficient_lumped(Z0, ZL, elements, freq):
    L, C = elements
    ZL_prime = ZL + 1j * 2 * pi * freq * L
    Z_in = 1 / (1 / ZL_prime + 1j * 2 * pi * freq * C)
    gamma = (Z_in - Z0) / (Z_in + Z0)
    return np.abs(gamma)

# Define objective function to minimize reflection coefficient for microstrip line
def reflection_coefficient_microstrip(Z0, ZL, l, freq, epsilon_r):
    beta = 2 * pi * freq / (3e8 / np.sqrt(epsilon_r))
    Z_in = Z0 * ((ZL + 1j * Z0 * np.tan(beta * l)) / (Z0 + 1j * ZL * np.tan(beta * l)))
    gamma = (Z_in - Z0) / (Z_in + Z0)
    return np.abs(gamma)

# Genetic algorithm setup
def evaluate_lumped(individual, Z0, ZL, freq):
    L, C = individual
    return reflection_coefficient_lumped(Z0, ZL, [L, C], freq),

def evaluate_microstrip(individual, Z0, ZL, freq, epsilon_r):
    l = individual[0]
    return reflection_coefficient_microstrip(Z0, ZL, l, freq, epsilon_r),

def optimize_lumped(Z0, ZL, freq):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 1e-9, 1e-6)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_lumped, Z0=Z0, ZL=ZL, freq=freq)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=1e-9, up=1e-6, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)
    ngen, cxpb, mutpb = 50, 0.5, 0.2

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, log

def optimize_microstrip(Z0, ZL, freq, epsilon_r):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 1e-3, 0.1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_microstrip, Z0=Z0, ZL=ZL, freq=freq, epsilon_r=epsilon_r)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=1e-3, up=0.1, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)
    ngen, cxpb, mutpb = 50, 0.5, 0.2

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, log

# Simulation parameters
Z0 = 50  # Characteristic impedance in ohms
ZL = 100 + 50j  # Load impedance in ohms
freq = 1000000000  # Frequency in Hz
epsilon_r = 4.4  # Relative permittivity for the microstrip line

# Optimize lumped elements
best_elements, log_lumped = optimize_lumped(Z0, ZL, freq)
L_opt, C_opt = best_elements
print(f"Optimal Inductor (L): {L_opt:.6e} H")
print(f"Optimal Capacitor (C): {C_opt:.6e} F")

# Optimize microstrip line
best_length, log_microstrip = optimize_microstrip(Z0, ZL, freq, epsilon_r)
l_opt = best_length[0]
print(f"Optimal Microstrip Line Length (l): {l_opt:.6e} meters")

# Plot convergence for lumped elements
gen = log_lumped.select("gen")
min_fits = log_lumped.select("min")
avg_fits = log_lumped.select("avg")

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

plt.title("Convergence of Genetic Algorithm (Lumped Elements)")
plt.show()

# Plot convergence for microstrip line
gen = log_microstrip.select("gen")
min_fits = log_microstrip.select("min")
avg_fits = log_microstrip.select("avg")

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

plt.title("Convergence of Genetic Algorithm (Microstrip Line)")
plt.show()
