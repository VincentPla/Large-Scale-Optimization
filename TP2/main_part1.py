__author__ = "Dimitri Bouche, Vincent Plassier"
__email__ = "dimitri.bouche@ensae.fr, vincent.plassier@ens-paris-saclay.fr"


import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import funcs_part1
importlib.reload(funcs_part1)

# Plot parameters
plt.rcParams.update({"font.size": 25})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})

plt.show()
plt.ion()

########### PRELIMINARIES #############################################################################################
# Load the true signal
path = os.getcwd()
xtrue = np.loadtxt(path + "/x.txt")

# Exponential sampling
Tmin = 1
Tmax = 1000
N = 200
exp_samp = funcs_part1.exponential_sampling(Tmin, Tmax, N)

# Plot true signal
plt.figure(1)
plt.xscale("log")
plt.plot(exp_samp, xtrue)
plt.title("True signal")

# Regular sampling
M = 50
tmin = 0
tmax = 1.5
reg_samp = funcs_part1.regular_sampling(tmax, tmin, M)

# Construct K and D
K = funcs_part1.construct_K(exp_samp, reg_samp)
D = funcs_part1.construct_D(N)

# Generate noisy data
noisy_y = funcs_part1.generate_noisy_y(K, xtrue, noise_coeff=10)

# Plot noisy data
plt.figure(2)
plt.plot(reg_samp, noisy_y)
plt.title("Noisy y")


########### SMOOTHNESS PRIOR ONLY #####################################################################################
# Solution to first problem (smoothness prior only)
beta = 10
sol1 = funcs_part1.sol_reg1(K, D, noisy_y, beta)

# Plot solution
plt.figure(3)
plt.xscale("log")
plt.plot(exp_samp, sol1)

# Tune beta
beta_grid = np.geomspace(0.00001, 1000)
tuner = [funcs_part1.normalized_qerror(funcs_part1.sol_reg1(K, D, noisy_y, b), xtrue) for b in beta_grid]
# Plot
plt.figure(4)
plt.xscale("log")
plt.plot(beta_grid, tuner)
plt.xlabel("Regularization (log scale)")
plt.ylabel("Normalized square error")
plt.title("Tuning beta")
beta_star1 = beta_grid[np.argmin(tuner)]
# Plot the resulting signal (with optimal beta)
# Plot solution
sol_star1 = funcs_part1.sol_reg1(K, D, noisy_y, beta_star1)
plt.figure(5)
plt.xscale("log")
plt.plot(exp_samp, sol_star1)
plt.title("Reconstructed - Tuned beta - Smoothness prior only")


########### SMOOTHNESS PRIOR AND CONSTRAINTS ##########################################################################
# Solution to the second problem (smoothness and constraints)
# Constraints
bounds=(min(xtrue), max(xtrue))

# Random initialization
x0 = np.clip(np.random.normal(0.01, 0.05, 200), bounds[0], bounds[1])
# x0 = np.zeros(200)

# Chose gamma
gamma = 1 / (np.linalg.norm(K) ** 2)

# Chose beta
beta = 1

# Projected gradient
iterates2 = funcs_part1.projected_gradient(K, D, x0, noisy_y, beta, gamma, bounds, maxit=10000, epsilon=1e-6)
sol2 = iterates2[-1]

# Monitor convergence
errors2 = [funcs_part1.normalized_qerror(x, xtrue) for x in iterates2]
plt.figure(6)
plt.plot(errors2)
plt.xlabel("Iteration")
plt.ylabel("Normalized square error")
plt.title("Convergence of projected gradient")

# Plot solution
plt.figure(7)
plt.xscale("log")
plt.plot(exp_samp, sol2)


# Tune beta
beta_grid = np.geomspace(0.00001, 1000)
tuner = []
for b in beta_grid:
    iterates2 = funcs_part1.projected_gradient(K, D, x0, noisy_y, b, gamma, bounds, maxit=10000, epsilon=1e-6)
    sol2 = iterates2[-1]
    tuner.append(funcs_part1.normalized_qerror(sol2, xtrue))
# Plot
plt.figure(8)
plt.xscale("log")
plt.plot(beta_grid, tuner)
plt.xlabel("Regularization (log scale)")
plt.ylabel("Normalized square error")
plt.title("Tuning beta")
beta_star2 = beta_grid[np.argmin(tuner)]
# Plot the resulting signal (with optimal beta)
# Plot solution
iterates2 = funcs_part1.projected_gradient(K, D, x0, noisy_y, beta_star2, gamma, bounds, maxit=10000, epsilon=1e-6)
sol_star2 = iterates2[-1]
plt.figure(9)
plt.xscale("log")
plt.plot(exp_samp, sol_star2)
plt.title("Reconstructed - Tuned beta - Smoothness prior and constraints")


########### SPARSITY PRIOR ############################################################################################
# Solution to the second problem (smoothness and constraints)

# Random initialization
# x0 = np.clip(np.random.normal(0.01, 0.05, 200), bounds[0], bounds[1])
x0 = np.zeros(N)

# Chose gamma
gamma = 2 / (np.linalg.norm(K) ** 2)

# Projected gradient
beta = 10e-1
iterates3 = funcs_part1.proximal_gradient(K, x0, noisy_y, beta, gamma, epsilon=5e-6, maxit=10000)
sol3 = iterates3[-1]

# Monitor convergence
errors3 = [funcs_part1.normalized_qerror(x, xtrue) for x in iterates3]
plt.figure(10)
plt.plot(errors3)
plt.xlabel("Iteration")
plt.ylabel("Normalized square error")
plt.title("Convergence of proximal gradient")

# Plot solution
plt.figure(11)
plt.xscale("log")
plt.plot(exp_samp, sol3)

# Tune beta
beta_grid = np.geomspace(0.00001, 1000)
tuner = []
for b in beta_grid:
    iterates3 = funcs_part1.proximal_gradient(K, x0, noisy_y, b, gamma, epsilon=5e-6, maxit=10000)
    sol3 = iterates3[-1]
    tuner.append(funcs_part1.normalized_qerror(sol3, xtrue))
# Plot
plt.figure(12)
plt.xscale("log")
plt.plot(beta_grid, tuner)
plt.xlabel("Regularization (log scale)")
plt.ylabel("Normalized square error")
plt.title("Tuning beta")
beta_star3 = beta_grid[np.argmin(tuner)]
# Plot the resulting signal (with optimal beta)
# Plot solution
iterates3 = funcs_part1.proximal_gradient(K, x0, noisy_y, beta_star3, gamma, epsilon=5e-6, maxit=10000)
sol_star3 = iterates3[-1]
plt.figure(13)
plt.xscale("log")
plt.plot(exp_samp, sol_star3)
plt.title("Reconstructed - Tuned beta - Sparsity prior")
