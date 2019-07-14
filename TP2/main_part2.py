__author__ = "Dimitri Bouche, Vincent Plassier"
__email__ = "dimitri.bouche@ensae.fr, vincent.plassier@ens-paris-saclay.fr"


import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import funcs_part1
import funcs_part2
importlib.reload(funcs_part1)
importlib.reload(funcs_part2)

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



########### FORWARD BACKWARD ##########################################################################################

bounds=(min(xtrue), max(xtrue))

# Random initialization
x0 = np.clip(np.random.normal(0.01, 0.05, 200), bounds[0], bounds[1])
# x0 = np.zeros(200)

# Chose gamma
gamma = 0.01 / (np.linalg.norm(K) ** 2)

# Chose beta
beta = 10e-3

iterates = funcs_part2.proximal_gradient_ent(K, x0, noisy_y, beta=beta, gamma=gamma, lamb=1, maxit=1000, epsilon=1e-6)

# Monitor convergence
errors = [funcs_part1.normalized_qerror(x, xtrue) for x in iterates]
plt.figure(3)
plt.plot(errors )
plt.xlabel("Iteration")
plt.ylabel("Normalized square error")
plt.title("Convergence of forward backward")

print(errors[-1])


plt.figure(4)
plt.xscale("log")
plt.plot(exp_samp, iterates[-1])
plt.title("Reconstructed - Entropy regularization - Forward backward")


# # Tune beta
# beta_grid = np.geomspace(1e-10, 1e-1)
# tuner = []
# for b in beta_grid:
#     iterates = funcs_part2.proximal_gradient_ent(K, x0, noisy_y, beta=beta, gamma=gamma, lamb=1, maxit=1000, epsilon=1e-5)
#     sol = iterates[-1]
#     tuner.append(funcs_part1.normalized_qerror(sol, xtrue))
# # Plot
# plt.figure(5)
# plt.xscale("log")
# plt.plot(beta_grid, tuner)
# plt.xlabel("Regularization (log scale)")
# plt.ylabel("Normalized square error")
# plt.title("Tuning beta")
# beta_star2 = beta_grid[np.argmin(tuner)]
# # Plot the resulting signal (with optimal beta)
# # Plot solution
# iterates2 = funcs_part1.projected_gradient(K, D, x0, noisy_y, beta_star2, gamma, bounds, maxit=1000, epsilon=1e-5)
# sol_star2 = iterates2[-1]
# plt.figure(6)
# plt.xscale("log")
# plt.plot(exp_samp, sol_star2)
# plt.title("Reconstructed - Tuned beta - Entropy regularization")



########### DOUGLAS RACHFORD ##########################################################################################

# Chose beta
beta = 10e-3

iterates2 = funcs_part2.douglas_rachford(K, x0, noisy_y, beta=beta, gamma=gamma, lamb=1, maxit=1000, epsilon=1e-5)


# Monitor convergence
errors2 = [funcs_part1.normalized_qerror(x, xtrue) for x in iterates2]
plt.figure(7)
plt.plot(errors2)
plt.xlabel("Iteration")
plt.ylabel("Normalized square error")
plt.title("Convergence of Douglas Rachford")


plt.figure(8)
plt.xscale("log")
plt.plot(exp_samp, iterates2[-1])
plt.title("Reconstructed - Entropy regularization - Douglas Rachford")

print(errors2[-1])



########### PPXA ##########################################################################################

eta, gamma, sigma = 1, 10, .01 * K.dot(xtrue)[0]
R = eta * M * sigma ** 2
x1 , x2 = np.zeros(N), np.zeros(M)

iterates3 = funcs_part2.PPXA(prox1, prox2, gamma, R, x1, x2, K, noisy_y, maxit=1000)


# Monitor convergence
errors3 = [funcs_part1.normalized_qerror(x, xtrue) for x in iterates3]
plt.figure(9)
plt.plot(errors3)
plt.xlabel("Iteration")
plt.ylabel("Normalized square error")
plt.title("Convergence of PPXA")


plt.figure(10)
plt.xscale("log")
plt.plot(exp_samp, iterates3[-1])
plt.title("Reconstructed - PPXA")

print(errors3[-1])
