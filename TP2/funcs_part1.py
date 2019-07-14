__author__ = "Dimitri Bouche, Vincent Plassier"
__email__ = "dimitri.bouche@ensae.fr, vincent.plassier@ens-paris-saclay.fr"

import numpy as np


def exponential_sampling(Tmin, Tmax, N):
    r = np.arange(N)
    return Tmin * np.exp(-r * np.log(Tmin / Tmax) / (N-1))


def regular_sampling(tmin, tmax, M):
    r = np.arange(M)
    return tmin + r / (M-1) * (tmax-tmin)


def construct_K(exp_samp, reg_samp):
    tT = np.tensordot(reg_samp, exp_samp, 0)
    return np.exp(- tT)


def construct_D(N):
    return np.roll(np.eye(N), shift=1, axis=1) - np.eye(N)


def generate_noisy_y(K, xtrue, noise_coeff=0.01):
    Kdx = K.dot(xtrue)
    sigma = noise_coeff * Kdx[0]
    M = K.shape[0]
    w = np.random.multivariate_normal(np.zeros(M), sigma ** 2 * np.eye(M))
    return Kdx + w


def sol_reg1(K, D, y, beta):
    return np.linalg.solve(K.T.dot(K) + beta * D.T.dot(D), K.T.dot(y))


def normalized_qerror(xpred, xtrue):
    return (np.linalg.norm(xpred - xtrue) / np.linalg.norm(xtrue)) ** 2


def gradient2(K, D, beta, x, noisy_y):
    return K.T.dot(K).dot(x) + beta * D.T.dot(D).dot(x) - K.T.dot(noisy_y)


def hessian2(K, D, beta):
    return K.T.dot(K) + beta * D.T.dot(D)


def projected_gradient(K, D, x0, noisy_y,
                       beta, gamma, bounds,
                       lamb=1, maxit=5000, epsilon=1e-5):
    it = 0
    div = np.inf
    iterates = []
    x = x0.copy()
    iterates.append(x0.copy())
    while it < maxit and div > epsilon:
        # Compute gradient
        grad = gradient2(K, D, beta, iterates[it], noisy_y)
        # Compute projection
        y = iterates[it] - gamma * grad
        proj = np.clip(y, bounds[0], bounds[1])
        # Update and record iterates
        iterates.append(iterates[it] + lamb * (proj - iterates[it]))
        it += 1
        # div = np.linalg.norm(iterates[it] - iterates[it - 1])
        div = np.linalg.norm(grad)
    return iterates


def gradient3(K, x, noisy_y):
    return K.T.dot(K).dot(x) - K.T.dot(noisy_y)


def hessian3(K):
    return K.T.dot(K)


def proximal_gradient(K, x0, noisy_y,
                      beta, gamma, lamb=1,
                      maxit=2000, epsilon=1e-3):
    it = 0
    div = np.inf
    iterates = []
    iterates.append(x0.copy())
    while it < maxit and div > epsilon:
        # Compute gradient
        grad = gradient3(K, iterates[it], noisy_y)
        y = iterates[it] - gamma * grad
        prox = np.sign(y) * np.maximum(0, np.abs(y) - gamma * beta)
        iterates.append(iterates[it] + lamb * (prox - iterates[it]))
        it += 1
        div = np.linalg.norm(iterates[it] - iterates[it - 1])
    return iterates
#
# def proximal_gradient(K, x0, noisy_y,
#                       beta, gamma, lamb=1,
#                       maxit=2000, epsilon=1e-3):
#     it = 0
#     div = np.inf
#     z0 = x0
#     while it < maxit and div > epsilon:
#         # Compute gradient
#         grad = gradient3(K, z0, noisy_y)
#         y = z0 - gamma * grad
#         x1 = np.sign(y) * np.maximum(0, np.abs(y) - gamma * beta)
#         z1 = x1 + lamb* (x1 - x0)
#         x0, z0 = x1, z1
#         it += 1
#         div = np.linalg.norm(grad)
#     return x0
