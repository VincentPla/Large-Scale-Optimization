__author__ = "Dimitri Bouche, Vincent Plassier"
__email__ = "dimitri.bouche@ensae.fr, vincent.plassier@ens-paris-saclay.fr"


import scipy.special as special
import numpy as np


def gradient(K, x, noisy_y):
    return K.T.dot(K).dot(x) - K.T.dot(noisy_y)


def halleys_method(x, epsilon=1e-8, maxit=50):
    w = 1
    it = 0
    div = np.inf
    while it < maxit and div > epsilon:
        wprev = w
        f = w * np.exp(w) - x
        w = w - f / (np.exp(w) * (w + 1) - (w + 2) * f / (2 * w + 2))
        it += 1
        div = np.abs(w - wprev)
    return w


def prox_ent(gamma, beta, x):
    prox = np.zeros(x.shape[0])
    nu = gamma * beta
    for i in range(x.shape[0]):
        u = x[i] / nu - 1 - np.log(nu)
        if u > 100:
            prox[i] = nu * (u - np.log(u))
        elif u < -20:
            return 0
        else:
            # prox[i] = nu * special.lambertw((1 / nu) * np.exp(x[i] / nu - 1))
            prox[i] = nu * halleys_method(u)
    return prox


def proximal_gradient_ent(K, x0, noisy_y,
                          beta, gamma=0.0001,
                          lamb=1, maxit=1000, epsilon=1e-5):
    it = 0
    div = np.inf
    iterates = []
    iterates.append(x0.copy())
    while it < maxit and div > epsilon:
        # Compute gradient
        grad = gradient(K, iterates[it], noisy_y)
        y = iterates[it] - gamma * grad
        prox = prox_ent(gamma, beta, y)
        iterates.append(iterates[it] + lamb * (prox - iterates[it]))
        it += 1
        div = np.linalg.norm(iterates[it - 1] - iterates[it])
    return iterates


def prox_f(x, K, nu, noisy_y):
    s = noisy_y.shape[0]
    n = x.shape[0]
    inv = np.linalg.inv(K.T.dot(K) + (1 / nu) * np.eye(n))
    return inv.dot(K.T.dot(noisy_y) + (1 / nu) * x)

    
def prox1(x, gamma): 
    return np.real(special.lambertw(np.exp(x / gamma -np.log(gamma) -1))) / gamma


def prox2(x, y, R):
    return y + (x - y) * np.minimum(1, R / np.linalg.norm(x - y))


def douglas_rachford(K, x0, noisy_y,
                     beta=0.0001, gamma=0.0001,
                     lamb=1, maxit=1000, epsilon=1e-5):
    it = 0
    div = np.inf
    iterates = [x0]
    while it < maxit and div > epsilon:
        y = prox_f(iterates[it], K, gamma, noisy_y)
        z = prox_ent(gamma, beta, 2 * y - iterates[it])
        iterates.append(iterates[it] + lamb * (z - y))
        it += 1
        div = np.linalg.norm(iterates[it - 1] - iterates[it])
    return iterates


def PPXA(prox1, prox2, gamma, R, x1, x2, K, noisy_y, maxit=10 ** 3):
    L_inv = np.linalg.inv(np.eye(K.shape[1]) + K.T.dot(K))
    iterates = [L_inv.dot(x1 + K.T.dot(x2))]
    for it in range(maxit):
        y1, y2 = prox1(x1, gamma), prox2(x2, noisy_y, R)
        c = L_inv.dot(y1 + K.T.dot(y2))
        alpha, v = it / (it + 1), iterates[-1]
        x1, x2 = x1 + alpha * (2 * c - v - y1), x2 + alpha * (K.dot(2 * c - v) - y2)
        iterates.append(v + alpha * (c - v))
    return iterates
