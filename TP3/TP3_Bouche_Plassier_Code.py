__author__ = "Dimitri Bouche, Vincent Plassier"
__email__ = "dimitri.bouche@ensar.fr, vincent.plassier@ens-paris-saclay.fr"


import scipy.io as io
import scipy.sparse as sparse
import scipy.sparse.linalg as sparselinalg
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import os
import numpy as np
import time


# Plot parameters
plt.rcParams.update({"font.size": 25})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})


# Sizes parameters, will remain fixed
n1, n2 = 90, 90
N = n1 * n2

m1, m2 = 90, 180
M = m1 * m2


def build_y(H, x, sigma=1):
    noise = np.random.normal(0, sigma, (H.shape[0], 1))
    return H.dot(x) + noise


def psi_prime(u, delta):
    return u / (delta * np.sqrt(delta ** 2 + u ** 2))


def gradient(H, G, x, y, delta, lamb):
    Hx = H.dot(x)
    Gx = G.dot(x)
    return H.T.dot(Hx) - H.T.dot(y)[:, 0] + lamb * G.T.dot(psi_prime(Gx, delta))


def lipschitz_constant(H, G, lamb, delta):
    uH, sH, vtH = sparselinalg.svds(H, k=1)
    uG, sG, vtG = sparselinalg.svds(G, k=1)
    return sH[0] + (lamb / delta ** 2) * sG[0]


def gradient_descent(H, G, x0, y, delta, lamb, gamma, eps=1e-3 * np.sqrt(N), maxit=3000):
    it = 0
    grad_norm = np.inf
    x = x0.copy()
    grad_norms = []
    timing_list = []
    while it <= maxit and grad_norm > eps:
        start = time.clock()
        grad = gradient(H, G, x, y, delta, lamb)
        x -= gamma * grad
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        it += 1
        end = time.clock()
        timing_list.append(end - start)
        print(grad_norm)
    return x, grad_norms, np.cumsum(timing_list)


def create_linear_operator(x, H, G, lamb, delta):
    Gx = G.dot(x)
    D = sparse.diags(psi_prime(np.abs(Gx), delta) / np.abs(Gx)).tocsc()

    def matvec(u):
        Hu = H.dot(u)
        Gu = G.dot(u)
        return H.T.dot(Hu) + lamb * G.T.dot(D).dot(Gu)
    N = H.shape[1]
    return sparse.linalg.LinearOperator(shape=(N, N), matvec=matvec, rmatvec=matvec)


def mm_quadratic(x0, y, H, G, lamb, delta, theta=1.0, maxit=2000, eps=1e-3 * np.sqrt(N)):
    it = 0
    grad_norm = np.inf
    x = x0.copy()
    grad_norms = []
    timing_list = []
    while it <= maxit and grad_norm > eps:
        start = time.clock()
        linop = create_linear_operator(x, H, G, lamb, delta)
        grad = gradient(H, G, x, y, delta, lamb)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        # We set a very low number of iteration which results in a low precision in the solution,
        # But an approximate direction is enough in that case and it speeds up the process dramatically
        xsol, success = sparselinalg.bicg(linop, b=grad, maxiter=50)
        x -= theta * xsol
        it += 1
        end = time.clock()
        timing_list.append(end - start)
        print(grad_norm)
    return x, grad_norms, np.cumsum(timing_list)


def pinv_memory_grad(x, D, H, G, delta, lamb):
    HD = (H.dot(D))
    Gx = G.dot(x)
    Diag = sparse.diags(psi_prime(np.abs(Gx), delta) / np.abs(Gx)).tocsc()
    GD = G.dot(D)
    P = linalg.pinv(HD.T.dot(HD) + lamb * GD.T.dot(Diag.dot(GD)))
    return P


def mm_memory_gradient(x0, y, H, G, lamb, delta, theta=1.0, maxit=3000, eps=1e-3 * np.sqrt(N)):
    it = 0
    grad_norm = np.inf
    x = x0.copy()
    grad_norms = []
    timing_list = []
    while it <= maxit and grad_norm > eps:
        start = time.clock()
        grad = gradient(H, G, x, y, delta, lamb)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        if it == 0:
            D = -grad.reshape((N, 1))
        else:
            D = np.concatenate((-grad.reshape((N, 1)), (x - xprev).reshape((N, 1))), axis=1)
        Pinv = pinv_memory_grad(x, D, H, G, delta, lamb)
        u = -Pinv.dot(D.T.dot(grad))
        xprev = x.copy()
        x += theta * D.dot(u)
        it += 1
        end = time.clock()
        timing_list.append(end - start)
        print(grad_norm)
    return x, grad_norms, np.cumsum(timing_list)


def blockMM_linear_op(G, GD, H, lamb, J):
    H_J, G_J = H[:, J], G[:, J]
    def matvec(u):
        Hu = H_J.dot(u)
        Gu = G_J.dot(u)
        return H_J.T.dot(Hu) + lamb * GD.dot(Gu)
    return sparse.linalg.LinearOperator(shape=(len(J), len(J)), matvec=matvec, rmatvec=matvec)


def mm_block_coordinate(x0, y, J, H, G, theta, lamb, delta, discount=1, maxit=1000, maxiter=10, eps=1e-4 * np.sqrt(N)):
    timing_list = []
    grad_norms = []
    start = time.clock()
    x = x0.copy()
    gradf = gradient(H, G, x, y, delta, lamb)
    grad_norm = np.linalg.norm(gradf)
    K, r = N // J * np.arange(J + 1, dtype=int), np.zeros(J + 1, dtype=int)
    r[: N % J] = np.arange(1, N % J + 1)
    K += r
    it = 0
    while grad_norm > eps and it < maxit:
        if it != 0:
            start = time.clock()
        grad_norms.append(grad_norm)
        # print('grad_norm', grad_norm)
        Gx = G.dot(x)
        D = sparse.diags(psi_prime(np.abs(Gx), delta) / np.abs(Gx)).tocsr()
        GD = G.T.dot(D)
        j = it % J
        rows_up = np.arange(K[j], K[j + 1])
        linop = blockMM_linear_op(G, GD[rows_up], H, lamb, rows_up)
        xsol, success = sparselinalg.bicg(linop, b=gradf[rows_up], maxiter=maxiter)
        x[rows_up] = x[rows_up] - theta * xsol
        it, theta = it + 1, theta * discount
        gradf = gradient(H, G, x, y, delta, lamb)
        grad_norm = np.linalg.norm(gradf)
        end = time.clock()
        timing_list.append(end - start)
        print(grad_norm)
    return x, grad_norms, np.cumsum(timing_list)


def mm_parallel_coordinate(x0, y, H, G, lamb, delta, theta, maxit=100, eps=1e-3 * np.sqrt(N)):
    timing_list = []
    grad_norms = []
    start = time.clock()
    H_abs, G_abs = np.abs(H), np.abs(G)
    H1, G1 = np.sum(H_abs, axis=1), np.sum(G_abs, axis=1)
    b1 = sparse.csc_matrix.multiply(H_abs, H1).sum(axis=0)
    gradf = gradient(H, G, x0, y, delta, lamb)
    grad_norm = np.linalg.norm(gradf)
    it = 0
    while grad_norm > eps and it < maxit:
        # print('grad_norm', grad_norm)
        grad_norms.append(grad_norm)
        if it != 0:
            start = time.clock()
        Gx = G.dot(x0)
        psi_prime_x = psi_prime(Gx, delta) / Gx
        b1 = sparse.csc_matrix.multiply(H_abs, H1).sum(axis=0)
        b2_1 = sparse.csc_matrix.multiply(G_abs.T, psi_prime_x)
        b2 = sparse.csc_matrix.multiply(b2_1.T, G1).sum(axis=0)
        b = 1 / np.array(b1 + lamb * b2).flatten()
        # theta = 0.1 / np.sqrt(np.linalg.norm(b))
        x0 = (x0 - theta * np.dot(np.diag(b), gradf)).copy()
        it = it + 1
        gradf = gradient(H, G, x0, y, delta, lamb)
        grad_norm = np.linalg.norm(gradf)
        end = time.clock()
        timing_list.append(end - start)
        print(grad_norm)
    return x0, grad_norms, np.cumsum(timing_list)


def signal_noise_ratio(xrecons, xtrue):
    return 10 * np.log10(np.linalg.norm(xtrue)**2 / np.linalg.norm(xtrue - xrecons)**2)


# ############### LOAD AND BUILD THE NECESSARY DATA ####################################################################
path = os.getcwd() + "/data/"
H = io.loadmat(path + "H.mat")["H"]
x = io.loadmat(path + "x.mat")["x"]
G = io.loadmat(path + "G.mat")["G"]

# Show x
plt.figure()
plt.imshow(x.reshape((n1, n2), order="F"))

# Build y
y = build_y(H, x)

# Show rectangular y
plt.figure()
plt.imshow(y.reshape((m1, m2), order="F"))

# Parameters given
lamb = 0.13
delta = 0.02

# Initial point, set very small random values to avoid division by zero error
# when defining the first majorant in memory gradient
x0 = np.abs(np.random.normal(0, 0.001, N))


# #################### GRADIENT DESCENT ###############################################################################
# Pace for gradient descent
L = lipschitz_constant(H, G, lamb, delta)
gamma = 0.1 / L

# Gradient descent
xgd, conv_gd, timer_gd  = gradient_descent(H, G, x0, y, delta, lamb, gamma, maxit=5000)

# Plot reconstructed signal with gradient descent
plt.figure()
plt.imshow(xgd.reshape((n1, n2), order="F"))


# ################### MM QUADRATIC #####################################################################################
theta = 1.5

xmmq, conv_mmq, timer_mmq = mm_quadratic(x0, y, H, G, lamb, delta, theta, maxit=3000)

# Plot reconstructed signal with gradient descent
plt.figure()
plt.imshow(xmmq.reshape((n1, n2), order="F"))


# ################## MM MEMORY GRADIENT ################################################################################
theta = 1.5

xmmg, conv_mmg, timer_mmg  = mm_memory_gradient(x0, y, H, G, lamb, delta, theta, maxit=3000, eps=1e-3 * np.sqrt(N))

# Plot reconstructed signal with gradient descent
plt.figure()
plt.imshow(xmmg.reshape((n1, n2), order="F"))


# ################## MM BLOCK COORDINATES ##############################################################################
J = 4
theta = 1.5

xmmb, conv_mmb, timer_mmb = mm_block_coordinate(x0, y, J, H, G, theta, lamb, delta, maxit=1000, maxiter=50, eps=1e-3 * np.sqrt(N))

# Plot reconstructed signal with gradient descent
plt.figure()
plt.imshow(xmmb.reshape((n1, n2), order="F"))


# ################### MM PARALLEL ######################################################################################
theta = 2.0

xmmp, conv_mmp, timer_mmp = mm_parallel_coordinate(x0, y, H, G, lamb, delta, theta, maxit=5000, eps=1e-3 * np.sqrt(N))


# ################## SPEED COMPARISON ##################################################################################
# Plot performance
plt.figure()
plt.semilogy()
plt.plot(timer_gd, conv_gd, label="Gradient descent")
plt.plot(timer_mmq, conv_mmq, label="MM Quadratic")
plt.plot(timer_mmg, conv_mmg, label="MM Memory gradient")
plt.plot(timer_mmb, conv_mmb, label="MM Block")
# plt.plot(timer_mmp, conv_mmp, label="MM Parallel")
plt.xlabel("CPU time elapsed (time.clock() on Unix)")
plt.ylabel("$L_2$ norm of gradient")
plt.legend()
plt.title("Convergence comparison")

plt.figure()
plt.semilogy()
plt.plot(np.cumsum(timer_mmp), conv_mmp, label="MM Parallel")
plt.xlabel("CPU time elapsed (time.clock() on Unix)")
plt.ylabel("$L_2$ norm of gradient")
plt.legend()
plt.title("Convergence of MM parallel")


# ################## OPTIMIZATION OF PARAMETERS ########################################################################
lambda_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
delta_grid = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.5]
theta = 1.0

optimal_val = - np.inf
optimal_params = None
for lamb in lambda_grid:
    for delta in delta_grid:
        xmmg, conv_mmg, timer_mmg = mm_memory_gradient(x0, y, H, G, lamb, delta, theta, maxit=10000, eps=1e-3 * np.sqrt(N))
        sigratio = signal_noise_ratio(xmmg, x)
        if sigratio > optimal_val:
            optimal_val = sigratio
            optimal_params = (lamb, delta)
    print(lamb)

# optimal version
xmmg_opti, conv_mmg_opti, timer_mmg_opti = mm_memory_gradient(x0, y, H, G, optimal_params[0], optimal_params[1],
                                               theta, maxit=3000, eps=1e-3 * np.sqrt(N))

# version with more detail but more noise
xmmg_noisy, conv_mmg_noisy, timer_mmg_noisy = mm_memory_gradient(x0, y, H, G, 1.0, 0.1,
                                               theta, maxit=3000, eps=1e-3 * np.sqrt(N))


fig, axes = plt.subplots(ncols=2)
axes[0].imshow(xmmg_opti.reshape((n1, n2), order="F"))
axes[1].imshow(xmmg_noisy.reshape((n1, n2), order="F"))
axes[0].set_title("Optimal SNR - $\lambda=1$ and $\delta=0.01$")
axes[1].set_title("$\lambda=1$ and $\delta=0.1$")