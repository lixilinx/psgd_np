import numpy as np
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 
from tensor_rank_decomposition import trd_cost_grad
import time

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

Qx1, Qx2 = 0.1*np.eye(R), np.eye(I)
Qy1, Qy2 = 0.1*np.eye(R), np.eye(J) 
Qz1, Qz2 = 0.1*np.eye(R), np.eye(K) 
sqrt_eps = np.sqrt(np.finfo(np.float64).eps)
kron_Loss = []
kron_Times = []
kron_Iter = np.linspace(1,5000,5000)
kron_Iter.tolist()

t0 = time.time()
for num_iter in range(5000):    
    loss, grads = trd_cost_grad(T, [x, y, z])
    kron_Loss.append(loss)
    t1 = time.time()
    kron_Times.append(t1-t0)
    dx, dy, dz = sqrt_eps*np.random.randn(R, I), sqrt_eps*np.random.randn(R, J), sqrt_eps*np.random.randn(R, K)
    _, perturbed_grads = trd_cost_grad(T, [x + dx, y + dy, z + dz])
    Qx1, Qx2 = psgd.update_precond_kron(Qx1, Qx2, dx, perturbed_grads[0] - grads[0])
    Qy1, Qy2 = psgd.update_precond_kron(Qy1, Qy2, dy, perturbed_grads[1] - grads[1])
    Qz1, Qz2 = psgd.update_precond_kron(Qz1, Qz2, dz, perturbed_grads[2] - grads[2])
    x -= 0.5*psgd.precond_grad_kron(Qx1, Qx2, grads[0])
    y -= 0.5*psgd.precond_grad_kron(Qy1, Qy2, grads[1])
    z -= 0.5*psgd.precond_grad_kron(Qz1, Qz2, grads[2])

#plt.subplot(121)
#plt.loglog(Loss)
#plt.subplot(122)
#plt.loglog(Times,Loss)
