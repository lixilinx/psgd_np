import numpy as np
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 
from tensor_rank_decomposition import trd_cost_grad

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

Qx, Qy, Qz = 0.1*np.ones_like(x), 0.1*np.ones_like(y), 0.1*np.ones_like(z)
sqrt_eps = np.sqrt(np.finfo(float).eps)
Loss = []
for num_iter in range(5000):    
    loss, grads = trd_cost_grad(T, [x, y, z])
    Loss.append(loss)
    dx, dy, dz = sqrt_eps*np.random.randn(R, I), sqrt_eps*np.random.randn(R, J), sqrt_eps*np.random.randn(R, K)
    _, perturbed_grads = trd_cost_grad(T, [x + dx, y + dy, z + dz])
    Qx = psgd.update_precond_diag(Qx, dx, perturbed_grads[0] - grads[0])
    Qy = psgd.update_precond_diag(Qy, dy, perturbed_grads[1] - grads[1])
    Qz = psgd.update_precond_diag(Qz, dz, perturbed_grads[2] - grads[2])
    x -= 0.5*psgd.precond_grad_diag(Qx, grads[0])
    y -= 0.5*psgd.precond_grad_diag(Qy, grads[1])
    z -= 0.5*psgd.precond_grad_diag(Qz, grads[2])
    
plt.semilogy(Loss)