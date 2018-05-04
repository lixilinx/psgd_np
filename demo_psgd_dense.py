import numpy as np
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 
from tensor_rank_decomposition import trd_cost_grad

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

Q = 0.1*np.eye(R*(I + J + K))
sqrt_eps = np.sqrt(np.finfo(np.float64).eps)
Loss = []
for num_iter in range(5000):
    loss, grads = trd_cost_grad(T, [x, y, z])
    Loss.append(loss)
    dx, dy, dz = sqrt_eps*np.random.randn(R, I), sqrt_eps*np.random.randn(R, J), sqrt_eps*np.random.randn(R, K)
    _, perturbed_grads = trd_cost_grad(T, [x + dx, y + dy, z + dz])
    Q = psgd.update_precond_dense(Q, [dx, dy, dz], [perturbed_grads[0] - grads[0],
                                                    perturbed_grads[1] - grads[1],
                                                    perturbed_grads[2] - grads[2]])
    pre_grads = psgd.precond_grad_dense(Q, grads)
    x -= pre_grads[0]
    y -= pre_grads[1]
    z -= pre_grads[2]
    
plt.semilogy(Loss)
