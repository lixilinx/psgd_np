import numpy as np
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 
from tensor_rank_decomposition import trd_cost_grad

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

sqrt_eps = np.sqrt(np.finfo(np.float64).eps)
num_para = R*(I + J + K)
r = 5 # order of SPLU preconditioner 
L12 = 0.1*np.concatenate([np.eye(r), np.zeros([num_para - r, r])], axis=0)
l3 = 0.1*np.ones([num_para - r, 1])
U12 = np.concatenate([np.eye(r), np.zeros([r, num_para - r])], axis=1)
u3 = np.ones([num_para - r, 1])
Loss = []
for num_iter in range(5000):
    loss, grads = trd_cost_grad(T, [x, y, z])
    Loss.append(loss)
    dx, dy, dz = sqrt_eps*np.random.randn(R, I), sqrt_eps*np.random.randn(R, J), sqrt_eps*np.random.randn(R, K)
    _, perturbed_grads = trd_cost_grad(T, [x + dx, y + dy, z + dz])
    L12, l3, U12, u3 = psgd.update_precond_splu(L12, l3, U12, u3, [dx, dy, dz],                          
                                                [perturbed_grads[0] - grads[0],                              
                                                 perturbed_grads[1] - grads[1],                           
                                                 perturbed_grads[2] - grads[2]])
    pre_grads = psgd.precond_grad_splu(L12, l3, U12, u3, grads)
    x -= 0.5*pre_grads[0]
    y -= 0.5*pre_grads[1]
    z -= 0.5*pre_grads[2]
    
plt.semilogy(Loss)