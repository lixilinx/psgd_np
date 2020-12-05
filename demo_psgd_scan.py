import numpy as np
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 
from tensor_rank_decomposition import trd_cost_grad
import time

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

qx1, qx2 = np.stack([np.ones(R), np.zeros(R)]), 0.1*np.ones((1, I))
qy1, qy2 = np.stack([np.ones(R), np.zeros(R)]), 0.1*np.ones((1, J))
qz1, qz2 = np.stack([np.ones(R), np.zeros(R)]), 0.1*np.ones((1, K))
sqrt_eps = np.sqrt(np.finfo(np.float64).eps)
scan_Loss = []
scan_Times = []
scan_Iter = np.linspace(1,5000,5000)
scan_Iter.tolist()

t0 = time.time()
for num_iter in range(5000):    
    loss, grads = trd_cost_grad(T, [x, y, z])
    scan_Loss.append(loss)
    t1 = time.time()
    scan_Times.append(t1-t0)
    dx, dy, dz = sqrt_eps*np.random.randn(R, I), sqrt_eps*np.random.randn(R, J), sqrt_eps*np.random.randn(R, K)
    _, perturbed_grads = trd_cost_grad(T, [x + dx, y + dy, z + dz])
    qx1, qx2 = psgd.update_precond_scan(qx1, qx2, dx, perturbed_grads[0] - grads[0])
    qy1, qy2 = psgd.update_precond_scan(qy1, qy2, dy, perturbed_grads[1] - grads[1])
    qz1, qz2 = psgd.update_precond_scan(qz1, qz2, dz, perturbed_grads[2] - grads[2])
    x -= 0.5*psgd.precond_grad_scan(qx1, qx2, grads[0])
    y -= 0.5*psgd.precond_grad_scan(qy1, qy2, grads[1])
    z -= 0.5*psgd.precond_grad_scan(qz1, qz2, grads[2])

#plt.subplot(121)
#plt.loglog(Loss)
#plt.subplot(122)
#plt.loglog(Times,Loss)
