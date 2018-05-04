import numpy as np
import matplotlib.pyplot as plt

from tensor_rank_decomposition import trd_cost_grad

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

Loss = []
for num_iter in range(5000):
    loss, grads = trd_cost_grad(T, [x, y, z])
    Loss.append(loss)
    x -= 0.01*grads[0]
    y -= 0.01*grads[1]
    z -= 0.01*grads[2]
    
plt.semilogy(Loss)
