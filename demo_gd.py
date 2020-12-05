import numpy as np
import matplotlib.pyplot as plt
import time

from tensor_rank_decomposition import trd_cost_grad

I, J, K, R = 3, 5, 7, 11
np.random.seed(1)
T = np.random.randn(I, J, K)
x, y, z = np.random.randn(R, I), np.random.randn(R, J), np.random.randn(R, K)

gd_Loss = []
gd_Times = []
gd_Iter = np.linspace(1,5000,5000)
gd_Iter.tolist()
t0 = time.time()
for num_iter in range(5000):
    loss, grads = trd_cost_grad(T, [x, y, z])
    gd_Loss.append(loss)
    t1 = time.time()
    gd_Times.append(t1-t0)
    x -= 0.01*grads[0]
    y -= 0.01*grads[1]
    z -= 0.01*grads[2]

#plt.subplot(121)
#plt.loglog(Loss)
#plt.subplot(122)
#plt.loglog(Times,Loss)
