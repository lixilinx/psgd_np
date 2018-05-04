"""fit tensor t_{ijk} with sum_r x_{ri}*y_{rj}*z_{rk}
"""
import numpy as np
# return decomposition cost and grads wrt x, y, and z
def trd_cost_grad(T, xyz):
    eta = 1e-6 # without regulization, decomposition has scaling ambiguity
    I, J, K = T.shape
    x, y, z = xyz
    R = x.shape[0]
    
    A = np.zeros_like(T)
    for r in range(R):
        A += np.reshape(x[r], (I,1,1))*np.reshape(y[r], (1,J,1))*np.reshape(z[r], (1,1,K))
    E = T - A
    cost = np.sum(E*E) + eta*(np.sum(x*x) + np.sum(y*y) + np.sum(z*z))
    # actually, these are 0.5*gradients
    grad_x, grad_y, grad_z = eta*x, eta*y, eta*z
    for r in range(R):
        grad_x[r] -= np.sum(E*np.reshape(y[r], (1,J,1))*np.reshape(z[r], (1,1,K)), axis=(1,2))
        grad_y[r] -= np.sum(E*np.reshape(x[r], (I,1,1))*np.reshape(z[r], (1,1,K)), axis=(0,2))
        grad_z[r] -= np.sum(E*np.reshape(x[r], (I,1,1))*np.reshape(y[r], (1,J,1)), axis=(0,1))
        
    return cost, [grad_x, grad_y, grad_z]
        
