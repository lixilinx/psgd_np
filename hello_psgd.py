"""A 'hello world' example of PSGD on minimizing Rosenbrock function
"""
import numpy as np
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd

def Rosenbrock(x):
    valu = 100.0*(x[1] - x[0]**2)**2 + (1.0 - x[0])**2
    grad = np.array([-400.0*x[0]*(x[1] - x[0]**2) - 2.0*(1.0 - x[0]),
                     200.0*(x[1] - x[0]**2)])
    
    return valu, grad


x = np.array([-1.0, 1.0])
eps = np.finfo(type(x[0])).eps
Q = 0.1*np.eye(2)   # initialize Q with small values; otherwise, diverge
values = []
for i in range(500):
    v, g = Rosenbrock(x)
    values.append(v)
    
    dx = np.sqrt(eps)*np.random.randn(2)
    _, perturbed_g = Rosenbrock(x + dx)
    
    # dense preconditioner assumes a list of delta_x and delta_g 
    Q = psgd.update_precond_dense(Q, [dx], [perturbed_g - g], 0.2)
    pre_g = psgd.precond_grad_dense(Q, [g])
    x = x - 0.5*pre_g[0]
    
plt.semilogy(values)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("hello_psgd.png")
plt.show()
