import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

import preconditioned_stochastic_gradient_descent as psgd 
from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# initialize preconditioner with identity matrix
num_para = sum([np.prod(W.size()) for W in Ws])
r = 10 # order of SPLU preconditioner 
L12 = np.concatenate([np.eye(r), np.zeros([num_para - r, r])], axis=0)
l3 = np.ones([num_para - r, 1])
U12 = np.concatenate([np.eye(r), np.zeros([r, num_para - r])], axis=1)
u3 = np.ones([num_para - r, 1])
# begin iteration here
step_size = 0.02
grad_norm_clip_thr = 1.0
Loss = []
for num_iter in range(10000):
    x, y = get_batches( )
    
    # calculate loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws, create_graph=True)
    Loss.append(loss.data.numpy())
    
    # update preconditioners
    delta = [torch.randn(W.size()) for W in Ws]
    grad_delta = sum([torch.sum(g*d) for (g, d) in zip(grads, delta)])
    hess_delta = grad(grad_delta, Ws)
    L12, l3, U12, u3 = psgd.update_precond_splu(L12, l3, U12, u3, 
                                                [d.data.numpy() for d in delta],                                                                
                                                [h.data.numpy() for h in hess_delta])
    
    # update Ws
    pre_grads = psgd.precond_grad_splu(L12, l3, U12, u3, 
                                       [g.data.numpy() for g in grads])
    grad_norm = np.sqrt(sum([np.sum(g*g) for g in pre_grads]))
    if grad_norm > grad_norm_clip_thr:
        step_adjust = grad_norm_clip_thr/grad_norm
    else:
        step_adjust = 1.0
    for i in range(len(Ws)):
        Ws[i].data = Ws[i].data - step_adjust*step_size*torch.FloatTensor(pre_grads[i])
        
    if num_iter % 100 == 0:
        print('training loss: {}'.format(Loss[-1]))
    
plt.semilogy(Loss)