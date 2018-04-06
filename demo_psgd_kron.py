import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable, grad

import preconditioned_stochastic_gradient_descent as psgd 
from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# initialize preconditioners with identity matrices
Qs = [[np.eye(W.size()[0]), np.eye(W.size()[1])] for W in Ws]
# begin iteration here
step_size = 0.02
grad_norm_clip_thr = 1.0
Loss = []
for num_iter in range(10000):
    x, y = get_batches( )
    
    # calculate the loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws, create_graph=True)
    Loss.append(loss.data.numpy()[0])
    
    # update preconditioners
    Q_update_gap = max(int(np.floor(np.log10(num_iter + 1.0))), 1)
    if num_iter % Q_update_gap == 0:# let us update Q less frequently
        delta = [Variable(torch.randn(W.size())) for W in Ws]
        grad_delta = sum([torch.sum(g*d) for (g, d) in zip(grads, delta)])
        hess_delta = grad(grad_delta, Ws)
        Qs = [psgd.update_precond_kron(q[0], q[1], dw.data.numpy(), dg.data.numpy()) for (q, dw, dg) in zip(Qs, delta, hess_delta)]
    
    # update Ws
    pre_grads = [psgd.precond_grad_kron(q[0], q[1], g.data.numpy()) for (q, g) in zip(Qs, grads)]
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