import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# begin iteration here
step_size = 0.02
grad_norm_clip_thr = 1.0
Loss = []
for num_iter in range(10000):
    x, y = get_batches( )
    
    # calculate loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws)
    Loss.append(loss.data.numpy())
        
    # update Ws
    grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in grads]))
    grad_norm = grad_norm.data[0]
    if grad_norm > grad_norm_clip_thr:
        step_adjust = grad_norm_clip_thr/grad_norm
    else:
        step_adjust = 1.0
    for i in range(len(Ws)):
        Ws[i].data = Ws[i].data - step_adjust*step_size*grads[i].data
        
    if num_iter % 100 == 0:
        print('training loss: {}'.format(Loss[-1]))
    
plt.semilogy(Loss)
