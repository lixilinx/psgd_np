### Numpy implementation of PSGD
This package implements preconditioned stochastic gradient descent (PSGD) with Numpy. We have implemented dense preconditioner (dense), diagonal preconditioner (diag), sparse LU decomposition preconditioner (splu), Kronecker product preconditioner (kron), and SCaling-And-Normalization preconditioner (scan). 

Try 'hello_psgd.py' to see whether it works in your configurations. PSGD should find the global minimum of Rosenbrock function after about 200 iterations, as shown below: 

![alt text](https://github.com/lixilinx/psgd_np/blob/master/hello_psgd.png)

'rnn_add_problem_data_model_loss.py' is a benchmark problem written in Pytorch, and 'demo_psgd_....py' demonstrates the usage of different preconditioners. Pytorch (http://pytorch.org/) is required to run these demos. Verified on version 0.4 (there are considerable changes from 0.3 to 0.4) 

Please refer to the Tensorflow implementations (https://github.com/lixilinx/psgd_tf) for more information and benchmark problems, and https://arxiv.org/abs/1803.09383 for technical details. 
