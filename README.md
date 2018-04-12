*Coming updates: I am implementing a sparse LU decomposition preconditioner. It is a limited-memory preconditioner, performs well, and works with a list of tensor parameters having arbitrary shapes.* 
### Numpy implementation of PSGD
This package implements preconditioned stochastic gradient descent (PSGD) with Numpy. We have implemented diagonal preconditioner (diag, related to equilibrated SGD), SCaling-And-Normalization preconditioner (scan, related to batch normalization), Kronecker product preconditioner (kron), and dense preconditioner (dense). Please refer to the Tensorflow implementations (https://github.com/lixilinx/psgd_tf) for more information related to PSGD.

Try 'hello_psgd.py' to see whether it works in your configurations. PSGD should find the global minimum of Rosenbrock function after about 200 iterations, as shown below: 

![alt text](https://github.com/lixilinx/psgd_np/blob/master/hello_psgd.png)

'rnn_add_problem_data_model_loss.py' is a benchmark problem, and 'demo_psgd_....py' demonstrates the usage of different preconditioners. Pytorch (http://pytorch.org/) is required to run these demos. Running 'compare_psgd.py' should give the following typical results: 

![alt text](https://github.com/lixilinx/psgd_np/blob/master/compare_psgd.png)

Comparisons on more benchmark problems are given in the Tensorflow implementations. 
