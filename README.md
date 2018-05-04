### Numpy implementation of PSGD
Please try 'hello_psgd.py' to see whether it works in your configurations. PSGD should find the global minimum of Rosenbrock function after about 200 iterations, as shown below: 

![alt text](https://github.com/lixilinx/psgd_np/blob/master/hello_psgd.png)

We demonstrate the usage of different preconditioners with a tensor rank decomposition problem. Running file 'compare_psgd.py' should give the following comparison results:

![alt text](https://github.com/lixilinx/psgd_np/blob/master/comparison.png)

We see that all preconditioners help to improve convergence. Note that these demos are for mathematical optimizations, and do not explicitly use any second order derivative. However, PSGD is originally designed for stochastic optimizations, and belongs to second order methods. Please check https://github.com/lixilinx/psgd_tf and https://github.com/lixilinx/psgd_torch for more information and examples implemented in Tensorflow and Pytorch.
