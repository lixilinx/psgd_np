"""Comparing PSGD with five different preconditioners on rnn_add_problem:
SGD:            PSGD with identity matrix as preconditioner
PSGD, diag:     PSGD with diagonal preconditioner (related to equilibrated SGD)
PSGD, scan:     PSGD with SCaling-And-Normalization preconditioner (related to batch normalization)
PSGD, kron:     PSGD with Kronecker-product preconditioner
PSGD, dense:    PSGD with dense preconditioner 
"""
import matplotlib.pyplot as plt

exec(open('demo_sgd.py').read())
exec(open('demo_psgd_diag.py').read())
exec(open('demo_psgd_scan.py').read())
exec(open('demo_psgd_kron.py').read())
exec(open('demo_psgd_dense.py').read())

plt.legend(['SGD', 'PSGD, diag', 'PSGD, scan', 'PSGD, kron', 'PSGD, dense'])
plt.savefig('comparison.pdf')