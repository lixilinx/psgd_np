"""Comparing PSGD with six different preconditioners:
GD:             PSGD with identity matrix as preconditioner
PSGD, diag:     PSGD with diagonal preconditioner
PSGD, scan:     PSGD with SCaling-And-Normalization preconditioner
PSGD, kron:     PSGD with Kronecker-product preconditioner
PSGD, splu:     PSGD with sparse LU decomposition preconditioner
PSGD, dense:    PSGD with dense preconditioner 
"""
import matplotlib.pyplot as plt

exec(open('demo_gd.py').read())
exec(open('demo_psgd_diag.py').read())
exec(open('demo_psgd_scan.py').read())
exec(open('demo_psgd_kron.py').read())
exec(open('demo_psgd_splu.py').read())
exec(open('demo_psgd_dense.py').read())

plt.legend(['Gradient descent', 'PSGD, diag', 'PSGD, scan', 'PSGD, kron', 'PSGD, splu', 'PSGD, dense'])
plt.savefig('comparison.png')