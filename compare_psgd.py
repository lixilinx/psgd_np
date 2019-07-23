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

ax1=plt.subplot(121)
ax1.plot(gd_Iter,gd_Loss,":")
ax1.plot(dense_Iter,dense_Loss,"-.")
ax1.plot(diag_Iter,diag_Loss,"--")
ax1.plot(kron_Iter,kron_Loss,"-")
ax1.plot(scan_Iter,scan_Loss,linestyle=(0, (5, 1)))
ax1.plot(splu_Iter,splu_Loss,linestyle=(0, (5, 10)) )
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.legend(['Gradient\n descent', 'PSGD,\n diag', 'PSGD,\n scan', 
            'PSGD,\n kron', 'PSGD,\n splu', 'PSGD,\n dense'],
             loc='lower left')

ax2=plt.subplot(122,sharey=ax1)
ax2.plot(gd_Times,gd_Loss,":")
ax2.plot(dense_Times,dense_Loss,"-.")
ax2.plot(diag_Times,diag_Loss,"--")
ax2.plot(kron_Times,kron_Loss,"-")
ax2.plot(scan_Times,scan_Loss,linestyle=(0, (5, 1)))
ax2.plot(splu_Times,splu_Loss,linestyle=(0, (5, 10)) )
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig('comparison.png')
