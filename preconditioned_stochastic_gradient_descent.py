# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:58:57 2017

Updated in April 2018: added diagonal and SCaling-And-Normalization (scan) preconditioners
                       added splu preconditioner; removed diagonal loading

Numpy functions for preconditioned SGD

@author: XILIN
"""
import numpy as np
import scipy

_tiny = np.finfo('float32').tiny   # to avoid dividing by zero
#_diag_loading = 1e-9   # to avoid numerical difficulty when solving triangular systems


###############################################################################
def update_precond_dense(Q, dxs, dgs, step=0.01, diag=0):
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner with positive diagonal entries 
    dxs: list of perturbations of parameters
    dgs: list of perturbations of gradients
    step: normalized step size in [0, 1]
    diag: see the code for details
    """
    dx = np.concatenate([np.reshape(x, [-1, 1]) for x in dxs])
    dg = np.concatenate([np.reshape(g, [-1, 1]) for g in dgs])
    
    #max_diag = np.max(np.diag(Q))
    #Q = Q + np.diag(np.clip(_diag_loading*max_diag - np.diag(Q), 0.0, max_diag))
    
    a = Q.dot(dg)
    b = scipy.linalg.solve_triangular(Q, dx, trans=1, lower=False)
    
    grad = np.triu(a.dot(a.T) - b.dot(b.T))
    if diag:
        step0 = step/(np.max(np.abs(np.diag(grad))) + _tiny)
    else:
        step0 = step/(np.max(np.abs(grad)) + _tiny)
        
    return Q - step0*grad.dot(Q)


def precond_grad_dense(Q, grads):
    """
    return preconditioned gradient using dense preconditioner
    Q: Cholesky factor of preconditioner
    grads: list of gradients
    """
    grad = [np.reshape(g, [-1, 1]) for g in grads]
    lens = [g.shape[0] for g in grad]
    grad = np.concatenate(grad)
    grad = Q.T.dot(Q.dot(grad))
    
    pre_grads = []
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(np.reshape(grad[idx : idx + lens[i]], grads[i].shape))
        idx = idx + lens[i]
        
    return pre_grads



###############################################################################
def update_precond_diag(Q, dx, dg, step=0.01):
    """
    update diagonal preconditioner
    Q: diagonal preconditioner
    dx: perturbation of parameter
    dg: perturbation of gradient
    step: normalized step size
    """
    grad = np.sign(np.abs(Q*dg) - np.abs(dx/Q))
    return Q - step*grad*Q


def precond_grad_diag(Q, grad):
    """
    return preconditioned gradient using a diagonal preconditioner
    Q: diagonal preconditioner
    grad: gradient
    """
    return Q*Q*grad



###############################################################################
def update_precond_kron(Ql, Qr, dX, dG, step=0.01, diag=0):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: normalized step size in range [0, 1] 
    diag: see the code for details
    """
    max_diag_l = np.max(np.diag(Ql))
    max_diag_r = np.max(np.diag(Qr))
    
    #Ql = Ql + np.diag(np.clip(_diag_loading*max_diag_l - np.diag(Ql), 0.0, max_diag_l))
    #Qr = Qr + np.diag(np.clip(_diag_loading*max_diag_r - np.diag(Qr), 0.0, max_diag_r))
    
    rho = np.sqrt(max_diag_l/max_diag_r)
    Ql = Ql/rho
    Qr = rho*Qr
    
    A = Ql.dot( dG.dot( Qr.T ) )
    Bt = scipy.linalg.solve_triangular(Ql, 
                                       np.transpose(scipy.linalg.solve_triangular(Qr, dX.T, trans=1, lower=False)), 
                                       trans=1, lower=False)
    
    grad1 = np.triu(A.dot(A.T) - Bt.dot(Bt.T))
    grad2 = np.triu(A.T.dot(A) - Bt.T.dot(Bt))
    
    if diag:
        step1 = step/(np.max(np.abs(np.diag(grad1))) + _tiny)
        step2 = step/(np.max(np.abs(np.diag(grad2))) + _tiny)
    else:
        step1 = step/(np.max(np.abs(grad1)) + _tiny)
        step2 = step/(np.max(np.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.dot(Ql), Qr - step2*grad2.dot(Qr)
    

def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    if Grad.shape[0] > Grad.shape[1]:
        return Ql.T.dot( Ql.dot( Grad.dot( Qr.T.dot(Qr) ) ) )
    else:
        return (((Ql.T.dot(Ql)).dot(Grad)).dot(Qr.T)).dot(Qr)
    


###############################################################################
# SCAN preconditioner is super sparse, sparser than a diagonal preconditioner! 
# For an (M, N) matrix, it only requires 2*M+N-1 parameters to represent it
# Make sure that input feature vector is augmented by 1 at the end, and the affine transformation is given by
#               y = x*(affine transformation matrix)
#
def update_precond_scan(ql, qr, dX, dG, step=0.01):
    """
    update SCaling-And-Normalization (SCAN) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1,0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step is the normalized step size in natrual gradient descent  
    """
    # diagonal loading is removed, here we just want to make sure that Ql and Qr have similar dynamic range
    max_l = np.max(np.abs(ql))
    max_r = np.max(qr) # qr always is positive
    rho = np.sqrt(max_l/max_r)
    ql = ql/rho
    qr = rho*qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = np.transpose(ql[0:1])*dG
    A = A + np.matmul(np.transpose(ql[1:]), dG[-1:]) # Ql*dG 
    A = A*qr # Ql*dG*Qr 
    
    Bt = np.transpose(1.0/ql[0:1])*dX
    Bt[-1:] = Bt[-1:] - np.matmul(ql[1:]/(ql[0:1]*ql[0,-1]), dX) 
    #Bt = np.concatenate([Bt[:-1],
    #                     Bt[-1:] - np.matmul(ql[1:]/(ql[0:1]*ql[0,-1]), dX)], axis=0) # Ql^(-T)*dX
    Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = np.sum(A*A, axis=1) - np.sum(Bt*Bt, axis=1)
    grad1_bias = np.matmul(A[:-1], A[-1:].T) - np.matmul(Bt[:-1], Bt[-1:].T) 
    grad1_bias = np.reshape(grad1_bias, [-1])
    grad1_bias = np.concatenate([grad1_bias, [0.0]], axis=0)  

    step1 = step/(max(np.max(np.abs(grad1_diag)), np.max(np.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = np.sum(A*A, axis=0, keepdims=True) - np.sum(Bt*Bt, axis=0, keepdims=True)
    step2 = step/(np.max(np.abs(grad2)) + _tiny)
    new_qr = qr - step2*grad2*qr
    
    return np.stack((new_ql0, new_ql1)), new_qr


def precond_grad_scan(ql, qr, Grad):
    """
    return preconditioned gradient using SCaling-And-Normalization (SCAN) preconditioner
    Suppose Grad has shape (M, N)
    ql: shape (2, M), defines a matrix has the same form as that for input feature normalization 
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    preG = np.transpose(ql[0:1])*Grad
    preG = preG + np.matmul(np.transpose(ql[1:]), Grad[-1:]) # Ql*Grad 
    preG = preG*(qr*qr) # Ql*Grad*Qr^T*Qr
    add_last_row = np.matmul(ql[1:], preG) # use it to modify the last row
    preG = np.transpose(ql[0:1])*preG
    preG[-1:] = preG[-1:] + add_last_row
    #preG = np.concatenate([preG[:-1],
    #                       preG[-1:] + add_last_row], axis=0) # Ql^T*Ql*Grad*Qr^T*Qr
    return preG



###############################################################################                        
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=0.01):
    """
    update sparse LU preconditioner P = Q^T*Q, where 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: step size
    """
    # make sure that L and U have similar dynamic range
    max_l = max(np.max(np.abs(L12)), np.max(l3))
    max_u = max(np.max(np.abs(U12)), np.max(u3))
    rho = np.sqrt(max_l/max_u)
    L12 = L12/rho
    l3 = l3/rho
    U12 = rho*U12
    u3 = rho*u3
    # extract blocks
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]
    
    dx = np.concatenate([np.reshape(x, [-1, 1]) for x in dxs], 0) # a tall column vector
    dg = np.concatenate([np.reshape(g, [-1, 1]) for g in dgs], 0) # a tall column vector
    
    # U*dg
    Ug1 = np.dot(U1, dg[:r]) + np.dot(U2, dg[r:])
    Ug2 = u3*dg[r:]
    # Q*dg
    Qg1 = np.dot(L1, Ug1)
    Qg2 = np.dot(L2, Ug1) + l3*Ug2
    # inv(U^T)*dx
    iUtx1 = scipy.linalg.solve_triangular(np.transpose(U1), dx[:r], lower=True)
    iUtx2 = (dx[r:] - np.dot(np.transpose(U2), iUtx1))/u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2/l3
    iQtx1 = scipy.linalg.solve_triangular(np.transpose(L1),                     
                                          iUtx1 - np.dot(np.transpose(L2), iQtx2), lower=False)
    # L^T*Q*dg
    LtQg1 = np.dot(np.transpose(L1), Qg1) + np.dot(np.transpose(L2), Qg2)
    LtQg2 = l3*Qg2
    # P*dg
    Pg1 = np.dot(np.transpose(U1), LtQg1)
    Pg2 = np.dot(np.transpose(U2), LtQg1) + u3*LtQg2
    # inv(L)*inv(Q^T)*dx
    iLiQtx1 = scipy.linalg.solve_triangular(L1, iQtx1, lower=True)
    iLiQtx2 = (iQtx2 - np.dot(L2, iLiQtx1))/l3
    # inv(P)*dx
    iPx2 = iLiQtx2/u3
    iPx1 = scipy.linalg.solve_triangular(U1, iLiQtx1 - np.dot(U2, iPx2), lower=False)
    
    # update L
    grad1 = np.dot(Qg1, np.transpose(Qg1)) - np.dot(iQtx1, np.transpose(iQtx1))
    grad1 = np.tril(grad1)
    grad2 = np.dot(Qg2, np.transpose(Qg1)) - np.dot(iQtx2, np.transpose(iQtx1))
    grad3 = Qg2*Qg2 - iQtx2*iQtx2
    max_abs_grad = np.max(np.abs(grad1))
    max_abs_grad = max(max_abs_grad, np.max(np.abs(grad2)))
    max_abs_grad = max(max_abs_grad, np.max(np.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newL1 = L1 - np.dot(step0*grad1, L1)
    newL2 = L2 - np.dot(step0*grad2, L1) - step0*grad3*L2
    newl3 = l3 - step0*grad3*l3

    # update U
    grad1 = np.dot(Pg1, np.transpose(dg[:r])) - np.dot(dx[:r], np.transpose(iPx1))
    grad1 = np.triu(grad1)
    grad2 = np.dot(Pg1, np.transpose(dg[r:])) - np.dot(dx[:r], np.transpose(iPx2))
    grad3 = Pg2*dg[r:] - dx[r:]*iPx2
    max_abs_grad = np.max(np.abs(grad1))
    max_abs_grad = max(max_abs_grad, np.max(np.abs(grad2)))
    max_abs_grad = max(max_abs_grad, np.max(np.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newU1 = U1 - np.dot(U1, step0*grad1)
    newU2 = U2 - np.dot(U1, step0*grad2) - step0*np.transpose(grad3)*U2
    newu3 = u3 - step0*grad3*u3

    return np.concatenate([newL1, newL2], axis=0), newl3, np.concatenate([newU1, newU2], axis=1), newu3


def precond_grad_splu(L12, l3, U12, u3, grads):
    """
    return preconditioned gradient with sparse LU preconditioner
    where P = Q^T*Q, 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    grads: a list of gradients to be preconditioned
    """
    grad = [np.reshape(g, [-1, 1]) for g in grads] # a list of column vector
    lens = [g.shape[0] for g in grad] # length of each column vector
    grad = np.concatenate(grad, 0)  # a tall column vector
    
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]    
    
    # U*g
    Ug1 = np.dot(U1, grad[:r]) + np.dot(U2, grad[r:])
    Ug2 = u3*grad[r:]
    # Q*g
    Qg1 = np.dot(L1, Ug1)
    Qg2 = np.dot(L2, Ug1) + l3*Ug2
    # L^T*Q*g
    LtQg1 = np.dot(np.transpose(L1), Qg1) + np.dot(np.transpose(L2), Qg2)
    LtQg2 = l3*Qg2
    # P*g
    pre_grad = np.concatenate([np.dot(np.transpose(U1), LtQg1),                          
                               np.dot(np.transpose(U2), LtQg1) + u3*LtQg2], axis=0)
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(np.reshape(pre_grad[idx : idx + lens[i]], np.shape(grads[i])))
        idx = idx + lens[i]
    
    return pre_grads



"""
Other forms of preconditioners, e.g., banded Q (P is banded too), can be useful as well. 
In many deep learning models, affine map 

    f(x) = W*[x; 1]
    
is the building block with matrix W containing the parameters to be optimized. 
Kronecker product preconditioners (including SCAN) are particularly useful for such applications.  
"""



#### Testing code 
#if __name__ == '__main__':
#    import matplotlib.pyplot as plt
#    
#    """
#    Verification of update_precond_dense()
#    Eigenvalues of the preconditioned system should be close to 1 or -1
#    """
#    dim = 3
#    hess0 = np.random.randn(dim, dim)
#    hess0 = hess0 + hess0.T     # the true Hessian
#    Q = np.eye(dim)
#    all_eigs = []
#    for _ in range(10000):
#        dx = 1e-3*np.random.randn(dim)
#        dg = hess0.dot(dx) # dg is noiseless here
#        Q = update_precond_dense(Q, [dx], [dg])
#        eigs, _ = np.linalg.eig( Q.T.dot(Q).dot(hess0) )
#        eigs.sort()
#        all_eigs.append(list(eigs))
#        if np.max(np.abs(np.abs(eigs) - 1)) < 0.1:
#            break
#        
#    plt.figure(1)
#    plt.plot(np.array(all_eigs))
#    plt.xlabel('Number of iterations')
#    plt.ylabel('Eigenvalues of preconditioned system')
#    plt.title('Dense preconditioner estimation')
#    
#    
#    
#    """
#    Verification of update_precond_kron()
#    Eigenvalues of the preconditioned system should be close to 1 or -1 as the 
#    true Hessian is decomposable 
#    """
#    dim1, dim2 = 2, 3
#    dim = dim1*dim2
#    hess1 = np.random.randn(dim1, dim1)
#    hess2 = np.random.randn(dim2, dim2)
#    hess0 = np.kron(hess2 + hess2.T, hess1 + hess1.T)   # the true Hessian
#    Ql, Qr = np.eye(dim1), np.eye(dim2)
#    all_eigs = []
#    for _ in range(10000):
#        dx = 1e-3*np.random.randn(dim)
#        dg = hess0.dot(dx) # dg is noiseless here
#        dX = dx.reshape(dim2, dim1).T   
#        dG = dg.reshape(dim2, dim1).T # numpy assumes row-major order, so dG is defined in this way
#        Ql, Qr = update_precond_kron(Ql, Qr, dX, dG)
#        eigs, _ = np.linalg.eig( np.kron(Qr.T.dot(Qr), Ql.T.dot(Ql)).dot(hess0) )
#        eigs.sort()
#        all_eigs.append(list(eigs))
#        if np.max(np.abs(np.abs(eigs) - 1)) < 0.1:
#            break
#        
#    plt.figure(2)
#    plt.plot(np.array(all_eigs))
#    plt.xlabel('Number of iterations')
#    plt.ylabel('Eigenvalues of preconditioned system')
#    plt.title('Kron product preconditioner estimation')
#
#
#
#    """
#    Verification of update_precond_kron()
#    Eigenvalues of the preconditioned system should be well normalized 
#    """
#    dim1, dim2 = 2, 3
#    dim = dim1*dim2
#    hess0 = np.random.randn(dim, dim)
#    hess0 = hess0 + hess0.T     # this is an arbitrary Hessian
#    Ql, Qr = np.eye(dim1), np.eye(dim2)
#    all_eigs = []
#    for i in range(1000):
#        dx = 1e-3*np.random.randn(dim)
#        dg = hess0.dot(dx) # dg is noiseless here
#        dX = dx.reshape(dim2, dim1).T
#        dG = dg.reshape(dim2, dim1).T
#        Ql, Qr = update_precond_kron(Ql, Qr, dX, dG)
#        eigs, _ = np.linalg.eig( np.kron(Qr.T.dot(Qr), Ql.T.dot(Ql)).dot(hess0) )
#        eigs.sort()
#        all_eigs.append(list(eigs))
#        
##        if i==0 or i==999: # remove this comment to see the improvement on eigenvalue spread reduction
##            eigs=np.log(np.abs(eigs))
##            print(np.std(eigs))
#        
#    plt.figure(3)
#    plt.plot(np.array(all_eigs))
#    plt.xlabel('Number of iterations')
#    plt.ylabel('Eigenvalues of preconditioned system')
#    plt.title('Normalizing eigenvalues using Kronecker product preconditioner')
#    
#    
#    
#    """
#    direct sum approximation
#    Eigenvalues of the preconditioned system should be well normalized 
#    """
#    dim1, dim2 = 2, 3
#    dim = dim1 + dim2
#    hess0 = np.random.randn(dim, dim)
#    hess0 = hess0 + hess0.T     # this is an arbitrary Hessian
#    Q1, Q2 = np.eye(dim1), np.eye(dim2)
#    all_eigs = []
#    for i in range(1000):
#        dx = 1e-3*np.random.randn(dim)
#        dg = hess0.dot(dx) # dg is noiseless here
#        Q1 = update_precond_dense(Q1, dx[:dim1], dg[:dim1])
#        Q2 = update_precond_dense(Q2, dx[dim1:], dg[dim1:])
#        eigs, _ = np.linalg.eig( scipy.linalg.block_diag(Q1.T.dot(Q1), Q2.T.dot(Q2)).dot(hess0) )
#        eigs.sort()
#        all_eigs.append(list(eigs))
#        
##        if i==0 or i==999:  # remove this comment to see the improvement on eigenvalue spread reduction
##            eigs=np.log(np.abs(eigs))
##            print(np.std(eigs))
#        
#    plt.figure(4)
#    plt.plot(np.array(all_eigs))
#    plt.xlabel('Number of iterations')
#    plt.ylabel('Eigenvalues of preconditioned system')
#    plt.title('Normalizing eigenvalues using direct sum preconditioner')
