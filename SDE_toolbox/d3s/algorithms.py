# -*- coding: utf-8 -*-
import numpy as _np
import scipy as _sp
import scipy.sparse.linalg

'''
Original code
    https://github.com/sklus/d3s

'''

def gedmd(X, Y, Z, psi, evs=5, operator='K'):
    '''
    Generator EDMD for the Koopman operator. The matrices X and Y
    contain the input data. For stochastic systems, Z contains the
    diffusion term evaluated in all data points X. If the system is
    deterministic, set Z = None.
    '''
    PsiX = psi(X)
    dPsiY = _np.einsum('ijk,jk->ik', psi.diff(X), Y)
    if not (Z is None): # stochastic dynamical system
        n = PsiX.shape[0] # number of basis functions
        ddPsiX = psi.ddiff(X) # second-order derivatives
        S = _np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
        for i in range(n):
            dPsiY[i, :] += 0.5*_np.sum( ddPsiX[i, :, :, :] * S, axis=(0,1) )
    
    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T
    if operator == 'P': C_1 = C_1.T

    A = _sp.linalg.pinv(C_0) @ C_1
    
    d, V = sortEig(A, evs, which='SM')
    
    return (A, d, V)

# auxiliary functions
def sortEig(A, evs=5, which='LM'):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = _sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = _sp.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])
