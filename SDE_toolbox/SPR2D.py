#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:07:41 2024

@author: pierrehouzelstein

Stochastic phase reduction in 2D by means of a finite differences algorithm

- Build LDagger via finite differences
- Get Psi(x, y) by eigendecomposition of LDagger
- Averaging method to get aPsi, DPsi, DTheta s.t. 
dPsi(psi) = aPsi(psi)*dt + sqrt(2*DPsi(psi))*dWt
"""

import time
import inspect
import shutil

from .PhaseBuilder2D import buildPsi2D


def SPR2D(path, x, y, f2D, driftParams, g2D, noiseParams, 
          nEvals = 31, nbins = 40, nEvalsRed=11):
    """
    Builds the stochastic phase reduction of a 2D stochastic oscillator with Ito SDE
    dX = f(X)*dt + g(X)*dWt, X = [x, y]^T

    Parameters
    ----------
    path : string
        Path to folder in which data will be written. If does not exist, will be written.
    x : array
        Grid range in the x direction.
    y : array
        Grid range in the y direction.
    driftFunc2D : function
        Drift function that takes X = [x, y] and driftParams as inputs.
        driftFunc2D(X, driftParams) = f(x, y).
    noiseAmpFunc2D : function
        Noise amplitude function that takes X = [x, y] and noiseParams as inputs.
        noiseAmpFunc2D(X, noiseParams) = g(x, y).
    driftParams : dictionary
        Dictionary containing the parameters for driftFunc2D.
    noiseParams : dictionary
        Dictionary containing the parameters for driftFunc2D.
    nEvals : int, optional
        Number of eigenvalues to compute when diagonalizing LDagger. 
        The default is 31.
    nbins : int, optional
        Number of isocrones over which to interpolate when computing the phase reductions. 
        The default is 40.
    nEvals : int, optional
        Number of eigenvalues to compute when diagonalizing the reduced LDagger. The default is 11.

    Returns
    -------
    None.

    """

    #Check that the drift/noise amplitude functions have the parameters they need
    nf = len(inspect.getfullargspec(f2D).args)
    ng = len(inspect.getfullargspec(g2D).args)
    
    assert nf == 1 or nf ==2, 'f  must be of type f(X) or f(X, dic)'
    assert ng == 1 or ng ==2, 'g  must be of type g(X) or g(X, dic)'
    
    if nf == 2:
        assert type(driftParams) == dict, 'Must pass a driftParams dict in args'
        def f(X):
            return f2D(X, driftParams)
    if ng == 2:
        assert type(noiseParams) == dict, 'Must pass a noiseParams dict in args'
        def g(X):
            return g2D(X, noiseParams)
    
    start = time.time()
    
    ###########################################################################
    
    print('Building stochastic asymptotic phase')
    print('')
    buildPsi2D(x, y, f, g, path, nEvals)
    
    ###########################################################################
    
    #Delete heavy folder with grid data
    savepath = f'{path}/gridData'
    shutil.rmtree(savepath)
    
    dur = time.time() - start
    print('Done!')
    print(f'Elapsed time - {dur:0.02f} s')
    
    return

