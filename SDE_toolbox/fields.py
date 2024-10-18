#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:01:18 2024

@author: pierrehouzelstein
"""

import numpy as np

#Drifts 
def Hopf2D(X, params):
    x, y = X
    
    beta = params['beta']; gamma = params['gamma']
    delta = params['delta']; kappa = params['kappa']
    
    r = (x**2 + y**2)
    fX = (delta - kappa*r)*x - (gamma - beta*r)*y
    fY = (gamma - beta*r)*x + (delta - kappa*r)*y
    return np.array([fX, fY])

def SNIC2D(X, params):
    """
    2D canonical SNIC bifurcation
    
    dx/dt = f(x) + g(x)dW
    
    Passing parameters as dic allows to jit compile using numba
    """
    x, y = X
    
    beta = params['beta']; m = params['m']
    
    r = (x**2 + y**2)
    fX = beta*x - m*y - x*r + ((y**2)/np.sqrt(r))
    fY = m*x + beta*y - y*r - ((x*y)/np.sqrt(r))
    return np.array([fX, fY])

#Noise

def additive2D(X, params):
    
    sxx = params['sxx']; syx = params['syx']
    sxy = params['sxy']; syy = params['syy']
    
    x, y = X
    gxx = sxx*np.ones(np.shape(x))
    gxy = sxy*np.ones(np.shape(x))
    gyx = syx*np.ones(np.shape(y))
    gyy = syy*np.ones(np.shape(y))
    
    return np.array([gxx, gxy, gyx, gyy])

def HodggkinHuxley(X,params):
    
    
    
    return