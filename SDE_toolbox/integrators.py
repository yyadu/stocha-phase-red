#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:33:50 2024

@author: pierrehouzelstein
"""

from numba import jit, float64, complex128
import numpy as np
from numba.typed import Dict
from numba.core import types
from tqdm import tqdm

from .utils import unWrapPhase, fourier_fit_single_point

##############################################################################
##############################################################################

def jitDic(dic):
    """
    Turn ordinary dictionary into numba compatible dictionary
    """
    jittedDic = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for key in dic:
        jittedDic[key] = dic[key]
    return jittedDic

###############################################################################
################  Integrators  ################################################
###############################################################################

def _EulerODE(f):
    """
    Euler scheme for dx = f(x)*dt
    """
    
    @jit(nopython=True, cache=True)
    def solveODE(x0, t, driftParams):
        nDims = len(x0)
        nSteps = len(t); dt = t[1] - t[0]
        sol = np.zeros((nDims, nSteps))
        sol[:,0] = x0
        for i in range(1, nSteps):
            x0 += f(x0, driftParams)*dt
            sol[:, i] = x0
        return sol

    return solveODE

def EulerODE(x0, t, f, driftParams, valueType = 'float'):
    """
    Run one realization of dx = f(x)dt
    """
    fJitted = jit(f, nopython=True, cache=True)
    solver = _EulerODE(fJitted)
    driftParams = jitDic(driftParams)
    return solver(x0, t, driftParams)

def _EulerSDE(f, g):
    """
    Euler scheme for dx = f(x)*dt + g(x)*dWt
    """
    @jit(nopython=True, cache=True)
    def solveSDE(x0, t, driftParams, diffParams):
        nDims = len(x0)
        nSteps = len(t); dt = t[1] - t[0]
        GWN = np.random.normal(0, 1, (nDims, nSteps))*np.sqrt(dt)
        sol = np.zeros((nDims, nSteps))
        sol[:,0] = x0
        for i in range(1, nSteps):
            dW = g(x0, diffParams).reshape((nDims, nDims)) @ np.ascontiguousarray(GWN[:,i])
            x0 += f(x0, driftParams)*dt + dW
            sol[:, i] = x0
        return sol
    return solveSDE

def EulerSDE(x0, t, f, driftParams, g, noiseParams):
    """
    Run one realization of dx = f(x)dt
    """
    fJitted = jit(f, nopython=True, cache=True)
    gJitted = jit(g, nopython=True, cache=True)
    solver = _EulerSDE(fJitted, gJitted)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)
    return solver(x0, t, driftParams, noiseParams)

def _EulerSDEPert(f, g):
    """
    Euler scheme for dx = f(x)*dt + g(x)*dWt
    """
    @jit(nopython=True, cache=True)
    def solveSDEPert(x0, t, I, driftParams, diffParams):
        nDims = len(x0)
        nSteps = len(t); dt = t[1] - t[0]
        GWN = np.random.normal(0, 1, (nDims, nSteps))*np.sqrt(dt)
        sol = np.zeros((nDims, nSteps))
        sol[:,0] = x0
        for i in range(1, nSteps):
            gX = g(x0, diffParams).reshape((nDims, nDims))
            x0 += f(x0, driftParams)*dt + I[:,i] + np.dot(gX, np.ascontiguousarray(GWN[:,i]))
            sol[:, i] = x0
        return sol
    return solveSDEPert

def _HeunSDE(f, g):
    """
    Heun scheme for dx = f(x)*dt + g*dWt
    !!! Ito/Strato dilemma -> use additive noise if Ito
    """
    @jit(nopython=True, cache=True)
    def solveSDE(x0, t, driftParams, diffParams):
        nDims = len(x0)
        nSteps = len(t); dt = t[1] - t[0]
        GWN = np.random.normal(0, 1, (nDims, nSteps))*np.sqrt(dt)
        sol = np.zeros((nDims, nSteps))
        sol[:,0] = x0
        
        for i in range(1, nSteps):
            
            #Predictor
            dW = g(x0, diffParams).reshape((nDims, nDims)) @ np.ascontiguousarray(GWN[:,i])
            xPred = x0 + f(x0, driftParams)*dt + dW
            
            #Update
            x0 += 0.5*( f(x0, driftParams) + f(xPred, driftParams))*dt + dW
            
            sol[:, i] = x0
            
        return sol
    return solveSDE

def HeunSDE(x0, t, f, driftParams, g, noiseParams):
    """
    Run one realization of dx = f(x)dt
    """
    fJitted = jit(f, nopython=True, cache=True)
    gJitted = jit(g, nopython=True, cache=True)
    solver = _HeunSDE(fJitted, gJitted)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)
    return solver(x0, t, driftParams, noiseParams)

#Integrate phase function with drift/diffusion fitted with Fourier coefs

@jit((float64[:])(float64[:], float64, complex128[:], complex128[:]), 
                      nopython=True, cache=True)
def EulerPhase(t, Psi0, aCoefs, DCoefs):

    nSteps = len(t); dt = t[1] - t[0]
    GWN = np.random.normal(0, 1, nSteps)*np.sqrt(dt)

    Psi = np.zeros(nSteps)
    Psi[0] = Psi0
    for i in (range(1, nSteps)):
        #First order prediction
        a = fourier_fit_single_point(Psi0, aCoefs)
        D = fourier_fit_single_point(Psi0, DCoefs)
        Psi0 += a*dt + np.sqrt(2*D)*GWN[i]
        Psi[i] = np.mod(Psi0, 2*np.pi)
        
    return Psi

@jit((float64[:])(float64[:], float64, complex128[:], complex128[:]), 
                      nopython=True, cache=True)
def HeunPhase(t, Psi0, aCoefs, DCoefs):

    nSteps = len(t); dt = t[1] - t[0]
    GWN = np.random.normal(0, 1, nSteps)*np.sqrt(dt)

    Psi = np.zeros(nSteps)
    Psi[0] = Psi0
    for i in (range(1, nSteps)):
        #First order prediction
        a = fourier_fit_single_point(Psi0, aCoefs)
        D = fourier_fit_single_point(Psi0, DCoefs)
        dW = np.sqrt(2*D)*GWN[i]
        
        Pred = Psi0 + a*dt + dW
        aPred = fourier_fit_single_point(Pred, aCoefs)
        
        #Update
        Psi0 += 0.5*(a + aPred)*dt + dW

        Psi[i] = np.mod(Psi0, 2*np.pi)
        
    return Psi

###############################################################################

def longTermStats(savepath, t, aCoefs, DCoefs, nIter):
    
    #IC
    Psi0 = np.random.rand()*2*np.pi
    
    #start at 10% of trajectory
    start = int(0.1*len(t))
    
    
    #Integrate data
    dPsi = np.zeros(nIter)
    print('Long terms statistics of the reduced phase...')
    for i in tqdm(range(nIter)):

        #Asymptotic phase
        Psi = EulerPhase(t, Psi0, aCoefs, DCoefs)
        #Psi = HeunPhase(t, Psi0, aCoefs, DCoefs)
        
        Psi = unWrapPhase(Psi)
        
        #plt.plot(Psi)
        
        
        dPsi[i] = (Psi[-1] - Psi[start])
        
    #plt.show()
        
    #Duration 
    T = t[-1] - t[start]
    
    #First moment
    omega_f = dPsi/T
    omega = np.mean(omega_f)
    stdErrOmega = np.std(omega_f, ddof=1) / np.sqrt(np.size(omega_f))
    
    #Second moment
    Df = ((dPsi - omega*T)**2)/(2*T)
    D = np.mean(Df)
    stdErrD = np.std(Df, ddof=1) / np.sqrt(np.size(Df))
    
    results = np.array([omega, D])
    stdErrs = np.array([stdErrOmega, stdErrD])
    np.savetxt(f'{savepath}/longTermStatsRed', results)
    np.savetxt(f'{savepath}/longTermStatsRedstdErr', stdErrs)
    
    print(f'Long term statistics of reduction: w = {omega:0.3f} ± {stdErrOmega:0.3f} - D = {D:0.3f} ± {stdErrD:0.3f}')

    return

