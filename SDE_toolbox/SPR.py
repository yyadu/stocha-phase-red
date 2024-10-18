#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:13:38 2024

@author: pierrehouzelstein

General phase reduction by means of empirical averages
Given any phase function of the form PsiFunc(X) = psi
and SDE dx = f(x)dt  + g(x)dWt
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from numba import jit, types, float64
import os
from scipy.sparse.linalg import eigs

from .LDaggerFinDiff import buildLDagger1D
from .integrators import jitDic, _EulerSDE
from .utils import phase_fit, unWrapPhase, fourier_fit_array

def SPR(path, x0, t, phaseFunc, f, driftParams, g, 
             noiseParams, nbins=60, nIter=100, nEvals = 11):
    """
    Numerical evaluation of the coefficients by means of the short term
    statistics
    """
    
    #Path in which to save data
    if not os.path.exists(path):
        os.makedirs(path)
    
    #Check 
    assert type(driftParams) == dict, 'Must pass a driftParams dict in args'
    assert type(noiseParams) == dict, 'Must pass a noiseParams dict in args'

        
    #Prepare integrator
    fJitted = jit(f, nopython=True, cache=True); gJitted = jit(g, nopython=True, cache=True)
    solver = _EulerSDE(fJitted, gJitted)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)

    #Phase bins
    bins = np.linspace(0, 2*np.pi, nbins + 2)[1:-1] 
    phaseArray = bins[1:]-bins[1]*0.5
    nSteps = len(t); dt = t[1] - t[0]

    phaseTraj = np.zeros((nIter, nSteps))
    print(f'Generating {nIter} trajectories...')
    for i in tqdm(range(nIter)):
        #Integrate realization
        X = solver(x0, t, driftParams, noiseParams)
        #Phase evolution
        phaseTraj[i,:] = phaseFunc(X)
        
    phaseTraj2 = np.copy(phaseTraj)

    #Drift
    K1_Full = np.zeros(len(bins)-1)
    counter_Full = np.zeros(len(bins)-1)
    for i in range(nIter):
        K1, counter = computeK1(phaseTraj[i,:], bins)
        K1_Full += K1
        counter_Full += counter
    K1_Final = K1_Full/(counter_Full*dt)

    #Now K2
    K2_Full = np.zeros(len(bins)-1)
    counter_Full = np.zeros(len(bins)-1)
    for i in range(nIter):
        K2, counter = computeK2(phaseTraj[i,:], bins, phaseArray)
        K2_Full += K2
        counter_Full += counter
    
    K2_Final = (K2_Full/(counter_Full) - (K1_Final*dt)**2)/(2*dt)
    K2_Final[K2_Final<0] = 1e-5

    #Now, we fit the data 
    K1_coefs = phase_fit(phaseArray, K1_Final, nTerms = 100)
    K2_coefs = phase_fit(phaseArray, K2_Final, nTerms = 100)
    
    plt.figure(dpi = 300)
    plt.title('Asymptotic phase reduction')
    plt.xlabel(r"$\psi$")
    plt.axhline(0, c='k')
    plt.axvline(np.pi, c='k')
    plt.plot(phaseArray, K1_Final, 'b', label = r'$a(\psi$)')
    plt.plot(phaseArray, np.sqrt(2*K2_Final), 'r', label = r'$\sqrt{2D(\psi}$')
    plt.xlim(0, 2*np.pi)
    plt.legend()
    plt.show()
    
    #Save
    savepath = f'{path}/empirical'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    with open(f'{savepath}/PsiRedCoefs.pck', 'wb') as file_handle:
        pickle.dump(K1_coefs, file_handle)
        pickle.dump(K2_coefs, file_handle)
        
    ###########################################################################
    ####  Build  the reduced LDagger operator  ################################
    ###########################################################################
    
    def aFunc(X):
        return fourier_fit_array(X, K1_coefs)
    
    def sqrt2DFunc(X):
        return np.sqrt(2*fourier_fit_array(X, K2_coefs))

    #PsiGrid1D = np.linspace(0, 2*np.pi, 3000 + 1)[:-1]
    # LDagger1D = buildLDagger1D(PsiGrid1D, aFunc, sqrt2DFunc)
    # evalsRed, _ = eigs(LDagger1D, k=nEvals, sigma = -0.1)
    # np.savetxt(f'{savepath}/eValsLDRed', evalsRed)
    
    # evals = np.loadtxt(f'{path}/eValsLD', dtype = complex)
    
    # plt.figure(dpi = 300)
    # plt.axhline(0, c='k')
    # plt.axvline(0, c='k')
    # plt.plot(evals.real, evals.imag, 'bo', label = 'Ruelle-Pollicott resonances')
    # plt.plot(evalsRed.real, evalsRed.imag, 'r.', label = 'Reduced Ruelle-Pollicott resonances')
    # plt.title(r'Spectra of $\mathcal{L}^\dagger$ and $\mathcal{L}^\dagger_\psi$')
    # plt.legend()
    # plt.show()

    ###########################################################################
        
    print('Use generated data to compute long terms statistics')
    dphase = np.zeros(nIter)
    
    #Start at 10% of the integration length to be in steady state
    start = int(0.1*nSteps)
    
    for i in range(nIter):
        phase = unWrapPhase(phaseTraj2[i,:])
        dphase[i] = (phase[-1] - phase[0])
    #plt.show()
    #Duration 
    T = t[-1] - t[start]
    
    #First moment
    omega_f = dphase/T
    omega = np.mean(omega_f)
    stdErrOmega = np.std(omega_f, ddof=1) / np.sqrt(np.size(omega_f))
    
    #Second moment
    Df = ((dphase - omega*T)**2)/(2*T)
    D = np.mean(Df)
    stdErrD = np.std(Df, ddof=1) / np.sqrt(np.size(Df))
    
    results = np.array([omega, D])
    stdErrs = np.array([stdErrOmega, stdErrD])
    np.savetxt(f'{path}/longTermStatsFull', results)
    np.savetxt(f'{path}/longTermStatsFullstdErr', stdErrs)
    
    print(f'Long term statistics: w = {omega:0.3f} ± {stdErrOmega:0.3f} - D = {D:0.3f} ± {stdErrD:0.3f}')

    return

###############################################################################

@jit(types.Tuple((float64[:], float64[:]))(float64[:], float64[:]), nopython=True, fastmath=True, cache=True, parallel=True)
def computeK1(phaseFun, bins):
    """
    Compute the first short-term moment of time series phaseFun
    """

    K1 = np.zeros(len(bins)); counter = np.zeros(len(bins))
    index = np.digitize(phaseFun, bins, right=False)

    for i in range(0, len(phaseFun)-1):
        
        if phaseFun[i] - phaseFun[i+1] > np.pi:
            K1[index[i]] += (phaseFun[i+1] + 2*np.pi) - phaseFun[i] 
                
        elif phaseFun[i] - phaseFun[i+1] < -np.pi:
            K1[index[i]] += (phaseFun[i+1] - 2*np.pi) - phaseFun[i]  
                
        else:
            K1[index[i]] += phaseFun[i+1] - phaseFun[i]
            
        counter[index[i]] += 1     
        
    return K1[1:], counter[1:]

@jit(types.Tuple((float64[:], float64[:]))(float64[:], float64[:], float64[:]),nopython=True,fastmath=True,cache=True,parallel=True)
def computeK2(phaseFun, bins, phaseArray):
    """
    Compute the second short-term moment of time series phaseFun
    """

    K2 = np.zeros(len(bins)); counter = np.zeros(len(bins))
    index = np.digitize(phaseFun, bins, right=False)

    for i in range(0, len(phaseFun)-1):
        
        if phaseFun[i] - phaseFun[i+1] > np.pi:
            K2[index[i]] += ((phaseFun[i+1] + 2*np.pi) - phaseFun[i])**2
                
        elif phaseFun[i] - phaseFun[i+1] < -np.pi:
            K2[index[i]] += ((phaseFun[i+1] - 2*np.pi) - phaseFun[i])**2
                
        else:
            K2[index[i]] += (phaseFun[i+1] - phaseFun[i])**2

        counter[index[i]] += 1      

    return K2[1:], counter[1:]