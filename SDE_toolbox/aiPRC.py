#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:59:48 2024

@author: pierrehouzelstein
"""

import numpy as np
from .integrators import _EulerSDE, _EulerSDEPert, jitDic
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats
from tqdm import tqdm

def PRC_directMethod(X0, tMax, phaseFunc, f, driftParams, g, noiseParams, ampList, nIter, nbins = 60, STEP = 10):
    """
    Apply pulse perturbation at random phases
    Store phase and phase shift
    Bin results, average
    """

    #Values
    nDims = len(X0)
    assert len(ampList) == nDims, 'Need as many pulse values as system dimensions'
    
    #Duration of longest integration
    nSteps = len(tMax);  dt = tMax[1] - tMax[0]
    
    #Start sampling after 10%: steady state
    start = int(0.1*nSteps); end = start + 1
    
    #Prepare integration solver
    fJitted = jit(f, nopython=True, cache=True)
    gJitted = jit(g, nopython=True, cache=True)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)
    solver = _EulerSDEPert(fJitted, gJitted)
    
    #Store
    nPoints = int((nSteps-start)/STEP)
    PsiList = np.zeros((nIter,nPoints))
    deltaPsi = np.zeros((nIter,nPoints))

    for n in range(nIter):
        print(f'Iter # {n+1}')
        for i in tqdm(range(nPoints)):
            #Integrate from 0 to increasing time (so that we get a random phase)
            t = np.linspace(0, end*dt, end)
            
            #Create pulse pert
            index = int(end - 2)
            I = np.zeros((nDims, end))
            I[:, index] = ampList
            
            #Integrate perturbed trajectory
            X = solver(X0, t, I, driftParams, noiseParams)
            
            #Get phase
            Psi = phaseFunc(X)
            
            #At pulse
            Psip = Psi[index - 1]
            #Post pulse
            Psipp = Psi[index]

            PsiList[n,i] = Psip
            deltaPsi[n,i] = Psipp - Psip

            end += STEP
            
            if end > nSteps:
                #Back to start
                end = start
                
    PsiList = PsiList.flatten()
    deltaPsi = deltaPsi.flatten()

    #Get mean directions by using exponential form: e^i(deltaPsi)
    av_trig, bin_edges, binnumber = stats.binned_statistic(PsiList, [np.cos(deltaPsi), np.sin(deltaPsi)], statistic='mean', bins=nbins)
    # Compute the number of elements in each bin
    count_stat, _, _ = stats.binned_statistic(PsiList, np.cos(deltaPsi), statistic='count', bins=nbins)

    
    Cbar = av_trig[0,:]; Sbar = av_trig[1,:]
    sPRC = np.zeros(np.shape(Cbar))
    for i in range(len(sPRC)):
        if Cbar[i] >= 0:
            sPRC[i] = np.arctan2(Sbar[i], Cbar[i])
        else:
            sPRC[i] = np.arctan2(Sbar[i], Cbar[i]) + np.pi

    bin_width = (bin_edges[1] - bin_edges[0])
    PsiBins = bin_edges[1:] - bin_width/2
    
    #Circular std dev
    Rbar = np.sqrt(Cbar**2 + Sbar**2)
    std_dev = np.sqrt(-2*np.log(Rbar))
    std_err = std_dev/np.sqrt(count_stat)

    plt.title('Recovered aiPRC - direct perturbation')
    plt.axhline(0, c='k')
    #plt.plot(PsiBins, sPRC, 'b-.')
    plt.errorbar(PsiBins, sPRC, yerr=std_err, c = 'b', fmt='-.', capsize=5, label='Mean with Std Error')
    plt.xlabel(r'$\psi$')
    plt.ylabel(r'$\Delta \psi$')
    plt.xlim(0, 2*np.pi)
    plt.show()

    return PsiBins, sPRC, std_err

def aiPRC_empirical(x0, t, phaseFunc, dphaseFunc, 
                    f, driftParams, g, noiseParams, 
                    nbins = 60):
    

    #Prepare integrator
    fJitted = jit(f, nopython=True, cache=True); gJitted = jit(g, nopython=True, cache=True)
    solver = _EulerSDE(fJitted, gJitted)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)

    #Phase bins
    X = solver(x0, t, driftParams, noiseParams)
    Psi = phaseFunc(X)
    dxPsi = dphaseFunc(X)
    
    ###########################################################################
    ####   Circular mean   ####################################################
    ###########################################################################
    
    sPRC, bin_edges, binnumber = stats.binned_statistic(Psi, dxPsi, statistic='mean', bins=nbins)
    # Compute the number of elements in each bin
    count_stat, _, _ = stats.binned_statistic(Psi, dxPsi, statistic='count', bins=nbins)
    #Std
    std_dev, _, _ = stats.binned_statistic(Psi, dxPsi, statistic='std', bins=nbins)
    std_err = std_dev/np.sqrt(count_stat)

    bin_width = (bin_edges[1] - bin_edges[0])
    PsiBins = bin_edges[1:] - bin_width/2

    plt.title('Recovered aiPRC - gradient averaging')
    plt.axhline(0, c='k')
    #plt.plot(PsiBins, sPRC, 'b-.')
    plt.errorbar(PsiBins, sPRC, yerr=std_err, c = 'b', fmt='-.', capsize=5, label='Mean with Std Error')
    plt.xlabel(r'$\psi$')
    plt.ylabel(r'$\Delta \psi$')
    plt.xlim(0, 2*np.pi)
    plt.show()
    
    return PsiBins, sPRC, std_err