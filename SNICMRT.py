#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:28:30 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import SDE_toolbox as SDE
import pickle

def loadParams(D, m):
    
    f = SDE.SNIC2D
    driftParams = {'beta': 1., 'm': m}
    g = SDE.additive2D
    sxx = syy = np.sqrt(2*D)
    noiseParams = {'sxx': sxx, 'sxy': 0., 'syx': 0., 'syy': syy}
    
    return f, driftParams, g, noiseParams

def buildMRT(D, m):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    f, driftParams, g, noiseParams = loadParams(D, m)
    
    #Grid
    x = np.linspace(-1.5, 1.5, 300)
    y = np.linspace(-1.5, 1.5, 300)

    #Build phase
    SDE.buildMRTPhase2D(path, x, y, f, driftParams, g, noiseParams)

    return

def reduction(D, m, MRTFunc):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    f, driftParams, g, noiseParams = loadParams(D, m)
    
    print('Building reduction')
    x0 = np.random.rand(2)
    dt = 1e-2; T = 100; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    SDE.MRT_SPR(path, x0, t, MRTFunc, f, driftParams, g, 
             noiseParams, nbins=50, nIter=1000, nEvals = 11)
    
    return

def longTermStats(D, m):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    #Long term stats 
    savepath = f'{path}/empiricalMRT'
    with open(f'{savepath}/MRTRedCoefs.pck', 'rb') as file_handle:
        aCoefs = pickle.load(file_handle)
        DCoefs = pickle.load(file_handle)

    #Params for integration
    dt = 1e-2; T = 100; nSteps = int(T/dt); nIter = 300
    t = np.linspace(0, T, nSteps)
    #SDE.longTermStats(savepath, t, aCoefs, DCoefs, nIter)
    print('')
    
    #Long term stats using the results from Lindner and Schimansky-Geier, PRL 2002
    TBar = np.loadtxt(f'./{path}/MRTPeriod')
    SDE.getRotationDiffusionMRT(TBar, DCoefs, savepath)
    
    return

def buildPRC(D,m, MRTFunc):

    f, driftParams, g, noiseParams = loadParams(D, m)
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    print('Computing PRC  empirically (using ergodicity)')
    x0 = np.random.rand(2)
    dt = 1e-2; T = 2000; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    #X direction
    dxMRTFunc = SDE.dxMRTFunc2D(path)
    PsiBinsX, sPRCX, stdErrX = SDE.aiPRC_empirical(x0, t, MRTFunc, dxMRTFunc, 
                        f, driftParams, g, noiseParams, 
                        nbins = 60)
    
    #Y direction
    dyMRTFunc = SDE.dyMRTFunc2D(path)
    PsiBinsY, sPRCY, stdErrY = SDE.aiPRC_empirical(x0, t, MRTFunc, dyMRTFunc, 
                        f, driftParams, g, noiseParams, 
                        nbins = 60)

    savepath = f'{path}/empiricalMRT'
    with open(f'{savepath}/empiricalMRTPRC.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(stdErrX, file_handle)
        
        pickle.dump(PsiBinsY, file_handle)
        pickle.dump(sPRCY, file_handle)
        pickle.dump(stdErrY, file_handle)
    
    return

def phaseResponse(D, m, MRTFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, m)
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    #Compute empirical phase response in the x/y directions
    X0 = np.random.rand(2)
    T = 100; dt = 1e-3; nSteps = int(T/dt)
    tMax = np.linspace(0, T, nSteps)
    nIter = 1
    
    Ix = 0.01; Iy = 0.; ampList = np.array([Ix, Iy])
    print('Computing PRC in the x-direction via direct perturbation')
    PsiBinsX, sPRCX, std_errX = SDE.PRC_directMethod(X0, tMax, MRTFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins = 50)
    print('')
    Ix = 0.; Iy = 0.01; ampList = np.array([Ix, Iy])
    print('Computing PRC in the y-direction via direct perturbation')
    PsiBinsY, sPRCY, std_errY = SDE.PRC_directMethod(X0, tMax, MRTFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins = 50)
    print('')
    
    with open(f'{path}/phaseResponseMRT.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(std_errX, file_handle)
        
        pickle.dump(PsiBinsY, file_handle)
        pickle.dump(sPRCY, file_handle)
        pickle.dump(std_errY, file_handle)
        
    return


def generateDataMRT(D, m):
    
    print('')
    print(f'Running m  = {m:0.3f}')
    print('')
    
    #Build the phase (only need to do this once)
    #buildMRT(D, m)
    
    #Load phase function
    MRTFunc = SDE.MRTFunc2D(f'./SNIC/m={m:0.3f}_D={D:0.2f}')
    
    #Empirical reduction
    #reduction(D, m, MRTFunc)

    #Stats
    longTermStats(D, m)
    
    #PRC
    #buildPRC(D,m, MRTFunc)
    
    #Phase response
    #phaseResponse(D, m, MRTFunc)

    return

def main():
    
    #System params
    m1 = 0.999; m2 = 1.03
    
    #D = 0.01
    #generateDataMRT(D, m1)
    #generateDataMRT(D, m2)
    
    #Noise amplitude
    DList = np.linspace(0.01, 0.1, 10)
    #DList = [0.09]
    for D in DList:
        
        #SNIC below
        generateDataMRT(D, m1)
        #SNIC above
        generateDataMRT(D, m2)
    
    
    return

if __name__ == '__main__':
    main()