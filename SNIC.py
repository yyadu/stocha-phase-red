#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:46:38 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import SDE_toolbox as SDE
import pickle
from tqdm import tqdm
from scipy.optimize import curve_fit

def loadParams(D, m):
    
    f = SDE.SNIC2D
    driftParams = {'beta': 1., 'm': m}
    g = SDE.additive2D
    sxx = syy = np.sqrt(2*D)
    noiseParams = {'sxx': sxx, 'sxy': 0., 'syx': 0., 'syy': syy}
    
    return f, driftParams, g, noiseParams

def sampleTrajectory(D, m, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, m)
    
    #Integration params: ICs + duration
    x01 = np.random.rand(2); x02 = np.copy(x01)
    T = 100; dt = 1e-3
    t = np.arange(0, T, dt)

    #Run ODE version
    start = time.time()
    res1 = SDE.EulerODE(x01, t, f, driftParams)
    end = time.time()
    print(f'2D SNIC ODE - elapsed time: {end - start: 0.4f}')

    #Run SDE
    start = time.time()
    res2 = SDE.EulerSDE(x02, t, f, driftParams, g, noiseParams)
    end = time.time()
    print(f'2D SNIC SDE - elapsed time: {end - start: 0.4f}')
    print()
    
    #Compare
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)
    axs[0].plot(t, res1[0,:])
    axs[0].plot(t, res2[0,:])
    axs[0].set_ylabel('x')
    #Phase
    axs[1].plot(t, res1[1,:])
    axs[1].plot(t, res2[1,:])
    axs[1].set_ylabel('y')
    axs[1].set_xlabel('t')
    plt.suptitle(f'SNIC - D={D:0.2f} - Comparing ODE/SDE behaviour')
    plt.show()
    
    #Transform time series and compare: phase captures periodicity
    
    psiTraj = PsiFunc(res2)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)
    axs[0].plot(t, res2[0,:], label = 'x(t)')
    axs[0].plot(t, res2[1,:], label = 'y(t)')
    axs[0].legend()
    #Phase
    axs[1].plot(t, psiTraj, label = r'$\psi(t)$')
    axs[1].set_xlabel('t')
    axs[1].legend()
    plt.suptitle('Recovered phase')
    plt.show()
    
    return

def buildPsi(D, m):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    f, driftParams, g, noiseParams = loadParams(D, m)
    
    #Grid
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)

    #Build phase
    SDE.buildPsi2D(path, x, y, f, driftParams, g, noiseParams, 
              nEvals = 31, nbins = 40, nEvalsRed=11)

    return

def reduction(D, m, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, m)
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    print('Building reduction')
    x0 = np.random.rand(2)
    dt = 1e-3; T = 100; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    SDE.SPR(path, x0, t, PsiFunc, 
            f, driftParams, g, noiseParams, 
                nbins=80, nIter=2000, nEvals = 11)
    
    return

def longTermStats(D, m):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    #Long term stats 
    savepath = f'{path}/empirical'
    with open(f'{savepath}/PsiRedCoefs.pck', 'rb') as file_handle:
        aCoefs = pickle.load(file_handle)
        DCoefs = pickle.load(file_handle)

    #Params for integration
    dt = 1e-3; T = 100; nSteps = int(T/dt); 
    nIter = 1000
    t = np.linspace(0, T, nSteps)
    SDE.longTermStats(savepath, t, aCoefs, DCoefs, nIter)
    print('')
    
    #Long term stats using the results from Lindner and Schimansky-Geier, PRL 2002
    SDE.getRotationDiffusion(aCoefs, DCoefs, savepath)
    
    return

def buildPRC(D,m, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, m)
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    print('Computing PRC  empirically (using ergodicity)')
    x0 = np.random.rand(2)
    dt = 1e-2; T = 2000; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    #X direction
    dxPsiFunc = SDE.dxPsiFunc2D(path)
    PsiBinsX, sPRCX, stdErrX = SDE.aiPRC_empirical(x0, t, PsiFunc, dxPsiFunc, 
                        f, driftParams, g, noiseParams, 
                        nbins = 60)
    
    #Y direction
    dyPsiFunc = SDE.dyPsiFunc2D(path)
    PsiBinsY, sPRCY, stdErrY = SDE.aiPRC_empirical(x0, t, PsiFunc, dyPsiFunc, 
                        f, driftParams, g, noiseParams, 
                        nbins = 60)

    savepath = f'{path}/empirical'
    with open(f'{savepath}/empiricalPRC.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(stdErrX, file_handle)
        
        pickle.dump(PsiBinsY, file_handle)
        pickle.dump(sPRCY, file_handle)
        pickle.dump(stdErrY, file_handle)
    
    return 

def phaseResponse(D, m, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, m)
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    
    #Compute empirical phase response in the x/y directions
    X0 = np.random.rand(2)
    T = 50; dt = 1e-3; nSteps = int(T/dt)
    tMax = np.linspace(0, T, nSteps)
    nIter = 1
    
    Ix = 0.01; Iy = 0.; ampList = np.array([Ix, Iy])
    print('Computing PRC in the x-direction via direct perturbation')
    PsiBinsX, sPRCX, std_errX = SDE.PRC_directMethod(X0, tMax, PsiFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins = 50)
    print('')
    Ix = 0.; Iy = 0.01; ampList = np.array([Ix, Iy])
    print('Computing PRC in the y-direction via direct perturbation')
    PsiBinsY, sPRCY, std_errY = SDE.PRC_directMethod(X0, tMax, PsiFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins = 50)
    print('')
    
    with open(f'{path}/phaseResponsePsi.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(std_errX, file_handle)
        
        pickle.dump(PsiBinsY, file_handle)
        pickle.dump(sPRCY, file_handle)
        pickle.dump(std_errY, file_handle)
        
    return


def generateData(D, m):
    
    print('')
    print(f'Running m  = {m:0.3f}')
    print('')

    
    #Build the phase (only need to do this once)
    #buildPsi(D, m)
    
    #Load phase function
    PsiFunc = SDE.PsiFunc2D(f'./SNIC/m={m:0.3f}_D={D:0.2f}')
    
    #Visualize traj and phase
    sampleTrajectory(D, m, PsiFunc)

    #Empirical reduction
    reduction(D, m, PsiFunc)
    
    #Stats
    longTermStats(D, m)
    
    #PRC
    buildPRC(D,m, PsiFunc)
    
    #Phase response
    phaseResponse(D, m, PsiFunc)

    return

def main():
    
    #System params
    m1 = 0.999; m2 = 1.03

    DList = np.linspace(0.01, 0.1, 10)
    for D in DList:
        print(f'D = {D:0.2f}')
        #Above
        generateData(D, m2)
        
        #Below 
        generateData(D, m1)


    return

if __name__ == '__main__':
    main()