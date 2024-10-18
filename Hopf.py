#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:37:57 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import SDE_toolbox as SDE
import pickle

def loadParams(D, delta):

    f = SDE.Hopf2D
    driftParams = {'beta': 0.5, 'gamma': 4, 'delta': delta, 'kappa': 1}
    g = SDE.additive2D
    sxx = syy = np.sqrt(2*D)
    noiseParams = {'sxx': sxx, 'sxy': 0., 'syx': 0., 'syy': syy}
    
    return f, driftParams, g, noiseParams

def sampleTrajectory(D, delta, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, delta)
    
    #Integration params: ICs + duration
    x01 = np.random.rand(2); x02 = np.copy(x01)
    T = 50; dt = 1e-3
    t = np.arange(0, T, dt)

    #Run ODE version
    start = time.time()
    res1 = SDE.EulerODE(x01, t, f, driftParams)
    end = time.time()
    print(f'2D Hopf ODE - elapsed time: {end - start: 0.4f}')

    #Run SDE
    start = time.time()
    res2 = SDE.EulerSDE(x02, t, f, driftParams, g, noiseParams)
    end = time.time()
    print(f'2D Hopf SDE - elapsed time: {end - start: 0.4f}')
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
    plt.suptitle(f'Hopf - D={D:0.2f} - Comparing ODE/SDE behaviour')
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

def buildPsi(D, delta):
    
    path = f'./Hopf/d={delta:0.3f}_D={D:0.2f}'
    f, driftParams, g, noiseParams = loadParams(D, delta)
    
    #Grid
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)

    #Build phase
    SDE.buildPsi2D(path, x, y, f, driftParams, g, noiseParams, 
              nEvals = 60, nbins = 40, nEvalsRed=11)

    return

def reduction(D, delta, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, delta)
    path = f'./Hopf/d={delta:0.3f}_D={D:0.2f}'
    
    print('Building reduction')
    x0 = np.random.rand(2)
    dt = 1e-3; T = 100; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    SDE.SPR(path, x0, t, PsiFunc, 
            f, driftParams, g, noiseParams, 
                nbins=80, nIter=1000, nEvals = 11)
    
    return

def longTermStats(D, delta):
    
    path = f'./Hopf/d={delta:0.3f}_D={D:0.2f}'
    
    #Long term stats 
    savepath = f'{path}/empirical'
    with open(f'{savepath}/PsiRedCoefs.pck', 'rb') as file_handle:
        aCoefs = pickle.load(file_handle)
        DCoefs = pickle.load(file_handle)

    #Params for integration
    dt = 1e-3; T = 100; nSteps = int(T/dt); nIter = 1000
    t = np.linspace(0, T, nSteps)
    SDE.longTermStats(savepath, t, aCoefs, DCoefs, nIter)
    print('')
    
    #Long term stats using the results from Lindner and Schimansky-Geier, PRL 2002
    SDE.getRotationDiffusion(aCoefs, DCoefs, savepath)
    
    return

def buildPRC(D, delta, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, delta)
    path = f'./Hopf/d={delta:0.3f}_D={D:0.2f}'
    
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

def phaseResponse(D, delta, PsiFunc):
    
    f, driftParams, g, noiseParams = loadParams(D, delta)
    path = f'./Hopf/d={delta:0.3f}_D={D:0.2f}'
    
    #Compute empirical phase response in the x/y directions
    X0 = np.random.rand(2)
    T = 50; dt = 1e-4; nSteps = int(T/dt)
    tMax = np.linspace(0, T, nSteps)
    nIter = 1
    
    Ix = 0.01; Iy = 0.; ampList = np.array([Ix, Iy])
    print('Computing PRC in the x-direction via direct perturbation')
    PsiBinsX, sPRCX, std_errX = SDE.PRC_directMethod(X0, tMax, PsiFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins =60, STEP=100)
    print('')
    Ix = 0.; Iy = 0.01; ampList = np.array([Ix, Iy])
    print('Computing PRC in the y-direction via direct perturbation')
    PsiBinsY, sPRCY, std_errY = SDE.PRC_directMethod(X0, tMax, PsiFunc, 
            f, driftParams, g, noiseParams, ampList, nIter, nbins = 60, STEP=100)
    print('')
    
    with open(f'{path}/phaseResponsePsi.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(std_errX, file_handle)
        
        pickle.dump(PsiBinsY, file_handle)
        pickle.dump(sPRCY, file_handle)
        pickle.dump(std_errY, file_handle)
        
    return


def generateData(D, delta):
    
    print('')
    print(f'Running delta  = {delta:0.3f}')
    print('')

    
    #Build the phase (only need to do this once)
    #buildPsi(D, delta)
    
    #Load phase function
    PsiFunc = SDE.PsiFunc2D(f'./Hopf/d={delta:0.3f}_D={D:0.2f}')
    
    #Visualize traj and phase
    sampleTrajectory(D, delta, PsiFunc)

    #Empirical reduction
    reduction(D, delta, PsiFunc)
    
    #Stats
    longTermStats(D, delta)
    
    #PRC
    buildPRC(D, delta, PsiFunc)
    
    #Phase response
	phaseResponse(D, delta, PsiFunc)

    return

def main():

    delta1 = -0.01; delta2 = 1
    
    DList = np.linspace(0.01, 0.1, 10)
    for D in DList:
        print(f'D = {D:0.2f}')
        generateData(D, delta1)
        generateData(D, delta2)

    return

if __name__ == '__main__':
    main()