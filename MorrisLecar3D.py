#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:05:43 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import SDE_toolbox as SDE
import pickle
from tqdm import tqdm



bm = -1.2; gm = 18
bz = -21; gz = 15
by = -10; gy = 10; phiy = 0.15

ENa = 50; EK = -100; EL = -70

gFast = 20; gKdr = 20; gL = 2

def MorrisLecar3D(X, params):
    
    Iapp = params['Iapp']
    gSub = params['gSub']
    ESub = params['ESub']
    phiz = params['phiz'] 
    
    V, y, z = X
    
    m0 = 0.5 * ( 1 + np.tanh( ( V - bm ) / gm ) )
    y0 = 0.5 * ( 1 + np.tanh( ( V - by ) / gy ) )
    z0 = 0.5 * ( 1 + np.tanh( ( V - bz ) / gz ) )
    
    tauy = 1 / np.cosh(( V - by ) / ( 2 * gy ))
    tauz = 1 / np.cosh(( V - bz ) / ( 2 * gz ))
    
    INa = gFast * m0 * ( V - ENa )
    IKdr = gKdr * y * ( V - EK )
    ISub = gSub * z * ( V - ESub )
    IL = gL * ( V - EL )
    
    fV = Iapp - INa - IKdr - ISub - IL
    fy = phiy * ( y0 - y ) / tauy
    fz = phiz * ( z0 - z ) / tauz
    
    return np.array([fV, fy, fz])

def additive3D(X, params):
    
    x, y, z = X
    sxx = params['sxx']
    
    gxx = sxx*np.ones(np.shape(x))
    gxy = gxz =  0*np.ones(np.shape(x))
    gyx = gyy = gyz =  0*np.ones(np.shape(y))
    gzx = gzy = gzz = 0*np.ones(np.shape(z))

    return np.array([gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz])

def sampleTrajectory(D, MLParams, PsiFunc):
    
    noiseParams = {'sxx': np.sqrt(2*D)}
    
    #Integration params: ICs + duration
    x01 = np.random.rand(3); x01[0] = 0.; x02 = np.copy(x01)

    T = 30; dt = 1e-2; nSteps = int(T/dt)
    tTr = 50; nTr = int(tTr/dt)
    t = np.linspace(0, T+tTr, nSteps+nTr)


    #start = time.time()
    #res2 = SDE.EulerSDE(x02, t, MorrisLecar3D, class1below, additive3D, noiseParams)
    res2 = SDE.EulerSDE(x02, t, MorrisLecar3D, MLParams, additive3D, noiseParams)
    #end = time.time()

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)
    #Voltage
    axs[0].plot(t, res2[0,:], label = ('V'))
    axs[0].legend()
    #Phase
    axs[-1].plot(t, PsiFunc(res2), label = r'$\psi(t)$')
    #axs[-1].plot(t[discontinuities], PsiFunc(res2)[discontinuities], 'ro')
    axs[-1].set_xlabel('t')
    axs[-1].legend()
    plt.suptitle(f'3D Morris-Lecar model - D = {D}')
    plt.show()
    
    # path = f'./MorrisLecar3D/c1bD={D}'
    # with open(f'{path}/sampleTraj.pck', 'wb') as file_handle:
    #     pickle.dump(t[:-nTr], file_handle)
    #     pickle.dump(res2[:,nTr:], file_handle)

    return

    
    
def reduction(D, MLParams, path, PsiFunc):
    
    noiseParams = {'sxx': np.sqrt(2*D)}
    
    print('Empirical reduction')
    x0 = np.zeros(3)#; x0[0] = -65.
    dt = 1e-2; T = 100; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    
    SDE.SPR(path, x0, t, PsiFunc, 
            MorrisLecar3D, MLParams, additive3D, noiseParams, 
                nbins=80, nIter=2000, nEvals = 11)
    print('')
    
    return

def longTermStats(path):
    
    savepath = f'{path}/empirical'
    with open(f'{savepath}/PsiRedCoefs.pck', 'rb') as file_handle:
        aCoefs = pickle.load(file_handle)
        DCoefs = pickle.load(file_handle)

    #Params for integration
    dt = 1e-2; T = 1000; nSteps = int(T/dt); nIter = 1000
    t = np.linspace(0, T, nSteps)
    SDE.longTermStats(savepath, t, aCoefs, DCoefs, nIter)
    print('')
    
    # #Long term stats using the results from Lindner and Schimansky-Geier, PRL 2002
    #SDE.getRotationDiffusion(aCoefs, DCoefs, savepath)
    # #Commented due to overflows
    
    return

def buildPRC(D, MLParams, path):
    
    f = MorrisLecar3D; g = additive3D
    driftParams = MLParams; noiseParams = {'sxx': np.sqrt(2*D)}
    
    #Empirically 
    
    print('Computing PRC in the V-direction via gradient averaging')
    x0 = np.zeros(3); x0[0] = -65.
    T = 10000; dt = 1e-2; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    PsiBinsV, dVPsi = SDE.aiPRCgEDMD(path, x0, t, MorrisLecar3D, MLParams, additive3D, noiseParams,
                derivand = 0, nbins = 80)
    
    with open(f'{path}/empiricalPRC.pck', 'wb') as file_handle:

        pickle.dump(PsiBinsV, file_handle)
        pickle.dump(dVPsi, file_handle)
        

    
    return

def phaseResponse(D, MLParams, PsiFunc, path):
    
    noiseParams = {'sxx': np.sqrt(2*D)}
    
    x0 = np.zeros(3); x0[0] = -65.
    T =100; dt = 1e-3; nSteps = int(T/dt)
    tMax = np.linspace(0, T, nSteps)
    nIter = 1
    
    IV = 1; Iy = 0.; Iz = 0; ampList = np.array([IV, Iy, Iz])
    print('Computing PRC in the V-direction via direct perturbation')
    PsiBinsX, sPRCX, stdErrX = SDE.PRC_directMethod(x0, tMax, PsiFunc, 
            MorrisLecar3D, MLParams, additive3D, noiseParams, ampList, nIter, nbins = 50)
    print('')
    
    with open(f'{path}/PsiResp.pck', 'wb') as file_handle:
        pickle.dump(PsiBinsX, file_handle)
        pickle.dump(sPRCX, file_handle)
        pickle.dump(stdErrX, file_handle)


    return


def main():
    
    D = 20.
    
    f = MorrisLecar3D
    g = additive3D
    
    driftParams = {'ESub': 50, 'phiz': 0.5, 'gSub': 2, 'Iapp': 29.} #class1below
    path = f'./MorrisLecar3D/c1bD={D}'
    
    # driftParams = {'ESub': 50, 'phiz': 0.5, 'gSub': 2, 'Iapp': 40.} #class 1 above
    # path = f'./MorrisLecar3D/c1aD={D}'
    
    #class3Params = {'ESub':  -100, 'phiz': 0.15, 'gSub': 7, 'Iapp': 70}
    #path = f'./MorrisLecar3D/c3D={D}'

    nMon = 2

    gridSizes = [500, 100, 100]

    noiseParams = {'sxx': np.sqrt(2*D)}

    print('Building Psi')
    T = 3000; dt = 0.01
    nTr = int((0.1*T)/dt)
    nSteps = int(T/dt) + nTr
    t = np.linspace(0, T + 0.1*T, nSteps)
    x0 = np.zeros(3)

    SDE.PsiBuildergEDMD(path, x0, t, f, driftParams, g, noiseParams, 
                        gridSizes, nMon, nEvals = 30, basis='monomials')

    sampleTrajectory(D, driftParams, PsiFunc= SDE.PsigEDMD(path))
    


    reduction(D, driftParams, path, PsiFunc= SDE.PsigEDMD(path))
    
    longTermStats(path)
    
    
    buildPRC(D, driftParams, path)
    
    phaseResponse(D, driftParams, SDE.PsigEDMD(path), path)
    

    


    
    return

if __name__ == '__main__':
    main()