#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:10:03 2024

@author: pierrehouzelstein
"""

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pickle
from scipy.sparse.linalg import lsqr, spsolve
import os
import shutil
from numba import jit
from  tqdm import tqdm
from scipy.sparse.linalg import eigs
from scipy.integrate import simpson

from .LDaggerFinDiff import buildLDagger1D
from .utils import loadQFunc, loadP0Func, loadSigmaFunc, dxMRTFunc2D, dyMRTFunc2D
from .PhaseBuilder2D import computeGradients
from .integrators import jitDic, _EulerSDE
from .SPR import computeK2
from .utils import phase_fit, unWrapPhase, fourier_fit_array, giveMeIsocrone
from .aiPRC import aiPRC_empirical


def buildMRTPhase2D(path, x, y, f, driftParams, g, noiseParams, 
                    rMinCoef=5, iterLim = 1000,
                    x0 = np.random.rand(2), t = np.linspace(0, 100, int(1000/1e-2)),
                    nbins = 60, nIter = 2000, nEvals = 11):
    
    QrFunc, dxQrFunc, dyQrFunc, QimFunc, dxQimFunc, dyQimFunc = loadQFunc(path)
    P0Func, dxP0Func, dyP0Func = loadP0Func(path)
    
    def PsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        return np.mod(np.arctan2(Qim, Qr), 2*np.pi)
    
    def dxPsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dxQr = dxQrFunc((y, x)); dxQim = dxQimFunc((y, x))
        return ( Qr*dxQim - Qim*dxQr ) / ( Qr**2 + Qim**2 )

    def dyPsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dyQr = dyQrFunc((y, x)); dyQim = dyQimFunc((y, x))
        return ( Qr*dyQim - Qim*dyQr ) / ( Qr**2 + Qim**2 )
    
    def uFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        return np.sqrt(Qr**2 + Qim**2)
    
    xLen = len(x); yLen = len(y)
    X, Y = np.meshgrid(x, y)
    
    #Probability space and grads
    P0data = P0Func((Y, X))
    dxP0data = dxP0Func((Y, X))
    dyP0data = dyP0Func((Y, X))
    
    #Find the phaseless set: (x, y) s.t. u(x,y) = 0 
    u = uFunc([X, Y])
    j, i = np.unravel_index(u.argmin(), u.shape)
    xPhaseless = x[i]; yPhaseless = y[j]

    #Elements of the ggT matrix
    gxx, gxy, gyx, gyy = g([X, Y], noiseParams)
    Gyx = gxx*gyx + gxy*gyy
    Gyy = gyy**2 +gyx**2
    
    #Def the probability current Jy
    xField, yField = f([X, Y], driftParams)
    Jy = yField.reshape(yLen,xLen)*P0data - 0.5*(Gyx*dxP0data + Gyy*dyP0data)
    
    #And now we can integrate along the section
    nPoints = 10000
    xSection = np.linspace(xPhaseless, x[-1], nPoints)
    dx = xSection[1] - xSection[0]
    ySection = np.ones(nPoints)*yPhaseless
    
    #Interpolate proba current
    f_jy = interp.RectBivariateSpline(y, x, Jy);
    yValues = f_jy(ySection, xSection, grid=False)
    
    #Integrate
    Tbar = 1/np.trapz(yValues, dx=dx)
    print(f'MRT period: T = {Tbar}')
    np.savetxt(f'{path}/MRTPeriod', [Tbar])
    
    def dxlnuFunc(X):
        x, y = X
        
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dxQr = dxQrFunc((y, x)); dxQim = dxQimFunc((y, x))
        
        num = Qr*dxQr + Qim*dxQim
        den = Qr**2 + Qim**2
        return num/den
    
    def dylnuFunc(X):
        X, y = X
        
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dyQr = dyQrFunc((y, x)); dyQim = dyQimFunc((y, x))
        
        num = Qr*dyQr + Qim*dyQim
        den = Qr**2 + Qim**2
        return num/den
    
    def aFunc(X, omega, gxx, gyy):
        """
        Drift of the phase SDE
        """
        x, y = X
        
        dx = -dxlnuFunc(X)*dxPsiFunc(X)*gxx**2
        dy = -dylnuFunc(X)*dyPsiFunc(X)*gyy**2
        return omega + (dx + dy)
    
    #Now that we have the mean period, we can get the phase difference
    omega = np.loadtxt(f'{path}/omega')
    aTerm = aFunc([X, Y], omega, gxx, gyy)
    bTerm = (2*np.pi/Tbar) - aTerm
    
    #Remove the points too close to the phaseless set
    entradas = 0
    rMin = rMinCoef*dx
    for i in range(0, xLen):
        for j in range(0, yLen):
            r = np.sqrt((x[i]-xPhaseless)**2 + (y[j]-yPhaseless)**2)
            if r<rMin: 
                bTerm[j,i]=0; entradas = entradas + 1
                
    #Solve Ldagger[DeltaPsi] = bTerm: least squares regression
    with open(f'{path}/LDagger.pck', 'rb') as file_handle:
        LDagger = pickle.load( file_handle)
    
    #Initial guess
    deltaPhi_1 = spsolve(LDagger, bTerm.reshape(xLen*yLen)); 
    error = (LDagger * deltaPhi_1).reshape(yLen, xLen) - bTerm; 
    eMax1 = max(abs(error.reshape(xLen*yLen))); 
    
    deltaPhi_2, istop, itn, r1norm = lsqr(LDagger, bTerm.reshape(xLen*yLen), iter_lim = iterLim, conlim = 0, atol= 0, btol= 0)[:4]
    error = (LDagger * deltaPhi_2).reshape(yLen, xLen) - bTerm; 
    eMax2 = max(abs(error.reshape(xLen*yLen))); 
    
    if (eMax2 - eMax1) > 0: 
        deltaPhi = deltaPhi_1; iterNum = 0; eMax = eMax1
    else: 
        deltaPhi = deltaPhi_2; 
        iterNum = iterLim; 
        eMax = eMax2

    #regression
    print('')
    print('Iterative construction of MRT phase')
    print('iterNum = %s\t eMax = %s' % (iterNum, eMax));
    while eMax > 1e-4 and iterNum < 10000:
        deltaPhi, istop, itn, r1norm = lsqr(LDagger, bTerm.reshape(xLen*yLen), iter_lim = iterLim, conlim = 0, atol= 0, btol= 0, x0 = deltaPhi.reshape(xLen*yLen))[:4]
        error = (LDagger * deltaPhi).reshape(yLen,xLen) - bTerm; 
        deltaPhi = deltaPhi.reshape(yLen,xLen)
        eMax = max(abs(error.reshape(xLen*yLen))); 
        iterNum = iterNum + iterLim
        print('iterNum = %s\t eMax = %s' % (iterNum, eMax)); 
    
    #Use the computed difference to get MRT phase
    Psi = PsiFunc([X,Y])
    mrt = np.mod((Psi.reshape(yLen,xLen) + deltaPhi.reshape(yLen,xLen)), 2*np.pi)
    
    # #Get data for first zero level - Use it to define 0-phase
    
    #Find the 0-level
    SigmaFunc, _, _  = loadSigmaFunc(path)
    sigma = SigmaFunc((Y, X))
    cs = plt.contour(x, y, sigma, levels=[0.0], zorder=2, colors='k', linewidths=3)
    plt.show()
    
    xLC = cs.allsegs[0][0][:,0]; yLC = cs.allsegs[0][0][:,1]
    #Shift so max value is at index 0
    index = np.argmax(xLC); 
    xLC = np.roll(xLC, -index); yLC = np.roll(yLC, -index)
    
    #Interpolate 
    _cosMRTFunc = interp.RegularGridInterpolator((x, y), np.cos(mrt), bounds_error=False, fill_value=None)
    _sinMRTFunc = interp.RegularGridInterpolator((x, y), np.sin(mrt), bounds_error=False, fill_value=None)

    #Find phase of max and shift
    c = _cosMRTFunc((yLC[0], xLC[0]))
    s = _sinMRTFunc((yLC[0], xLC[0]))
    phase = np.mod(np.arctan2(s, c), 2*np.pi)
    mrt = np.mod(mrt - phase, 2*np.pi)
    
    plt.figure(dpi = 300)
    plt.title('MRT phase')
    plt.pcolormesh(X, Y, mrt, cmap = 'hsv')
    plt.colorbar()
    plt.show()

    #Build the spatial derivatives of the MRT
    #Create folder to save data
    savepath = f'{path}/gridData'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    computeGradients(savepath, x, y, np.cos(mrt), 'cosMRT')
    computeGradients(savepath, x, y, np.sin(mrt), 'sinMRT')
    
    dxcosdata = np.loadtxt(f'{savepath}/dxcosMRT')
    dycosdata = np.loadtxt(f'{savepath}/dycosMRT')
    dxsindata = np.loadtxt(f'{savepath}/dxsinMRT')
    dysindata = np.loadtxt(f'{savepath}/dysinMRT')
    
    #Interpolate
    _cosMRTFunc = interp.RegularGridInterpolator((x, y), np.cos(mrt), bounds_error=False, fill_value=None)
    _sinMRTFunc = interp.RegularGridInterpolator((x, y), np.sin(mrt), bounds_error=False, fill_value=None)
    _dxcosMRTFunc = interp.RegularGridInterpolator((x, y), dxcosdata, bounds_error=False, fill_value=None)
    _dycosMRTFunc = interp.RegularGridInterpolator((x, y), dycosdata, bounds_error=False, fill_value=None)
    _dxsinMRTFunc = interp.RegularGridInterpolator((x, y), dxsindata, bounds_error=False, fill_value=None)
    _dysinMRTFunc = interp.RegularGridInterpolator((x, y), dysindata, bounds_error=False, fill_value=None)
    
    funcpath  = f'{path}/functions'
    if not os.path.exists(funcpath):
        os.makedirs(funcpath)
        
    with open(f'{funcpath}/MRTFunc.pck', 'wb') as file_handle:
        pickle.dump(_cosMRTFunc, file_handle)
        pickle.dump(_dxcosMRTFunc, file_handle)
        pickle.dump(_dycosMRTFunc, file_handle)
        
        pickle.dump(_sinMRTFunc, file_handle)
        pickle.dump(_dxsinMRTFunc, file_handle)
        pickle.dump(_dysinMRTFunc, file_handle)
    
    #Delete heavy folder with grid data
    shutil.rmtree(savepath)
    
    return

def MRT_SPR(path, x0, t, phaseFunc, f, driftParams, g, 
             noiseParams, nbins, nIter, nEvals):
    """
    Numerical evaluation of the coefficients by means of the short term
    statistics
    """
    
    #Path in which to save data
    if not os.path.exists(path):
        os.makedirs(path)
        
    Tbar = np.loadtxt(f'{path}/MRTPeriod')
    
    #Check 
    assert type(driftParams) == dict, 'Must pass a driftParams dict in args'
    assert type(noiseParams) == dict, 'Must pass a noiseParams dict in args'

        
    #Prepare integrator
    fJitted = jit(f, nopython=True, cache=True); gJitted = jit(g, nopython=True, cache=True)
    solver = _EulerSDE(fJitted, gJitted)
    driftParams = jitDic(driftParams)
    noiseParams = jitDic(noiseParams)

    #Phase bins
    bins = np.linspace(0, 2*np.pi, nbins + 1)[:-1] 
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

    #For MRT, this one is constant
    K1_Final = (2*np.pi / Tbar )*np.ones(len(bins)-1)

    #Now K2
    K2_Full = np.zeros(len(bins)-1)
    counter_Full = np.zeros(len(bins)-1)
    for i in range(nIter):
        K2, counter = computeK2(phaseTraj[i,:], bins, phaseArray)
        K2_Full += K2
        counter_Full += counter
    
    K2_Final = (K2_Full/(counter_Full) - (K1_Final*dt)**2)/(2*dt)

    #Now, we fit the data
    K1_coefs = phase_fit(phaseArray, K1_Final, nTerms = 50)
    K2_coefs = phase_fit(phaseArray, K2_Final, nTerms = 50)
    
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
    savepath = f'{path}/empiricalMRT'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    with open(f'{savepath}/MRTRedCoefs.pck', 'wb') as file_handle:
        pickle.dump(K1_coefs, file_handle)
        pickle.dump(K2_coefs, file_handle)
        
    ###########################################################################
    ####  Build  the reduced LDagger operator  ################################
    ###########################################################################
    
    def aFunc(X):
        return fourier_fit_array(X, K1_coefs)
    
    def sqrt2DFunc(X):
        return np.sqrt(2*fourier_fit_array(X, K2_coefs))
    
    PsiGrid1D = np.linspace(0, 2*np.pi, 3000 + 1)[:-1]
    LDagger1D = buildLDagger1D(PsiGrid1D, aFunc, sqrt2DFunc)
    evalsRed, _ = eigs(LDagger1D, k=nEvals, sigma = -0.1)
    np.savetxt(f'{savepath}/eValsLDRed', evalsRed)
    
    evals = np.loadtxt(f'{path}/eValsLD', dtype = complex)
    
    plt.figure(dpi = 300)
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(evals.real, evals.imag, 'bo', label = 'Ruelle-Pollicott resonances')
    plt.plot(evalsRed.real, evalsRed.imag, 'r.', label = 'Reduced Ruelle-Pollicott resonances')
    plt.title(r'Spectra of $\mathcal{L}^\dagger$ and $\mathcal{L}^\dagger_\psi$')
    plt.legend()
    plt.show()

    ###########################################################################
        
    print('Use generated data to compute long terms statistics')
    dphase = np.zeros(nIter)
    
    #Start at 10% of the integration length to be in steady state
    start = int(0.1*nSteps)
    
    for i in range(nIter):
        phase = unWrapPhase(phaseTraj2[i])
        dphase[i] = (phase[-1] - phase[start])
        
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
    np.savetxt(f'{path}/longTermStatsMRT', results)
    np.savetxt(f'{path}/longTermStatsMRTstdErr', stdErrs)
    
    print(f'Long term statistics: w = {omega:0.3f} ± {stdErrOmega:0.3f} - D = {D:0.3f} ± {stdErrD:0.3f}')

    return

