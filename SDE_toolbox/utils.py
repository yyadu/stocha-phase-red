#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:53:07 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from numba import jit, float64, complex128, int64

def find_lambda_1(evals, imThr = 1e-10):
    #Find robustly oscillatory eigenvalue in a list of eigenvalues
    minRe = 1e10
    lambda_1 = None
    for i in range(len(evals)):
        im = np.imag(evals[i]); re = np.real(evals[i])
        #..non purely real...slowest rotating....slowest decaying
        if abs(im) > imThr and abs(re) <= minRe and re < 0:
            index1 = i; minRe = abs(re)#; minIm = abs(im)
            lambda_1 = evals[i]
    return lambda_1, index1


###############################################################################
#############   Load computed functions   #####################################
###############################################################################

def loadP0Func(path):
    
    with open(f'{path}/functions/P0Func.pck', 'rb') as file:
        P0Func = pickle.load(file)
        dxP0Func = pickle.load(file)
        dyP0Func = pickle.load(file)
        
    return P0Func, dxP0Func, dyP0Func

def loadQFunc(path):
    
    with open(f'{path}/functions/QFunc.pck', 'rb') as file:
        QrFunc = pickle.load(file)
        dxQrFunc = pickle.load(file)
        dyQrFunc = pickle.load(file)
        
        QimFunc = pickle.load(file)
        dxQimFunc = pickle.load(file)
        dyQimFunc = pickle.load(file)
        
    return QrFunc, dxQrFunc, dyQrFunc, QimFunc, dxQimFunc, dyQimFunc

def loadMRTFunc(path):
    
    with open(f'{path}/functions/MRTFunc.pck', 'rb') as file:
        cosMRTFunc = pickle.load(file)
        dxcosMRTFunc = pickle.load(file)
        dycosMRTFunc = pickle.load(file)
        
        sinMRTFunc = pickle.load(file)
        dxsinMRTFunc = pickle.load(file)
        dysinMRTFunc = pickle.load(file)
        
    return cosMRTFunc, dxcosMRTFunc, dycosMRTFunc, sinMRTFunc, dxsinMRTFunc, dysinMRTFunc


def loadSigmaFunc(path):
        
    with open(f'{path}/functions/SigmaFunc.pck', 'rb') as file:
        sigmaFunc = pickle.load(file)
        dxsigmaFunc = pickle.load(file)
        dysigmaFunc = pickle.load(file)
    
    return sigmaFunc, dxsigmaFunc, dysigmaFunc

def PsiFunc2D(path):
    QrFunc, _, _, QimFunc, _, _ = loadQFunc(path)
    def PsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        return np.mod(np.arctan2(Qim, Qr), 2*np.pi)
    return PsiFunc

def dxPsiFunc2D(path):
    QrFunc, dxQrFunc, _, QimFunc, dxQimFunc, _ = loadQFunc(path)
    def dxPsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dxQr = dxQrFunc((y, x)); dxQim = dxQimFunc((y, x))
        return ( Qr*dxQim - Qim*dxQr ) / ( Qr**2 + Qim**2 )
    return dxPsiFunc

def dyPsiFunc2D(path):
    QrFunc, _, dyQrFunc, QimFunc, _, dyQimFunc = loadQFunc(path)
    def dyPsiFunc(X):
        x, y = X
        Qr = QrFunc((y, x)); Qim = QimFunc((y, x))
        dyQr = dyQrFunc((y, x)); dyQim = dyQimFunc((y, x))
        return ( Qr*dyQim - Qim*dyQr ) / ( Qr**2 + Qim**2 )
    return dyPsiFunc

def MRTFunc2D(path):
    cosFunc, _, _, sinFunc, _, _ = loadMRTFunc(path)
    def MRTFunc(X):
        x, y = X
        cos = cosFunc((y, x)); sin = sinFunc((y, x))
        return np.mod(np.arctan2(sin, cos), 2*np.pi)
    return MRTFunc

def dxMRTFunc2D(path):
    cosFunc, dxcosFunc, _, sinFunc, dxsinFunc, _ = loadMRTFunc(path)
    def dxMRTFunc(X):
        x, y = X
        cos = cosFunc((y, x)); sin = sinFunc((y, x))
        dxcos = dxcosFunc((y, x)); dxsin = dxsinFunc((y, x))
        return ( cos*dxsin - sin*dxcos ) / ( cos**2 + sin**2 )
    return dxMRTFunc

def dyMRTFunc2D(path):
    cosFunc, _, dycosFunc, sinFunc, _, dysinFunc = loadMRTFunc(path)
    def dyMRTFunc(X):
        x, y = X
        cos = cosFunc((y, x)); sin = sinFunc((y, x))
        dycos = dycosFunc((y, x)); dysin = dysinFunc((y, x))
        return ( cos*dysin - sin*dycos ) / ( cos**2 + sin**2 )
    return dyMRTFunc

###############################################################################
###############################################################################

@jit(nopython=True, cache=True)
def unWrapPhase(phaseTL):
    """
    Take phase on the [0, 2pi[ line and stetch it on the real line
    """
    entries = 10
    while entries != 0:
        entries = 0
        for i in range(0, len(phaseTL)-1):
            if phaseTL[i] - phaseTL[i+1] > np.pi:
                phaseTL[i+1] = phaseTL[i+1] + 2*np.pi
                entries = entries + 1
            elif phaseTL[i] - phaseTL[i+1] < -np.pi:
                phaseTL[i+1] = phaseTL[i+1] - 2*np.pi
                entries = entries + 1
    return phaseTL

###############################################################################
###############################################################################

def giveMeIsocrone(isocrones, angle, x, y):
    """
    Find level set of phase function on the grid
    """

    M, N = isocrones.shape
    #print modf(angle%(pi/2))[0]
    if abs(np.modf(angle%(np.pi/2))[0]) < 0.01:
        isocrones = np.mod(isocrones + np.pi/8, 2*np.pi); angle = np.mod(angle + np.pi/8, 2*np.pi);
    
    plt.figure(figsize = (4,4), dpi = 300)
    cs_cos = plt.contour(x, y, np.cos(isocrones).reshape(N,M), levels=[np.cos(angle)], zorder=2, colors='k', linewidths=3)
    cs_sin = plt.contour(x, y, np.sin(isocrones).reshape(N,M), levels=[np.sin(angle)], zorder=2, linestyles = 'dashed', colors='r', linewidths=3)
    plt.close()

    traj = cs_cos.allsegs[0][0]
    X_isocos = traj[:,0]; Y_isocos = traj[:,1]

    traj = cs_sin.allsegs[0][0]
    X_isosin = traj[:,0]; Y_isosin = traj[:,1]

    same_points = []
    tolerance_y = 1e-3; tolerance_x = 1e-3
    for i, ya in enumerate(Y_isocos):
        for j, yb in enumerate(Y_isosin):
            if abs(ya - yb) < tolerance_y:
                if abs(X_isocos[i] - X_isosin[j]) < tolerance_x:
                    same_points.append([X_isocos[i], ya])
        
    same_points = np.array(same_points)
    try:
        IsoX = same_points[:,0]; IsoY = same_points[:,1]
        #Sort by radius to avoid plotting issues
        r = np.sqrt(IsoX**2 + IsoY**2)
        ind2sort = np.argsort(r)
        IsoX = IsoX[ind2sort]
        IsoY = IsoY[ind2sort]
    except:
        IsoX, IsoY = 0, 0

    return IsoX, IsoY

###############################################################################
###############################################################################

@jit((complex128)(float64[:], float64[:], int64), nopython=True, cache=True)
def cn(y, theta, n):
    """
    Get the nth Fourier coef from a "phase series"
    y: data
    theta: phase € [0, 2pi[ (and not a time)
    """
    c = y*np.exp(-1j*n*theta)
    return c.sum()/c.size

@jit((complex128[:])(float64[:], float64[:], int64), nopython=True, cache=True)
def get_coefs(y, theta, Nh):
    """
    Get the Fourier coefs  from a "phase series"
    y: data
    theta: phase € [0, 2pi[ (and not a time)
    Nh: max order
    """
    coefs = np.zeros(Nh, dtype = "complex")
    for i in range(Nh):
        coefs[i] = cn(y, theta, i)
        
    coefs[0] = coefs[0]/2
    return coefs

def phase_fit(theta, data, nTerms):
    """
    Fit a function f(Psi)
    """

    #PBCs
    concTheta = np.concatenate((theta - 2*np.pi, theta, theta + 2*np.pi))
    concFunc =  np.concatenate((data, data, data))
    
    #Interpolate
    interpoFunk = interp1d(concTheta, concFunc)

    #Range of angle values for Fourier transform: even, without 2pi for periodicity
    n = 10; N = 2**n + 1
    angle = np.linspace(0, 2*np.pi, N)[:-1]

    #Get regular data from interpolation
    data2 = interpoFunk(angle)

    #Get the Fourier coefficients
    coefs = get_coefs(data2, angle, nTerms)

    return coefs

###############################################################################

@jit((float64)(float64, complex128[:]), nopython=True, cache=True)
def fourier_fit_single_point(theta, coefs):
    """
    Fitted function using Fourier transform
    x: point of interest
    coefs: Fourier coefs
    """
    Nh =  coefs.size
    f = 0
    for i in range(Nh):
        f += 2*coefs[i]*np.exp(1j*i*theta)
    return f.real

@jit((float64[:])(float64[:], complex128[:]), nopython=True, cache=True)
def fourier_fit_array(theta, coefs):
    """
    Fitted function using Fourier transform
    coefs: Fourier coefs
    """
    Nh =  coefs.size
    f = np.zeros((Nh, len(theta)), dtype = 'complex')
    for i in range(Nh):
        f[i,:] = 2*coefs[i]*np.exp(1j*i*theta)
    out = f.sum(axis = 0)
    
    return out.real