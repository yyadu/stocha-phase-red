#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:51:08 2024

@author: pierrehouzelstein

cf Lindner and Schimansky-Geier, PRL 2002

"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import fourier_fit_array
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import pickle

###############################################################################
#### Theoretical drift and diffusion using a and D coefs ######################
###############################################################################

#Effective potential
def VPsi(Psi, aCoefs, DCoefs, nSteps):
    PsiArray = np.linspace(0, Psi, nSteps)
    a = fourier_fit_array(PsiArray, aCoefs)
    D = fourier_fit_array(PsiArray, DCoefs)
    return trapz(-a/D, PsiArray)

def IPsi(Psi, VFunc, DCoefs, sign, nSteps):

    assert sign == 1 or sign == -1, 'sign must be ±1'
    
    #Range
    if sign > 0:
        PsiTilde = np.linspace(Psi, Psi + 2*np.pi, nSteps);
    else:
        PsiTilde = np.linspace(Psi - 2*np.pi, Psi, nSteps);
        
    #Term in the integral
    DPsi = fourier_fit_array(PsiTilde, DCoefs)
    V = VFunc(PsiTilde)
    integrand = np.exp(sign*V)/np.sqrt(DPsi)

    integral = trapz(integrand, PsiTilde)
    VOut = VFunc(Psi)

    return sign*np.exp(-sign*VOut)*integral

def velocity(DCoefs, VFunc, IPlusFunc, nSteps):
    """
    Use the previously computed values to get mean velocity
    """
    
    PsiTilde = np.linspace(0, 2*np.pi, nSteps)
    
    num = 2*np.pi * ( 1 - np.exp(VFunc(2*np.pi)))
    
    D = fourier_fit_array(PsiTilde, DCoefs)
    IP = IPlusFunc(PsiTilde)
    intgd = IP/np.sqrt(D)
    
    den = trapz(intgd, PsiTilde)
    
    return num/den

def diffusion(DCoefs, VFunc, IPlusFunc, IMinusFunc, nSteps):
    """
    Use the previously computed values to get mean diffusion
    """
    
    PsiTilde = np.linspace(0, 2*np.pi, nSteps)
    D = fourier_fit_array(PsiTilde, DCoefs)
    
    numIntgd = (IPlusFunc(PsiTilde)**2)*IMinusFunc(PsiTilde) / np.sqrt(D)
    num = 4*(np.pi**2)*trapz(numIntgd, PsiTilde)
    
    denIntgd = IPlusFunc(PsiTilde) / np.sqrt(D)
    den = (trapz(denIntgd, PsiTilde))**3
    
    return num/den

def getRotationDiffusion(aCoefs, DCoefs, savepath, filename = 'theorPsiRedCoefs'):

    ###############################################################################
    # Compute the potential V
    
    nSteps1 = 1000
    PsiV = np.linspace(-2*np.pi, 4*np.pi, nSteps1)
    V = np.zeros(nSteps1)
    for i in range(nSteps1):
        #Compute the integral at each point
        V[i] = VPsi(PsiV[i], aCoefs, DCoefs, nSteps=1000)
    VFunc = interp1d(PsiV, V)
    
    plt.plot(PsiV, V, label = r'$V(\psi)$')
    plt.xlabel(r'$\psi$')
    plt.legend()
    plt.show()
    
    ###############################################################################
    # Compute the coefficients I+ and I-
    nSteps2 = 2000
    PsiI = np.linspace(0, 2*np.pi, nSteps2)
    IPlus = np.zeros(nSteps2)
    IMinus = np.zeros(nSteps2)
    for i in range(nSteps2):
        #Here we need a very fine grid size
        IPlus[i] = IPsi(PsiI[i], VFunc, DCoefs, sign=1, nSteps=20000)
        IMinus[i] = IPsi(PsiI[i], VFunc, DCoefs, sign=-1, nSteps=20000)
        
    plt.plot(PsiI, IPlus, label = r'$I_+(\psi)$')
    plt.plot(PsiI, IMinus, label = r'$I_-(\psi)$')
    plt.xlabel(r'$\psi$')
    plt.legend()
    plt.show()
        
    IPlusFunc = interp1d(PsiI, IPlus)
    IMinusFunc = interp1d(PsiI, IMinus)
    
    nSteps = 10000
    Veff = abs(velocity(DCoefs, VFunc, IPlusFunc, nSteps))
    Deff = abs(diffusion(DCoefs, VFunc, IPlusFunc, IMinusFunc, nSteps))
    
    print('Theoretical values for the mean rotation rate and phase diffusion coefficients:')
    print(f'omega = {Veff:0.3f}; Deff =  {Deff:0.3f}')
    
    with open(f'{savepath}/{filename}.pck', 'wb') as file_handle:
        pickle.dump(Veff, file_handle)
        pickle.dump(Deff, file_handle)

    return Veff, Deff
