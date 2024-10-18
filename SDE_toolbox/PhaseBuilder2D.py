#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:40:27 2024

@author: pierrehouzelstein


"""

import numpy as np
from scipy.sparse.linalg import eigs
from scipy import integrate
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import os
import pickle

from .utils import unWrapPhase
from .LDaggerFinDiff import buildLDagger2D

import time
import inspect
import shutil

def buildPsi2D(path, x, y, f2D, driftParams, g2D, noiseParams, 
          nEvals = 31, nbins = 40, nEvalsRed=11):
    """
    Builds the stochastic phase reduction of a 2D stochastic oscillator with Ito SDE
    dX = f(X)*dt + g(X)*dWt, X = [x, y]^T

    Parameters
    ----------
    path : string
        Path to folder in which data will be written. If does not exist, will be written.
    x : array
        Grid range in the x direction.
    y : array
        Grid range in the y direction.
    driftFunc2D : function
        Drift function that takes X = [x, y] and driftParams as inputs.
        driftFunc2D(X, driftParams) = f(x, y).
    noiseAmpFunc2D : function
        Noise amplitude function that takes X = [x, y] and noiseParams as inputs.
        noiseAmpFunc2D(X, noiseParams) = g(x, y).
    driftParams : dictionary
        Dictionary containing the parameters for driftFunc2D.
    noiseParams : dictionary
        Dictionary containing the parameters for driftFunc2D.
    nEvals : int, optional
        Number of eigenvalues to compute when diagonalizing LDagger. 
        The default is 31.
    nbins : int, optional
        Number of isocrones over which to interpolate when computing the phase reductions. 
        The default is 40.
    nEvals : int, optional
        Number of eigenvalues to compute when diagonalizing the reduced LDagger. The default is 11.

    Returns
    -------
    None.

    """

    #Check that the drift/noise amplitude functions have the parameters they need
    nf = len(inspect.getfullargspec(f2D).args)
    ng = len(inspect.getfullargspec(g2D).args)
    
    assert nf == 1 or nf ==2, 'f  must be of type f(X) or f(X, dic)'
    assert ng == 1 or ng ==2, 'g  must be of type g(X) or g(X, dic)'
    
    if nf == 2:
        assert type(driftParams) == dict, 'Must pass a driftParams dict in args'
        def f(X):
            return f2D(X, driftParams)
    if ng == 2:
        assert type(noiseParams) == dict, 'Must pass a noiseParams dict in args'
        def g(X):
            return g2D(X, noiseParams)

    ###########################################################################
    
    print('Building stochastic asymptotic phase on 2D grid')
    print('')
    
    PsiBuilder2D(x, y, f, g, path, nEvals)
    
    ###########################################################################
    
    #Delete heavy folder with grid data
    savepath = f'{path}/gridData'
    shutil.rmtree(savepath)

    print('Done!')
    return

###############################################################################
##############  Main function  ################################################
###############################################################################

def PsiBuilder2D(x, y, f, g, path, nEvals):
    """
    Build Psi(x, y) the stochastic asymptotic phase: argument of Q, the slowest
    decaying eigenfunction of the Kolmogorov backward operator LDagger
    """
    
    #Create folder to save data
    savepath = f'{path}/gridData'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    #Compute LDagger
    LDagger = buildLDagger2D(x, y, f, g)
    with open(f'{path}/LDagger.pck', 'wb') as file_handle:
        pickle.dump(LDagger, file_handle)

    #Diagonalize
    print('Diagonalizing LDagger - this might take a while...')
    eigenVals, eigenVects = eigs(LDagger, k=nEvals, sigma = -1)
    np.savetxt(f'{path}/eValsLD', eigenVals)
    
    #Representation of the Fokker-Planck operator
    eigenValsFP, eigenVectsFP = eigs(LDagger.H, k=3, sigma=0)

    #Stationary density
    giveMeP0(savepath, x, y, eigenValsFP, eigenVectsFP)
    
    #Isostables, isocrones
    real_evals, xLC, yLC = findAllIsostables(savepath, x, y, eigenVals, eigenVects)
    mu, omega, isocrones = find_Q(savepath, x, y, eigenVals, eigenVects)
    
    #Save rotation
    np.savetxt(f'{path}/omega', [omega])
    
    #Quality factor: must be > 1
    print(f'Quality factor: {abs(omega/mu):0.2f}')
    
    #Plot isocrones
    X, Y = np.meshgrid(x, y)
    plt.figure(dpi = 300)
    plt.title('Isocrones')
    plt.pcolormesh(X, Y, isocrones, cmap = 'hsv')
    plt.colorbar()
    plt.show()
    
    #Compute gradients
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/P0' ), 'P0')
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/isocrones' ), 'Psi')
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/isostable_1/isostables' ), 'Sigma')
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/uValues'), 'u');
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/lnuValues'), 'lnu');
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/R[Q]'), 'R[Q]')
    computeGradients(savepath, x, y, np.loadtxt(f'{savepath}/Im[Q]'), 'Im[Q]')
    
    #Interpolate functions on the grid and save them
    
    #Probability
    P0data = np.loadtxt(f'{savepath}/P0')
    P0Func = interp.RegularGridInterpolator((x, y), P0data, bounds_error=False, fill_value=None)
    
    dxP0data = np.loadtxt(f'{savepath}/dxP0')
    dxP0Func = interp.RegularGridInterpolator((x, y), dxP0data, bounds_error=False, fill_value=None)
    
    dyP0data = np.loadtxt(f'{savepath}/dyP0')
    dyP0Func = interp.RegularGridInterpolator((x, y), dyP0data, bounds_error=False, fill_value=None)

    #Q(x,y) (see Pérez-Cervera et al. 2023)
    Qrdata = np.loadtxt(f'{savepath}/R[Q]')
    QrFunc = interp.RegularGridInterpolator((x, y), Qrdata, bounds_error=False, fill_value=None)
    
    Qimdata = np.loadtxt(f'{savepath}/Im[Q]')
    QimFunc = interp.RegularGridInterpolator((x, y), Qimdata, bounds_error=False, fill_value=None)
    
    dxQrdata = np.loadtxt(f'{savepath}/dxR[Q]')
    dxQrFunc = interp.RegularGridInterpolator((x, y), dxQrdata, bounds_error=False, fill_value=None)
    
    dyQrdata = np.loadtxt(f'{savepath}/dyR[Q]')
    dyQrFunc = interp.RegularGridInterpolator((x, y), dyQrdata, bounds_error=False, fill_value=None)
    
    dxQimdata = np.loadtxt(f'{savepath}/dxIm[Q]')
    dxQimFunc = interp.RegularGridInterpolator((x, y), dxQimdata, bounds_error=False, fill_value=None)
    
    dyQimdata = np.loadtxt(f'{savepath}/dyIm[Q]')
    dyQimFunc = interp.RegularGridInterpolator((x, y), dyQimdata, bounds_error=False, fill_value=None)
    
    #Stochastic amplitude (see Pérez-Cervera et al. 2021)
    sigmadata = np.loadtxt(f'{savepath}/isostable_1/isostables')
    sigmaFunc = interp.RegularGridInterpolator((x, y), sigmadata, bounds_error=False, fill_value=None)
    
    dxsigmadata = np.loadtxt(f'{savepath}/dxSigma')
    dxsigmaFunc = interp.RegularGridInterpolator((x, y), dxsigmadata, bounds_error=False, fill_value=None)
    
    dysigmadata = np.loadtxt(f'{savepath}/dySigma')
    dysigmaFunc = interp.RegularGridInterpolator((x, y), dysigmadata, bounds_error=False, fill_value=None)
    
    funcpath  = f'{path}/functions'
    if not os.path.exists(funcpath):
        os.makedirs(funcpath)
        
    #Save interpolated functions
    with open(f'{funcpath}/P0Func.pck', 'wb') as file_handle:
        pickle.dump(P0Func, file_handle)
        pickle.dump(dxP0Func, file_handle)
        pickle.dump(dyP0Func, file_handle)
        
    with open(f'{funcpath}/QFunc.pck', 'wb') as file_handle:
        pickle.dump(QrFunc, file_handle)
        pickle.dump(dxQrFunc, file_handle)
        pickle.dump(dyQrFunc, file_handle)
        
        pickle.dump(QimFunc, file_handle)
        pickle.dump(dxQimFunc, file_handle)
        pickle.dump(dyQimFunc, file_handle)
        
    with open(f'{funcpath}/SigmaFunc.pck', 'wb') as file_handle:
        pickle.dump(sigmaFunc, file_handle)
        pickle.dump(dxsigmaFunc, file_handle)
        pickle.dump(dysigmaFunc, file_handle)

    return

###############################################################################
##############  Helper functions  #############################################
###############################################################################

def giveMeP0(path, x, y, eigenValsFP, eigenVectsFP):
    """
    Find stationary distribution using Fokker-Planck operator eigenvalues
    """
    
    #Find the eigenvalue closest to 0+ 1j*0
    minRe = 50.1; minIm = 50.1
    for i in range(len(eigenValsFP)):
        re = np.real(eigenValsFP[i]); im = np.imag(eigenValsFP[i])
        if abs(im) < 0.001 and abs(re) < 0.001 and abs(re) < minRe and abs(im) < minIm:
            index = i; minRe = abs(re); minIm = abs(im)
    minEval = minRe + 1j*minIm
    P0 = np.real(eigenVectsFP[:,index]); 
    #Space size
    N = len(x); M = len(y) 
    P0 = P0.reshape(M,N)
    #Normalize
    dx = x[1] - x[0]; dy = y[1] - y[0]
    norm = performIntegration(P0, dx, dy)
    P0 /= norm
    
    #save
    strToSave = f'{path}/P0'
    np.savetxt(strToSave, P0)
    
    #Plot
    X, Y = np.meshgrid(x, y)
    
    plt.figure(dpi = 300)
    plt.pcolormesh(X, Y, P0)
    plt.colorbar()
    plt.title(r'$P_0$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{path}/P0')
    plt.show()

    return minEval

def performIntegration(array2integrate, dx, dy):
    """
    A method to integrate samples over 2d array with constant step
    """
    M, N = array2integrate.shape
    xComponent = np.zeros(M)
    for i in range(0, M):
        xComponent[i] = integrate.simpson(array2integrate[i,:], dx=dx)
    norm = integrate.simpson(xComponent, dx=dy); 
    return norm

def findAllIsostables(path, x, y, eigenVals, eigenVects):
    #Find all isostables
    error = False
    upper_bound = None
    real_evals = []
    xLC_list = []; yLC_list = []
    while error == False:
        try:
            #Get eigenvalue/associated eigenfunction
            lambdaFloquet, LC = find_Sigma(path, x, y, eigenVals, eigenVects, upper_bound)
            #Update upper bound for next iteration
            upper_bound = lambdaFloquet
            real_evals.append(upper_bound)
            xLC_list.append(LC[0,:]); yLC_list.append(LC[1,:])
        except:
            error = True
            
    return np.array(real_evals), np.array(xLC_list, dtype=object), np.array(yLC_list, dtype=object)

def find_Sigma(path, x, y, eigenVals, eigenVects, upper_bound = None):
    """
    Find the purely real eigenvalue smaller than the upper bound
    Find the associated eigenfunction Sigma
    Find the zero level
    """
    #Default: upper bound = Q eigenvalue
    if upper_bound == None:
        #Find mu
        minRe = 10.1
        for i in range(len(eigenVals)):
            im = np.imag(eigenVals[i]); re = np.real(eigenVals[i])
            if im > 0.0 and abs(re) < minRe and abs(re) > 0.0:
                    index = i; minRe = abs(re)
        upper_bound = eigenVals[index].real
    
    #Find lambda Floquet
    minRe = -1e3
    lambdaFloquet = None
    purely_real_evals = []
    for i in range(len(eigenVals)):
        im = np.imag(eigenVals[i]); re = np.real(eigenVals[i])
        if re > minRe and re < upper_bound and abs(im) < 0.0001:
            lambdaFloquet = re
            minRe = re
            index = i
        if abs(im) < 0.0001:
           purely_real_evals.append(re) 
           
    #Raise error if no lambdaFloquet found
    if lambdaFloquet == None:
        raise Exception('No lambdaFloquet found: try with a larger number of eigenvalues')
    
    #Count the number of eigenvalues with larger real part so we know which set we're on
    isostable_number = len([i for i in purely_real_evals if i > lambdaFloquet])
            
    #Associated eigenfunction
    isostables = np.real(eigenVects[:,index]);
    # para que el maximo sea 1 en el phaseless set
    if isostables[0] > 0: isostables = isostables/min(isostables)
    else: isostables = isostables/max(isostables)
    
    #Reshape to space
    N = len(x); M = len(y) 
    isostables = isostables.reshape(M,N)
    
    #read P0 to normalize
    P0 = np.loadtxt(path + '/P0')
    dx = x[1] - x[0]; dy = y[1] - y[0]
    norm = performIntegration(P0*isostables**2, dx, dy)
    isostables = isostables/np.sqrt(norm)
    
    #Find the 0-level
    cs = plt.contour(x, y, isostables, levels=[0.0], zorder=2, colors='k', linewidths=3)
    plt.show()
    
    
    xLC = cs.allsegs[0][0][:,0]; yLC = cs.allsegs[0][0][:,1]
    LC = np.array([xLC, yLC])
    
    #Save data in file for chosen level set
    zero_level_path = path + '/isostable_' + str(isostable_number)
    if not os.path.exists(zero_level_path):
        os.makedirs(zero_level_path)

    np.savetxt(zero_level_path + '/isostables', isostables)
    np.savetxt(zero_level_path + '/zero_level', LC)
    
    return lambdaFloquet, LC

def find_Q(path, x, y, eigenVals, eigenVects):
    """
    Find the eigenvalue with smallest real part and non zero imaginary part
    Find the associated eigenfunction Q
    Use it to compute the stochastic asymptotic phase Psi
    """
    
    #Find lambda = mu + i*omega
    minRe = 10.1
    for i in range(len(eigenVals)):
        im = np.imag(eigenVals[i]); re = np.real(eigenVals[i])
        if im > 0.0 and abs(re) < minRe and abs(re) > 0.0:
                index = i; minRe = abs(re)
    mu = np.real(eigenVals[index]); omega = np.imag(eigenVals[index])
    
    #Associated eigenfunction
    Q = (eigenVects[:,index])
    #Reshape to space
    N = len(x); M = len(y) 
    Q = Q.reshape(M,N)
    
    #Get the phase
    isocrones = np.mod(np.arctan2(np.imag(Q), np.real(Q)), 2*np.pi)

    #Get data for first zero level
    LC = np.loadtxt(path + '/isostable_1/zero_level')
    xLC = LC[0,:]; yLC = LC[1,:]
    
    #Shift so max value is at index 0
    index = np.argmax(xLC); 
    xLC = np.roll(xLC, -index); yLC = np.roll(yLC, -index)
    
    #Interpolate 
    f_cos = interp.interp2d(x, y, np.cos(isocrones), kind='quintic')
    f_sin = interp.interp2d(x, y, np.sin(isocrones), kind='quintic')
    
    #Find phase of max
    phase = np.mod(np.arctan2(f_sin(xLC[0], yLC[0])[0], f_cos(xLC[0], yLC[0])[0]), 2*np.pi)
    
    #0-phase at max x (e.g. spiking)
    isocrones = np.mod(isocrones - phase, 2*np.pi)
    
    #Shift Q phase
    Q = Q*np.exp(-phase*1j); 
    QRe = np.real(Q); QIm = np.imag(Q)
    
    #read P0 to normalize
    P0 = np.loadtxt(path + '/P0')
    dx = x[1] - x[0]; dy = y[1] - y[0]
    norm = performIntegration(P0*(QRe**2 + QIm**2), dx, dy)
    QRe = QRe/np.sqrt(norm); QIm = QIm/np.sqrt(norm)

    #Create the grid
    X, Y = np.meshgrid(x, y, sparse=True)
    
    #Save data
    np.savetxt(path + '/isocrones', isocrones)
    np.savetxt(path + '/R[Q]', QRe)
    np.savetxt(path + '/Im[Q]', QIm)
    np.savetxt(path + '/uValues', np.sqrt(QRe**2 + QIm**2))
    np.savetxt(path + '/lnuValues', np.log(np.sqrt(QRe**2 + QIm**2)))

    return mu, omega, isocrones

def computeGradients(path, x, y, data, strName, phase = False):
     
    N = len(x); dx = x[1] - x[0]; M = len(y); dy = y[1] - y[0] 
    gradX = np.zeros((M,N)); gradY = np.zeros((M,N))

    for i in range(0, M):
        firstColumn = data[i,:];
        if phase: firstColumn = unWrapPhase(firstColumn)
        coeffVeector = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])/dx
        gradX[i,0] = sum(firstColumn[:7]*coeffVeector)
        gradX[i,1] = sum(firstColumn[1:8]*coeffVeector)
        gradX[i,2] = sum(firstColumn[2:9]*coeffVeector)
        gradX[i,-3] = sum(firstColumn[-9:-2]*np.flip(-coeffVeector))
        gradX[i,-2] = sum(firstColumn[-8:-1]*np.flip(-coeffVeector))
        gradX[i,-1] = sum(firstColumn[-7:]*np.flip(-coeffVeector))
        coeffVeector = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])/dx
        for j in range(3, len(firstColumn)-3):
            gradX[i,j] = sum(firstColumn[j-3:j+4]*coeffVeector)  
            
    for i in range(0, N):
        firstColumn = data[:,i];
        if phase: firstColumn = unWrapPhase(firstColumn)
        coeffVeector = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])/dy
        gradY[0,i] = sum(firstColumn[:7]*coeffVeector)
        gradY[1,i] = sum(firstColumn[1:8]*coeffVeector)
        gradY[2,i] = sum(firstColumn[2:9]*coeffVeector)
        gradY[-3,i] = sum(firstColumn[-9:-2]*np.flip(-coeffVeector))
        gradY[-2,i] = sum(firstColumn[-8:-1]*np.flip(-coeffVeector))
        gradY[-1,i] = sum(firstColumn[-7:]*np.flip(-coeffVeector))
        coeffVeector = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])/dy
        for j in range(3, len(firstColumn)-3):
            gradY[j,i] = sum(firstColumn[j-3:j+4]*coeffVeector)

    np.savetxt(path+'/dx'+ strName, gradX)
    np.savetxt(path+'/dy'+ strName, gradY)
    
    return

