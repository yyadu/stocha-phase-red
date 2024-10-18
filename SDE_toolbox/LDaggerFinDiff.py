#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 18:01:49 2024

@author: pierrehouzelstein

Finite differences methods to compute the building blocks of the Ldagger operator

Methods: buildLDagger1D, buildLDagger2D
"""

from scipy.sparse import kron, diags, eye
from numpy import zeros, ones, hstack, array, meshgrid
from numba import jit

###############################################################################
############  LDagger approximation methods for use in package  ##############
###############################################################################

def buildLDagger1D(X, f, g):
    """
    Use finite differences to compute the Kolmogorov backward operator
    associated to 1D Ito SDE dx = f(x)dt + g(x)dW
    """
    dx = X[1] - X[0]

    #Vector space
    fX = f(X)

    #Diffusion matrix
    gx = g(X)
    sx = (gx**2)/2
    
    fdX = fXdX1D(fX, len(X), dx)
    sdXX = sdXX1D(sx, len(X), dx)
    
    LDagger = fdX + sdXX

    return LDagger

def buildLDagger2D(x, y, f, g):
    """
    Use finite differences to compute the Kolmogorov backward operator in 2D
    associated to Ito SDE dx = f(x)dt + g(x)dW
    SUPPORTED: isotropic additive/multiplicative noise
    NOT SUPPORTED: anisotropic noise
    """
    
    #Grid specs
    xLen = len(x); yLen = len(y)
    dx = x[1] - x[0]; dy = y[1] - y[0]
    
    #Create the grid
    X, Y = meshgrid(x, y)
    #Vector space
    xField, yField = f([X, Y])
    xField, yField = xField.reshape(xLen*yLen), yField.reshape(xLen*yLen)
    
    #Diffusion matrix
    gxx, gyx, gxy, gyy = g([X, Y])
    gxx, gyy = gxx.reshape(xLen*yLen), gyy.reshape(xLen*yLen)
    sxx = 0.5*gxx**2; syy = 0.5*gyy**2
    
    #Gradient matrices
    fXdX = fXdX2D(xField/dx, xLen, yLen, True); 
    sXXdXX = sXXdXX2D(sxx/(dx**2), xLen, yLen, True)
    
    
    fYdY = fYdY2D(yField/dy, xLen, yLen, True); 
    sYYdYY = sYYdYY2D(syy/(dy**2), xLen, yLen, True)

    #Representation of Ldagger in 2D space
    lDagger = fXdX + fYdY + sXXdXX + sYYdYY
    
    return lDagger

###############################################################################
############  1D finite differences for gradient approximations  ##############
###############################################################################

@jit(nopython=True, cache=True) 
def fXdX1D(vector, N, h):
    """
    Finite differences method to compute f(x)*dx in one dim
    """
    
    vectorCentral = zeros(N); 
    vectorCentral[-4] = 1/(280*h); 
    vectorCentral[-3] = -4/(105*h); 
    vectorCentral[-2] = 1/(5*h); 
    vectorCentral[-1] = -4/(5*h); 
    vectorCentral[4] = -1/(280*h);
    vectorCentral[3] = 4/(105*h); 
    vectorCentral[2] = -1/(5*h); 
    vectorCentral[1] = 4/(5*h)
    matrixSolving = zeros((N, N))
    
    for i in range(0, N):
        matrixSolving[i,:] = vectorCentral[:]
        vectorCentral = hstack((array([vectorCentral[-1]]), vectorCentral[:-1]))

    for i in range(0, N):
        matrixSolving[i,:] = matrixSolving[i,:]*vector[i]

    return matrixSolving


@jit(nopython=True, cache=True) 
def sdXX1D(sigmaVector, N, h):
    """
    Finite differences method to compute sigma(x)*dxx in one dim
    """
    vectorCentral = zeros(N); 
    vectorCentral[-4] = -1/(560*h*h); 
    vectorCentral[-3] = 8/(315*h*h); 
    vectorCentral[-2] = -1/(5*h*h); 
    vectorCentral[-1] = 8/(5*h*h); 
    vectorCentral[4] = -1/(560*h*h); 
    vectorCentral[3] = 8/(315*h*h); 
    vectorCentral[2] = -1/(5*h*h); 
    vectorCentral[1] = 8/(5*h*h); 
    vectorCentral[0] = -205/(72*h*h)
    
    matrixSolving = zeros((N, N))
    for i in range(0, N):
        matrixSolving[i,:] = vectorCentral[:]
        vectorCentral = hstack((array([vectorCentral[-1]]), vectorCentral[:-1]))

    for i in range(0, N):
        matrixSolving[i,:] = matrixSolving[i,:]*sigmaVector[i]
    
    return matrixSolving

###############################################################################
############  2D finite differences for gradient approximations  ##############
##  https://en.wikipedia.org/wiki/Finite_difference_coefficient#cite_note-2  ##
###############################################################################

def fXdX2D(xVector, N, M, boundary = False):
    """
    Approximation of the fx*∂/∂x operator in 2D
    """
    
    # Las derivadas centrales
    center4 = diags((1/280.)*ones(N-4),-4) - diags((1/280.)*ones(N-4),4) - diags((4/105.)*ones(N-3),-3) + diags((4/105.)*ones(N-3),3) + diags((1/5.)*ones(N-2),-2) - diags((1/5.)*ones(N-2),2) - diags((4/5.)*ones(N-1),-1) + diags((4/5.)*ones(N-1),1)
    vectorFake = zeros(N); vectorFake[4:N-4] = 1
    center4 = center4.multiply(vectorFake.reshape(N,1))

    center3 = -diags((1/60.)*ones(N-3),-3) + diags((1/60.)*ones(N-3),3) + diags((3/20.)*ones(N-2),-2) - diags((3/20.)*ones(N-2),2) - diags((3/4.)*ones(N-1),-1) + diags((3/4.)*ones(N-1),1)
    vectorFake = zeros(N); vectorFake[3] = 1; vectorFake[-4] = 1
    center3 = center3.multiply(vectorFake.reshape(N,1))
    
    center2 = diags((1/12.)*ones(N-2),-2) - diags((1/12.)*ones(N-2),2) - diags((2/3.)*ones(N-1),-1) + diags((2/3.)*ones(N-1),1)
    vectorFake = zeros(N); vectorFake[2] = 1; vectorFake[-3] = 1 
    center2 = center2.multiply(vectorFake.reshape(N,1))
    
    center1 = -diags((1/2)*ones(N-1),-1) + diags((1/2)*ones(N-1),1)
    vectorFake = zeros(N); vectorFake[1] = 1; vectorFake[-2] = 1 
    center1 = center1.multiply(vectorFake.reshape(N,1))
    
    # Bloque de arriba y de abajo (forward/backward)
    
    upTerm = -diags((49/20.)*ones(N),0) + diags(6*ones(N-1),1) - diags((15./2)*ones(N-2),2) + diags((20/3.)*ones(N-3),3) - diags((15/4.)*ones(N-4),4) + diags((6/5.)*ones(N-5),5) - diags((1/6.)*ones(N-6),6)
    vectorFake = zeros(N); vectorFake[0] = 1
    upTerm = upTerm.multiply(vectorFake.reshape(N,1))
    downTerm = diags((49/20.)*ones(N),0) - diags(6*ones(N-1),-1) + diags((15./2)*ones(N-2),-2) - diags((20/3.)*ones(N-3),-3) + diags((15/4.)*ones(N-4),-4) - diags((6/5.)*ones(N-5),-5) + diags((1/6.)*ones(N-6),-6)
    vectorFake = zeros(N); vectorFake[-1] = 1
    downTerm = downTerm.multiply(vectorFake.reshape(N,1))   
   
    if boundary: 
        xTerm = center4 + center3 + center2 + center1
    else:
        xTerm = center4 + center3 + center2 + center1 + upTerm + downTerm
    
    xTerm = kron(eye(M), xTerm)    
    xTerm = xTerm.multiply(xVector.reshape(N*M,1))
    
    return xTerm

def fYdY2D(yVector, N, M, boundary = False):
    """
    Approximation of the fy*∂/∂y operator in 2D
    """
    
    # Las derivadas centrales
    center4 = diags((1/280.)*ones(N*M-4*N),-4*N) - diags((1/280.)*ones(N*M-4*N),4*N) -diags((4/105.)*ones(N*M-3*N),-3*N) + diags((4/105.)*ones(N*M-3*N),3*N) + diags((1/5.)*ones(N*M-2*N),-2*N) - diags((1/5.)*ones(N*M-2*N),2*N) - diags((4/5.)*ones(N*M-N),-N) + diags((4/5.)*ones(N*M-N),N)
    vectorFake = zeros(N*M); vectorFake[4*N:N*(M-4)] = 1
    center4 = center4.multiply(vectorFake.reshape(N*M,1))    
    
    center3 = -diags((1/60.)*ones(N*M-3*N),-3*N) + diags((1/60.)*ones(N*M-3*N),3*N) + diags((3/20.)*ones(N*M-2*N),-2*N) - diags((3/20.)*ones(N*M-2*N),2*N) - diags((3/4.)*ones(N*M-N),-N) + diags((3/4.)*ones(N*M-N),N)
    vectorFake = zeros(N*M); vectorFake[3*N:4*N] = 1; vectorFake[N*(M-4):N*(M-3)] = 1 
    center3 = center3.multiply(vectorFake.reshape(N*M,1))
    
    center2 = diags((1/12.)*ones(N*M-2*N),-2*N) - diags((1/12.)*ones(N*M-2*N),2*N) - diags((2/3.)*ones(N*M-N),-N) + diags((2/3.)*ones(N*M-N),N)
    vectorFake = zeros(N*M); vectorFake[2*N:3*N] = 1; vectorFake[N*(M-3):N*(M-2)] = 1 
    center2 = center2.multiply(vectorFake.reshape(N*M,1))
    
    center1 = -diags((1/2.)*ones(N*M-N),-N) + diags((1/2.)*ones(N*M-N),N)
    vectorFake = zeros(N*M); vectorFake[N:2*N] = 1; vectorFake[N*(M-2):N*(M-1)] = 1 
    center1 = center1.multiply(vectorFake.reshape(N*M,1))
    
    # Bloque de arriba y de abajo (forward/backward)
    upTerm = -diags((49/20.)*ones(N*M),0) + diags(6*ones(N*M-N),N) - diags((15/2)*ones(N*M-2*N),2*N) + diags((20/3.)*ones(N*M-3*N),3*N) - diags((15/4.)*ones(N*M-4*N),4*N) + diags((6/5.)*ones(N*M-5*N),5*N) - diags((1/6.)*ones(N*M-6*N),6*N)
    vectorFake = zeros(N*M); vectorFake[0:N] = 1
    upTerm = upTerm.multiply(vectorFake.reshape(N*M,1))
    downTerm = diags((49/20.)*ones(N*M),0) - diags(6*ones(N*M-N),-N) + diags((15/2)*ones(N*M-2*N),-2*N) - diags((20/3.)*ones(N*M-3*N),-3*N) + diags((15/4.)*ones(N*M-4*N),-4*N) - diags((6/5.)*ones(N*M-5*N),-5*N) + diags((1/6.)*ones(N*M-6*N),-6*N)
    vectorFake = zeros(N*M); vectorFake[N*M-N:] = 1
    downTerm = downTerm.multiply(vectorFake.reshape(N*M,1))   
    
    if boundary: 
        yTerm = center4 + center3 + center2 + center1
    else:
        yTerm = center4 + center3 + center2 + center1 + upTerm + downTerm
        
    yTerm = yTerm.multiply(yVector.reshape(N*M,1))

    return yTerm

def sXXdXX2D(xVector, N, M, boundary = False):
    """
    Approximation of the sigmaXX*∂^2/∂x^2 operator in 2D
    """
    
    # Las derivadas centrales
    center4 = -diags((1/560.)*ones(N-4),-4) - diags((1/560.)*ones(N-4),4) + diags((8/315.)*ones(N-3),-3) + diags((8/315.)*ones(N-3),3) - diags((1/5.)*ones(N-2),-2) - diags((1/5.)*ones(N-2),2) + diags((8/5.)*ones(N-1),-1) + diags((8/5.)*ones(N-1),1) - diags((205/72.)*ones(N),0)
    vectorFake = zeros(N); vectorFake[4:N-4] = 1
    center4 = center4.multiply(vectorFake.reshape(N,1))
    
    center3 = diags((1/90.)*ones(N-3),-3) + diags((1/90.)*ones(N-3),3) - diags((3/20.)*ones(N-2),-2) - diags((3/20.)*ones(N-2),2) + diags((3/2.)*ones(N-1),-1) + diags((3/2.)*ones(N-1),1) - diags((49/18.)*ones(N),0)
    vectorFake = zeros(N); vectorFake[3] = 1; vectorFake[-4] = 1
    center3 = center3.multiply(vectorFake.reshape(N,1))
    
    center2 = -diags((1/12.)*ones(N-2),-2) - diags((1/12.)*ones(N-2),2) + diags((4/3.)*ones(N-1),-1) + diags((4/3.)*ones(N-1),1) - diags((5/2.)*ones(N),0)
    vectorFake = zeros(N); vectorFake[2] = 1; vectorFake[-3] = 1 
    center2 = center2.multiply(vectorFake.reshape(N,1))

    center1 = diags(ones(N-1),-1) + diags(ones(N-1),1) - diags(2*ones(N),0)
    vectorFake = zeros(N); vectorFake[1] = 1; vectorFake[-2] = 1 
    center1 = center1.multiply(vectorFake.reshape(N,1))
    
    center0 = -diags(2*ones(N),0) + diags(2*ones(N-1),-1) + diags(2*ones(N-1),1)
    vectorFake = zeros(N); vectorFake[0] = 1; vectorFake[-1] = 1;
    center0 = center0.multiply(vectorFake.reshape(N,1))
    
    
    # Bloque de arriba y de abajo (forward/backward)
    upTerm = diags((469/90.)*ones(N),0) - diags((223/10.)*ones(N-1),1) + diags((879/20.)*ones(N-2),2) - diags((949/18.)*ones(N-3),3) + diags((41)*ones(N-4),4) - diags((201/10.)*ones(N-5),5) + diags((1019/180)*ones(N-6),6) - diags((7/10)*ones(N-7),7)
    vectorFake = zeros(N); vectorFake[0] = 1
    upTerm = upTerm.multiply(vectorFake.reshape(N,1))
    downTerm = diags((469/90.)*ones(N),0) - diags((223/10.)*ones(N-1),-1) + diags((879/20.)*ones(N-2),-2) - diags((949/18.)*ones(N-3),-3) + diags((41)*ones(N-4),-4) - diags((201/10.)*ones(N-5),-5) + diags((1019/180)*ones(N-6),-6) - diags((7/10)*ones(N-7),-7)
    vectorFake = zeros(N); vectorFake[-1] = 1
    downTerm = downTerm.multiply(vectorFake.reshape(N,1))   

    if boundary: 
        xTerm = center4 + center3 + center2 + center1 + center0
    else:
        xTerm = center4 + center3 + center2 + center1 + upTerm + downTerm

    xTerm = kron(eye(M), xTerm)    
    xTerm = xTerm.multiply(xVector.reshape(N*M,1))    
    
    return xTerm

def sYYdYY2D(yVector, N, M, boundary = False):
    """
    Approximation of the sigmaYY*∂2/∂y^2 operator in 2D
    """
    
    center4 = -diags((1/560.)*ones(N*M-4*N),-4*N) - diags((1/560.)*ones(N*M-4*N),4*N) + diags((8/315.)*ones(N*M-3*N),-3*N) + diags((8/315.)*ones(N*M-3*N),3*N) - diags((1/5.)*ones(N*M-2*N),-2*N) - diags((1/5.)*ones(N*M-2*N),2*N) + diags((8/5.)*ones(N*M-N),-N) + diags((8/5.)*ones(N*M-N),N) - diags((205/72.)*ones(N*M),0)
    vectorFake = zeros(N*M); vectorFake[4*N:N*(M-4)] = 1
    center4 = center4.multiply(vectorFake.reshape(N*M,1))

    center3 = diags((1/90.)*ones(N*M-3*N),-3*N) + diags((1/90.)*ones(N*M-3*N),3*N) - diags((3/20.)*ones(N*M-2*N),-2*N) - diags((3/20.)*ones(N*M-2*N),2*N) + diags((3/2.)*ones(N*M-N),-N) + diags((3/2.)*ones(N*M-N),N) - diags((49/18.)*ones(N*M),0)
    vectorFake = zeros(N*M); vectorFake[3*N:4*N] = 1; vectorFake[N*(M-4):N*(M-3)] = 1
    center3 = center3.multiply(vectorFake.reshape(N*M,1))

    center2 = -diags((1/12.)*ones(N*M-2*N),-2*N) - diags((1/12.)*ones(N*M-2*N),2*N) + diags((4/3.)*ones(N*M-N),-N) + diags((4/3.)*ones(N*M-N),N) - diags((5/2.)*ones(N*M),0)
    vectorFake = zeros(N*M); vectorFake[2*N:3*N] = 1; vectorFake[N*(M-3):N*(M-2)] = 1 
    center2 = center2.multiply(vectorFake.reshape(N*M,1))
    
    center1 = diags(ones(N*M-N),-N) + diags(ones(N*M-N),N) - diags(2*ones(N*M),0)
    vectorFake = zeros(N*M); vectorFake[N:2*N] = 1; vectorFake[N*(M-2):N*(M-1)] = 1 
    center1 = center1.multiply(vectorFake.reshape(N*M,1))    
    
    center0 = -diags(2*ones(N*M),0) + diags(2*ones(N*M-N),-N) + diags(2*ones(N*M-N),N)
    vectorFake = zeros(N*M); vectorFake[0:N] = 1; vectorFake[N*(M-1):] = 1;
    center0 = center0.multiply(vectorFake.reshape(N*M,1))
    
    
    upTerm = diags((469/90)*ones(N*M),0) - diags((223/10)*ones(N*M-N),N) + diags((879/20)*ones(N*M-2*N),2*N) - diags((949/18)*ones(N*M-3*N),3*N) + diags((41)*ones(N*M-4*N),4*N) - diags((201/10)*ones(N*M-5*N),5*N) + diags((1019/180)*ones(N*M-6*N),6*N) - diags((7/10)*ones(N*M-7*N),7*N)
    vectorFake = zeros(N*M); vectorFake[0:N] = 1
    upTerm = upTerm.multiply(vectorFake.reshape(N*M,1))
    downTerm = diags((469/90)*ones(N*M),0) - diags((223/10)*ones(N*M-N),-N) + diags((879/20)*ones(N*M-2*N),-2*N) - diags((949/18)*ones(N*M-3*N),-3*N) + diags((41)*ones(N*M-4*N),-4*N) - diags((201/10)*ones(N*M-5*N),-5*N) + diags((1019/180)*ones(N*M-6*N),-6*N) - diags((7/10)*ones(N*M-7*N),-7*N)
    vectorFake = zeros(N*M); vectorFake[N*M-N:] = 1
    downTerm = downTerm.multiply(vectorFake.reshape(N*M,1))   

    if boundary: 
        yTerm = center4 + center3 + center2 + center1 + center0
    else:
        yTerm = center4 + center3 + center2 + center1 + upTerm + downTerm
        
    yTerm = yTerm.multiply(yVector.reshape(N*M,1))

    return yTerm



