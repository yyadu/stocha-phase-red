#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:17:07 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats
from .integrators import EulerSDE
from .utils import find_lambda_1, unWrapPhase

from .d3s.observables import monomials, FourierBasis
from .d3s.algorithms import gedmd

# Function to find the closest grid point index
def find_closest_index(arr, value):
    return np.abs(arr - value).argmin()

def buildGrid(coordinates, grid_resolution):
    """
    Build grid that spans the explored phase space for an arbitrary number of dimensions.
    
    Parameters:
    coordinates (list of np.ndarray): List of arrays, each array corresponds to one dimension.
    grid_resolution (int): Number of grid points along each dimension.
    
    Returns:
    np.ndarray: Array of occupied grid points, shape will be (n_dimensions, n_occupied_points).
    """
    # Stack all coordinate arrays along a new dimension (columns)
    trajectory = np.vstack(coordinates).T
    
    # Get the bounds of the trajectory for each dimension
    min_vals = trajectory.min(axis=0)
    max_vals = trajectory.max(axis=0)
    
    # Create the grid for each dimension
    grids = [np.linspace(min_vals[dim], max_vals[dim], grid_resolution) for dim in range(trajectory.shape[1])]
    
    # Initialize the grid occupancy array
    grid_shape = tuple([grid_resolution] * trajectory.shape[1])
    grid_occupancy = np.zeros(grid_shape, dtype=bool)
    
    # Mark the grid cells containing the trajectory
    for point in trajectory:
        indices = tuple(find_closest_index(grids[dim], point[dim]) for dim in range(trajectory.shape[1]))
        grid_occupancy[indices] = True
    
    # Get the indices of the occupied grid cells
    occupied_indices = np.argwhere(grid_occupancy)
    
    # Get occupied grid cells
    grid = np.zeros((trajectory.shape[1], len(occupied_indices)))
    for n, idx_tuple in enumerate(occupied_indices):
        grid[:, n] = [grids[dim][idx_tuple[dim]] for dim in range(trajectory.shape[1])]
    
    return grid

def PsiBuildergEDMD(path, x0, t, f, driftParams, g, noiseParams, 
                    grid_resolution, nMon, nEvals = 30, basis='monomials'):
    
    assert basis == 'monomials' or basis == 'fourier', 'Basis functions: fourier or monomials'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    #Library of observables
    if basis == 'monomials':
        F = monomials(nMon)
    elif basis == 'fourier':
        F = FourierBasis(nMon)
    
    
    #Generate long trajectory which we'll use to define a grid
    traj = EulerSDE(x0, t, f, driftParams, g, noiseParams)
    #Remove non stationary part
    nDims, nSteps = traj.shape
    traj = traj[:,int(0.1*nSteps):]
    
    print(f'Full phase space grid: {grid_resolution**2} points')
    grid = buildGrid(traj, grid_resolution)
    nDims, nPoints = grid.shape
    print(f'Final grid: {nPoints} points')
    
    #Build LDagger on grid
    
    #Drift
    Y = f(grid, driftParams)
    #Diffusion
    Z = g(grid, noiseParams).reshape((nDims, nDims, nPoints))
    
    #Main dish
    K, d, V = gedmd(grid, Y, Z, F, evs=nEvals, operator='K')
    inds = np.where(np.real(d) > 0)
    d2 = np.delete(d, inds)
    
    np.savetxt(f'{path}/eValsLD', d2)
    
    #Visualize eigenvalues
    plt.figure(dpi=300)
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(d2.real, d2.imag, 'r.', label = r'$\lambda$')
    lambda_1, index1 = find_lambda_1(d)
    plt.plot(lambda_1.real, lambda_1.imag, 'bo', label = r'$\lambda_1$')
    plt.title(r'Recovered $\mathcal{L}^\dagger$ eigenvalues')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    Q = V[:, index1].T
    Fc = F(traj[:,:500])

    Qt = Q @ Fc
    Psi = np.mod(np.arctan2(Qt.imag, Qt.real), 2*np.pi)
    PsiU = np.copy(Psi)
    PsiU = unWrapPhase(PsiU)
    if PsiU[-1] < PsiU[0]: 
        Q = np.conj(Q)
        
    savepath  = f'{path}/functions'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    with open(f'{savepath}/QgEDMD.pck', 'wb') as file_handle:
        pickle.dump(Q, file_handle)
        pickle.dump(F, file_handle)

    with open(f'{path}/grid.pck', 'wb') as file_handle:
        pickle.dump(grid, file_handle)

    return

def PsigEDMD(path):
    
    savepath  = f'{path}/functions'
    with open(f'{savepath}/QgEDMD.pck', 'rb') as file_handle:
        Q = pickle.load(file_handle)
        F = pickle.load(file_handle)
        
    def PsiFunc(X):
        Qt = Q @ F(X)
        return np.mod(np.arctan2(Qt.imag, Qt.real), 2*np.pi)

    return PsiFunc

def diffPsigEDMD(path, derivand):
    """
    Since Q(x) = sum a_i*f(x), dxQ(x) = sum a_i*dxf(x)
    Psi(x) = arctan(Im(Q), R(Q)) 
    => chain rule: dxPsi = ( R(Q)*dxIm(Q) - Im(Q)dxR(Q)) / (R(Q)^2 + Im(Q)^2)
    
    derivand gives the variable by which we derive
    e.g.  (x1, x2, x3)
    0 => dPsi/dx1
    """
    
    savepath  = f'{path}/functions'
    with open(f'{savepath}/QgEDMD.pck', 'rb') as file_handle:
        Q = pickle.load(file_handle)
        F = pickle.load(file_handle)
    dxF = F.diff
    
    def diffFunc(X):
        
        Qt = Q @ F(X)
        dxQt = Q @ dxF(X)[:,derivand,:]
        
        Qr = Qt.real; Qim = Qt.imag
        dxQr = dxQt.real; dxQim = dxQt.imag
        
        dxPsi = ( Qr*dxQim - Qim*dxQr ) / ( Qr**2 + Qim**2 )
        
        return dxPsi
    
    return diffFunc

def aiPRCgEDMD(path, x0, t, f, driftParams, g, noiseParams, derivand, nbins = 60):
    

    #Prepare phase functions
    Psif = PsigEDMD(path)
    dxPsif = diffPsigEDMD(path, derivand)
    
    X = EulerSDE(x0, t, f, driftParams, g, noiseParams)
    Psi = Psif(X)
    dxPsi = dxPsif(X)
    
    ###########################################################################
    ####   Circular mean   ####################################################
    ###########################################################################
    
    av_trig, bin_edges, binnumber = stats.binned_statistic(Psi, [np.cos(dxPsi), np.sin(dxPsi)], statistic='mean', bins=nbins)
    Cbar = av_trig[0,:]; Sbar = av_trig[1,:]
    bin_width = (bin_edges[1] - bin_edges[0])
    PsiBins = bin_edges[1:] - bin_width/2

    sPRC = np.zeros(np.shape(Cbar))
    for i in range(len(sPRC)):
        if Cbar[i] >= 0:
            sPRC[i] = np.arctan2(Sbar[i], Cbar[i])
        else:
            sPRC[i] = np.arctan2(Sbar[i], Cbar[i]) + np.pi

    plt.title('Recovered aiPRC - gradient averaging')
    plt.axhline(0, c='k')
    plt.plot(PsiBins, sPRC, 'b-.')
    plt.xlabel(r'$\psi$')
    plt.ylabel(r'$\Delta \psi$')
    plt.xlim(0, 2*np.pi)
    plt.show()
    
    return PsiBins, sPRC

###############################################################################



