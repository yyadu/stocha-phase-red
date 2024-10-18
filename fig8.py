#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:18:27 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import SDE_toolbox as SDE

import pickle

fontsize = 14

def samplePanel(fig, gs, ax1, ax2, ax3, D):

    path = f'./MorrisLecar3D/c1bD={D}'
    PsiFunc = SDE.PsigEDMD(path)
    N = 1
    
    with open(f'{path}/sampleTraj.pck', 'rb') as file_handle:
        t = pickle.load(file_handle)
        traj = pickle.load(file_handle)
    
    ax1.tick_params(right=False, top=False, bottom=False,
                   labelbottom=False)
    ax2.tick_params(right=False, top=False, bottom=False,
                   labelbottom=False)

    ax1.plot(t[::N], traj[0,::N], label = 'V(t)')
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize = fontsize)

    ax2.plot(t[::N], traj[1,::N], label = 'Y(t)')
    ax2.plot(t[::N], traj[2,::N], label = 'Z(t)')
    ax2.tick_params(labelsize=14)
    ax2.legend(fontsize = fontsize)
    

    ax3.plot(t[::N], PsiFunc(traj[:, ::N]), label=r'$\Psi(\mathbf{X}(t))$')
    ax3.set_xlabel('t (ms)', fontsize=14)
    ax3.tick_params(labelsize=14)
    ax3.legend(fontsize = fontsize)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax1.set_title('a) 3D Morris-Lecar model - noise induced oscillations', fontsize = fontsize)

    return

def phasePanel(ax, D):
    
    path = f'./MorrisLecar3D/c1bD={D}'
    
    cm = plt.cm.hsv
    
    with open(f'{path}/grid.pck', 'rb') as file_handle:
        grid = pickle.load(file_handle)

    PsiFunc = SDE.PsigEDMD(path)
    PsiGrid = PsiFunc(grid)

    ###########################################################################
    ########### 3D Plot ########################################################
    ###########################################################################

    # Add x, y gridlines 
    ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.3, 
        alpha = 0.2)

    sc = ax.scatter3D(grid[0,:], grid[1,:], grid[2,:], c = PsiGrid , cmap = cm, s = 1, alpha = 0.7)

    fontsize = 15
    ax.set_xlabel('V', fontweight ='bold', fontsize = fontsize)
    ax.set_ylabel('Y', fontweight ='bold', fontsize = fontsize)
    ax.set_zlabel('Z', fontweight ='bold', fontsize = fontsize)
    
    ax.set_yticks([0, 0.1, 0.2, 0.3]) 
    ax.tick_params(labelsize=14)
    
    ax.view_init(elev=20, azim=-110);
    
    cb = plt.colorbar(sc, ax = ax, shrink = 0.7, aspect = 20)
    cb.set_label(r'$\Psi(\mathbf{X})$', fontsize = fontsize)
    ax.set_title(r'b) Recovered $\Psi(\mathbf{X})$ in 3D-space', fontsize = fontsize) 

    return

def statsPanel(ax):
    
    sigmaList  = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    omegas = np.zeros((3, len(sigmaList)))
    Ds = np.zeros((3, len(sigmaList)))
    stdErrs = np.ones((4, len(sigmaList)))
    for i, D in enumerate(sigmaList):
        path = f'./MorrisLecar3D/c1bD={D}'
        
        omegaFull, DFull = np.loadtxt(f'{path}/longTermStatsFull')
        omegas[0,i] = omegaFull; Ds[0,i] = DFull
        
        omegaRed, DRed = np.loadtxt(f'{path}/empirical/longTermStatsRed')
        omegas[1,i] = omegaRed; Ds[1,i] = DRed
        
        #with open(f'{path}/empirical/theorPsiRedCoefs.pck', 'rb') as file_handle:
            #Veff = pickle.load(file_handle)
            #Deff = pickle.load(file_handle)
        #omegas[2,i] = Veff; Ds[2,i] = Deff
        
        #stdErrs[0:2,i] = np.loadtxt(f'{path}/longTermStatsFullstdErr')
        #stdErrs[2:,i] = np.loadtxt(f'{path}/empirical/longTermStatsRedstdErr')
        
    # ax.errorbar(sigmaList, omegas[0,:], yerr=stdErrs[0,:], fmt='-', c ='aqua', lw =3, label = r'$\omega^{\psi}_{eff}$ (full)')
    # ax.errorbar(sigmaList, Ds[0,:], yerr=stdErrs[1,:], fmt='-', c ='lime', lw = 3, alpha=0.5, label = r'$D^{\psi}_{eff}$ (full)')

    # ax.errorbar(sigmaList, omegas[1,:], yerr=stdErrs[2,:], lw=1, fmt='-', c ='royalblue', label = r'$\omega^{\psi}_{eff}$ (reduction)')
    # ax.errorbar(sigmaList, Ds[1,:], yerr=stdErrs[3,:], fmt='-', c ='mediumseagreen', lw=1, label = r'$D^{\psi}_{eff}$ (reduction)')
    
    ax.plot(sigmaList, omegas[0,:], ls='-', c ='aqua', lw =3, label = r'$\omega^{\psi}_{eff}$ (full)')
    ax.plot(sigmaList, Ds[0,:], ls='-', c ='lime', lw = 3, alpha=0.5, label = r'$D^{\psi}_{eff}$ (full)')

    ax.plot(sigmaList, omegas[1,:], lw=1, ls='-', c ='royalblue', label = r'$\omega^{\psi}_{eff}$ (reduction)')
    ax.plot(sigmaList, Ds[1,:], ls='-', c ='mediumseagreen', lw=1, label = r'$D^{\psi}_{eff}$ (reduction)')
    
    ax.plot(20, omegas[0][1], 'r*', ms = 12)
    ax.plot(20, Ds[0][1], 'r*', ms = 12)
    
    #ax.plot(DList, omegas[2,:], 'o', c ='darkblue', label = r'$\omega^{\psi}_{eff}$ (theory)', markersize = 6)
    #ax.plot(DList, Ds[2,:], 'o', c ='darkgreen', label = r'$D^{\psi}_{eff}$ (theory)', markersize = 6)
    
    ax.set_title('d) Long term statistics', fontsize = fontsize) 
    ax.legend(fontsize = 11, ncol=2)
    ax.tick_params(labelsize=14)
    ax.set_xlabel(r'Noise intensity $D$', fontsize = fontsize)
    
    return

def reductionPanel(ax, D):
    
    path = f'./MorrisLecar3D/c1bD={D}'
    
    savepath = f'{path}/empirical'
    with open(f'{savepath}/PsiRedCoefs.pck', 'rb') as file_handle:
        aCoefs = pickle.load(file_handle)
        DCoefs = pickle.load(file_handle)
        
    psi = np.linspace(0, 2*np.pi, 100)
    a = SDE.fourier_fit_array(psi, aCoefs)
    D = SDE.fourier_fit_array(psi, DCoefs)
    
    
    ax.axhline(0, c = 'k')
    ax.plot(psi, a, lw = 4, label = r'$a_\psi(\psi)$')
    ax.plot(psi, D, lw = 4, label = r'$D_\psi(\psi)$')
    ax.set_xlim(0, 2*np.pi)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper left')
    ax.set_title('c) Reduction coefficients', fontsize = fontsize)
    ax.set_xlabel(r'$\psi$', fontsize=fontsize)
    
    return



def PRCPanel(ax, D):
    IV = 1
    
    path = f'./MorrisLecar3D/c1bD={D}'
    
    with open(f'{path}/empiricalPRC.pck', 'rb') as file_handle:
        
        #Expected
        PsiBinsV = pickle.load(file_handle)
        dVPsi = pickle.load(file_handle)
        
    with open(f'{path}/PsiResp.pck', 'rb') as file_handle:
        #Recovered
        PsiBinsX = pickle.load(file_handle)
        sPRX = pickle.load(file_handle)
        stdErrX = pickle.load(file_handle)
        
        
    ax.axhline(0, c='k')
    ax.plot(PsiBinsV, IV*dVPsi, c ='aqua', lw = 6, label = r'$\epsilon_V\langle \partial_V \psi \rangle$')
    #ax.plot(PsiBinsX[1:], sPRX[1:], 'o', c ='royalblue', label = 'Numerics')
    ax.errorbar(PsiBinsX[1:], sPRX[1:], yerr=stdErrX[1:], fmt='.', c ='royalblue', capsize=3, label='Numerics')
    ax.legend()
    ax.set_title('e) aiPRC - Voltage perturbation', fontsize = fontsize)

    plt.xlim(0, 2*np.pi)

    ax.set_xlabel(r'$\psi$', fontsize = fontsize)
    ax.tick_params(labelsize=14)
    ax.set_ylabel(r'$\Delta \psi$', fontsize = fontsize)
    
    return

def main():
    
    # Create a figure with a 4x2 grid
    fig = plt.figure(figsize=(10, 10), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig)
    
    # Create a nested grid for ax1, ax2, ax3
    nested_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0:3, 0], hspace=0.05)  # Tight vertical space
    
    #################   PANEL 1   #################
    ax1 = fig.add_subplot(nested_gs[0])
    ax2 = fig.add_subplot(nested_gs[1], sharex=ax1)
    ax3 = fig.add_subplot(nested_gs[2], sharex=ax1)
    samplePanel(fig, gs, ax1, ax2, ax3, D=20.)
    
    #################   PANEL 2   #################
    ax4 = fig.add_subplot(gs[0:2, 1], projection='3d')
    phasePanel(ax4, D=20.)
    
    #################   PANEL 3   #################
    ax5 = fig.add_subplot(gs[2, 1])
    reductionPanel(ax5, D=20.)
    
    #################   PANEL 4   #################
    ax6 = fig.add_subplot(gs[3, 0])
    statsPanel(ax6)
    
    #################   PANEL 5   #################
    ax7 = fig.add_subplot(gs[3, 1])
    PRCPanel(ax7, D=20.)
    
    # Adjust overall spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Control general spacing between other subplots
    
    plt.savefig('./FIG8.jpg')
    plt.show()
    
    return

if __name__ == '__main__':
    main()