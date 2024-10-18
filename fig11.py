#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:30:12 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def HopfPanel(ax, delta, title):
    
    sigmaList = np.linspace(0.01, 0.1, 10)

    omegas = np.zeros((3, len(sigmaList)))
    Ds = np.zeros((3, len(sigmaList)))
    
    stdErrs = np.ones((4, len(sigmaList)))
    
    for i, sigma in enumerate(sigmaList):
        path = f'./Hopf/d={delta:0.3f}_D={sigma:0.2f}'
        
        omegaFull, DFull = np.loadtxt(f'{path}/longTermStatsMRT')
        omegas[0,i] = omegaFull; Ds[0,i] = DFull
        
        omegaRed, DRed = np.loadtxt(f'{path}/empiricalMRT/longTermStatsRed')
        omegas[1,i] = omegaRed; Ds[1,i] = DRed
        
        with open(f'{path}/empiricalMRT/theorPsiRedCoefs.pck', 'rb') as file_handle:
            Veff = pickle.load(file_handle)
            Deff = pickle.load(file_handle)
        omegas[2,i] = Veff; Ds[2,i] = Deff
        
        stdErrs[0:2,i] = np.loadtxt(f'{path}/longTermStatsMRTstdErr')
        stdErrs[2:,i] = np.loadtxt(f'{path}/empiricalMRT/longTermStatsRedstdErr')

    #ax.plot(sigmaList, omegas[0,:], '-', c ='aqua', lw = 6, label = r'$\omega^{\theta}_{eff}$ (full)')
    ax.errorbar(sigmaList, omegas[0,:], yerr=stdErrs[0,:], fmt='-', c ='aqua', lw = 3, label = r'$\omega^{\theta}_{eff}$ (full)')
    ax.errorbar(sigmaList, Ds[0,:], yerr=stdErrs[1,:], fmt='-', c ='lime', lw = 3, alpha=0.5, label = r'$D^{\theta}_{eff}$ (full)')
    
    # ax.plot(sigmaList, Ds[0,:], '-', c ='lime', lw = 6, alpha=0.5, label = r'$D^{\theta}_{eff}$ (full)')
    
    ax.errorbar(sigmaList, omegas[1,:], yerr=stdErrs[2,:], fmt='-', c ='royalblue', lw = 1, label = r'$\omega^{\theta}_{eff}$ (reduction)')
    ax.errorbar(sigmaList, Ds[1,:], yerr=stdErrs[3,:], fmt='-', c ='mediumseagreen', lw=1, label = r'$D^{\theta}_{eff}$ (reduction)')
    # ax.plot(sigmaList, omegas[1,:], '-', c ='royalblue', label = r'$\omega^{\theta}_{eff}$ (reduction)')
    # ax.plot(sigmaList, Ds[1,:], '-', c ='mediumseagreen', label = r'$D^{\theta}_{eff}$ (reduction)')
    
    ax.plot(sigmaList, omegas[2,:], 'o', c ='darkblue', label = r'$\omega^{\theta}_{eff}$ (theory)', markersize = 6)
    ax.plot(sigmaList, Ds[2,:], 'o', c ='darkgreen', label = r'$D^{\theta}_{eff}$ (theory)', markersize = 6)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'Noise intensity $D$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    return


def SNICPanel(ax, m, title):
    
    sigmaList = np.linspace(0.01, 0.1, 10)

    omegas = np.zeros((3, len(sigmaList)))
    Ds = np.zeros((3, len(sigmaList)))
    stdErrs = np.ones((4, len(sigmaList)))
    for i, sigma in enumerate(sigmaList):
        path = f'./SNIC/m={m:0.3f}_D={sigma:0.2f}'
        
        omegaFull, DFull = np.loadtxt(f'{path}/longTermStatsMRT')
        omegas[0,i] = omegaFull; Ds[0,i] = DFull
        
        omegaRed, DRed = np.loadtxt(f'{path}/empiricalMRT/longTermStatsRed')
        omegas[1,i] = omegaRed; Ds[1,i] = DRed
        
        with open(f'{path}/empiricalMRT/theorPsiRedCoefs.pck', 'rb') as file_handle:
            Veff = pickle.load(file_handle)
            Deff = pickle.load(file_handle)
        omegas[2,i] = Veff; Ds[2,i] = Deff
        
        stdErrs[0:2,i] = np.loadtxt(f'{path}/longTermStatsMRTstdErr')
        stdErrs[2:,i] = np.loadtxt(f'{path}/empiricalMRT/longTermStatsRedstdErr')
        
    ax.errorbar(sigmaList, omegas[0,:], yerr=stdErrs[0,:], fmt='-', c ='aqua', lw =3, label = r'$\omega^{\theta}_{eff}$ (full)')
    ax.errorbar(sigmaList, Ds[0,:], yerr=stdErrs[1,:], fmt='-', c ='lime', lw = 3, alpha=0.5, label = r'$D^{\theta}_{eff}$ (full)')

    # ax.plot(sigmaList, omegas[0,:], '-', c ='aqua', lw = 6, label = r'$\omega^{\theta}_{eff}$ (full)')
    # ax.plot(sigmaList, Ds[0,:], '-', c ='lime', lw = 6, alpha=0.5, label = r'$D^{\theta}_{eff}$ (full)')
    
    ax.errorbar(sigmaList, omegas[1,:], yerr=stdErrs[2,:], lw=1, fmt='-', c ='royalblue', label = r'$\omega^{\theta}_{eff}$ (reduction)')
    ax.errorbar(sigmaList, Ds[1,:], yerr=stdErrs[3,:], fmt='-', c ='mediumseagreen', lw=1, label = r'$D^{\theta}_{eff}$ (reduction)')
    # ax.plot(sigmaList, omegas[1,:], '-', c ='royalblue', label = r'$\omega^{\theta}_{eff}$ (reduction)')
    # ax.plot(sigmaList, Ds[1,:], '-', c ='mediumseagreen', label = r'$D^{\theta}_{eff}$ (reduction)')
    
    ax.plot(sigmaList, omegas[2,:], 'o', c ='darkblue', label = r'$\omega^{\theta}_{eff}$ (theory)', markersize = 6)
    ax.plot(sigmaList, Ds[2,:], 'o', c ='darkgreen', label = r'$D^{\theta}_{eff}$ (theory)', markersize = 6)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'Noise intensity $D$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    return

def main():
    
    fig, axs =  plt.subplots(2, 2, figsize=(10, 8), dpi = 300)
    ax1 = axs[0,0]; ax2 = axs[0,1]; ax3 = axs[1,0]; ax4 = axs[1,1]
    
    # Plot in each subplot
    HopfPanel(ax1, delta=1, title = 'a) Hopf above bifurcation')
    HopfPanel(ax2, delta=-0.01, title = 'b) Hopf below bifurcation')
    SNICPanel(ax3, m=1.03, title = 'c) SNIC above bifurcation')
    SNICPanel(ax4, m=0.999, title = 'd) SNIC below bifurcation')
    plt.tight_layout()

    # Collect handles and labels for the legend from one of the axes
    handles, labels = ax1.get_legend_handles_labels()
    
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14)
    
    # Adjust layout so the legend fits
    plt.subplots_adjust(top=0.85)
    
    # Display the plot
    plt.savefig('./FIG11.jpg')
    plt.show()
    
    
    
    return

if __name__ == '__main__':
    main()