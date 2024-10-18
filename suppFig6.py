#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:45:57 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle

fontsize = 20

def plotPsiPanel(ax, filepath):
    
    sigmaList = np.linspace(0.01, 0.1, 10)

    omegas = np.zeros((3, len(sigmaList)))
    Ds = np.zeros((3, len(sigmaList)))
    
    for i, sigma in enumerate(sigmaList):
        #path = f'./Hopf/d={delta:0.3f}_D={sigma:0.2f}'
        path = f'{filepath}_D={sigma:0.2f}'
        
        omegaFull, DFull = np.loadtxt(f'{path}/longTermStatsFull')
        omegas[0,i] = omegaFull; Ds[0,i] = DFull
        
        omegaRed, DRed = np.loadtxt(f'{path}/empirical/longTermStatsRed')
        omegas[1,i] = omegaRed; Ds[1,i] = DRed
        
        with open(f'{path}/empirical/theorPsiRedCoefs.pck', 'rb') as file_handle:
            Veff = pickle.load(file_handle)
            Deff = pickle.load(file_handle)
        omegas[2,i] = Veff; Ds[2,i] = Deff

    ax.plot(sigmaList, omegas[0,:], '-', c ='aqua', lw = 6, label = r'$\omega^{\vartheta}_{eff}$ (full)')
    ax.plot(sigmaList, Ds[0,:], '-', c ='lime', lw = 6, alpha=0.5, label = r'$D^{\vartheta}_{eff}$ (full)')
    
    ax.plot(sigmaList, omegas[1,:], '-', c ='royalblue', label = r'$\omega^{\vartheta}_{eff}$ (reduction)')
    ax.plot(sigmaList, Ds[1,:], '-', c ='mediumseagreen', label = r'$D^{\vartheta}_{eff}$ (reduction)')
    
    ax.plot(sigmaList, omegas[2,:], 'o', c ='darkblue', label = r'$\omega^{\vartheta}_{eff}$ (theory)', markersize = 6)
    ax.plot(sigmaList, Ds[2,:], 'o', c ='darkgreen', label = r'$D^{\vartheta}_{eff}$ (theory)', markersize = 6)

    return

def plotMRTPanel(ax, filepath):
    
    sigmaList = np.linspace(0.01, 0.1, 10)

    omegas = np.zeros((3, len(sigmaList)))
    Ds = np.zeros((3, len(sigmaList)))
    
    for i, sigma in enumerate(sigmaList):
        #path = f'./Hopf/d={delta:0.3f}_D={sigma:0.2f}'
        path = f'{filepath}_D={sigma:0.2f}'
        
        omegaFull, DFull = np.loadtxt(f'{path}/longTermStatsMRT')
        omegas[0,i] = omegaFull; Ds[0,i] = DFull
        
        omegaRed, DRed = np.loadtxt(f'{path}/empiricalMRT/longTermStatsRed')
        omegas[1,i] = omegaRed; Ds[1,i] = DRed
        
        with open(f'{path}/empiricalMRT/theorPsiRedCoefs.pck', 'rb') as file_handle:
            Veff = pickle.load(file_handle)
            Deff = pickle.load(file_handle)
        omegas[2,i] = Veff; Ds[2,i] = Deff

    ax.plot(sigmaList, omegas[0,:], '-', c ='aqua', lw = 6, label = r'$\omega^{\theta}_{eff}$ (full)')
    ax.plot(sigmaList, Ds[0,:], '-', c ='lime', lw = 6, alpha=0.5, label = r'$D^{\theta}_{eff}$ (full)')
    
    ax.plot(sigmaList, omegas[1,:], '-', c ='royalblue', label = r'$\omega^{\theta}_{eff}$ (reduction)')
    ax.plot(sigmaList, Ds[1,:], '-', c ='mediumseagreen', label = r'$D^{\theta}_{eff}$ (reduction)')
    
    ax.plot(sigmaList, omegas[2,:], 'o', c ='darkblue', label = r'$\omega^{\theta}_{eff}$ (theory)', markersize = 6)
    ax.plot(sigmaList, Ds[2,:], 'o', c ='darkgreen', label = r'$D^{\theta}_{eff}$ (theory)', markersize = 6)

    return

def plot():
    
    fig = plt.figure(dpi=300, figsize = (14, 16))  # Increase figsize to accommodate two sets of plots
    
    gs = gridspec.GridSpec(4, 2)
    
    #Hopf above
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title(r'1a) Hopf above - $\psi$', fontsize=fontsize)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.tick_params(labelsize=18)
    path = './Hopf/d=1.000'
    plotPsiPanel(ax0, path)
    
    ax1 = plt.subplot(gs[0, 1])
    ax1.set_title(r'1b) Hopf above - $\theta$', fontsize=fontsize)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(labelsize=18)
    plotMRTPanel(ax1, path)
    
    
    #Hopf below
    ax2 = plt.subplot(gs[1, 0])
    ax2.set_title(r'2a) Hopf below - $\psi$', fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.tick_params(labelsize=18)
    path = './Hopf/d=-0.010'
    plotPsiPanel(ax2, path)
    
    ax3 = plt.subplot(gs[1, 1])
    ax3.set_title(r'2b) Hopf below - $\theta$', fontsize=fontsize)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.tick_params(labelsize=18)
    plotMRTPanel(ax3, path)
    
    #SNIC above
    ax4 = plt.subplot(gs[2, 0])
    ax4.set_title(r'3a) SNIC above - $\psi$', fontsize=fontsize)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.tick_params(labelsize=18)
    path = './SNIC/m=1.030'
    plotPsiPanel(ax4, path)
    
    ax5 = plt.subplot(gs[2, 1])
    ax5.set_title(r'3b) SNIC above - $\theta$', fontsize=fontsize)
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.tick_params(labelsize=18)
    plotMRTPanel(ax5, path)
    
    #SNIC below
    ax6 = plt.subplot(gs[3, 0])
    ax6.set_title(r'4a) SNIC below - $\psi$', fontsize=fontsize)
    ax6.set_xlabel(r'$\sigma$', fontsize = fontsize)
    ax6.tick_params(labelsize=18)
    path = './SNIC/m=0.999'
    plotPsiPanel(ax6, path)

    ax7 = plt.subplot(gs[3, 1])
    ax7.set_title(r'4b) SNIC below - $\theta$', fontsize=fontsize)
    ax7.set_xlabel(r'$\sigma$', fontsize = fontsize)
    ax7.tick_params(labelsize=18)
    plotMRTPanel(ax7, path)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Collect handles and labels for the legend from one of the axes
    handles, labels = ax0.get_legend_handles_labels()
    
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=20)
    
    # Adjust layout so the legend fits
    plt.subplots_adjust(top=0.85)
    
    #plt.tight_layout()
    plt.savefig('./SuppFig6.jpg')
    plt.show()
    
    return

def main():
    
    plot()
    
    return

if __name__ == '__main__':
    main()