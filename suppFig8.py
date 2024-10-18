#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:03:57 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

sigma = 0.01

fontsize = 14

def HopfPanel(ax, delta, title, Ix = 0.05, Iy = 0.05):
    
    path = f'./Hopf/d={delta:0.3f}_D={sigma:0.2f}'
    
    with open(f'{path}/phaseResponseMRT.pck', 'rb') as file_handle:
        
        PsiBinsX = pickle.load(file_handle)
        sPRX = pickle.load(file_handle)
        stdErrX = pickle.load(file_handle)
        
        PsiBinsY = pickle.load(file_handle)
        sPRY = pickle.load(file_handle)
        stdErrY = pickle.load(file_handle)
        
    #Gradient averaging
    with open(f'{path}/empiricalMRT/empiricalMRTPRC.pck', 'rb') as file_handle:
        PsiBinsX2 = pickle.load(file_handle)
        sPRX2 = pickle.load(file_handle)
        PRCstdErrX = pickle.load(file_handle)
        
        PsiBinsY2 = pickle.load(file_handle)
        sPRY2 = pickle.load(file_handle)
        PRCstdErrY = pickle.load(file_handle)

    ax.axhline(0, c='k')
    ax.plot(PsiBinsX2 , Ix*sPRX2 , c ='aqua', lw = 6, label = r'$\epsilon_x\langle \partial_x \theta \rangle$')
    #ax.plot(PsiBinsX[1:], sPRX[1:], 'o', c ='royalblue', label = 'Numerics')
    ax.errorbar(PsiBinsX, sPRX, yerr=stdErrX, fmt='.', c ='royalblue', capsize=4, label='Numerics')
    
    ax.plot(PsiBinsY2, Iy*sPRY2, c ='lime', lw = 6, label = r'$\epsilon_y\langle \partial_y \theta \rangle$')
    #ax.plot(PsiBinsY[1:], sPRY[1:], 'o', c ='darkgreen', label = 'Numerics')
    ax.errorbar(PsiBinsY, sPRY, yerr=stdErrY, fmt='.', c ='darkgreen', capsize=3, label='Numerics')
    
    
    ax.set_xlim(0, 2*np.pi)

    ax.set_xlabel(r'$\theta$', fontsize = fontsize)
    ax.set_ylabel(r'$\Delta \theta$', fontsize = fontsize)
    ax.set_title(title, fontsize = fontsize)
    
    ax.tick_params(axis='both', labelsize=12)
    
    
    return

def SNICPanel(ax, m, title, Ix = 0.05, Iy = 0.05):
    
    path = f'./SNIC/m={m:0.3f}_D={sigma:0.2f}'
    
    #Empirical phase response
    with open(f'{path}/phaseResponseMRT.pck', 'rb') as file_handle:
        
        PsiBinsX = pickle.load(file_handle)
        sPRX = pickle.load(file_handle)
        stdErrX = pickle.load(file_handle)
        
        PsiBinsY = pickle.load(file_handle)
        sPRY = pickle.load(file_handle)
        stdErrY = pickle.load(file_handle)
        
    #Gradient averaging
    with open(f'{path}/empiricalMRT/empiricalMRTPRC.pck', 'rb') as file_handle:
        PsiBinsX2 = pickle.load(file_handle)
        sPRX2 = pickle.load(file_handle)
        PRCstdErrX = pickle.load(file_handle)
        
        PsiBinsY2 = pickle.load(file_handle)
        sPRY2 = pickle.load(file_handle)
        PRCstdErrY = pickle.load(file_handle)
        
    ax.axhline(0, c='k')
    ax.plot(PsiBinsX2 , Ix*sPRX2 , c ='aqua', lw = 6, label = r'$\epsilon_x\langle \partial_x \theta \rangle$')
    #ax.plot(PsiBinsX[1:], sPRX[1:], 'o', c ='royalblue', label = 'Numerics')
    ax.errorbar(PsiBinsX, sPRX, yerr=stdErrX, fmt='.', c ='royalblue', capsize=3, label='Numerics')
    
    ax.plot(PsiBinsY2, Iy*sPRY2, c ='lime', lw = 6, label = r'$\epsilon_y\langle \partial_y \theta \rangle$')
    #ax.plot(PsiBinsY[1:], sPRY[1:], 'o', c ='darkgreen', label = 'Numerics')
    ax.errorbar(PsiBinsY, sPRY, yerr=stdErrY, fmt='.', c ='darkgreen', capsize=3, label='Numerics')
    
    
    ax.set_xlim(0, 2*np.pi)

    ax.set_xlabel(r'$\theta$', fontsize = fontsize)
    ax.set_ylabel(r'$\Delta \theta$', fontsize = fontsize)
    ax.set_title(title, fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=12 )

    return

def main():
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi = 300)
    ax1 = axs[0,0]; ax2 = axs[0,1]; ax3 = axs[1,0]; ax4 = axs[1,1]
    
    # Plot in each subplot
    HopfPanel(ax1, delta=1, title = 'a) Hopf above bifurcation', Ix = 0.01, Iy = 0.01)
    HopfPanel(ax2, delta=-0.01, title = 'b) Hopf below bifurcation', Ix = 0.01, Iy = 0.01)
    SNICPanel(ax3, m=1.03, title = 'c) SNIC above bifurcation', Ix = 0.01, Iy = 0.01)
    SNICPanel(ax4, m=0.999, title = 'd) SNIC below bifurcation', Ix = 0.01, Iy = 0.01)
    plt.tight_layout()

    # Collect handles and labels for the legend from one of the axes
    handles, labels = ax1.get_legend_handles_labels()
    
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize='large')
    
    # Adjust layout so the legend fits
    plt.subplots_adjust(top=0.85)
    
    # Display the plot
    
    plt.savefig('./suppFig8.jpg')
    plt.show()
    
    return

if __name__ == '__main__':
    main()