#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:19:33 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle

import SDE_toolbox as SDE


def plotPanel(ax, m, D, title):
    
    path = f'./SNIC/m={m:0.3f}_D={D:0.2f}'
    with open(f'{path}/empirical/PsiRedCoefs.pck', 'rb') as file:
        aCoefs = pickle.load(file)
        DCoefs = pickle.load(file)

    psi = np.linspace(0, 2*np.pi, 100)
    aPsi = SDE.fourier_fit_array(psi, aCoefs)
    DPsi = SDE.fourier_fit_array(psi, DCoefs)
    
    ax.plot(psi, aPsi, label=r'$a_\psi(\psi)$')
    ax.plot(psi, DPsi, label=r'$D_\psi(\psi)$')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(r'$\psi$', fontsize = 15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.axhline(0, c='k')
    ax.set_xlim(0, 2*np.pi)
    
    return

def plotFigure5(D, m1, m2, m3):

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi = 300, sharey=True)
    ax1 = axs[0]; ax2 = axs[1]; ax3 = axs[2]
    
    # Plot in each subplot
    plotPanel(ax1, m1, D, title = rf'a) $m$ = {m1}')
    plotPanel(ax2, m2, D, title = rf'b) $m$ = {m2}')
    plotPanel(ax3, m3, D, title = rf'c) $m$ = {m3}')

    # Collect handles and labels for the legend from one of the axes
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)
    
    # Adjust layout so the legend fits
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)

    plt.savefig('./FIG5.jpg')
    plt.show()
    
    return

def main():
    
    D = 0.01
    
    m1 = 0.999; m2 = 1.013; m3 = 1.03
    plotFigure5(D, m1, m2, m3)

    return

if __name__ == '__main__':
    main()