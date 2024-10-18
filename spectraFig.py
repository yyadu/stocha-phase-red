#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:48:59 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import SDE_toolbox as SDE

def main():
    
    D1 = 0.01
    D2 = 0.08
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300, sharey='row')
    
    titles = ['a) Hopf above bifurcation', 'b) Hopf below bifurcation',
             'c) SNIC above bifurcation', 'd) SNIC below bifurcation']
    
    delta1 = 1; delta2 = -0.01
    m1 = 1.03; m2 = 0.999
    paths1 = [f'./Hopf/d={delta1:0.3f}_D={D1:0.2f}', f'./Hopf/d={delta2:0.3f}_D={D1:0.2f}',
             f'./SNIC/m={m1:0.3f}_D={D1:0.2f}', f'./SNIC/m={m2:0.3f}_D={D1:0.2f}']
    
    paths2 = [f'./Hopf/d={delta1:0.3f}_D={D2:0.2f}', f'./Hopf/d={delta2:0.3f}_D={D2:0.2f}',
             f'./SNIC/m={m1:0.3f}_D={D2:0.2f}', f'./SNIC/m={m2:0.3f}_D={D2:0.2f}']
    
    for i, ax in enumerate(axs.ravel()):
        ax.axhline(0, c='k')
        ax.axvline(0, c='k')
        ax.set_title(titles[i], fontsize=15)
        
        #Low noise
        eVals = np.loadtxt(f'{paths1[i]}/eValsLD',  dtype=complex)
        eVals = sorted(eVals, key=lambda x: x.real)#[-32:]
        
        lambda1, _ = SDE.find_lambda_1(eVals)
        if lambda1.imag < 0: lambda1=lambda1.conj()
        
        ax.plot(np.real(eVals), np.imag(eVals), 'bo', label = 'Low noise')
        
        #High noise
        eVals = np.loadtxt(f'{paths2[i]}/eValsLD',  dtype=complex)
        eVals = sorted(eVals, key=lambda x: x.real)#[-32:]
        lambda1, _ = SDE.find_lambda_1(eVals)
        if lambda1.imag < 0: lambda1=lambda1.conj()
        
        ax.plot(np.real(eVals), np.imag(eVals), 'r*', label = 'High noise', markersize = 7)
        
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel(r"$\mathcal{Re}[{\lambda}]$", fontsize=16)
        ax.set_ylabel(r"$\mathcal{Im}[{\lambda}]$", fontsize=16)
    
    # Collect handles and labels for the legend from one of the axes
    handles, labels = axs[0,0].get_legend_handles_labels()
    
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper left', fontsize=14)
    
    axs[0,0].set_xlim(-2.1, 0.05)
    axs[0,1].set_xlim(-2.1, 0.05)
    
    axs[0,0].set_ylim(-10, 10)
    axs[0,1].set_ylim(-10, 10)
    
    axs[1,0].set_xlim(-1.3, 0.05)
    axs[1,1].set_xlim(-1.3, 0.05)
    axs[1,0].set_ylim(-2, 2)
    axs[1,1].set_ylim(-2, 2)
    
        
    
    plt.suptitle('Sample spectra of the $\mathcal{L}^\dagger$ operator', fontsize = 18)

    # Adjust layout
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    
    plt.savefig('./spectra.jpg')
    
    # Show the figure
    plt.show()
    
    
    
    
    
    
    
    return

if __name__ == '__main__':
    main()