#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:16:20 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

import SDE_toolbox as SDE

alpha = 0.5; fontsize = 20; theta = np.linspace(0, 2*np.pi, 20)
fontsize2 = 15

def phasePanel(fig, ax, x, y, phaseFunc, cbLabel = r'$\psi$'):
    
    
    X, Y  = np.meshgrid(x, y)
    isocrones = phaseFunc((X, Y))
    
    im = ax.pcolormesh(x, y, isocrones, cmap = 'hsv', alpha=alpha)

    Cycler2 = plt.cycler("color", plt.cm.hsv(np.linspace(0, 1, 20)))
    ax.set_prop_cycle(Cycler2)
    
    for i in range(20):
        isoX, isoY = SDE.giveMeIsocrone(isocrones, theta[i], x, y)
        ax.plot(isoX, isoY, '.', markersize = 1)
        
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(cbLabel, fontsize=fontsize)
    
    ax.set(aspect='equal')
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('y', fontsize=fontsize)
    ax.set_xticks(ticks=[-1., 0, 1], labels=[-1., 0, 1], fontsize=fontsize)
    ax.set_yticks(ticks=[-1., 0, 1], labels=[-1., 0, 1], fontsize=fontsize)
    
    return

def P0Panel(fig, ax, x, y, path):
    
    with open(f'{path}/functions/P0Func.pck', 'rb') as file:
        P0Func = pickle.load(file)
    
    X, Y  = np.meshgrid(x, y)
    P0 = P0Func((Y, X))

    im = ax.pcolormesh(x, y, P0)
    
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(r'$P_0$', fontsize = fontsize)

    
    ax.set(aspect='equal')
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('y', fontsize=fontsize)
    ax.set_xticks(ticks=[-1., 0, 1], labels=[-1., 0, 1], fontsize=fontsize)
    ax.set_yticks(ticks=[-1., 0, 1], labels=[-1., 0, 1], fontsize=fontsize)
    
    return

def PsiPanel(ax, path):
    
    with open(f'{path}/empirical/PsiRedCoefs.pck', 'rb') as file:
        aCoefs = pickle.load(file)
        DCoefs = pickle.load(file)
    
    psi = np.linspace(0, 2*np.pi, 100)
    a = SDE.fourier_fit_array(psi, aCoefs)
    D = SDE.fourier_fit_array(psi, DCoefs)
    
    ax.get_xaxis().set_visible(False)
    ax.axhline(0, c = 'k')
    ax.plot(psi, a, lw = 4, label = r'$a_\psi(\psi)$')
    ax.plot(psi, D, lw = 4, label = r'$D_\psi(\psi)$')
    ax.set_xlim(0, 2*np.pi)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper right')
    
    return 

def MRTPanel(ax, path):
    
    with open(f'{path}/empirical/MRTRedCoefs.pck', 'rb') as file:
        aCoefs = pickle.load(file)
        DCoefs = pickle.load(file)
    theta = np.linspace(0, 2*np.pi, 100)
    a = SDE.fourier_fit_array(theta, aCoefs)
    D = SDE.fourier_fit_array(theta, DCoefs)

    ax.axhline(0, c = 'k')
    ax.plot(theta, a, lw = 4, label = r'$a_\theta(\theta)$')
    ax.plot(theta, D, lw = 4, label = r'$D_\theta(\theta)$')
    ax.set_xlim(0, 2*np.pi)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper right')
    ax.set_xlabel(r'$\vartheta$', fontsize=fontsize)
    
    return

def Fig_1(path1, path2):
    
    #Fontsize of all ticks and labels
    fontsize = 20
    
    #Grid space (phases/thetas)
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)

    # Create a figure with a 4x2 grid
    fig = plt.figure(figsize=(13, 20), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig)
    
    #################   PANEL 1: Psi(x, y)   #################

    PsiFunc  = SDE.PsiFunc2D(path1)
    ax1 = fig.add_subplot(gs[0, 0])
    phasePanel(fig, ax1, x, y, PsiFunc)
    ax1.set_title(r'1.a) Asymptotic stochastic phase $\Psi(\mathbf{x})$', fontsize = fontsize)

    #################   PANEL 2 : Theta(x,y)  #################
    MRTFunc = SDE.MRTFunc2D(path1)
    ax2 = fig.add_subplot(gs[0, 1])
    phasePanel(fig, ax2, x, y, MRTFunc, cbLabel=r'$\theta$')
    ax2.set_title(r'1.b) MRT phase $\Theta(\mathbf{x})$', fontsize = fontsize)
    
    #################   PANEL 3 : P0(x,y)   #################
    
    ax3 = fig.add_subplot(gs[1, 0])
    P0Panel(fig, ax3, x, y, path1)
    ax3.set_title(r'1.c) Stationary density $P_0(\mathbf{x})$', fontsize = fontsize)
    
    #################   PANEL 4 : a and D   #################

    # Split ax4 into two subplots
    gs_ax4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1])
    ax4_1 = fig.add_subplot(gs_ax4[0])
    ax4_2 = fig.add_subplot(gs_ax4[1])
    
    ######  asymptotic  #####
    
    PsiPanel(ax4_1, path1)
    ax4_1.set_title('1.d) Asymptotic phase reduction', fontsize=fontsize)
    
    ######  MRT  #####
    
    MRTPanel(ax4_2, path1)
    ax4_2.set_title('1.e) MRT phase reduction', fontsize=fontsize)

    #################   PANEL 5: Psi(x, y)   #################
    PsiFunc  = SDE.PsiFunc2D(path2)
    ax5 = fig.add_subplot(gs[2, 0])
    phasePanel(fig, ax5, x, y, PsiFunc)
    ax5.set_title(r'2.a) Asymptotic stochastic phase $\Psi(\mathbf{x})$', fontsize = fontsize)
    
    #################   PANEL 6 : Theta(x,y)  #################
    MRTFunc = SDE.MRTFunc2D(path1)
    ax6 = fig.add_subplot(gs[2, 1])
    phasePanel(fig, ax6, x, y, MRTFunc, cbLabel=r'$\theta$')
    ax6.set_title(r'2.b) MRT phase $\Theta(\mathbf{x})$', fontsize = fontsize)
    
    #################   PANEL 7 : P0(x,y)  #################
    ax7 = fig.add_subplot(gs[3, 0])
    P0Panel(fig, ax7, x, y, path2)
    ax7.set_title(r'2.c) Stationary density $P_0(\mathbf{x})$', fontsize = fontsize)
    
    #################   PANEL 8 : a and D   #################

    # Split ax8 into two subplots
    gs_ax8 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3, 1])
    ax8_1 = fig.add_subplot(gs_ax8[0])
    ax8_2 = fig.add_subplot(gs_ax8[1])
    
    ######  asymptotic  #####
    
    PsiPanel(ax8_1, path2)
    ax8_1.set_title('2.d) Asymptotic phase reduction', fontsize=fontsize)
    
    ######  MRT  #####

    MRTPanel(ax8_2, path2)
    ax8_2.set_title('2.e) MRT phase reduction', fontsize=fontsize)

    plt.tight_layout()  # Adjust layout to fit global titles
    

    return

def main():
    
    D1= 0.01; D2 = 0.08
    
    ###########################################################################
    delta = 1
    path1 = f'./Hopf/d={delta:0.3f}_D={D1:0.2f}'
    path2 = f'./Hopf/d={delta:0.3f}_D={D2:0.2f}'

    Fig_1(path1, path2)
    plt.savefig('./Fig1.jpg')
    plt.show()
    
    ###########################################################################
    delta = -0.01
    path1 = f'./Hopf/d={delta:0.3f}_D={D1:0.2f}'
    path2 = f'./Hopf/d={delta:0.3f}_D={D2:0.2f}'
    
    Fig_1(path1, path2)
    plt.savefig('./Fig2.jpg')
    plt.show()
    
    ###########################################################################
    
    m = 1.03
    path1 = f'./SNIC/m={m:0.3f}_D={D1:0.2f}'
    path2 = f'./SNIC/m={m:0.3f}_D={D2:0.2f}'
    
    Fig_1(path1, path2)
    plt.savefig('./Fig3.jpg')
    plt.show()
    
    ###########################################################################
    
    m = 0.999
    path1 = f'./SNIC/m={m:0.3f}_D={D1:0.2f}'
    path2 = f'./SNIC/m={m:0.3f}_D={D2:0.2f}'
    
    Fig_1(path1, path2)
    plt.savefig('./Fig4.jpg')
    plt.show()
    
    return

if __name__ == '__main__':
    main()