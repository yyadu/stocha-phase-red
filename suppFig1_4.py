#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:57:16 2024

@author: pierrehouzelstein
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import SDE_toolbox as SDE



def plot(times, xTraj1, yTraj1, xTraj2, yTraj2, x, y):


    plt.figure(dpi=300, figsize=(16, 16))  # Increase figsize to accommodate two sets of plots
    
    gs = gridspec.GridSpec(4, 2)  # 4 rows instead of 2 to stack vertically
    
    N = 20
    labelsize = 30
    fontsize = 20
    
    # First set of plots (upper)
    ax0 = plt.subplot(gs[0:2, 0])  # Phase space trajectories
    ax0.set_title(f'a) Phase space - D = {0.01}', fontsize=labelsize)
    ax1 = plt.subplot(gs[0, 1])    # x(t)
    ax1.set_title(f'b) Time series - D = {0.01}', fontsize=labelsize)
    #ax1.get_xaxis().set_visible(False)
    ax2 = plt.subplot(gs[1, 1])    # y(t)
    
    
    # Second set of plots (lower)
    ax0_lower = plt.subplot(gs[2:4, 0])  # Phase space trajectories (lower)
    ax0_lower.set_title('c) Phase space - D = 0.08', fontsize=labelsize)
    ax1_lower = plt.subplot(gs[2, 1])    # x(t) (lower)
    ax1_lower.set_title('d) Time series - D = 0.08', fontsize=labelsize)
    #ax1_lower.get_xaxis().set_visible(False)
    ax2_lower = plt.subplot(gs[3, 1])    # y(t) (lower)

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, N))
    
    nSteps = len(times)
    nSegs = int(nSteps / N)
    for j in range(10):
        for i in range(N):
            ax0.plot(xTraj1[j, i * nSegs:(i + 1) * nSegs], yTraj1[j, i * nSegs:(i + 1) * nSegs], color=colors[i])
    
    for i in range(N):
        ax1.plot(times[i * nSegs:(i + 1) * nSegs], xTraj1[0, i * nSegs:(i + 1) * nSegs], color=colors[i])
        ax2.plot(times[i * nSegs:(i + 1) * nSegs], yTraj1[0, i * nSegs:(i + 1) * nSegs], color=colors[i])
        
    for j in range(10):
        for i in range(N):
            ax0_lower.plot(xTraj2[j, i * nSegs:(i + 1) * nSegs], yTraj2[j, i * nSegs:(i + 1) * nSegs], color=colors[i])
    
    for i in range(N):
        ax1_lower.plot(times[i * nSegs:(i + 1) * nSegs], xTraj2[0, i * nSegs:(i + 1) * nSegs], color=colors[i])
        ax2_lower.plot(times[i * nSegs:(i + 1) * nSegs], yTraj2[0, i * nSegs:(i + 1) * nSegs], color=colors[i])
    
    # Set labels and limits for both sets of plots
    for ax in [ax0, ax0_lower]:
        ax.set_xlabel('x', fontsize=labelsize)
        ax.set_ylabel('y', fontsize=labelsize)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.tick_params(labelsize=fontsize)
    
    for ax, label in zip([ax1, ax1_lower], ['x(t)', 'x(t)']):
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('t', fontsize=labelsize)
        ax.set_ylabel(label, fontsize=labelsize)
        ax.tick_params(labelsize=fontsize)
    
    for ax, label in zip([ax2, ax2_lower], ['y(t)', 'y(t)']):
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('t', fontsize=labelsize)
        ax.set_ylabel(label, fontsize=labelsize)
        ax.tick_params(labelsize=fontsize)
    
    
    plt.tight_layout()

    
    return

def dataHopf(D, delta, t):
    
    #Parameters for the drift function
    f = SDE.Hopf2D
    driftParams = {'beta': 0.5, 'gamma': 4, 'delta': delta, 'kappa': 1}
    g = SDE.additive2D
    sxx = syy = np.sqrt(2*D)
    noiseParams = {'sxx': sxx, 'sxy': 0., 'syx': 0., 'syy': syy}
    
    nSteps = len(t)
    xTraj = np.zeros((10, nSteps))
    yTraj = np.zeros((10, nSteps))
    x0 = np.random.rand(2)
    for i in range(10):
        traj = SDE.EulerSDE(x0, t, f, driftParams, g, noiseParams)
        xTraj[i,:] = traj[0,:]
        yTraj[i,:] = traj[1,:]

    return xTraj, yTraj

def dataSNIC(D, m, t):
    
    #Parameters for the drift function
    f = SDE.SNIC2D
    driftParams = {'beta': 1., 'm': m}
    g = SDE.additive2D
    sxx = syy = np.sqrt(2*D)
    noiseParams = {'sxx': sxx, 'sxy': 0., 'syx': 0., 'syy': syy}
    
    nSteps = len(t)
    xTraj = np.zeros((10, nSteps))
    yTraj = np.zeros((10, nSteps))
    x0 = np.random.rand(2)
    for i in range(10):
        traj = SDE.EulerSDE(x0, t, f, driftParams, g, noiseParams)
        xTraj[i,:] = traj[0,:]
        yTraj[i,:] = traj[1,:]

    return xTraj, yTraj


def plotAll():
    
    #Traj
    dt = 1e-3; T = 30; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)

    x = y = np.linspace(-1.5, 1.5, 300)

    #Hopf above
    xTraj1, yTraj1 = dataHopf(0.01, 1, t)
    xTraj2, yTraj2 = dataHopf(0.08, 1, t)

    plot(t, xTraj1, yTraj1, xTraj2, yTraj2, x, y)
    plt.savefig('./SuppFig1.jpg')
    plt.show()
    
    #Hopf below
    xTraj1, yTraj1 = dataHopf(0.01, -0.01, t)
    xTraj2, yTraj2 = dataHopf(0.08, -0.01, t)

    plot(t, xTraj1, yTraj1, xTraj2, yTraj2, x, y)
    plt.savefig('./SuppFig2.jpg')
    plt.show()
    
    #SNIC
    dt = 1e-3; T = 150; nSteps = int(T/dt)
    t = np.linspace(0, T, nSteps)
    m1 = 0.999; m2 = 1.03
    
    #Above
    xTraj1, yTraj1 = dataSNIC(0.01, m2, t)
    xTraj2, yTraj2 = dataSNIC(0.08, m2, t)
    
    plot(t, xTraj1, yTraj1, xTraj2, yTraj2, x, y)
    plt.savefig('./SuppFig3.jpg')
    plt.show()
    
    #Below
    xTraj1, yTraj1 = dataSNIC(0.01, m1, t)
    xTraj2, yTraj2 = dataSNIC(0.08, m1, t)
    
    plot(t, xTraj1, yTraj1, xTraj2, yTraj2, x, y)
    plt.savefig('./SuppFig4.jpg')
    plt.show()
    
    return

def main():
    
    plotAll()
    
    
    return

if __name__ == '__main__':
    main()