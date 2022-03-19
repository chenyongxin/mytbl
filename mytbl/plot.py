#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot stuff.

@author: CHEN Yongxin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def rectangular_patches(xyc, dxy, **kwargs):
    """
    Generate 2D rectangular patches.
    
    Parameters
    ----------
    xyc: n-by-2 array 
        Center coordinate.
    dxy: n-by-2 array 
        Side length.
    **kwargs : optional key arguments
        Arguments for patch.
        
    Returns
    -------
    patches: list
        A list of patch object.
    """
    assert(xyc.shape == dxy.shape)
    patches = []
    for i in range(len(xyc)):
        patches.append(
            mpl.patches.Rectangle(xyc[i,:]-0.5*dxy[i,:], dxy[i,0], dxy[i,1], **kwargs))
    return patches

def plot_rectangular_frame(xyc, dxy, **kwargs):
    """
    Plot rectangular frames.

    Parameters
    ----------
    xyc: n-by-2 array 
        Center coordinate.
    dxy: n-by-2 array 
        Side length.
    **kwargs : optional key arguments
        Arguments for plot.

    Returns
    -------
    None.

    """
    assert(xyc.shape == dxy.shape)
    for i in range(len(xyc)):
        plt.plot((xyc[i,0]-dxy[i,0]/2, xyc[i,0]-dxy[i,0]/2), 
                 (xyc[i,1]-dxy[i,1]/2, xyc[i,1]+dxy[i,1]/2), **kwargs)
        
        plt.plot((xyc[i,0]+dxy[i,0]/2, xyc[i,0]+dxy[i,0]/2), 
                 (xyc[i,1]-dxy[i,1]/2, xyc[i,1]+dxy[i,1]/2), **kwargs)
        
        plt.plot((xyc[i,0]-dxy[i,0]/2, xyc[i,0]+dxy[i,0]/2), 
                 (xyc[i,1]-dxy[i,1]/2, xyc[i,1]-dxy[i,1]/2), **kwargs)
        
        plt.plot((xyc[i,0]-dxy[i,0]/2, xyc[i,0]+dxy[i,0]/2), 
                 (xyc[i,1]+dxy[i,1]/2, xyc[i,1]+dxy[i,1]/2), **kwargs)

if __name__ == "__main__":
    plt.figure()
    ax = plt.gca()
    xyc, dxy = [], []
    n = 2
    for i in range(n):
        xyc.append((i,i))
        dxy.append((1,1))
    
    xyc = np.array(xyc)
    dxy = np.array(dxy)
    
    # Plot patches
    patches = rectangular_patches(xyc, dxy, alpha=0.2)
    for patch in patches:
        ax.add_patch(patch)
    
    # Plot frames
    plot_rectangular_frame(xyc, dxy, alpha=0.2, color='k', linestyle='--')
    
    plt.xlim(-2,10)
    plt.ylim(-2,10)