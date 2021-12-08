#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data operation.

@author: CHEN Yongxin
"""

import numpy as np

def clip_xyz_mask(x, y, z, normal, origin):
    """
    Clip axes and return clipped x, y, z masks.
    
    Parameters
    ----------
    x, y, z: 1D array 
        Coordinates.
    normal: int {-3,-2,-1,1,2,3}
        Normal plane to discard clipped part.
    origin: float
        Original point.
    
    Returns
    -------
    xmask, ymask:
        x and y masks
    """
    assert(normal in (-3, -2, -1, 1, 2, 3))
    xmask = np.ones_like(x, dtype=bool) 
    ymask = np.ones_like(y, dtype=bool)
    zmask = np.ones_like(z, dtype=bool)
    
    if abs(normal) == 1:
        if np.sign(normal) > 0:
            xmask = x <= origin
        else:
            xmask = x >= origin
    elif abs(normal) == 2:
        if np.sign(normal) > 0:
            ymask = y <= origin
        else:
            ymask = y >= origin
    else:
        if np.sign(normal) > 0:
            zmask = z <= origin
        else:
            zmask = z >= origin
    return xmask, ymask, zmask

def field_xyz_clip(a, x, y, z, normal, origin):
    """
    Clip a field (3D array data).

    Parameters
    ----------
    a : 3D array
        Field.
    x,y,z : 1D array
        Axes.
    normal : scalar 
        One of {-3,-2,-1,1,2,3}.
    origin: float
        Original point of the clipping plane.

    Returns
    -------
    Grid: tuple
        A tuple of 3 axes components.
    data: new data
    """
    xmask, ymask, zmask = clip_xyz_mask(x, y, z, normal, origin)
    return (x[xmask], y[ymask], z[zmask]), \
        a.copy()[xmask, :, :][:, ymask, :][:, :, zmask]
    
def field_horizontal_phase_average(a, x, y, z, px=1, py=1):
    """
    Phase average for a field.

    return (xx, yy, zz), average
    Parameters
    ----------
    a : 3D array
        Field.
    x,y,z : 1D array
        Axes.
    px : int, optional
        Number of parts in x. The default is 1.
    py : int, optional
        Number of parts in y. The default is 1.

    Returns
    -------
    Grid: tuple
        A tuple of 3 axes components.
    data: 3D array
        Phase averaged data.
    """
    onx, ony, onz = x.size, y.size, z.size      # original grid size
    xx, yy, zz = x.copy(), y.copy(), z.copy()   # get original grid
    
    # Discard the last one if that is odd number
    nx, ny = onx, ony
    if onx%2 == 0:
        nx = onx -1 
        xx = xx[:nx]

    if ony%2 == 0:
        ny = ony -1
        yy = yy[:ny]
    
    npx, npy = (nx-1)//px, (ny-1)//py          # new number of cells in x and y
    xx,  yy  = x[:npx+1],  y[:npy+1]
    average = np.zeros((npx+1, npy+1, onz))    # averaged data
    
    for i in range(px):
        for j in range(py):
            average += a[npx*i:npx*(i+1)+1, npy*j:npy*(j+1)+1, :]
    average /= (px*py)
    
    return (xx, yy, zz), average
  

def get_velgrad(u, v, w, x, y, z):
    """
    Copmpute velocity gradient tensor.
    Data generated from vertex centered to be stored in cell center.
    Hence, the number of cell-cenetered data less 1 than vertex centered data 
    in each direction.

    Parameters
    ----------
    u,v,w: 3D array
        Fields
    x,y,z: 1D array
        Axes
        
    Returns
    -------
    velgrad: 5D array with dimensions [3, 3, nx-1, ny-1, nz-1]
        Velocity gradient.
    """
    def derivate(du, dx, j):
        dudx = np.copy(du)
        n = dx.size
        if j == 0:
            for i in range(n):
                dudx[i, :, :] /= dx[i]
        if j == 1:
            for i in range(n):
                dudx[:, i, :] /= dx[i]
        if j == 2:
            for i in range(n):
                dudx[:, :, i] /= dx[i]
        return dudx
    
    dx = []
    dx.append(x[1:]-x[:-1])
    dx.append(y[1:]-y[:-1])
    dx.append(z[1:]-z[:-1])
    
    du = []
    du.append((u[1:, :, :] - u[:-1, :, :])[:, :-1, :-1])
    du.append((u[:, 1:, :] - u[:, :-1, :])[:-1, :, :-1])
    du.append((u[:, :, 1:] - u[:, :, :-1])[:-1, :-1, :])
    
    du.append((v[1:, :, :] - v[:-1, :, :])[:, :-1, :-1])
    du.append((v[:, 1:, :] - v[:, :-1, :])[:-1, :, :-1])
    du.append((v[:, :, 1:] - v[:, :, :-1])[:-1, :-1, :])
    
    du.append((w[1:, :, :] - w[:-1, :, :])[:, :-1, :-1])
    du.append((w[:, 1:, :] - w[:, :-1, :])[:-1, :, :-1])
    du.append((w[:, :, 1:] - w[:, :, :-1])[:-1, :-1, :])
    
    velgrad = np.zeros((3,3) + du[0].shape)
    for i in range(3):
        for j in range(3):
            velgrad[i, j, :, :, :] = derivate(du[i*3+j], dx[j], j)
    
    return velgrad    