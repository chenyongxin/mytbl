#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slice a 3D field.

@author: CHEN Yongxin
"""

import numpy as np
import matplotlib.pyplot as plt

def nearest(x, x0):
    """
    Find the nearest point to x0 from a 1D array x and return index.

    Parameters
    ----------
    x : 1D numpy array
        Axis.
    x0 : scalar
        The point.

    Returns
    -------
    index: scalar
        The index of the nearest point from x to the point x0.
    """
    return np.argmin(np.abs(x - x0))

def field_slice(field, x, x0, direction):
    """
    Slice and return the nearest 2D slice of a field.

    Parameters
    ----------
    field : 3D array
    x : 1D numpy array
    x0 : scalar
    direction: scalar
        {0, 1, 2}.
    
    Returns
    -------
    A 2D slice.
    """
    assert(field.shape[direction] == x.size)
    assert(direction in (0,1,2))
    index = nearest(x, x0)
    if direction == 0:
        return field[index, :, :]
    elif direction == 1:
        return field[:, index, :]
    else:
        return field[:, :, index]

def field_slice_contour(field, x0, direction, x, y, z, **kwargs):
    """
    Visualize field slice contour and return the slice. 

    Parameters
    ----------
    field : 3D array
    x : 1D numpy array
    x0 : scalar
    direction: scalar, one of {0, 1, 2}.
    x, y, z:
        1D array for axes.
    **kwargs: keyword arguments for plt.contourf
    
    Returns
    -------
    A 2D slice.
    """
    assert(field.shape == (x.size, y.size, z.size))
    xyz = [x, y, z]
    a = field_slice(field, xyz[direction], x0, direction)
    
    xyz.pop(direction)
    X, Y = np.meshgrid(xyz[0], xyz[1], indexing='ij')
    plt.contourf(X, Y, a, **kwargs)
    plt.axis('equal')
    return a