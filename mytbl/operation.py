#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data operation.

@author: CHEN Yongxin
"""

import numpy as np

def nearest(x, x0):
    """ Find the nearest index of list x to x0. """
    return np.argmin(np.abs(x - x0))

def periodic_expand(a):
    """
    Expand a 3D field with one more grid in each direction with 
    a periodic condition.
    """
    nx, ny, nz = a.shape
    b = np.zeros((nx+1, ny+1, nz+1))
    b[:-1, :-1, :-1] = a
    b[-1, :, :] = b[0, :, :]
    b[:, -1, :] = b[:, 0, :]
    b[:, :, -1] = b[:, :, 0]
    return b

def periodic_expand_horizontal(a):
    """
    Expand a 3D field with one more grid in only horizontal directions. 
    In the vertical direction, zero gradient boundary condition is used at top.
 
    Parameters
    ----------
    a : 3D field
    
    Returns
    -------
    A 3D field with one more grid.
    """
    nx, ny, nz = a.shape
    b = np.zeros((nx+1, ny+1, nz+1))
    b[:-1, :-1, :-1] = a
    b[-1, :, :] = b[0, :,  :]
    b[:, -1, :] = b[:, 0,  :]
    b[:, :, -1] = b[:, :, -2]
    return b

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
  
def field_horizontal_tile(a, x, y, z, px=1, py=1):
    """
    Tile a field a in horizontal direction. The field a can be from phased
    averaged result. It is recommended to use cell centered result. 
    
    Parameters
    ----------
    a : 3D array
        Field.
    x,y,z : 1D array
        Axes. It is assumed that x, y, z are well translated, i.e. from 0 for 
        vertex centered data and Delta/2 for cell centered data.
    px : int, optional
        Number of parts in x. The default is 1.
    py : int, optional
        Number of parts in y. The default is 1.

    Returns
    -------
    Grid: tuple
        A tuple of 3 axes components.
    data: 3D array
        Tiled data.

    """
    # Make grid
    nx, ny, nz = x.size, y.size, z.size
    lx = x[-1]+x[0]
    ly = y[-1]+y[0]
    xx = np.zeros(nx*px)
    yy = np.zeros(ny*py)
    for i in range(px):
        xx[i*nx:(i+1)*nx] = lx*i + x
    for i in range(py):
        yy[i*ny:(i+1)*ny] = ly*i + y
    zz = z.copy()
    
    # Make field
    f = np.zeros((nx*px, ny*py, nz))
    for i in range(px):
        for j in range(py):
            f[i*nx:(i+1)*nx, j*ny:(j+1)*ny] = a
    return (xx, yy, zz), f
    

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

def cumulative_integrate(f, dx):
    """
    Compute cumulative stress along height: F(z) = \int_z^H f dx. 
    Integrate f in the ascend order.
    
    Parameters
    ----------
    f: array
        Function.
    dx: array
        Grid spacing.
        
    Returns
    -------
    r: array
        Cumulative stress.
    """
    assert(f.size == dx.size)
    r = np.zeros(f.size)
    r[-1] = f[-1]*dx[-1]
    for i in range(1, f.size):
        r[-1-i] = r[-i] + f[-1-i]*dx[-1-i]
    return r

def mask_domain_cuboids_streamwise(cens, sides, x, y, z, direction=True):
    """
    Mask domain with 1 (or -1) in the first grid point outside the cube array, 
    otherwise 0. Use this with care as one cannot use a cuboid that totally 
    deattaches from the domain.

    Parameters
    ----------
    cens : n-by-3 array
        Coordinate of a cube array.
    sides : n-by-3 array
        Side lengths of a cube array.
    x, y, z : 1D array
        Grid.
    direction : bool, optional
        If true, multiply normal vector in the streamwise direction, which 
        pointing to the fluid phase.
    Returns
    -------
    a : 3D array
        Mask array.
    """
    assert(cens.shape == sides.shape)
    a = np.zeros((x.size, y.size, z.size))
    
    # Strategy: only change the vicinity of a cube
    for cen, side in zip(cens, sides):
        imin = nearest(x, cen[0]-side[0]/2)
        imax = nearest(x, cen[0]+side[0]/2)
        jmin = nearest(y, cen[1]-side[1]/2)
        jmax = nearest(y, cen[1]+side[1]/2)
        kmin = nearest(z, cen[2]-side[2]/2)
        kmax = nearest(z, cen[2]+side[2]/2)
        
        one_neg = 1 if imin > 1        else 0
        one_pos = 1 if imax < x.size-1 else 0
        
        neg = cen[0]-side[0]/2
        pos = cen[0]+side[0]/2
        
        if x[0] <= neg and neg <= x[-1]:
            # There is no big difference.
            #a[imin-one_neg, jmin:jmax, kmin:kmax] = -1 if direction else 1  
            a[imin, jmin:jmax, kmin:kmax] = -1 if direction else 1
        if x[0] <= pos and pos <= x[-1]: 
            #a[imax+one_pos, jmin:jmax, kmin:kmax] =  1
            a[imax, jmin:jmax, kmin:kmax] =  1 
    return a

def mask_horizontal_domain_aligned_cuboid_array_wake(cens, sides, x, y):
    """
    Mask a horizontal domain (x-y plane) given an array of aligned cubes.
    Return 1/True for the cube wake while 0/False for other points.

    Parameters
    ----------
    cens : n-by-2 array
        Coordinate of a cube array.
    sides : n-by-2 array
        Side lengths of a cube array.
    x, y : 1D array
        Grid.

    Returns
    -------
    a : 2D array
        Mask array.
    """
    assert(cens.shape == sides.shape)
    a = np.zeros((x.size, y.size), dtype=bool)
    
    # Add strips
    for cen, side in zip(cens, sides):
        jmin = nearest(y, cen[1]-side[1]/2)
        jmax = nearest(y, cen[1]+side[1]/2)
        a[:, jmin:jmax] = 1
    
    # Exclude the cubes
    for cen, side in zip(cens, sides):
        imin = nearest(x, cen[0]-side[0]/2)
        imax = nearest(x, cen[0]+side[0]/2)
        jmin = nearest(y, cen[1]-side[1]/2)
        jmax = nearest(y, cen[1]+side[1]/2)
        
        a[imin:imax+1, jmin:jmax+1] = 0 
    return a


def get_cube_pressure(cens, sides, x, y, z, p):
    """
    Get pressure force for a cube array in the streamwise direction.
    f_D_i = -1/A_T \Sigma p_j n_i dy_j. 
    Note here we should use cell centered pressure. 

    Parameters
    ----------
    cens : n-by-3 array
         Coordinate of a cube array.
    sides : n-by-3 array
        Side lengths of a cube array.
    x, y, z : 1D array
        Grid, vertex centered.
    p : 3D array
        Pressure field in cell center.
    Returns
    -------
    a: 1D array
        Pressure force of cube array along height.
    """
    assert(p.shape == (x.size-1, y.size-1, z.size-1))
    a = np.zeros(z.size-1)
    
    # Total area
    At = (x[-1]-x[0])*(y[-1]-y[0])
    
    # Cell center coordinate and make a mask (integral kernel)
    xc = 0.5*(x[1:]+x[:-1])
    yc = 0.5*(y[1:]+y[:-1])
    zc = 0.5*(z[1:]+z[:-1])
    dy = y[1:] - y[:-1]
    kernel = mask_domain_cuboids_streamwise(cens, sides, xc, yc, zc, direction=True)
    for j in range(kernel.shape[1]):
        kernel[:,j,:] *= dy[j]           # line integration 
    
    # Integrate along height
    for k in range(z.size-1):
        a[k] = np.sum(p[:,:,k]*kernel[:,:,k])
    a /= -At
    return a 
    

def get_nusgs(u, v, w, x, y, z, C=0.1):
    """
    Compute eddy viscosity using velocity and grid information and Vreman model.
    x, y, z are vertex centered coordinate, while eddy viscosity is stored in 
    the cell center.
    
    Parameters
    ----------
    u,v,w: 3D array
        Fields
    x,y,z: 1D array
        Axes
    C: scalar, optional
        Vreman model coefficient. Default value is 0.1.
        
    Returns
    -------
    nu_sgs: 3D array
        Eddy viscosity.
    """
    # Grid info
    nx, ny, nz = x.size, y.size, z.size
    
    # Get grid spacing
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    dz = z[1:]-z[:-1]
    dx2 = np.zeros((nx-1, ny-1, nz-1))
    dy2 = np.zeros((nx-1, ny-1, nz-1))
    dz2 = np.zeros((nx-1, ny-1, nz-1))
    for i in range(nx-1):
        dx2[i, :, :] = dx[i]**2
    for j in range(ny-1):
        dy2[:, j, :] = dy[j]**2
    for k in range(nz-1):
        dz2[:, :, k] = dz[k]**2
    
    # Get velocity gradient tensor
    grad = get_velgrad(u, v, w, x, y, z)
    
    a = np.transpose(grad, (1,0,2,3,4))
    b = np.zeros_like(a)
    for j in range(3):
        for i in range(3):
            b[i, j, :, :, :] += dx2 * a[0, i, :, :, :] * a[0, j, :, :, :] + \
                                dy2 * a[1, i, :, :, :] * a[1, j, :, :, :] + \
                                dz2 * a[2, i, :, :, :] * a[2, j, :, :, :]
    
    bb = b[0, 0, :, :, :] * b[1, 1, :, :, :] - b[0, 1, :, :, :]*b[0, 1, :, :, :] + \
         b[0, 0, :, :, :] * b[2, 2, :, :, :] - b[0, 2, :, :, :]*b[0, 2, :, :, :] + \
         b[1, 1, :, :, :] * b[2, 2, :, :, :] - b[1, 2, :, :, :]*b[1, 2, :, :, :]
    
    aa = np.einsum("ijklm, ijklm -> klm", a, a)
    ed = C * np.sqrt(np.divide(bb, aa, out=np.zeros_like(aa), where=aa!=0))
    return ed
