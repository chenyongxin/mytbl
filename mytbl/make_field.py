#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use HDF5 data to generate a single field -- Q-criteron or vorticity.
Convert velocity infomation in HDF/xxxx.h5 to Q-criteron field 
in VTK/q.xxxx.vtr or vorticity field in VTK/vort.xxxx.vtr.

@author: CHEN Yongxin
"""

import os
import sys
import numpy as np
from vtr import write_vtr
from myhdf5 import MyHDF5
from operation import get_velgrad

args = sys.argv.copy()
if len(args) < 3:
    print("Calling with `make_field which filename` (exclude extension .h5/.vtr)")
    sys.exit(1)
    
# filename/prefix
which = args[1]      # which field, "q" or "vort"
fname = args[2]      # e.g. 0001
assert(which in ("vort", "q"))

# Read HDF5 file
HDF_DIR = "HDF/"
h5file = HDF_DIR+fname+".h5"
assert(os.path.exists(h5file))
h5reader = MyHDF5(h5file)
u = h5reader.get('u')
v = h5reader.get('v')
w = h5reader.get('w')
h5reader.close()

# Read grid
GRID_DIR = "GRID/"
x = np.loadtxt(GRID_DIR+"x")
y = np.loadtxt(GRID_DIR+"y")
z = np.loadtxt(GRID_DIR+"z")

# Get cell center coordinate
xx = 0.5*(x[:-1] + x[1:]); nx = xx.size
yy = 0.5*(y[:-1] + y[1:]); ny = yy.size
zz = 0.5*(z[:-1] + z[1:]); nz = zz.size

VTK_DIR = "VTK/"
if which == "vort":
    
    # Make vorticity
    grad = get_velgrad(u, v, w, x, y, z)
    vx = grad[2,1,:,:,:] - grad[1,2,:,:,:]
    vy = grad[0,2,:,:,:] - grad[2,0,:,:,:]
    vz = grad[1,0,:,:,:] - grad[0,1,:,:,:]
    
    # Output file
    fields = {"vort_x": vx, "vort_y":vy, "vort_z":vz}
    write_vtr(VTK_DIR+"vort."+fname+".vtr", xx, yy, zz, fields) 
else:
    
    # Compute Q
    grad = get_velgrad(u, v, w, x, y, z)
    omega = (grad-np.transpose(grad, (1,0,2,3,4)))/2
    sij   = (grad+np.transpose(grad, (1,0,2,3,4)))/2
    a = np.einsum('nmijk,nmijk->ijk', omega, omega)        # Sum(Omega Omega)
    b = np.einsum('nmijk,nmijk->ijk', sij, sij)            # Sum(Sij Sij)
    q = (a-b)/2.
    
    # Output file
    fields = {"q": q}
    write_vtr(VTK_DIR+"q."+fname+".vtr", xx, yy, zz, fields)
    
