#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert HDF5 file HDF/fname.h5 with grid GRID/{x,y,z} to VTK/fname.vtr. 

@author: CHEN Yongxin
"""

import os
import sys
import numpy as np
from vtr import write_vtr
from myhdf5 import MyHDF5

args = sys.argv.copy()
if len(args) < 2:
    print("Calling with `hdf2vtk filename` (exclude extension .h5/.vtr)")
    sys.exit(1)

# filename/prefix
fname = args[1]

# Read HDF5 file
HDF_DIR = "HDF/"
h5file = HDF_DIR+fname+".h5"
assert(os.path.exists(h5file))
h5reader = MyHDF5(h5file)

# Read grid
GRID_DIR = "GRID/"
x = np.loadtxt(GRID_DIR+"x"); nx = x.size
y = np.loadtxt(GRID_DIR+"y"); ny = y.size
z = np.loadtxt(GRID_DIR+"z"); nz = z.size

# Output file
VTK_DIR = "VTK/"

# Get fields
fields = {}
for fieldname in h5reader.dataset:
    fields.update({fieldname: h5reader.get(fieldname)})

h5reader.close()

# Output data 
write_vtr(VTK_DIR+fname+".vtr", x, y, z, fields)