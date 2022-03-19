#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input/output from and to pavaview vtr or pvtr files.

@author: CHEN Yongxin
"""

import vtk
import numpy as np
from struct import pack

def read_vtr_reader(reader, filename):
    """
    Read vtr or pvtr file with different reader.
    
    Parameters
    ----------
    pvtr: str
        Path to .pvtr file.
    
    Returns
    -------
    grid: list of x, y, z
        3 arrays.    
    point: dict
        Point data.    
    cell: dict
        Cell data.
    """
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    
    # Get grid
    x = np.array(data.GetXCoordinates())
    y = np.array(data.GetYCoordinates())
    z = np.array(data.GetZCoordinates())
    
    # Get data pointer
    point_data = data.GetPointData()
    cell_data = data.GetCellData()
        
    # Build output dicts
    point, cell = {}, {}
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        array = np.array(point_data.GetAbstractArray(i))
        if array.ndim > 1:    # e.g velocity: (nx*ny*nz, 3)
            array = array.transpose()
        else:
            array = np.reshape(array, (1, array.size))
        point.update({name: array})
    
    for i in range(cell_data.GetNumberOfArrays()):
        name = cell_data.GetArrayName(i)
        array = np.array(cell_data.GetAbstractArray(i))
        if array.ndim > 1:
            array = array.transpose()
        else:
            array = np.reshape(array, (1, array.size))
        cell.update({name: array})
        
    return (x,y,z), point, cell

def read_pvtr(pvtr):
    """
    Read .pvtr file
    
    Parameters
    ----------
    pvtr: str
        Path to .pvtr file.
    
    Returns
    -------
    grid: list of x, y, z
        3 arrays.    
    point: dict
        Point data.    
    cell: dict
        Cell data.
    """
    return read_vtr_reader(vtk.vtkXMLPRectilinearGridReader(), pvtr)

def read_vtr(vtr):
    """
    Read .vtr file
    
    Parameters
    ----------
    vtr: str
        Path to .vtr file.

    Returns
    -------
    grid: list of x, y, z
        3 arrays.    
    point: dict
        Point data.    
    cell: dict
        Cell data.
    """
    return read_vtr_reader(vtk.vtkXMLRectilinearGridReader(), vtr)

def write_vtr(name, x, y, z, fields):
    """
    Write rectilinear grid .vtr file in binary.

    Parameters
    ----------
    name: str
        File name.
    x, y, z: array-like, float, (N,)
        x, y, z axis 1D grid point.
    fields: dict
        Output fields dictionary object of point data.
        Key: field's name.
        Value: numpy array, 3D. e.g. Value = np.zeros((nx, ny, nz)).
    """
    def encode(string): 
        return str.encode(string)
    
    nx, ny, nz = x.size, y.size, z.size          # dimensions
    off = 0                                      # offset
    ise, jse, kse = [1, nx], [1, ny], [1, nz]    # start and ending indices

    with open(name, 'wb') as fh:
        fh.write(encode( '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'))
        fh.write(encode(f'<RectilinearGrid WholeExtent="{ise[0]} {ise[1]} {jse[0]} {jse[1]} {kse[0]} {kse[1]}">\n'))
        fh.write(encode(f'<Piece Extent="{ise[0]} {ise[1]} {jse[0]} {jse[1]} {kse[0]} {kse[1]}">\n'))
        fh.write(encode( '<Coordinates>\n'))
        fh.write(encode(f'<DataArray type="Float32" Name="x" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += nx*4 + 4
        fh.write(encode(f'<DataArray type="Float32" Name="y" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += ny*4 + 4
        fh.write(encode(f'<DataArray type="Float32" Name="z" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += nz*4 + 4
        fh.write(encode( '</Coordinates>\n'))
        
        # additional info for fields
        if len(fields) > 0:
            fh.write(encode('<PointData>\n'))
            for key, value in fields.items():
                fh.write(encode('<DataArray type="Float32" Name="{}" format="appended" offset="{}" NumberOfComponents="1"/>\n'.
                                format(key, off)))
                off += value.size*4 + 4
            fh.write(encode('</PointData>\n'))

        fh.write(encode('</Piece>\n'))
        fh.write(encode('</RectilinearGrid>\n'))
        fh.write(encode('<AppendedData encoding="raw">\n'))
        fh.write(encode('_'))
        fh.write(pack("i",    4*nx))
        fh.write(pack("f"*nx,   *x))
        fh.write(pack("i",    4*ny))
        fh.write(pack("f"*ny,   *y))
        fh.write(pack("i",    4*nz))
        fh.write(pack("f"*nz,   *z))

        # write fields if present
        if len(fields) > 0:
            for value in fields.values():
                fh.write(pack("i", 4*value.size))
                fh.write(pack("f"*value.size, *(value.flatten(order='F'))))
                
        fh.write(encode('\n'))
        fh.write(encode('</AppendedData>\n'))
        fh.write(encode('</VTKFile>'))
        