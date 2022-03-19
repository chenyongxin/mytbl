#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output data to a file.

@author: CHEN Yongxin
"""

import numpy as np
import os, sys

CELL_WIDTH = 15

def outputs_array(name, array, items):
    """
    Light-weighted spreadsheet output with array input. The spreadsheet looks like
               
    items ->  |item1  item2  item3  ...   |
    data  ->  | xxx    xxx    xxx   ...   |
              | xxx    xxx    xxx   ...   |
              | ...                       |
              |                           |
    Parameters
    ----------
    name: str
        Name of file.
    array: array-like
        2D array.
    items: list
        List of items. Every item is a string.
    """
    if os.path.isfile(name):
        sys.exit("File {} already exists.".format(name))
        
    if len(items) != array.shape[1]:
        sys.exit("Number of items must match columns of data.")
    
    for item in items:
        if len(item) > CELL_WIDTH-1:
            sys.exit(f"Item length must less than {CELL_WIDTH-1} characters.")
    
    with open(name, 'w') as fh:
        dump_array(fh, array, items)

def outputs_dict(fname, data):
    """
    Output a dictionary data to a ASCII file.

    Parameters
    ----------
    fname : str
        File name.
    data : dict
        Dictionary to store data.

    Returns
    -------
    None.
    """
    assert(isinstance(data, dict))
    array, items = [], []
    for key, value in data.items():
        items.append(key)
        array.append(value)
    array = np.transpose(np.array(array))
    outputs_array(fname, array, items)

def dump_array(fh, array, items):
    """
    Dump array to a file with a specific file handle.
    
    Parameters
    ----------
    fh: object
        File handle.    
    array: array-like
        2D array.
    items: list
        List of items. Every item is a string.
    """
    # write header
    line = ""
    for item in items:
        line += " "*(CELL_WIDTH-len(item))+item 
    fh.write(line+'\n')

    # write array in scientific notation
    nrows, ncols = array.shape
    fmt = "{:" + "{}.{}e".format(CELL_WIDTH, CELL_WIDTH//2-1) + "}"   
    for i in range(nrows):
        for j in range(ncols):
            fh.write(str(fmt.format(array[i, j])))
        fh.write('\n')