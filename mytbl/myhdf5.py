#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handle HDF5 files. 

@author: CHEN Yongxin
"""

import numpy as np
import h5py as h5

class MyHDF5(object):
    """
    Customized HDF5 object. By default, all datasets are stored in the root 
    location as /dataset1, /dataset2, ... For the sake of simplification, it is
    recommended not to use groups. 
    """
    
    def __init__(self, name=None, mode='a'):
        """
        Initialize a HDF5 file object if file name is provided.
        """
        self.fh      = None                  # file handle
        self.dataset = []
        self.group   = []
        if name is not None:
            self.file(name=name, mode=mode)
        
    def file(self, name, mode='a'):
        """
        Open a file with default mode 'a' (append).
        
        Parameters
        ----------
        name: str
            Name of file.
        mode: char, optional
            Open file mode.
        """
        self.fh = h5.File(name=name, mode=mode)
        self.info()
        
    def info(self):
        """Update dataset and group info."""
        self.dataset, self.group = [], []
        def classify(name, obj):
            # Get group and dataset info.
            if isinstance(obj, h5.Dataset): self.dataset.append(name)
            if isinstance(obj, h5.Group):   self.group.append(name)
        self.fh.visititems(classify)
                
    def concatenate(self, dataset):
        """
        Concatenate data to an existing dataset by deleting old one and adding
        a new one with concatenating the new dataset. 
        
        Parameters
        ----------
        dataset: dict
            Dataset to be appended. 
            {'An Existing dataset name':array to be appended}.
        """
        assert(isinstance(dataset, dict))
        assert(len(dataset) == 1)
        datasetName = list(dataset.keys())[0]
        originalDataset = self.get(datasetName)
        newDataset = list(dataset.values())[0]
        del self.fh[datasetName]
        self.fh.create_dataset(datasetName, data=
                               np.concatenate((originalDataset, newDataset)))
        
    def get(self, dataName):
        """
        Get the value of dataset and return a numpy array.
        
        Parameters
        ----------
        dataName: str
            Name of dataset.
        
        Returns
        -------
        a: array
            Value of the dataset.
        """
        assert(dataName in self.dataset)
        return np.array(self.fh[dataName])
    
    def append(self, dataset):
        """
        Adding a new dataset.
        
        Parameter
        ---------
        dataset: dict
            Dataset to be appened. 
            {'New dataset name':array}
        """
        assert(isinstance(dataset, dict))
        assert(len(dataset) == 1)
        self.fh.create_dataset(name = list(dataset.keys())[0], 
                               data = list(dataset.values())[0])
        self.info()            # update dataset/group info
        
    def close(self):        
        """Close the HDF5 file."""        
        self.fh.close()