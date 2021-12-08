#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate aligned cube/cuboids.

@author: CHEN Yongxin
"""
import numpy as np

class Cube:
    """
    A cube or cuboid object.
    """
    def __init__(self, loc=None, length=None):
        """
        Initialise a cube object, with optional location and side length.        
        """
        self.__loc  = np.zeros(3)   # centre's location
        self.__side = np.zeros(3)   # side length
        if loc is not None:
            self.set_loc(loc)
        if length is not None:
            self.set_side(length)
        
    def translate(self, dx):
        """
        Translate the object with a specified increment.
        
        Parameters
        ----------
        dx: array
            3-entry numpy array
        """
        if(not isinstance(dx, np.ndarray)): 
            raise TypeError("Translation should be with numpy array.")
        if(dx.size != 3):
            raise ValueError("Translation should be with 3 entries.")
        self.__loc += dx
    
    def set_side(self, length):
        """
        Set length to cube's sides.
        
        Parameters
        ----------
        length: array
            3-entry numpy array for 3 sides.
        """
        if(not isinstance(length, np.ndarray)): 
            raise TypeError("Side length should be with numpy array.")
        if(length.size != 3):
            raise ValueError("Side length should be with 3 entries.")
        self.__side = length.copy()
    
    def set_loc(self, loc):
        """
        Set location to cube's location.
        
        Parameters
        ----------
        loc: array
            3-entry numpy array for xyz.
        """
        if(not isinstance(loc, np.ndarray)): 
            raise TypeError("Location should be with numpy array.")
        if(loc.size != 3):
            raise ValueError("Location should be with 3 entries.")
        self.__loc = loc.copy()
    
    def get_side(self):
        """
        Get sides of a cube by returning a numpy array.
        """
        return self.__side.copy()
    
    def get_loc(self):
        """
        Get central location of a cube by returning a numpy array.
        """
        return self.__loc.copy()
    
        
class Array:
    """
    An array of cubes/cuboids.
    """
    def __init__(self):
        """
        Initialise a cube/cuboid array.
        """
        self.__cubes = []
        
    def append(self, cube):
        """
        Append a cube to the list.
        """
        if(not isinstance(cube, Cube)): 
            raise TypeError("Should append a Cube object.")
        self.__cubes.append(cube)       
        
    def merge(self, bArray):
        """
        Merge another array of cube/cuboids (bArray) to the current array.
        """
        assert(isinstance(bArray, Array))
        for i in range(len(bArray.__cubes)):
            self.__cubes.append(bArray.__cubes[i])
        
    def pop(self, i):
        """
        Pop out ith cube in the array.
        """
        self.__cubes.pop(i)
        
    def write_txt(self, fname):
        """
        Write info of array of cube/cuboids to a file.

        Parameters
        ----------
        fname : string
            Write the array of cube/cuboids to a file named fname.

        Returns
        -------
        None.
        """
        with open(fname, 'w') as fh:
            fh.write(str(len(self.__cubes))+'\n')
            for cube in self.__cubes:
                for i in range(3):
                    fh.write("{0:10.5e} ".format(cube.get_loc()[i]))
                for i in range(3):
                    fh.write("{0:10.5e} ".format(cube.get_side()[i]))
                fh.write('\n')
                
    def write_stl(self, fname):
        """
        Write STL file.
        
        Parameters
        ----------
        fname: string
            File name.
        """
        def write_triangle(fh, v0, v1, v2):
            fh.write("facet normal 0 0 0\n")
            fh.write("outer loop\n")
            for v in [v0, v1, v2]:            
                fh.write("vertex {} {} {}\n".format(v[0], v[1], v[2]))
            fh.write("endloop\n")
            fh.write("endfacet\n")
        
        def write_plane(fh, v0, v1, v2, v3):
            write_triangle(fh, v0, v1, v2)
            write_triangle(fh, v2, v3, v0)
            
        def write_dual_plane(fh, xc, hdh, d):
            """
            Write two parallel planes with central coordinate xc, half side 
            length hdh and normal direction d. d=[0,1,2].
            """
            d  += 1
            d2  = d %3+1
            d3  = d2%3+1
            d  -= 1; o  = np.zeros(3); o[d]   = 1; o  *= hdh
            d2 -= 1; o2 = np.zeros(3); o2[d2] = 1; o2 *= hdh
            d3 -= 1; o3 = np.zeros(3); o3[d3] = 1; o3 *= hdh
            write_plane(fh, xc-o-o2-o3, xc-o-o2+o3, xc-o+o2+o3, xc-o+o2-o3)
            write_plane(fh, xc+o-o2-o3, xc+o-o2+o3, xc+o+o2+o3, xc+o+o2-o3)            
            
        def write_cuboid(fh, xc, hdh):
            for d in range(3):
                write_dual_plane(fh, xc, hdh, d)
        
        with open(fname, 'w') as fh:
            fh.write("solid cubes\n")
            for cube in self.__cubes:
                write_cuboid(fh, cube.get_loc(), cube.get_side()/2)
            fh.write("endsolid cubes")
                
if __name__ == "__main__":  
    
    print("Cubegen example")
    
    # Cube side lenth
    array = Array()
       
    h = 1.
    dx, dy, dz = h, h, h
    side = np.array([dx, dy, dz])
    sx, sy = 4*h, 2*h
    
    array.append(Cube(np.array([1, 1, dz/2.]), side))
    array.append(Cube(np.array([1, 3, dz/2.]), side))
    array.append(Cube(np.array([3, 0, dz/2.]), side))
    array.append(Cube(np.array([3, 2, dz/2.]), side))
    array.append(Cube(np.array([3, 4, dz/2.]), side))
    
    array.write_txt("cuboids.ccc")
    array.write_stl("cuboids.stl")