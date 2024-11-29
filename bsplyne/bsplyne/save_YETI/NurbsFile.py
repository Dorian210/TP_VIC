# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:52:47 2020

@author: mguerder
"""

import os

import numpy as np


class NurbsFile:
    """A class for writing .NB files."""

    def __init__(self):
        self.objects = []
        self.str_data = str()

    def concatenate_data(self):
        """Concatenate all data to one string."""
        self.write_dim()
        self.write_nb_cps()
        self.write_nb_patches()
        self.write_nb_elems()
        self.write_nb_elems_patch()
        self.write_kvs()
        self.write_jprq()
        self.write_nijk()
        self.write_weights()

        return self.str_data

    def write_dim(self):
        """Write dimensions."""
        string = str()
        string += '*Dimension\n'
        string += ', '.join([str(obj.dim) for obj in self.objects])
        string += '\n'
        self.str_data += string

    def write_nb_cps(self):
        """Write number of control points by element."""
        string = str()
        string += '*Number of CP by element\n'
        string += ', '.join([str(obj.nb_nodes_tot) for obj in self.objects])
        string += '\n'
        self.str_data += string

    def write_nb_patches(self):
        """Write the number of patche(s)."""
        string = str()
        string += '*Number of patch\n'
        string += '{}\n'.format(len(self.objects))
        self.str_data += string

    def write_nb_elems(self):
        """Write the total number of elements."""
        string = str()
        string += '*Total number of element\n'
        string += '{}\n'.format(sum([obj.nb_elem_tot for obj in self.objects]))
        self.str_data += string

    def write_nb_elems_patch(self):
        """Write the number of elements by patch."""
        string = str()
        string += '*Number of element by patch\n'
        string += ', '.join([str(obj.nb_elem_tot) for obj in self.objects])
        string += '\n'
        self.str_data += string

    def write_kvs(self):
        """Write the knovectors."""
        string = str()
        i_obj = 0
        for obj in self.objects:
            string += '*Patch({})\n'.format(i_obj + 1)
            for kv, nb_knot in zip(obj.knotvectors, obj.nb_knots):
                string += '{}\n'.format(nb_knot)
                string += ', '.join([str(x) for x in kv])
                string += '\n'
            i_obj += 1
        self.str_data += string

    def write_jprq(self):
        """Write the degrees."""
        string = str()
        string += '*Jpqr\n'
        for obj in self.objects:
            string += ', '.join([str(x) for x in obj.jpqr])
            string += '\n'
        self.str_data += string

    def write_nijk(self):
        """Write the Nijk."""
        string = str()
        string += '*Nijk\n'
        i_elem = 1
        for obj in self.objects:
            for line in obj.nijk:
                string += '{}, '.format(i_elem)
                string += ', '.join([str(x) for x in line])
                string += '\n'
                i_elem += 1
        self.str_data += string

    def write_weights(self):
        """Write the weights."""
        string = str()
        string += '*Weights\n'
        i_elem = 1
        for obj in self.objects:
            for line in obj.weights:
                string += '{}, '.format(i_elem)
                string += ', '.join([str(x) for x in line])
                string += '\n'
                i_elem += 1
        self.str_data += string
