# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:05:39 2021

@author: mguerder
"""

import os

from . import InputFile
from . import NurbsFile


def write_files(obj_list, filename, **kwargs):
    """
    Write .inp and .NB files for ``Yeti`` inputs.

    Parameters
    ----------
    obj_list : list
        List of objects. Objects can either be of class ``DefaultDomain`` or
        ``LagrangeElemU5``.
    filename : string
        Name of output files.
    job_name : string, optional
        Name of job written in the .inp file. The default is `job1`.
    part_name : string, optional
        Name of part written in the .inp file. The default is `part1`.
    directory: string, optional
        Output directory to write the files to. The default is `None` (takes
        the current working directory).

    """
    # --- Get keyword arguments
    job_name = kwargs.get('job_name', 'job1')
    job_name += '.txt'
    part_name = kwargs.get('part_name', 'part1')
    directory = kwargs.get('directory', None)
    # Change directory if necessary
    if directory is not None:
        os.chdir(directory)

    objects = []
    interfaces = []
    domains = []
    lagrange = []
    coupling = False
    for obj in obj_list:
        if obj.__class__.__name__ == 'InterfDomain':
            interfaces.append(obj)
            coupling = True
        elif obj.__class__.__name__ == 'LagrangeElem':
            lagrange.append(obj)
            coupling = True
        elif obj.__class__.__name__ == 'DefaultDomain':
            domains.append(obj)
        elif obj.__class__.__name__ == 'LagrangeElemU5':
            domains.append(obj)
        objects.append(obj)

    # .inp file
    InpFile = InputFile.InputFile(job_name, part_name, coupling)
    InpFile.domains = domains
    if coupling:
        InpFile.interfaces = interfaces
        InpFile.lagrange = lagrange

    string = InpFile.concatenate_data()
    with open(filename + '.inp', 'w') as file:
        file.write(string)

    # .nb file
    NbFile = NurbsFile.NurbsFile()
    NbFile.objects = objects

    string = NbFile.concatenate_data()
    with open(filename + '.NB', 'w') as file:
        file.write(string)
