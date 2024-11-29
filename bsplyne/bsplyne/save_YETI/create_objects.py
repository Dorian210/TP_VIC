# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:40:38 2021

@author: mguerder
"""

import numpy as np
from geomdl import BSpline
from geomdl import NURBS
from geomdl import knotvector
from geomdl import construct
from geomdl import fitting

from post_pro.datasetting import Domain
from post_pro.datasetting import Interface
from geometry.surfaces import utilities as surf_util


def create_cplist(coords):
    """Create a list of control points that is compatible with geomdl.

    Parameters
    ----------
    coords : list
        Control points list.

    Returns
    -------
    cp_list : list
        New list of control points, ordered for geomdl usage.
    ctrlpts_size : list of ints
        Number of control points in each direction.

    """
    coords_x, coords_y, coords_z = coords

    nb_cp_u = len(coords_x)
    nb_cp_v = len(coords_y)
    nb_cp_w = len(coords_z)
    ctrlpts_size = [nb_cp_u, nb_cp_v, nb_cp_w]

    cp_list = []
    for k in range(nb_cp_w):
        for i in range(nb_cp_u):
            for j in range(nb_cp_v):
                x = coords_x[i]
                y = coords_y[j]
                z = coords_z[k]
                cp_list.append([x, y, z])

    return cp_list, ctrlpts_size


def create_volume(coords, degs, ctrlpts_size=None, create_cp=True,
                  weights=None):
    """Create a geomdl volume according to the input data.

    Parameters
    ----------
    coords : list
        List containing the control points coordinates.
    degs : list, tuple of ints
        Degree in each parametric direction.
    ctrlpts_size : list of ints, optional
        Number of control points in each direction. The default is None.
    create_cp : bool, optional
        Argument used to create a list of control points. If set to False, the
        control points coordinates are taken as `coords'. The default is True.
    weights : list, optional
        List of weightsn if needed. The default is None.

    Returns
    -------
    volume : geomdl BSpline or NURBS Volume
        Resulting volume.

    """
    if create_cp:
        cp_list, ctrlpts_size = create_cplist(coords)
    else:
        cp_list = np.zeros_like(coords)
        nb_cp_u, nb_cp_v, nb_cp_w = ctrlpts_size
        ii = 0
        for k in range(nb_cp_w):
            for j in range(nb_cp_v):
                for i in range(nb_cp_u):
                    idx = j + i * nb_cp_v + k * nb_cp_u * nb_cp_v
                    cp_list[idx] = coords[ii]
                    ii += 1
        cp_list = cp_list.tolist()

    nb_cp_u, nb_cp_v, nb_cp_w = ctrlpts_size

    deg_u, deg_v, deg_w = degs

    if not weights:
        volume = BSpline.Volume()
        volume.degree = deg_u, deg_v, deg_w
        volume.cpsize = (nb_cp_u, nb_cp_v, nb_cp_w)
        volume.ctrlpts = cp_list
        volume.knotvector_u = knotvector.generate(deg_u, nb_cp_u)
        volume.knotvector_v = knotvector.generate(deg_v, nb_cp_v)
        volume.knotvector_w = knotvector.generate(deg_w, nb_cp_w)
    else:
        volume = NURBS.Volume()
        volume.degree = deg_u, deg_v, deg_w
        volume.cpsize = (nb_cp_u, nb_cp_v, nb_cp_w)
        volume.ctrlpts = cp_list
        volume.weights = weights
        volume.knotvector_u = knotvector.generate(deg_u, nb_cp_u)
        volume.knotvector_v = knotvector.generate(deg_v, nb_cp_v)
        volume.knotvector_w = knotvector.generate(deg_w, nb_cp_w)

    return volume


def create_surface(coords, degs, ctrlpts_size=None, create_cp=True,
                   weights=None):
    """Create a geomdl surface according to the input data.

    Parameters
    ----------
    coords : list
        List containing the control points coordinates.
    degs : list, tuple of ints
        Degree in each parametric direction.
    ctrlpts_size : list of ints, optional
        Number of control points in each direction. The default is None.
    create_cp : bool, optional
        Argument used to create a list of control points. If set to False, the
        control points coordinates are taken as `coords'. The default is True.

    Returns
    -------
    surface : geomdl BSpline.Surface
        Resulting surface.

    """
    if create_cp:
        cp_list, ctrlpts_size = create_cplist(coords)
    else:
        cp_list = np.zeros_like(coords)
        nb_cp_u, nb_cp_v, nb_cp_w = ctrlpts_size
        ii = 0
        for k in range(nb_cp_w):
            for j in range(nb_cp_v):
                for i in range(nb_cp_u):
                    idx = j + i * nb_cp_v + k * nb_cp_u * nb_cp_v
                    cp_list[idx] = coords[ii]
                    ii += 1
        cp_list = cp_list.tolist()

    ctrlpts_size.remove(1)
    nb_cp_u, nb_cp_v = ctrlpts_size

    deg_u, deg_v = degs

    if not weights:
        surface = BSpline.Surface()
        surface.degree = deg_u, deg_v
        surface.ctrlpts_size_u = nb_cp_u
        surface.ctrlpts_size_v = nb_cp_v
        surface.ctrlpts = cp_list
        surface.knotvector_u = knotvector.generate(deg_u, nb_cp_u)
        surface.knotvector_v = knotvector.generate(deg_v, nb_cp_v)
    else:
        surface = NURBS.Surface()
        surface.degree = deg_u, deg_v
        surface.ctrlpts_size_u = nb_cp_u
        surface.ctrlpts_size_v = nb_cp_v
        surface.ctrlpts = cp_list
        surface.weights = weights
        surface.knotvector_u = knotvector.generate(deg_u, nb_cp_u)
        surface.knotvector_v = knotvector.generate(deg_v, nb_cp_v)

    return surface


def create_coupling_objs(nb_doms, nb_interfs, ids_support_dom, geometries):
    """Create a list of objects to write to file.

    Parameters
    ----------
    nb_doms : int
        Number of domains.
    nb_interfs : int
        Number of interfaces.
    ids_support_dom : list of ints
        Connectivity array between domains. Each pair of domains that are to be
        coupled is contained herein.
    geometries : list
        A list containing all geometries (domains and interfaces).

    Returns
    -------
    objects_list : list
        A list containing all necessary information to write to file.

    """
    domains_list = []
    interfs_list = []
    lgrge_list = []
    nb_lgrge = nb_interfs // 2 if nb_interfs > 1 else 1

    i_obj = 0
    for geom in geometries:
        # Domains
        if i_obj < nb_doms:
            domain = Domain.DefaultDomain(geom, i_obj + 1, 'U1')
            domains_list.append(domain)
        # Interfaces
        else:
            interf = Interface.InterfDomain(geom, i_obj + 1)
            interfs_list.append(interf)
        i_obj += 1
    # Lagrange
    ids_lgrge = []
    for i_lgrge in range(nb_lgrge):
        interf = interfs_list[i_lgrge * 2]
        lgrge = Interface.LagrangeElem(i_obj + 1, interf)
        lgrge_list.append(lgrge)
        ids_lgrge.append(i_obj + 1)
        i_obj += 1
    # Set additionnal interfaces properties
    ids_lgrge = np.repeat(ids_lgrge, 2)
    for i_interf in range(nb_interfs):
        interf = interfs_list[i_interf]
        interf.is_master = (interf.id_interf + 1) % 2
        interf.id_support_dom = ids_support_dom[i_interf]
        interf.id_lgrge = ids_lgrge[i_interf]

    objects_list = domains_list + interfs_list + lgrge_list

    return objects_list


def create_interf_geom(vol_1, id_surf_1, vol_2, id_surf_2, nb_pts, degrees,
                       ctrlpts_size):
    """Compute the parametric surface interface between two parts.

    This function computes the parametric position of the interface between
    `part_1' and `part_2', on `part_2'.

    Parameters
    ----------
    vol_1 : geomdl BSpline or NURBS Volume
        First part to compute interface for.
    id_surf_1 : int
        Index of the parametric face involved in the interface on part_1'.
    vol_2 : geomdl BSpline or NURBS Volume
        Second part to compute interface for.
    id_surf_2 : int
        Index of the parametric face involved in the interface on part_2'.
    nb_pts : int
        Number of points to sample on each part.
    degrees : list of ints
        Degree in u- and v-direction for the interface surface.
    ctrlpts_size : list of ints
        Control points size in u- and v-direction for the interface surface.

    Returns
    -------
    fit_surf : geomdl BSpline Surface
        Resulting interface surface.

    """
    # isosurf = construct.extract_isosurface(vol_1)[id_surf_1]
    # isosurf.delta = 1. / nb_pts
    # evalsurf = construct.extract_isosurface(vol_2)[id_surf_2]
    # params = []
    # for point in isosurf.evalpts:
    #     uu = surf_util.point_inversion(evalsurf, point)
    #     if id_surf_2 == 0:  # zeta = 0
    #         uu = [float(u) for u in uu] + [0.0]
    #     elif id_surf_2 == 1:  # zeta = 1
    #         uu = [float(u) for u in uu] + [1.0]
    #     elif id_surf_2 == 2:  # eta = 0
    #         uu = [float(u) for u in uu]
    #         uu.insert(1, 0.0)
    #     elif id_surf_2 == 3:  # eta = 1
    #         uu = [float(u) for u in uu]
    #         uu.insert(1, 1.0)
    #     elif id_surf_2 == 4:  # xi = 0
    #         uu = [0.0] + [float(u) for u in uu]
    #     elif id_surf_2 == 5:  # xi = 1
    #         uu = [1.0] + [float(u) for u in uu]
    #     params.append(uu)
    # fit_surf = fitting.approximate_surface(params, nb_pts, nb_pts,
    #                                        degrees[0], degrees[1],
    #                                        ctrlpts_size_u=ctrlpts_size[0],
    #                                        ctrlpts_size_v=ctrlpts_size[1])

    adjust = [[2, 0.0], [2, 1.0], [1, 0.0], [1, 1.0], [0, 0.0], [0, 0.0]]
    idx_adjust, adjust_param = adjust[id_surf_2]

    isosurf = construct.extract_isosurface(vol_1)[id_surf_1]
    evalsurf = construct.extract_isosurface(vol_2)[id_surf_2]

    uu = np.linspace(0.0, 1.0, nb_pts)
    vv = np.linspace(0.0, 1.0, nb_pts)
    params = []
    for v in vv:
        for u in uu:
            param = [u, v]
            pt = isosurf.evaluate_single(param)
            invpt = surf_util.point_inversion(evalsurf, pt)
            invpt.insert(int(idx_adjust), adjust_param)
            params.append(invpt)

    fit_surf = fitting.interpolate_surface(params, nb_pts, nb_pts,
                                           degrees[0], degrees[1],
                                           centripetal=True)

    return fit_surf
