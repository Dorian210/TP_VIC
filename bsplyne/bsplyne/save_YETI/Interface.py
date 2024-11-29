# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:11:09 2021

@author: mguerder
"""

import numpy as np


# =============================================================================
# CLASS INTERFACE DOMAIN
# =============================================================================
class InterfDomain:
    """A class to define interfaces for coupling analyses."""

    def __init__(self, geometry, id_interf):
        # Interface ppties.
        self.id_interf = id_interf
        self.id_support_dom = 0
        self.id_lgrge = 0
        self.is_master = 0
        self.elem_type = 'U00'

        # Initialise geom. ppties.
        self.dim = 0
        self.nb_cp = []
        self.nb_cp_tot = 0
        self.jpqr = []
        self.geomdl_cp = []
        self.ctrlpts = []
        self.ien = []
        self.nb_elem = 0
        self.nb_elem_tot = 0
        self.nb_nodes = []
        self.nb_nodes_tot = 0
        self.nb_integ = []
        self.knotvectors = []
        self.nb_knots = []
        self.nijk = []
        self.weights = []

        # Compute geom. properties
        self.dim = len(geometry.cpsize)
        self.nb_cp = geometry.cpsize
        self.jpqr = geometry.degree
        if self.dim == 1:
            self.jpqr = [self.jpqr]
        self.geomdl_cp = geometry.ctrlpts
        self.nb_cp_tot = np.prod(self.nb_cp).tolist()
        self.nb_nodes = np.array(self.jpqr) + 1
        self.nb_nodes_tot = np.prod(self.nb_nodes, dtype=int)
        self.nb_integ = (max(self.jpqr) + 1) ** 3
        self.knotvectors = geometry.knotvector
        if self.dim == 1:
            self.knotvectors = [self.knotvectors]
        self.nb_knots = [len(kv) for kv in self.knotvectors]
        self.build_ctrlpts_array()
        self.build_connectivity_array()
        self.build_nijk()
        self.build_weights_array()

    def build_ctrlpts_array(self):
        """Build control points array."""
        cp_array = np.zeros((self.nb_cp_tot, 3))
        # Surface case
        if self.dim == 2:
            nb_cp_u, nb_cp_v = self.nb_cp[:]
            ctrlpts = self.geomdl_cp
            ii = 0  # Loop counter
            for i in range(nb_cp_u):
                for j in range(nb_cp_v):
                    idx = i + j * nb_cp_u
                    cp_array[idx, :] = ctrlpts[ii][:]
                    ii += 1
        # Volume case
        if self.dim == 3:
            nb_cp_u, nb_cp_v, nb_cp_w = self.nb_cp[:]
            ctrlpts = self.geomdl_cp
            ii = 0  # Loop counter
            for k in range(nb_cp_w):
                for i in range(nb_cp_u):
                    for j in range(nb_cp_v):
                        idx = i + j * nb_cp_u + k * nb_cp_u * nb_cp_v
                        cp_array[idx, :] = ctrlpts[ii][:]
                        ii += 1
        self.ctrlpts = cp_array.tolist()

    def build_connectivity_array(self):
        """Build IEN array for all patches."""
        # Number of elements in each direction
        self.nb_elem = np.array(self.nb_cp) - np.array(self.jpqr)
        # Total number of elements in patch
        self.nb_elem_tot = np.prod(self.nb_elem)
        # Number of nodes/element in each direction
        self.nb_nodes = np.array(self.jpqr) + 1
        # Connectivity array, xi-dir.
        ien_xi = np.tile(np.arange(0, self.nb_nodes[0]),
                         (self.nb_elem[0], 1)) + \
            np.tile(np.vstack(np.arange(0, self.nb_elem[0])),
                    (1, self.nb_nodes[0]))
        if self.dim == 1:
            ien_patch = ien_xi + 1
        else:
            # Connectivity array, eta-dir.
            ien_eta = np.tile(np.arange(0, self.nb_nodes[1]),
                              (self.nb_elem[1], 1)) + \
                np.tile(np.vstack(np.arange(0, self.nb_elem[1])),
                        (1, self.nb_nodes[1]))
            ien_eta *= self.nb_cp[0]
            # Combine xi- and eta-dir. arrays
            ien_xieta = \
                np.tile(ien_xi, (self.nb_elem[1], self.nb_nodes[1])) + \
                np.repeat(np.repeat(ien_eta, self.nb_nodes[0], axis=1),
                          self.nb_elem[0], axis=0)
            if self.dim == 2:
                ien_patch = ien_xieta + 1
            if self.dim == 3:
                # Connectivity array, zeta-dir.
                ien_zeta = np.tile(np.arange(0, self.nb_nodes[2]),
                                   (self.nb_elem[2], 1)) + \
                    np.tile(np.vstack(np.arange(0, self.nb_elem[2])),
                            (1, self.nb_node[2]))
                ien_zeta *= self.nb_cp[0] * self.nb_cp[1]
                # Fill connectivity array
                ien_patch = (
                    np.tile(ien_xieta, (self.nb_elem[2],
                                        self.nb_nodes[2])) +
                    np.repeat(np.repeat(ien_zeta, np.prod(self.nb_nodes[: 2]),
                                        axis=1),
                              np.prod(self.nb_elem[: 2]), axis=0))[:, ::-1] + 1

        self.ien = ien_patch  # Update IEN

    def build_nijk(self):
        """Build Nijk for all patches."""
        # Compute Nijk for each direction
        Nijk_by_dir = []
        for nb_knots_this_dir, jpqr_this_dir in zip(self.nb_knots, self.jpqr):
            idx_start = jpqr_this_dir + 1
            idx_stop = nb_knots_this_dir - jpqr_this_dir
            Nijk_by_dir.append(np.arange(idx_start, idx_stop))
        # Differenciate dim. and add to global Nijk
        if self.dim == 1:
            Nijk_xi = Nijk_by_dir
            for i in Nijk_xi:
                self.nijk.append(i)
        if self.dim == 2:
            Nijk_xi, Nijk_eta = Nijk_by_dir
            for j in Nijk_eta:
                for i in Nijk_xi:
                    self.nijk.append([i, j])
        elif self.dim == 3:
            Nijk_xi, Nijk_eta, Nijk_zeta = Nijk_by_dir
            for k in Nijk_zeta:
                for j in Nijk_eta:
                    for i in Nijk_xi:
                        self.nijk.append([i, j, k])

    def build_weights_array(self):
        """Build weights array for all patches."""
        for i_elem in range(self.nb_elem_tot):
            self.weights.append([1.0] * self.nb_nodes_tot)


# =============================================================================
# CLASS LAGRANGE ELEMENTS
# =============================================================================
class LagrangeElem:
    """A class to define Lagrange elements for coupling analyses."""

    def __init__(self, id_lgrge, interf):
        self.id_lgrge = id_lgrge
        self.elem_type = 'U4'
        self.dim = interf.dim
        self.nb_cp = [1]
        self.nb_cp_tot = 1
        self.jpqr = np.zeros_like(interf.jpqr).tolist()
        self.ctrlpts = [[0.0] * 3]
        self.ien = [[1]]
        self.nb_elem = [1]
        self.nb_elem_tot = 1
        self.nb_nodes = 1
        self.nb_nodes_tot = 1
        self.nb_integ = interf.nb_integ
        self.knotvectors = [[0.0, 1.0] for i in range(self.dim)]
        self.nb_knots = [len(kv) for kv in self.knotvectors]
        self.nijk = [[1] * self.dim]
        self.weights = [[1.0]]


# =============================================================================
# CLASS INTERFACE ELEMENTS (NEW)
# =============================================================================
class LagrangeElemU5:
    """A class to define U5 Lagrange elements for coupling analyses."""

    def __init__(self, id_lgrge,
                 id_master, id_face_master,
                 id_slave, id_face_slave):
        # !!! WARNING assuming 2-dimensional interface
        self.id_lgrge = id_lgrge
        self.elem_type = 'U5'
        self.id_master = id_master
        self.id_face_master = id_face_master
        self.id_slave = id_slave
        self.id_face_slave = id_face_slave
        self.dim = 2
        self.nb_cp = [1]
        self.nb_cp_tot = 1
        self.jpqr = [0, 0]
        self.ctrlpts = [[0.0] * 3]
        self.ien = [[1]]
        self.nb_elem = [1]
        self.nb_elem_tot = 1
        self.nb_nodes = 1
        self.nb_nodes_tot = 1
        self.nb_integ = 8
        self.knotvectors = [[0.0, 1.0] for i in range(self.dim)]
        self.nb_knots = [len(kv) for kv in self.knotvectors]
        self.nijk = [[1] * self.dim]
        self.weights = [[1.0]]
        self.bcs = None
        self.cload = None
        self.dload = None
        self.pfield = None
