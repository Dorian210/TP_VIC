# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:27:21 2020

@author: mguerder
"""


class InputFile:
    """A class to build .inp files."""

    def __init__(self, job_name, part_name, coupling):
        self.job_name = job_name
        self.part_name = part_name
        self.coupling = coupling
        self.domains = []
        self.interfaces = []
        self.lagrange = []
        self.str_data = str()
        self.id_1st_node = []
        self.id_1st_elem = []

    def concatenate_data(self):
        """Create string containing all data."""
        self.write_job_heading()
        self.write_part_heading()
        self.write_nodes()
        self.write_elements()
        self.write_element_sets()
        self.write_node_sets()
        self.write_UEL_ppties()
        self.write_part_footing()
        self.write_non_u_dload()
        self.write_assembly()
        self.write_mat()
        self.write_step()
        self.write_bcs()
        self.write_cloads()
        self.write_dloads()
        self.write_output()
        self.write_step_footing()

        return self.str_data

    def write_job_heading(self):
        """Write job heading."""
        string = '*HEADING\n'
        string += '**Job name:' + self.job_name + '\n'
        self.str_data += string

    def write_part_heading(self):
        """Write part heading."""
        string = '*Part, name=' + self.part_name + '\n'
        # Get element types in domains list
        elt_types_all = [dom.elem_type for dom in self.domains]
        elt_types_filtered = list(set(elt_types_all))
        # Create one dict for each element type & get properties accordingly
        dict_list = []
        for elem_type in elt_types_filtered:
            indices = []
            for idx, e in enumerate(elt_types_all):
                if e == elem_type:
                    indices.append(idx)
            dim = max([self.domains[x].dim for x in indices])
            nb_nodes = max([self.domains[x].nb_nodes_tot for x in indices])
            nb_integ = max([self.domains[x].nb_integ for x in indices])
            dict_list.append({'elt_type': elem_type, 'dim': dim,
                              'nb_nodes': nb_nodes, 'nb_integ': nb_integ})
        # Write domains elemt types
        for dico in dict_list:
            if dico['elt_type'] == 'U1' or dico['elt_type'] == 'U10':
                tensor = 'THREED'
            else:
                tensor = 'NONE'
            string += ('*USER ELEMENT, NODES={}, TYPE={}, COORDINATES={},'
                       'INTEGRATION={}, TENSOR={}\n'.format(
                           dico['nb_nodes'], dico['elt_type'], dico['dim'],
                           dico['nb_integ'], tensor))
            string += '1,2,3\n'
        if self.coupling:
            # Interfaces
            dim = max([interf.dim for interf in self.interfaces])
            nb_nodes = max([interf.nb_nodes_tot for interf in self.interfaces])
            nb_integ = max([interf.nb_integ for interf in self.interfaces])
            string += ('*USER ELEMENT, NODES={}, TYPE=U00, COORDINATES={},'
                       'INTEGRATION={}, TENSOR=NONE\n'.format(
                           nb_nodes, dim, nb_integ))
            string += '1,2,3\n'
            # Lagrange
            dim = max([lgrge.dim for lgrge in self.lagrange])
            nb_nodes = max([lgrge.nb_nodes_tot for lgrge in self.lagrange])
            nb_integ = max([lgrge.nb_integ for lgrge in self.lagrange])
            string += ('*USER ELEMENT, NODES={}, TYPE=U4, COORDINATES={},'
                       'INTEGRATION={}, TENSOR=NONE\n'.format(
                           nb_nodes, dim, nb_integ))
            string += '1,2,3\n'
        self.str_data += string

    def write_nodes(self):
        """Write nodes coordinates."""
        string = '*Node,nset=AllNode\n'
        i_node = 1  # Nodes counter
        for dom in self.domains:
            self.id_1st_node.append(i_node)
            for cp in dom.ctrlpts:
                string += '{}, '.format(i_node)
                string += ', '.join([str(x) for x in cp])
                string += '\n'
                i_node += 1
        if self.coupling:
            for interf in self.interfaces:
                self.id_1st_node.append(i_node)
                for cp in interf.ctrlpts:
                    string += '{}, '.format(i_node)
                    string += ', '.join([str(x) for x in cp])
                    string += '\n'
                    i_node += 1
            for lgrge in self.lagrange:
                self.id_1st_node.append(i_node)
                for cp in lgrge.ctrlpts:
                    string += '{}, '.format(i_node)
                    string += ', '.join([str(x) for x in cp])
                    string += '\n'
                    i_node += 1
        self.id_1st_node.append(i_node)
        self.str_data += string

    def write_elements(self):
        """Write elements connectivity array, for each element type."""
        string = str()
        i_elem = 1  # Element counter
        i_obj = 0  # Object counter
        for dom in self.domains:
            string += '*Element, type={}\n'.format(dom.elem_type)
            self.id_1st_elem.append(i_elem)
            frst_node = self.id_1st_node[i_obj]
            for indices in dom.ien:
                string += '{}, '.format(i_elem)
                string += ', '.join([str(x + frst_node - 1) for x in indices])
                string += '\n'
                i_elem += 1
            i_obj += 1
        if self.coupling:
            for interf in self.interfaces:
                string += '*Element,type=U00\n'
                self.id_1st_elem.append(i_elem)
                frst_node = self.id_1st_node[i_obj]
                for indices in interf.ien:
                    string += '{}, '.format(i_elem)
                    string += ', '.join([str(x + frst_node - 1)
                                         for x in indices])
                    string += '\n'
                    i_elem += 1
                i_obj += 1
            for lgrge in self.lagrange:
                string += '*Element,type=U4\n'
                self.id_1st_elem.append(i_elem)
                frst_node = self.id_1st_node[i_obj]
                for indices in lgrge.ien:
                    string += '{}, '.format(i_elem)
                    string += ', '.join([str(x + frst_node - 1)
                                         for x in indices])
                    string += '\n'
                    i_elem += 1
                i_obj += 1
        self.id_1st_elem.append(i_elem)
        self.str_data += string

    def write_element_sets(self):
        """Write element sets."""
        string = str()
        i_patch = 1  # Patch counter
        for dom in self.domains:
            i_obj = i_patch - 1  # Object counter
            string += '*ELSET,ELSET=EltPatch{},generate\n'.format(i_patch)
            string += '{}, {}, 1\n'.format(
                self.id_1st_elem[i_obj], self.id_1st_elem[i_obj + 1] - 1)
            i_patch += 1
        if self.coupling:
            for interf in self.interfaces:
                i_obj = i_patch - 1  # Object counter
                string += '*ELSET,ELSET=EltPatch{},generate\n'.format(i_patch)
                string += '{}, {}, 1\n'.format(
                    self.id_1st_elem[i_obj], self.id_1st_elem[i_obj + 1] - 1)
                i_patch += 1
            for lgrge in self.lagrange:
                i_obj = i_patch - 1  # Object counter
                string += '*ELSET,ELSET=EltPatch{},generate\n'.format(i_patch)
                string += '{}, {}, 1\n'.format(
                    self.id_1st_elem[i_obj], self.id_1st_elem[i_obj + 1] - 1)
                i_patch += 1
        self.str_data += string

    def write_node_sets(self):
        """Write node sets."""
        string = str()
        # Domains node sets faces
        i_obj = 0
        for dom in self.domains:
            if dom.elem_type == 'U0' or dom.elem_type == 'U5':
                pass
            else:
                frst_node = self.id_1st_node[i_obj]
                i_patch = i_obj + 1
                string += '*NSET,NSET=CPonFace1Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[0]) + '\n'
                string += '*NSET,NSET=CPonFace2Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[1]) + '\n'
                string += '*NSET,NSET=CPonFace3Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[2]) + '\n'
                string += '*NSET,NSET=CPonFace4Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[3]) + '\n'
                string += '*NSET,NSET=CPonFace5Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[4]) + '\n'
                string += '*NSET,NSET=CPonFace6Patch{}\n'.format(i_patch)
                string += ', '.join(str(x + frst_node - 1)
                                    for x in dom.ctrlpts_sets[5]) + '\n'
            i_obj += 1
        # Hull object sets
        elt_types_all = [dom.elem_type for dom in self.domains]
        indices = []
        for idx, elt_type in enumerate(elt_types_all):
            if elt_type == 'U0':
                indices.append(idx)
        if indices:
            i_hull = 1
            for i_obj in indices:
                string += '*NSET,NSET=CPonHull{},generate\n'.format(i_hull)
                string += '{}, {}, 1\n'.format(
                    self.id_1st_node[i_obj], self.id_1st_node[i_obj + 1] - 1)
                i_hull += 1
        # Interface sets
        if self.coupling:
            i_interf = 0  # Interface counter
            i_obj = len(self.domains)  # Object counter
            for interf in self.interfaces:
                string += '*NSET,NSET=CPinterf{},generate\n'.format(
                    i_interf + 1)
                string += '{}, {}, 1\n'.format(
                    self.id_1st_node[i_obj], self.id_1st_node[i_obj + 1] - 1)
                i_interf += 1
                i_obj += 1
        self.str_data += string

    def write_UEL_ppties(self):
        """Write material properties assigned to each patch."""
        string = str()
        i_patch = 0
        for dom in self.domains:
            string += '*UEL PROPERTY, ELSET=EltPatch{}, '.format(i_patch + 1)
            # for hull objects
            if dom.elem_type == 'U0' or dom.elem_type == 'U5':
                string += 'MATERIAL=MATvoid\n'
            else:
                string += 'MATERIAL=MAT\n'
            # for embedded solids
            if dom.elem_type == 'U10':
                string += '{}, {}\n'.format(i_patch + 1, dom.id_hull)
            elif dom.elem_type == 'U5':
                string += '{}, {}, {}, {}, {}\n'.format(
                    i_patch + 1,
                    dom.id_master, dom.id_face_master,
                    dom.id_slave, dom.id_face_slave)
            else:
                string += '{}\n'.format(i_patch + 1)
            i_patch += 1
        if self.coupling:
            for interf in self.interfaces:
                string += '*UEL PROPERTY, ELSET=EltPatch{}, '.format(
                    i_patch + 1)
                string += 'MATERIAL=MATvoid\n'
                string += '{}, {}, {}, {}\n'.format(
                    i_patch + 1, interf.id_support_dom, interf.id_lgrge,
                    interf.is_master)
                i_patch += 1
            for lgrge in self.lagrange:
                string += '*UEL PROPERTY, ELSET=EltPatch{}, '.format(
                    i_patch + 1)
                string += 'MATERIAL=MATvoid\n'
                string += '{}, 0\n'.format(i_patch + 1)
                i_patch += 1
        self.str_data += string

    def write_part_footing(self):
        """Write part footing."""
        string = '*End Part\n'
        self.str_data += string

    def write_non_u_dload(self):
        """Write non-uniform distributed loads."""
        string = str()
        if any([dom.pfield for dom in self.domains]):
            id_pfield = 0  # Initialise pressure field id
            for dom in self.domains:
                if dom.pfield:
                    for p in dom.pfield:
                        pres_arr, face_id = p
                        # Write header
                        string += '*DISTRIBUTION, ' + \
                            'NAME=pres_field{},'.format(id_pfield+1) + \
                            ' LOCATION=NODE\n'
                        # Loop on array
                        for item in pres_arr:
                            string += '{}, {}\n'.format(int(item[0]),
                                                        float(item[1]))
                        # Update pressure field id
                        id_pfield += 1
            self.str_data += string

    def write_assembly(self):
        """Write instances."""
        string = '** ASSEMBLY\n'
        string += '*Assembly, name=assembly1\n'
        string += '*Instance, name=I1, part={}\n'.format(self.part_name)
        string += '*End Instance\n'
        string += '*End Assembly\n'
        self.str_data += string

    def write_mat(self):
        """Write material properties."""
        string = '**MATERIAL\n'
        string += '*MATERIAL,NAME=MAT\n'
        string += '*Density\n'
        string += '4.51e-09\n'
        string += '*Elastic\n'
        string += '110000.000000000000000, 0.340000000000000\n'
        if (self.coupling or ('U0' in [dom.elem_type for dom in self.domains])
                          or ('U5' in [dom.elem_type for dom in self.domains])
            ):
            string += '*MATERIAL,NAME=MATvoid\n'
            string += '*Elastic\n'
            string += '0.00, 0.00\n'
        self.str_data += string

    def write_step(self):
        """Write steps."""
        string = '*STEP,extrapolation=NO,NLGEOM=NO\n'
        string += '*Static\n'
        self.str_data += string

    def write_bcs(self):
        """Write boundary conditions."""
        string = '** BOUNDARY CONDITIONS\n'
        string += '*Boundary\n'
        # Hull object
        elt_types_all = [dom.elem_type for dom in self.domains]
        indices = []
        for idx, elt_type in enumerate(elt_types_all):
            if elt_type == 'U0':
                indices.append(idx)
        if indices:
            i_hull = 1
            for i_obj in indices:
                string += 'I1.CPonHull{}, 1, 3, 0.0\n'.format(i_hull)
                i_hull += 1
        # Interfaces
        if self.coupling:
            i_interf = 1
            for interf in self.interfaces:
                string += 'I1.CPinterf{}, 1, 3, 0.0\n'.format(i_interf)
                i_interf += 1
        # Others
        for dom in self.domains:
            if dom.bcs:
                string += 'I1.{}, {}, {}, {}\n'.format(dom.bcs['nset'],
                                                       dom.bcs['dir1'],
                                                       dom.bcs['dir2'],
                                                       dom.bcs['mag'])
        self.str_data += string

    def write_cloads(self):
        """Write concentrated loads."""
        string = str()
        if any([dom.cload for dom in self.domains]):
            string += '*Cload\n'
            for dom in self.domains:
                if dom.cload:
                    string += 'I1.{}, {}, {}, {}\n'.format(dom.cload['nset'],
                                                           dom.cload['dir1'],
                                                           dom.cload['dir2'],
                                                           dom.cload['mag'])
        else:
            string += '**Cload\n'
            string += '**I1.CPToLoad, 1, 1, 0.0\n'

        self.str_data += string

    def write_dloads(self):
        """Write distributed loads."""
        string = str()
        if any([dom.dload for dom in self.domains]):
            string += '*Dload\n'
            for dom in self.domains:
                if dom.dload:
                    string += 'I1.{}, {}, {}\n'.format(dom.dload['elset'],
                                                       dom.dload['ltype'],
                                                       dom.dload['mag'])
        if any([dom.pfield for dom in self.domains]):
            if not any([dom.dload for dom in self.domains]):
                string += '*Dload\n'
            id_pfield = 0  # Initialise pressure field id
            i_patch = 0
            for dom in self.domains:
                if dom.pfield:
                    for p in dom.pfield:
                        pres_arr, face_id = p
                        string += 'I1.EltPatch{}, U4{}, pres_field{}\n'.format(
                            i_patch+1, face_id, id_pfield+1)
                        # Update pressure field id
                        id_pfield += 1
                # Update patch id
                i_patch += 1
        else:
            string += '**Dload\n'
            string += '**I1.EltPatch1, U60, 0.0\n'

        self.str_data += string

    def write_output(self):
        """Write output requests."""
        string = str()
        string += '** OUTPUT REQUESTS\n'
        string += '*node file, frequency=1\n'
        string += 'U,RF,CF\n'
        string += '*el file, frequency=1\n'
        string += 'SDV\n'
        self.str_data += string

    def write_step_footing(self):
        """Write step footing."""
        string = '*End step\n'
        self.str_data += string
