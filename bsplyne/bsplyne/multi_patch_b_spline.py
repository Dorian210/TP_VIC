# %%
import os
from itertools import permutations
from functools import lru_cache
from typing import Union, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sps
import numba as nb
from tqdm import tqdm, trange

from .b_spline import BSpline, _writePVD
from .b_spline_basis import BSplineBasis
from .save_YETI import Domain, write

# union-find algorithm for connectivity
@nb.njit
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

@nb.njit
def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

@nb.njit
def get_unique_nodes_inds(nodes_couples, nb_nodes):
    parent = np.arange(nb_nodes)
    rank = np.zeros(nb_nodes, dtype=np.int32)
    for a, b in nodes_couples:
        union(parent, rank, a, b)
    unique_nodes_inds = np.empty(nb_nodes, dtype=np.int32)
    for i in range(nb_nodes):
        unique_nodes_inds[i] = find(parent, i)
    return unique_nodes_inds

class MultiPatchBSplineConnectivity:
    """
    Contains all the methods to link multiple B-spline patches.
    It uses 3 representations of the data : 
      - a unique representation, possibly common with other meshes, containing 
        only unique nodes indices, 
      - a unpacked representation containing duplicated nodes indices, 
      - a separated representation containing duplicated nodes indices, 
        separated between patches. It is here for user friendliness.

    Attributes
    ----------
    unique_nodes_inds : numpy.ndarray of int
        The indices of the unique representation needed to create the unpacked one.
    shape_by_patch : numpy.ndarray of int
        The shape of the separated nodes by patch.
    nb_nodes : int
        The total number of unpacked nodes.
    nb_unique_nodes : int
        The total number of unique nodes.
    nb_patchs : int
        The number of patches.
    npa : int
        The dimension of the parametric space of the B-splines.
    """
    unique_nodes_inds: np.ndarray
    shape_by_patch: np.ndarray
    nb_nodes: int
    nb_unique_nodes: int
    nb_patchs: int
    npa: int
    
    def __init__(self, unique_nodes_inds, shape_by_patch, nb_unique_nodes):
        """

        Parameters
        ----------
        unique_nodes_inds : numpy.ndarray of int
            The indices of the unique representation needed to create the unpacked one.
        shape_by_patch : numpy.ndarray of int
            The shape of the separated nodes by patch.
        nb_unique_nodes : int
            The total number of unique nodes.
        """
        self.unique_nodes_inds = unique_nodes_inds
        self.shape_by_patch = shape_by_patch
        self.nb_nodes = np.sum(np.prod(self.shape_by_patch, axis=1))
        self.nb_unique_nodes = nb_unique_nodes #np.unique(self.unique_nodes_inds).size
        self.nb_patchs, self.npa = self.shape_by_patch.shape
    
    @classmethod
    def from_nodes_couples(cls, nodes_couples, shape_by_patch):
        """
        Create the connectivity from a list of couples of unpacked nodes.

        Parameters
        ----------
        nodes_couples : numpy.ndarray of int
            Couples of indices of unpacked nodes that are considered the same.
            Its shape should be (# of couples, 2)
        shape_by_patch : numpy.ndarray of int
            The shape of the separated nodes by patch.

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        nb_nodes = np.sum(np.prod(shape_by_patch, axis=1))
        unique_nodes_inds = get_unique_nodes_inds(nodes_couples, nb_nodes)
        different_unique_nodes_inds, inverse = np.unique(unique_nodes_inds, return_inverse=True)
        unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], different_unique_nodes_inds))) - 1)[inverse]
        nb_unique_nodes = np.unique(unique_nodes_inds).size
        return cls(unique_nodes_inds, shape_by_patch, nb_unique_nodes)
    
    @classmethod
    def from_separated_ctrlPts(cls, separated_ctrlPts, eps=1e-10, return_nodes_couples: bool=False):
        """
        Create the connectivity from a list of control points given as 
        a separated field by comparing every couple of points.

        Parameters
        ----------
        separated_ctrlPts : list of numpy.ndarray of float
            Control points of every patch to be compared in the separated 
            representation. Every array is of shape : 
            (``NPh``, nb elem for dim 1, ..., nb elem for dim ``npa``)
        eps : float, optional
            Maximum distance between two points to be considered the same, by default 1e-10
        return_nodes_couples : bool, optional
            If `True`, returns the `nodes_couples` created, by default False

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        NPh = separated_ctrlPts[0].shape[0]
        assert np.all([ctrlPts.shape[0]==NPh for ctrlPts in separated_ctrlPts[1:]]), "Physical spaces must contain the same number of dimensions !"
        shape_by_patch = np.array([ctrlPts.shape[1:] for ctrlPts in separated_ctrlPts], dtype='int')
        nodes_couples = []
        previous_pts = separated_ctrlPts[0].reshape((NPh, -1))
        previous_inds_counter = previous_pts.shape[1]
        previous_inds = np.arange(previous_inds_counter)
        for ctrlPts in separated_ctrlPts[1:]:
            # create current pts and inds
            current_pts = ctrlPts.reshape((NPh, -1))
            current_inds_counter = previous_inds_counter + current_pts.shape[1]
            current_inds = np.arange(previous_inds_counter, current_inds_counter)
            # get couples
            dist = np.linalg.norm(previous_pts[:, :, None] - current_pts[:, None, :], axis=0)
            previous_inds_inds, current_inds_inds = (dist<eps).nonzero()
            nodes_couples.append(np.hstack((previous_inds[previous_inds_inds, None], current_inds[current_inds_inds, None])))
            # add current to previous for next iteration
            previous_pts = np.hstack((previous_pts, current_pts))
            previous_inds_counter = current_inds_counter
            previous_inds = np.hstack((previous_inds, current_inds))
        if len(nodes_couples)>0:
            nodes_couples = np.vstack(nodes_couples)
        else:
            nodes_couples = np.empty((0, 2), dtype='int')
        if return_nodes_couples:
            return cls.from_nodes_couples(nodes_couples, shape_by_patch), nodes_couples
        else:
            return cls.from_nodes_couples(nodes_couples, shape_by_patch)
    
    def unpack(self, unique_field):
        """
        Extract the unpacked representation from a unique representation.

        Parameters
        ----------
        unique_field : numpy.ndarray
            The unique representation. Its shape should be :
            (field, shape, ..., `self`.`nb_unique_nodes`)

        Returns
        -------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        unpacked_field = unique_field[..., self.unique_nodes_inds]
        return unpacked_field
    
#     def unpack_patch_jacobians(self, field_size):
#         patch_jacobians = []
#         ind = 0
#         for patch_shape in self.shape_by_patch:
#             nb_row = np.prod(patch_shape)
#             next_ind = ind + nb_row
#             row = np.arange(nb_row)
#             col = self.unique_nodes_inds[ind:next_ind]
#             data = np.ones(nb_row)
#             mat = sps.coo_matrix((data, (row, col)), shape=(nb_row, self.nb_unique_nodes))
#             patch_jacobians.append(sps.block_diag([mat]*field_size))
#             ind = next_ind
#         return patch_jacobians
    
    def pack(self, unpacked_field, method='mean'):
        """
        Extract the unique representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape should be :
            (field, shape, ..., `self`.`nb_nodes`)
        method: str
            The method used to group values that could be different

        Returns
        -------
        unique_nodes : numpy.ndarray
            The unique representation. Its shape is :
            (field, shape, ..., `self`.`nb_unique_nodes`)
        """
        field_shape = unpacked_field.shape[:-1]
        unique_field = np.zeros((*field_shape, self.nb_unique_nodes), dtype=unpacked_field.dtype)
        if method=='first':
            unique_field[..., self.unique_nodes_inds[::-1]] = unpacked_field[..., ::-1]
        elif method=='mean':
            np.add.at(unique_field.T, self.unique_nodes_inds, unpacked_field.T)
            counts = np.zeros(self.nb_unique_nodes, dtype='uint')
            np.add.at(counts, self.unique_nodes_inds, 1)
            unique_field /= counts
        else:
            raise NotImplementedError(f"Method {method} is not implemented ! Consider using 'first' or 'mean'.")
        return unique_field
    
    def separate(self, unpacked_field):
        """
        Extract the separated representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)

        Returns
        -------
        separated_field : list of numpy.ndarray
            The separated representation. Every array is of shape : 
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)
        """
        field_shape = unpacked_field.shape[:-1]
        separated_field = []
        ind = 0
        for patch_shape in self.shape_by_patch:
            next_ind = ind + np.prod(patch_shape)
            separated_field.append(unpacked_field[..., ind:next_ind].reshape((*field_shape, *patch_shape)))
            ind = next_ind
        return separated_field
    
    def agglomerate(self, separated_field):
        """
        Extract the unpacked representation from a separated representation.

        Parameters
        ----------
        separated_field : list of numpy.ndarray
            The separated representation. Every array is of shape : 
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)

        Returns
        -------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        field_shape = separated_field[0].shape[:-self.npa]
        assert np.all([f.shape[:-self.npa]==field_shape for f in separated_field]), "Every patch must have the same field shape !"
        unpacked_field = np.concatenate([f.reshape((*field_shape, -1)) for f in separated_field], axis=-1)
        return unpacked_field
    
    def unique_field_indices(self, field_shape, representation="separated"):
        """
        Get the unique, unpacked or separated representation of a field's unique indices.

        Parameters
        ----------
        field_shape : tuple of int
            The shape of the field. For example, if it is a vector field, `field_shape` 
            should be (3,). If it is a second order tensor field, it should be (3, 3).
        representation : str, optional
            The user must choose between `"unique"`, `"unpacked"`, and `"separated"`.
            It corresponds to the type of representation to get, by default "separated"

        Returns
        -------
        unique_field_indices : numpy.ndarray of int or list of numpy.ndarray of int
            The unique, unpacked or separated representation of a field's unique indices.
            If unique, its shape is (*`field_shape`, `self`.`nb_unique_nodes`).
            If unpacked, its shape is : (*`field_shape`, `self`.`nb_nodes`).
            If separated, every array is of shape : (*`field_shape`, nb elem for dim 1, ..., nb elem for dim `npa`).
        """
        nb_indices = np.prod(field_shape)*self.nb_unique_nodes
        unique_field_indices_as_unique_field = np.arange(nb_indices, dtype='int').reshape((*field_shape, self.nb_unique_nodes))
        if representation=="unique":
            unique_field_indices = unique_field_indices_as_unique_field
            return unique_field_indices
        elif representation=="unpacked":
            unique_field_indices = self.unpack(unique_field_indices_as_unique_field)
            return unique_field_indices
        elif representation=="separated":
            unique_field_indices = self.separate(self.unpack(unique_field_indices_as_unique_field))
            return unique_field_indices
        else:
            raise ValueError(f'Representation "{representation}" not recognised. Representation must either be "unique", "unpacked", or "separated" !')
    
    def get_duplicate_unpacked_nodes_mask(self):
        unique, inverse, counts = np.unique(self.unique_nodes_inds, return_inverse=True, return_counts=True)
        duplicate_nodes_mask = np.zeros(self.nb_nodes, dtype='bool')
        duplicate_nodes_mask[counts[inverse]>1] = True
        return duplicate_nodes_mask
    
    def extract_exterior_borders(self, splines):
        if self.npa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
        duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
        separated_unique_nodes_inds = self.unique_field_indices(())
        arr = np.arange(self.npa).tolist()
        border_splines = []
        border_unique_nodes_inds = []
        border_shape_by_patch = []
        for i in range(self.nb_patchs):
            spline = splines[i]
            duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
            unique_nodes_inds_spline = separated_unique_nodes_inds[i]
            shape_by_patch_spline = self.shape_by_patch[i]
            for axis in range(self.npa):
                bases = np.hstack((spline.bases[(axis + 1):], spline.bases[:axis]))
                axes = arr[axis:-1] + arr[:axis]
                border_shape_by_patch_spline = np.hstack((shape_by_patch_spline[(axis + 1):], shape_by_patch_spline[:axis]))
                if not np.take(duplicate_nodes_mask_spline, 0, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases[::-1])
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=axis).transpose(axes[::-1]).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[::-1][None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {0} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
                if not np.take(duplicate_nodes_mask_spline, -1, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases)
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=axis).transpose(axes).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {-1} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
        border_splines = np.array(border_splines, dtype='object')
        border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
        border_shape_by_patch = np.concatenate(border_shape_by_patch)
        border_unique_to_self_unique_connectivity, inverse = np.unique(border_unique_nodes_inds, return_inverse=True)
        border_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity))) - 1)[inverse]
        border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
        border_connectivity = self.__class__(border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes)
        return border_connectivity, border_splines, border_unique_to_self_unique_connectivity
    
    def extract_interior_borders(self, splines):
        if self.npa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
        duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
        separated_unique_nodes_inds = self.unique_field_indices(())
        arr = np.arange(self.npa).tolist()
        border_splines = []
        border_unique_nodes_inds = []
        border_shape_by_patch = []
        for i in range(self.nb_patchs):
            spline = splines[i]
            duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
            unique_nodes_inds_spline = separated_unique_nodes_inds[i]
            shape_by_patch_spline = self.shape_by_patch[i]
            for axis in range(self.npa):
                bases = np.hstack((spline.bases[(axis + 1):], spline.bases[:axis]))
                axes = arr[axis:-1] + arr[:axis]
                border_shape_by_patch_spline = np.hstack((shape_by_patch_spline[(axis + 1):], shape_by_patch_spline[:axis]))
                if np.take(duplicate_nodes_mask_spline, 0, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases[::-1])
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=axis).transpose(axes[::-1]).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[::-1][None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {0} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
                if np.take(duplicate_nodes_mask_spline, -1, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases)
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=axis).transpose(axes).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {-1} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
        border_splines = np.array(border_splines, dtype='object')
        border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
        border_shape_by_patch = np.concatenate(border_shape_by_patch)
        border_unique_to_self_unique_connectivity, inverse = np.unique(border_unique_nodes_inds, return_inverse=True)
        border_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity))) - 1)[inverse]
        border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
        border_connectivity = self.__class__(border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes)
        return border_connectivity, border_splines, border_unique_to_self_unique_connectivity
    
    # def extract_exterior_surfaces(self, splines):
    #     if self.npa!=3:
    #         raise AssertionError("The parametric space must be 3D to extract surfaces !")
    #     duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
    #     duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
    #     separated_unique_nodes_inds = self.unique_field_indices(())
    #     arr = np.arange(self.npa).tolist()
    #     border_splines = []
    #     border_unique_nodes_inds = []
    #     border_shape_by_patch = []
    #     for i in range(self.nb_patchs):
    #         spline = splines[i]
    #         duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
    #         unique_nodes_inds_spline = separated_unique_nodes_inds[i]
    #         shape_by_patch_spline = self.shape_by_patch[i]
    #         
    #         # surface 1
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=0).all():
    #             bspline_border = BSpline.from_bases(spline.bases[:0:-1])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=0).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[:0:-1][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 1 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 2
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=0).all():
    #             bspline_border = BSpline.from_bases(spline.bases[1:])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=0).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[1:][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 2 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 3
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=1).all():
    #             bspline_border = BSpline.from_bases(spline.bases[::2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=1).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[::2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 3 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 4
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=1).all():
    #             bspline_border = BSpline.from_bases(spline.bases[::-2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=1).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[::-2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 4 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 5
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=2).all():
    #             bspline_border = BSpline.from_bases(spline.bases[1::-1])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=2).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[1::-1][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 5 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 6
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=2).all():
    #             bspline_border = BSpline.from_bases(spline.bases[:2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=2).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[:2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 6 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #     border_splines = np.array(border_splines, dtype='object')
    #     border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
    #     border_shape_by_patch = np.concatenate(border_shape_by_patch)
    #     border_unique_to_self_unique_connectivity, inverse = np.unique(border_unique_nodes_inds, return_inverse=True)
    #     border_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity))) - 1)[inverse]
    #     border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
    #     border_connectivity = self.__class__(border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes)
    #     return border_connectivity, border_splines, border_unique_to_self_unique_connectivity
    
    def subset(self, splines, patches_to_keep):
        new_splines = splines[patches_to_keep]
        separated_unique_nodes_inds = self.unique_field_indices(())
        new_unique_nodes_inds = np.concatenate([separated_unique_nodes_inds[patch].flat for patch in patches_to_keep])
        new_shape_by_patch = self.shape_by_patch[patches_to_keep]
        new_unique_to_self_unique_connectivity, inverse = np.unique(new_unique_nodes_inds, return_inverse=True)
        new_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], new_unique_to_self_unique_connectivity))) - 1)[inverse]
        new_nb_unique_nodes = np.unique(new_unique_nodes_inds).size
        new_connectivity = self.__class__(new_unique_nodes_inds, new_shape_by_patch, new_nb_unique_nodes)
        return new_connectivity, new_splines, new_unique_to_self_unique_connectivity
    
    def save_paraview(self, splines, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.npa
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]
        
        def process_patch(patch):
            groups = {"interior": {"ext": "vtu", "npart": patch, "nstep": n_step}, 
                      "elements_borders": {"ext": "vtu", "npart": patch, "nstep": n_step}, 
                      "control_points": {"ext": "vtu", "npart": patch, "nstep": n_step}}
            splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                        path, 
                                        name, 
                                        n_step=n_step, 
                                        n_eval_per_elem=n_eval_per_elem, 
                                        fields=separated_fields[patch], 
                                        groups=groups, 
                                        make_pvd=False, 
                                        verbose=False)
        
        if verbose:
            with tqdm(total=self.nb_patchs) as pbar:
                with ThreadPoolExecutor() as executor:
                    threads = [executor.submit(process_patch, patch) for patch in range(self.nb_patchs)]
                    for thread in as_completed(threads):
                        thread.result()
                        pbar.update(1)
        else:
            with ThreadPoolExecutor() as executor:
                executor.map(process_patch, range(self.nb_patchs))
        
        groups = {"interior": {"ext": "vtu", "npart": self.nb_patchs, "nstep": n_step}, 
                  "elements_borders": {"ext": "vtu", "npart": self.nb_patchs, "nstep": n_step}, 
                  "control_points": {"ext": "vtu", "npart": self.nb_patchs, "nstep": n_step}}
        _writePVD(os.path.join(path, name), groups)
    
    def save_YETI(self, splines, separated_ctrl_pts, path, name):
        if self.npa==2:
            el_type = "U3"
        elif self.npa==3:
            el_type = "U1"
        else:
            raise NotImplementedError("Can only save surfaces or volumes !")
        objects_list = []
        for patch in range(self.nb_patchs):
            geomdl_patch = splines[patch].getGeomdl(separated_ctrl_pts[patch])
            obj = Domain.DefaultDomain(geometry=geomdl_patch,
                                       id_dom=patch,
                                       elem_type=el_type)
            objects_list.append(obj)
        write.write_files(objects_list, os.path.join(path, name))
        

if __name__=='__main__':
    from bsplyne_lib import new_cube
    
    cube1, cube1_ctrlPts = new_cube([0.5, 0.5, 0.5], [0, 0, 1], 1)
    cube2, cube2_ctrlPts = new_cube([1.5, 0.5, 0.5], [1, 0, 0], 1)
    splines = [cube1, cube2]
    separated_ctrlPts = [cube1_ctrlPts, cube2_ctrlPts]
    
    connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrlPts)
    
    border_connectivity, border_splines, border_unique_to_self_unique_connectivity = connectivity.extract_exterior_borders(splines)
    
    print(connectivity.unique_field_indices((1,)))
    print(border_connectivity.unique_field_indices((1,)))
    
    
# %%

if __name__=="__main__":
    import os
    from itertools import permutations
    from functools import lru_cache
    from typing import Union, Iterable

    import numpy as np
    import scipy.sparse as sps
    from tqdm import trange

    from bsplyne import BSplineBasis, BSpline, MultiPatchBSplineConnectivity

class CouplesBSplineBorder:
    
    def __init__(self, spline1_inds, spline2_inds, axes1, axes2, front_sides1, front_sides2, transpose_2_to_1, flip_2_to_1, NPa):
        self.spline1_inds = spline1_inds
        self.spline2_inds = spline2_inds
        self.axes1 = axes1
        self.axes2 = axes2
        self.front_sides1 = front_sides1
        self.front_sides2 = front_sides2
        self.transpose_2_to_1 = transpose_2_to_1
        self.flip_2_to_1 = flip_2_to_1
        self.NPa = NPa
        self.nb_couples = self.flip_2_to_1.shape[0]
    
    @classmethod
    def extract_border_pts(cls, field, axis, front_side, field_dim=1, offset=0):
        npa = field.ndim - field_dim
        base_face = np.hstack((np.arange(axis + 1, npa), np.arange(axis)))
        if not front_side:
            base_face = base_face[::-1]
        border_field = field.transpose(axis + field_dim, *np.arange(field_dim), *(base_face + field_dim))[(-(1 + offset) if front_side else offset)]
        return border_field
    
    @classmethod
    def extract_border_spline(cls, spline, axis, front_side):
        base_face = np.hstack((np.arange(axis + 1, spline.NPa), np.arange(axis)))
        if not front_side:
            base_face = base_face[::-1]
        degrees = spline.getDegrees()
        knots = spline.getKnots()
        border_degrees = [degrees[i] for i in base_face]
        border_knots = [knots[i] for i in base_face]
        border_spline = BSpline(border_degrees, border_knots)
        return border_spline
    
    @classmethod
    def transpose_and_flip(cls, field, transpose, flip, field_dim=1):
        field = field.transpose(*np.arange(field_dim), *(transpose + field_dim))
        for i in range(flip.size):
            if flip[i]:
                field = np.flip(field, axis=(i + field_dim))
        return field
    
    @classmethod
    def transpose_and_flip_knots(cls, knots, spans, transpose, flip):
        new_knots = []
        for i in range(flip.size):
            if flip[i]:
                new_knots.append(sum(spans[i]) - knots[transpose[i]][::-1])
            else:
                new_knots.append(knots[transpose[i]])
        return new_knots
    
    @classmethod
    def transpose_and_flip_back_knots(cls, knots, spans, transpose, flip):
        transpose_back = np.argsort(transpose)
        flip_back = flip[transpose_back]
        return cls.transpose_and_flip_knots(knots, spans, transpose_back, flip_back)
    
    @classmethod
    def transpose_and_flip_spline(cls, spline, transpose, flip):
        spans = spline.getSpans()
        knots = spline.getKnots()
        degrees = spline.getDegrees()
        for i in range(flip.size):
            p = degrees[transpose[i]]
            knot = knots[transpose[i]]
            if flip[i]:
                knot = sum(spans[i]) - knot[::-1]
            spline.bases[i] = BSplineBasis(p, knot)
        return spline
    
    @classmethod
    def from_splines(cls, separated_ctrl_pts, splines):
        NPa = splines[0].NPa
        assert np.all([sp.NPa==NPa for sp in splines]), "Every patch should have the same parametric space dimension !"
        NPh = separated_ctrl_pts[0].shape[0]
        assert np.all([ctrl_pts.shape[0]==NPh for ctrl_pts in separated_ctrl_pts]), "Every patch should have the same physical space dimension !"
        npatch = len(splines)
        all_flip = np.unpackbits(np.arange(2**(NPa - 1), dtype='uint8')[:, None], axis=1, count=(NPa - 1 - 8), bitorder='little')[:, ::-1].astype('bool')
        all_transpose = np.array(list(permutations(np.arange(NPa - 1))))
        spline1_inds = []
        spline2_inds = []
        axes1 = []
        axes2 = []
        front_sides1 = []
        front_sides2 = []
        transpose_2_to_1 = []
        flip_2_to_1 = []
        for spline1_ind in range(npatch):
            spline1 = splines[spline1_ind]
            ctrl_pts1 = separated_ctrl_pts[spline1_ind]
            # print(f"sp1 {spline1_ind}")
            for spline2_ind in range(spline1_ind + 1, npatch):
                spline2 = splines[spline2_ind]
                ctrl_pts2 = separated_ctrl_pts[spline2_ind]
                # print(f"|sp2 {spline2_ind}")
                for axis1 in range(spline1.NPa):
                    degrees1 = np.hstack((spline1.getDegrees()[(axis1 + 1):], spline1.getDegrees()[:axis1]))
                    knots1 = spline1.getKnots()[(axis1 + 1):] + spline1.getKnots()[:axis1]
                    # print(f"||ax1 {axis1}")
                    for axis2 in range(spline2.NPa):
                        degrees2 = np.hstack((spline2.getDegrees()[(axis2 + 1):], spline2.getDegrees()[:axis2]))
                        knots2 = spline2.getKnots()[(axis2 + 1):] + spline2.getKnots()[:axis2]
                        spans2 = spline2.getSpans()[(axis2 + 1):] + spline2.getSpans()[:axis2]
                        # print(f"|||ax2 {axis2}")
                        for front_side1 in [False, True]:
                            pts1 = cls.extract_border_pts(ctrl_pts1, axis1, front_side1)
                            # print(f"||||{'front' if front_side1 else 'back '} side1")
                            for front_side2 in [False, True]:
                                pts2 = cls.extract_border_pts(ctrl_pts2, axis2, front_side2)
                                # print(f"|||||{'front' if front_side2 else 'back '} side2")
                                for transpose in all_transpose:
                                    # print(f"||||||transpose {transpose}")
                                    if (degrees1==[degrees2[i] for i in transpose]).all():
                                        # print(f"||||||same degrees {degrees1}")
                                        if list(pts1.shape[1:])==[pts2.shape[1:][i] for i in transpose]:
                                            # print(f"||||||same shapes {pts1.shape[1:]}")
                                            if np.all([knots1[i].size==knots2[transpose[i]].size for i in range(NPa - 1)]):
                                                # print(f"||||||same knots sizes {[knots1[i].size for i in range(NPa - 1)]}")
                                                for flip in all_flip:
                                                    # print(f"|||||||flip {flip}")
                                                    if np.all([(k1==k2).all() for k1, k2 in zip(knots1, cls.transpose_and_flip_knots(knots2, spans2, transpose, flip))]):
                                                        # print(f"|||||||same knots {knots1}")
                                                        pts2_turned = cls.transpose_and_flip(pts2, transpose, flip)
                                                        if np.allclose(pts1, pts2_turned):
                                                            # print("_________________GOGOGO_________________")
                                                            spline1_inds.append(spline1_ind)
                                                            spline2_inds.append(spline2_ind)
                                                            axes1.append(axis1)
                                                            axes2.append(axis2)
                                                            front_sides1.append(front_side1)
                                                            front_sides2.append(front_side2)
                                                            transpose_2_to_1.append(transpose)
                                                            flip_2_to_1.append(flip)
        spline1_inds = np.array(spline1_inds, dtype='int')
        spline2_inds = np.array(spline2_inds, dtype='int')
        axes1 = np.array(axes1, dtype='int')
        axes2 = np.array(axes2, dtype='int')
        front_sides1 = np.array(front_sides1, dtype='bool')
        front_sides2 = np.array(front_sides2, dtype='bool')
        transpose_2_to_1 = np.array(transpose_2_to_1, dtype='int')
        flip_2_to_1 = np.array(flip_2_to_1, dtype='bool')
        return cls(spline1_inds, spline2_inds, axes1, axes2, front_sides1, front_sides2, transpose_2_to_1, flip_2_to_1, NPa)
    
    def get_operator_allxi1_to_allxi2(self, spans1, spans2, couple_ind):
        ax1 = self.axes1[couple_ind]
        ax2 = self.axes2[couple_ind]
        front1 = self.front_sides1[couple_ind]
        front2 = self.front_sides2[couple_ind]
        transpose = self.transpose_2_to_1[couple_ind]
        flip = self.flip_2_to_1[couple_ind]
        
        A = np.zeros((self.NPa, self.NPa), dtype='float')
        A[ax2, ax1] = -1 if front1==front2 else 1
        arr = np.arange(self.NPa)
        j1 = np.hstack((arr[(ax1 + 1):], arr[:ax1]))
        j2 = np.hstack((arr[(ax2 + 1):], arr[:ax2]))
        A[j2[transpose], j1] = [-1 if f else 1 for f in flip]
        b = np.zeros(self.NPa, dtype='float')
        b[ax2] = (int(front1) + int(front2))*(1 if front2 else -1)
        b[j2[transpose]] = [1 if f else 0 for f in flip]
        
        alpha1, beta1 = np.array(spans1).T
        M1, p1 = np.diag(1/(beta1 - alpha1)), -alpha1/(beta1 - alpha1)
        alpha2, beta2 = np.array(spans2).T
        M2, p2 = np.diag(beta2 - alpha2), alpha2
        b = p2 + M2@b + M2@A@p1
        A = M2@A@M1
        
        return A, b
    
    def get_connectivity(self, shape_by_patch):
        indices = []
        start = 0
        for shape in shape_by_patch:
            end = start + np.prod(shape)
            indices.append(np.arange(start, end).reshape(shape))
            start = end
        nodes_couples = []
        for i in range(self.nb_couples):
            border_inds1 = self.__class__.extract_border_pts(indices[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i], field_dim=0)
            border_inds2 = self.__class__.extract_border_pts(indices[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i], field_dim=0)
            border_inds2_turned_and_fliped = self.__class__.transpose_and_flip(border_inds2, self.transpose_2_to_1[i], self.flip_2_to_1[i], field_dim=0)
            nodes_couples.append(np.hstack((border_inds1.reshape((-1, 1)), border_inds2_turned_and_fliped.reshape((-1, 1)))))
        if len(nodes_couples)>0:
            nodes_couples = np.vstack(nodes_couples)
        return MultiPatchBSplineConnectivity.from_nodes_couples(nodes_couples, shape_by_patch)
    
    def get_borders_couples(self, separated_field, offset=0):
        field_dim = separated_field[0].ndim - self.NPa
        borders1 = []
        borders2_turned_and_fliped = []
        for i in range(self.nb_couples):
            border1 = self.__class__.extract_border_pts(separated_field[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i], offset=offset, field_dim=field_dim)
            borders1.append(border1)
            border2 = self.__class__.extract_border_pts(separated_field[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i], offset=offset, field_dim=field_dim)
            border2_turned_and_fliped = self.__class__.transpose_and_flip(border2, self.transpose_2_to_1[i], self.flip_2_to_1[i], field_dim=field_dim)
            borders2_turned_and_fliped.append(border2_turned_and_fliped)
        return borders1, borders2_turned_and_fliped
    
    def get_borders_couples_splines(self, splines):
        borders1 = []
        borders2_turned_and_fliped = []
        for i in range(self.nb_couples):
            border1 = self.__class__.extract_border_spline(splines[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i])
            borders1.append(border1)
            border2 = self.__class__.extract_border_spline(splines[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i])
            border2_turned_and_fliped = self.__class__.transpose_and_flip_spline(border2, self.transpose_2_to_1[i], self.flip_2_to_1[i])
            borders2_turned_and_fliped.append(border2_turned_and_fliped)
        return borders1, borders2_turned_and_fliped
    
    def compute_border_couple_DN(self, couple_ind: int, splines: list[BSpline], XI1_border: list[np.ndarray], k1: list[int]):
        spline1 = splines[self.spline1_inds[couple_ind]]
        ax1 = self.axes1[couple_ind]
        front1 = self.front_sides1[couple_ind]
        spline2 = splines[self.spline2_inds[couple_ind]]
        ax2 = self.axes2[couple_ind]
        front2 = self.front_sides2[couple_ind]
        XI1 = XI1_border[(self.NPa - 1 - ax1):] + [np.array([spline1.bases[ax1].span[int(front1)]])] + XI1_border[:(self.NPa - 1 - ax1)]
        transpose_back = np.argsort(self.transpose_2_to_1[couple_ind])
        flip_back = self.flip_2_to_1[couple_ind][transpose_back]
        spans = spline2.getSpans()[(ax2 + 1):] + spline2.getSpans()[:ax2]
        XI2_border = [(sum(spans[i]) - XI1_border[transpose_back[i]]) if flip_back[i] else XI1_border[transpose_back[i]] for i in range(self.NPa - 1)]
        XI2 = XI2_border[(self.NPa - 1 - ax2):] + [np.array([spline2.bases[ax2].span[int(front2)]])] + XI2_border[:(self.NPa - 1 - ax2)]
        k2 = k1[:self.axes1[couple_ind]] + k1[(self.axes1[couple_ind] + 1):]
        k2 = [k2[i] for i in transpose_back]
        k2 = k2[:self.axes2[couple_ind]] + [k1[self.axes1[couple_ind]]] + k2[self.axes2[couple_ind]:]
        DN1 = spline1.DN(XI1, k=k1)
        DN2 = spline2.DN(XI2, k=k2)
        return DN1, DN2
    
    def compute_border_couple_DN(self, couple_ind: int, splines: list[BSpline], XI1_border: list[np.ndarray], k1: list[int]):
        spline1 = splines[self.spline1_inds[couple_ind]]
        spans1 = spline1.getSpans()
        ax1 = self.axes1[couple_ind]
        front1 = self.front_sides1[couple_ind]
        XI1 = XI1_border[(self.NPa - 1 - ax1):] + [np.array([spline1.bases[ax1].span[int(front1)]])] + XI1_border[:(self.NPa - 1 - ax1)]
        DN1 = spline1.DN(XI1, k=k1)
        
        spline2 = splines[self.spline2_inds[couple_ind]]
        spans2 = spline2.getSpans()
        ax2 = self.axes2[couple_ind]
        front2 = self.front_sides2[couple_ind]
        transpose = self.transpose_2_to_1[couple_ind]
        A, b = self.get_operator_allxi1_to_allxi2(spans1, spans2, couple_ind)
        XI2 = []
        for i in range(self.NPa):
            j = np.argmax(np.abs(A[i]))
            XI2.append(A[i, j]*XI1[j] + b[i])
        
        k = int(sum(k1))
        DN2 = spline2.DN(XI2, k=k)
        if k!=0:
            AT = 1
            for i in range(k):
                AT = np.tensordot(AT, A, 0)
            AT = AT.transpose(*2*np.arange(k), *(2*np.arange(k) + 1))
            DN2 = np.tensordot(DN2, AT, k)
            i1 = np.repeat(np.arange(self.NPa), k1)
            DN2 = DN2[tuple(i1.tolist())]
        
        # if isinstance(k1, int):
        #     DN2 = spline2.DN(XI2, k=k1)
        #     if k1!=0:
        #         AT = 1
        #         for i in range(k1):
        #             AT = np.tensordot(AT, A, 0)
        #         AT = AT.transpose(*2*np.arange(k1), *(2*np.arange(k1) + 1))
        #         DN2 = np.tensordot(DN2, AT, k1)
        # else:
        #     k2 = np.array(k1[:ax1] + k1[(ax1 + 1):], dtype='int')
        #     k2[transpose] = k2
        #     k2 = np.hstack((k2[:ax2], [k1[ax1]], k2[ax2:]))
        #     print(k1, k2)
        #     DN2 = spline2.DN(XI2, k=k2)
        #     i1 = np.repeat(np.arange(self.NPa), k1)
        #     i2 = np.repeat(np.arange(self.NPa), k2)
        #     print(np.prod(A[i2, i1]))
        #     k1 = int(sum(k1))
        #     AT = 1
        #     for i in range(k1):
        #         AT = np.tensordot(AT, A, 0)
        #     AT = AT.transpose(*2*np.arange(k1), *(2*np.arange(k1) + 1))
        #     print(AT[tuple(i2.tolist() + i1.tolist())])
        return DN1, DN2


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from bsplyne import BSpline
    np.set_printoptions(suppress=True)

    spl1 = BSpline([2, 2], [np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1])])
    ctrl1 = np.array(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing='ij'))
    spl2 = BSpline([2, 2], [np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1])])
    ctrl2 = np.array(np.meshgrid(np.linspace(2, 1, 4), np.linspace(1, 0, 4), indexing='ij')).transpose(0, 2, 1)
    fig, ax = plt.subplots()
    spl1.plotMPL(ctrl1, ax=ax)
    plt.arrow(ctrl1[0, 0, 0], ctrl1[1, 0, 0], 0.5*(ctrl1[0, 1, 0] - ctrl1[0, 0, 0]), 0.5*(ctrl1[1, 1, 0] - ctrl1[1, 0, 0]), width=0.02, zorder=2)
    plt.annotate(r"$\xi$", (ctrl1[0, 1, 0], ctrl1[1, 1, 0]), zorder=3)
    plt.arrow(ctrl1[0, 0, 0], ctrl1[1, 0, 0], 0.5*(ctrl1[0, 0, 1] - ctrl1[0, 0, 0]), 0.5*(ctrl1[1, 0, 1] - ctrl1[1, 0, 0]), width=0.02, zorder=2)
    plt.annotate(r"$\eta$", (ctrl1[0, 0, 1], ctrl1[1, 0, 1]), zorder=3)
    spl2.plotMPL(ctrl2, ax=ax)
    plt.arrow(ctrl2[0, 0, 0], ctrl2[1, 0, 0], 0.5*(ctrl2[0, 1, 0] - ctrl2[0, 0, 0]), 0.5*(ctrl2[1, 1, 0] - ctrl2[1, 0, 0]), width=0.02, zorder=2)
    plt.annotate(r"$\xi$", (ctrl2[0, 1, 0], ctrl2[1, 1, 0]), zorder=3)
    plt.arrow(ctrl2[0, 0, 0], ctrl2[1, 0, 0], 0.5*(ctrl2[0, 0, 1] - ctrl2[0, 0, 0]), 0.5*(ctrl2[1, 0, 1] - ctrl2[1, 0, 0]), width=0.02, zorder=2)
    plt.annotate(r"$\eta$", (ctrl2[0, 0, 1], ctrl2[1, 0, 1]), zorder=3)
    ctrls = [ctrl1, ctrl2]
    spls = [spl1, spl2]

    cpl = CouplesBSplineBorder.from_splines(ctrls, spls)
    conn = cpl.get_connectivity(np.array([[4, 4], [4, 4]]))
    [border], _ = cpl.get_borders_couples_splines(spls)
    XI, dXI = border.gauss_legendre_for_integration([1])
    XI = (np.array([0.49, 0.75]),)
    N1, N2 = cpl.compute_border_couple_DN(0, spls, list(XI), k1=[2, 0])
    print(ctrl1.reshape((2, -1))@N1.T, ctrl2.reshape((2, -1))@N2.T)
    print(N1.A, N2.A, sep="\n")
    
    cpl.get_operator_allxi1_to_allxi2(spl1.getSpans(), spl2.getSpans(), 0)
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):

        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)
            
        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)

    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        '''Add an 3d arrow to an `Axes3D` instance.'''

        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)


    setattr(Axes3D, 'arrow3D', _arrow3D)

    spl1 = BSpline([2, 2, 2], [np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1])])
    ctrl1 = np.array(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing='ij'))
    spl2 = BSpline([2, 2, 2], [np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1]), np.array([0, 0, 0, 0.5, 1, 1, 1])])
    ctrl2 = np.array(np.meshgrid(np.linspace(1, 0, 4), np.linspace(2, 1, 4), np.linspace(0, 1, 4), indexing='ij')).transpose(0, 1, 3, 2)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_artist(Arrow3D(*ctrl1[:, 0, 0, 0], *(0.5*(ctrl1[:, 1, 0, 0] - ctrl1[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl1[:, 1, 0, 0], r"$\xi_1$", ha="center", va="center", zorder=3)
    ax.add_artist(Arrow3D(*ctrl1[:, 0, 0, 0], *(0.5*(ctrl1[:, 0, 1, 0] - ctrl1[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl1[:, 0, 1, 0], r"$\eta_1$", ha="center", va="center", zorder=3)
    ax.add_artist(Arrow3D(*ctrl1[:, 0, 0, 0], *(0.5*(ctrl1[:, 0, 0, 1] - ctrl1[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl1[:, 0, 0, 1], r"$\zeta_1$", ha="center", va="center", zorder=3)
    ax.add_artist(Arrow3D(*ctrl2[:, 0, 0, 0], *(0.5*(ctrl2[:, 1, 0, 0] - ctrl2[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl2[:, 1, 0, 0], r"$\xi_2$", ha="center", va="center", zorder=3)
    ax.add_artist(Arrow3D(*ctrl2[:, 0, 0, 0], *(0.5*(ctrl2[:, 0, 1, 0] - ctrl2[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl2[:, 0, 1, 0], r"$\eta_2$", ha="center", va="center", zorder=3)
    ax.add_artist(Arrow3D(*ctrl2[:, 0, 0, 0], *(0.5*(ctrl2[:, 0, 0, 1] - ctrl2[:, 0, 0, 0])), mutation_scale=20, lw=1, arrowstyle="->"))
    ax.text(*ctrl2[:, 0, 0, 1], r"$\zeta_2$", ha="center", va="center", zorder=3)
    ax.set_xlim3d(-0.1, 1.1)
    ax.set_ylim3d(-0.1, 2.1)
    ax.set_zlim3d(-0.1, 1.1)
    ax.view_init(elev=30, azim=54)
    ax.set_aspect('equal')
    ctrls = [ctrl1, ctrl2]
    spls = [spl1, spl2]
    cpl = CouplesBSplineBorder.from_splines(ctrls, spls)
    print(cpl.__dict__)

    A, b = cpl.get_operator_allxi1_to_allxi2(spl1.getSpans(), spl2.getSpans(), 0)
    print(A, b, sep="\n")
    xi1 = ctrl1[tuple([slice(None)] + [(-1 if cpl.front_sides1[0] else 0) if i==cpl.axes1[0] else slice(None) for i in range(cpl.NPa)])]
    xi2 = ctrl2[tuple([slice(None)] + [(-1 if cpl.front_sides2[0] else 0) if i==cpl.axes2[0] else slice(None) for i in range(cpl.NPa)])]
    (A@xi1.reshape((3, -1)) + b[:, None] - xi2.reshape((3, -1))).reshape(xi1.shape)

# %%

if __name__=="__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from bsplyne import BSpline, MultiPatchBSplineConnectivity
    
    def plot_multipatch(ddl, connectivity):
        if ddl.shape[0]==2:
            fig, ax = plt.subplots()
            line_col = LineCollection
        elif ddl.shape[0]==3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            line_col = Line3DCollection
        else:
            raise NotImplementedError(f"Physical space must be 2D or 3D, not {ddl.shape[0]}D !")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        pts = connectivity.separate(connectivity.unpack(ddl))
        indices = connectivity.unique_field_indices(())
        for i, (p, txt) in enumerate(zip(pts[::-1], indices[::-1])):
            if 1==1:
                ax.scatter(*p)
                for j, t in enumerate(txt.flat):
                    ax.text(*[x.flat[j] for x in p], t, size=20, zorder=10, color='k')
                p = np.moveaxis(p, 0, -1)
                for _ in range(connectivity.npa):
                    ax.add_collection(line_col(p.reshape((-1, p.shape[-2], p.shape[-1])), colors=colors[i]))
                    p = np.moveaxis(p, 0, -2)
        plt.show()
    
    pts1 = np.array(np.meshgrid([0, 1], [0, 1, 2], [0, 1], indexing='ij')).transpose(0, 3, 2, 1)
    # pts1 = np.concatenate((pts1, np.zeros_like(pts1[0])[None]))
    sp1 = BSpline([1, 2, 1], [np.array([0, 0, 1, 1], dtype='float'), np.array([0, 0, 0, 1, 1, 1], dtype='float'), np.array([0, 0, 1, 1], dtype='float')])
    pts2 = np.array(np.meshgrid([0, 2], [0, 1, 2], [0, 1], indexing='ij'))
    # pts2 = np.concatenate((pts2, np.zeros_like(pts2[0])[None]))
    sp2 = BSpline([1, 2, 1], [np.array([0, 0, 1, 1], dtype='float'), np.array([0, 0, 0, 1, 1, 1], dtype='float'), np.array([0, 0, 1, 1], dtype='float')])
    pts = [pts1, pts2]
    splines = [sp1, sp2]
    
    couples = CouplesBSplineBorder.from_splines(pts, splines)
    shape_by_patch = np.array([p.shape[1:] for p in pts], dtype='int')
    connectivity = couples.get_connectivity(shape_by_patch)
    
    # connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(pts)
    
    ddl = connectivity.pack(connectivity.agglomerate(pts))
    
    plot_multipatch(ddl, connectivity)

# %%


# TODO : extract border, displace border, DN multipatch, knot insertion, order elevation
class MultiPatchBSpline:
    
    def __init__(self, splines, couples=None, connectivity=None):
        self.splines = splines
        self.npatch = len(self.splines)
        assert np.all([s.NPa==self.splines[0].NPa for s in self.splines[1:]]), "The parametric space should be of the same dimension on every patch"
        self.NPa = self.splines[0].NPa
        # assert np.all([s.NPh==self.splines[0].NPh for s in self.splines[1:]]), "The physical space should be of the same dimension on every patch"
        # self.NPh = self.splines[0].NPh
        # if couples is None:
        #     self.couples = CouplesBSplineBorder.from_splines(splines)
        # else:
        #     self.couples = couples
        if connectivity is None:
            self.connectivity = self.couples.get_connectivity(self.splines)
        else:
            self.connectivity = connectivity
    
    def get_border(self):
        if self.NPa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_coo_mask = self.get_duplicate_coo_mask()
        border_splines = []
        border_coo_inds = []
        ind = 0
        for i in range(self.npatch):
            s = self.splines[i]
            degrees = s.getDegrees()
            knots = np.array(s.getKnots(), dtype='object')
            coo = s.ctrlPts
            next_ind = ind + self.coo_sizes[i]
            coo_inds_s = self.coo_inds[ind:next_ind].reshape(coo.shape)
            duplicate_coo_mask_s = duplicate_coo_mask[ind:next_ind].reshape(coo.shape)
            ind = next_ind
            for axis in range(self.NPa):
                d = np.delete(degrees, axis)
                k = np.delete(knots, axis, axis=0)
                for side in range(2):
                    if not np.take(duplicate_coo_mask_s, -side, axis=(axis+1)).all():
                        border_splines.append(BSpline(np.take(coo, -side, axis=(axis+1)), d, k))
                        border_coo_inds.append(np.take(coo_inds_s, -side, axis=(axis+1)).flat)
        border_splines = np.array(border_splines, dtype='object')
        border_coo_inds = np.concatenate(border_coo_inds)
        border = self.__class__(border_splines, border_coo_inds, self.ndof)
        return border
    
    def move_border(self):
        # TODO with IDW
        raise NotImplementedError()
    
    def save_paraview(self, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if verbose:
            iterator = trange(self.npatch, desc=("Saving " + name))
        else:
            iterator = range(self.npatch)
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.npatch)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.connectivity.separate(self.connectivity.unpack(value))
                for patch in range(self.npatch):
                    separated_fields[patch][key] = separated_value[patch]
        
        groups = {}
        for patch in iterator:
            groups = self.splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                                      path, 
                                                      name, 
                                                      n_step=n_step, 
                                                      n_eval_per_elem=n_eval_per_elem, 
                                                      fields=separated_fields[patch], 
                                                      groups=groups, 
                                                      make_pvd=((patch + 1)==self.npatch), 
                                                      verbose=False)
    
    def save_stl(self):
        raise NotImplementedError()

if __name__=="__main__":
    from bsplyne_lib import BSpline, new_cube

    cube = new_cube([0.5, 0.5, 0.5], [0, 0, 1], 1)
    cube_degrees = cube.getDegrees()
    cube_knots = cube.getKnots()
    orientation = np.ones(3, dtype='float')/np.sqrt(3)
    length = 10
    splines = [cube]
    for axis in range(3):
        to_extrude = np.take(cube.ctrlPts, -1, axis=(axis + 1))
        ctrlPts = np.concatenate((to_extrude[:, None], 
                                (to_extrude + length*orientation[:, None, None])[:, None]), axis=1)
        degrees = np.array([1] + [cube_degrees[i] for i in range(3) if i!=axis], dtype='int')
        knots = [np.array([0, 0, 1, 1], dtype='float')] + [cube_knots[i] for i in range(3) if i!=axis]
        splines.append(BSpline(ctrlPts, degrees, knots))
    splines = np.array(splines, dtype='object')
    connectivity = MultiPatchBSplineConnectivity.from_splines(splines)
    couples = None
    volume = MultiPatchBSpline(splines, couples, connectivity)
    dof = volume.connectivity.get_dof(volume.splines)
    U_pts = volume.connectivity.dof_to_pts(dof + 1*np.random.rand(dof.size))
    fields = {"U": U_pts[None]}
    volume.save_paraview("./out_tests", "MultiPatch", fields=fields)


# %%

if __name__=="__main__":
    from bsplyne_lib import new_cube
    from bsplyne_lib.geometries_in_3D import _rotation_matrix
    
    length = 2.
    C1 = new_cube([length/4, length/4, length/4], [0, 0, 1], length/2)
    C2 = new_cube([length/4, length/4, 3*length/4], [0, 0, 1], length/2)
    center2 = np.array([length/4, length/4, 3*length/4])[:, None, None, None]
    C2.ctrlPts = np.tensordot(_rotation_matrix([0, 0, 1], np.pi*0/2), C2.ctrlPts - center2, 1) + center2
    splines = np.array([C1, C2], dtype='object')
    to_insert = [np.array([0.25, 0.5, 0.75], dtype='float'), 
                 np.array([0.25, 0.5, 0.75], dtype='float'), 
                 np.array([0.25, 0.5, 0.75], dtype='float')]
    to_elevate = [1, 1, 1]
    for sp in splines:
        sp.orderElevation(to_elevate)
        sp.knotInsertion(to_insert)
    volume = MultiPatchBSpline(splines)
    print(volume.splines[0].ctrlPts.shape)
    print(volume.couples.spline1_inds, 
          volume.couples.spline2_inds, 
          volume.couples.axes1, 
          volume.couples.axes2, 
          volume.couples.front_sides1, 
          volume.couples.front_sides2, 
          volume.couples.transpose_2_to_1, 
          volume.couples.flip_2_to_1)
    print(volume.connectivity.coo_inds)
    dof = volume.connectivity.get_dof(volume.splines)
    U_pts = volume.connectivity.dof_to_pts(dof + 1 + 0.1*np.random.rand(dof.size))
    fields = {"U": U_pts[None]}
    volume.save_paraview("./out_tests", "MultiPatch", fields=fields)
    

# %%

if __name__=="__main__":
    import numpy as np
    from stl import mesh
    from bsplyne_lib import new_quarter_strut

    spline = new_quarter_strut([0, 0, 0], [0, 0, 1], 1, 10)

    tri = []
    XI = spline.linspace(n_eval_per_elem=[10, 1, 100])
    shape = [xi.size for xi in XI]
    for axis in range(3):
        XI_axis = [xi for xi in XI]
        shape_axis = [shape[i] for i in range(len(shape)) if i!=axis]
        XI_axis[axis] = np.zeros(1)
        pts_l = spline(tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        XI_axis[axis] = np.ones(1)
        pts_r = spline(tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        for pts in [pts_l, pts_r]:
            A = pts[:,  :-1,  :-1].reshape((3, -1)).T[:, None, :]
            B = pts[:,  :-1, 1:  ].reshape((3, -1)).T[:, None, :]
            C = pts[:, 1:  ,  :-1].reshape((3, -1)).T[:, None, :]
            D = pts[:, 1:  , 1:  ].reshape((3, -1)).T[:, None, :]
            tri1 = np.concatenate((A, B, C), axis=1)
            tri2 = np.concatenate((D, C, B), axis=1)
            tri.append(np.concatenate((tri1, tri2), axis=0))
    tri = np.concatenate(tri, axis=0)
    data = np.empty(tri.shape[0], dtype=mesh.Mesh.dtype)
    data['vectors'] = tri
    m = mesh.Mesh(data, remove_empty_areas=True)

    m.save('new_stl_file.stl')

# %%

class MultiPatchBSpline:
    
    def __init__(self, splines, connectivity):
        self.splines = splines
        self.npatch = len(self.splines)
        assert np.all([s.NPa==self.splines[0].NPa for s in self.splines[1:]]), "The parametric space should be of the same dimension on every patch"
        self.NPa = self.splines[0].NPa
        assert np.all([s.NPh==self.splines[0].NPh for s in self.splines[1:]]), "The physical space should be of the same dimension on every patch"
        self.NPh = self.splines[0].NPh
        self.connectivity = connectivity
    
    def get_border(self):
        if self.NPa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_coo_mask = self.get_duplicate_coo_mask()
        border_splines = []
        border_coo_inds = []
        ind = 0
        for i in range(self.npatch):
            s = self.splines[i]
            degrees = s.getDegrees()
            knots = np.array(s.getKnots(), dtype='object')
            coo = s.ctrlPts
            next_ind = ind + self.coo_sizes[i]
            coo_inds_s = self.coo_inds[ind:next_ind].reshape(coo.shape)
            duplicate_coo_mask_s = duplicate_coo_mask[ind:next_ind].reshape(coo.shape)
            ind = next_ind
            for axis in range(self.NPa):
                d = np.delete(degrees, axis)
                k = np.delete(knots, axis, axis=0)
                for side in range(2):
                    if not np.take(duplicate_coo_mask_s, -side, axis=(axis+1)).all():
                        border_splines.append(BSpline(np.take(coo, -side, axis=(axis+1)), d, k))
                        border_coo_inds.append(np.take(coo_inds_s, -side, axis=(axis+1)).flat)
        border_splines = np.array(border_splines, dtype='object')
        border_coo_inds = np.concatenate(border_coo_inds)
        border = self.__class__(border_splines, border_coo_inds, self.ndof)
        return border
    
    def move_border(self):
        # TODO with IDW
        raise NotImplementedError()
    
    def save_paraview(self, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if verbose:
            iterator = trange(self.npatch, desc=("Saving " + name))
        else:
            iterator = range(self.npatch)
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.npatch)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.connectivity.separate(self.connectivity.unpack(value))
                for patch in range(self.npatch):
                    separated_fields[patch][key] = separated_value[patch]
        
        groups = {}
        for patch in iterator:
            groups = self.splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                                      path, 
                                                      name, 
                                                      n_step=n_step, 
                                                      n_eval_per_elem=n_eval_per_elem, 
                                                      fields=separated_fields[patch], 
                                                      groups=groups, 
                                                      make_pvd=((patch + 1)==self.npatch), 
                                                      verbose=False)
    
    def save_stl(self):
        raise NotImplementedError()