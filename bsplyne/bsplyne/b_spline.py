import os
import re
from typing import Iterable, Union
import xml.dom.minidom

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from wide_product import wide_product
import meshio as io

from bsplyne.b_spline_basis import BSplineBasis

class BSpline:
    """
    B-Spline from a `NPa`-D parametric space into a NPh-D physical space.
    NPh is not an attribute of this class.

    Attributes
    ----------
    NPa : int
        Parametric space dimension.
    bases : numpy.array of BSplineBasis
        `numpy`.`array` containing a `BSplineBasis` instance for each of the 
        `NPa` axis of the parametric space.

    """
    
    def __init__(self, degrees, knots):
        """
        
        Parameters
        ----------
        degrees : numpy.array of int
            Contains the degrees of the B-spline in each parametric dimension.
        knots : list of numpy.array of float
            Contains the knot vectors of the B-spline for each parametric 
            dimension.

        Returns
        -------
        BSpline : BSpline instance
            Contains the ``BSpline`` object created.

        Examples
        --------
        Creation of a 2D shape as a `BSpline` instance :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> BSpline(degrees, knots)

        """
        self.NPa = len(degrees)
        self.bases = np.empty(self.NPa, dtype='object')
        for idx in range(self.NPa):
            p = degrees[idx]
            knot = knots[idx]
            self.bases[idx] = BSplineBasis(p, knot)
    
    @classmethod
    def from_bases(cls, bases):
        """
        Create a ``BSpline`` instance from an array of ``BSplineBasis``.

        Parameters
        ----------
        bases : numpy.ndarray of BSplineBasis
            The array of ``BSplineBasis`` instances.

        Returns
        -------
        BSpline
            Contains the ``BSpline`` object created.
        """
        self = cls([], [])
        self.bases = bases
        self.NPa = self.bases.size
        return self
    
    def getDegrees(self):
        """
        Returns the degree of each basis in the parametric space.

        Parameters
        ----------
        None

        Returns
        -------
        degrees : numpy.array of int
            Contains the degrees of the B-spline.

        Examples
        --------
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getDegrees()
        array([2, 2])

        """
        degrees = np.array([basis.p for basis in self.bases], dtype='int')
        return degrees
    
    def getKnots(self):
        """
        Returns the knot vector of each basis in the parametric space.

        Parameters
        ----------
        None

        Returns
        -------
        knots : list of numpy.array of float
            Contains the knot vectors of the B-spline.

        Examples
        --------
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getKnots()
        [array([0. , 0. , 0. , 0.5, 1. , 1. , 1. ]),
         array([0. , 0. , 0. , 0.5, 1. , 1. , 1. ])]

        """
        knots = [basis.knot for basis in self.bases]
        return knots

    def getNbFunc(self):
        """
        Compute the number of basis functions of the spline.

        Returns
        -------
        int
            Number of basis functions.
        """
        return np.prod([basis.n + 1 for basis in self.bases])

    def getSpans(self):
        """
        Return the span of each basis in the parametric space.

        Parameters
        ----------
        None

        Returns
        -------
        spans : list of tuple(float, float)
            Contains the span of the B-spline.
        """
        spans = [basis.span for basis in self.bases]
        return spans
    
#     def get_indices(self, begining=0):
#         """
#         Create an array containing the indices of the control points of 
#         the B-spline.
# 
#         Parameters
#         ----------
#         begining : int, optional
#             First index of the arrayof indices, by default 0
# 
#         Returns
#         -------
#         indices : np.array of int
#             Indices of the control points in the same shape as the 
#             control points.
#         """
#         indices = np.arange(begining, begining + self.ctrlPts.size).reshape(self.ctrlPts.shape)
#         return indices

    def linspace(self, n_eval_per_elem=10):
        """
        Generate `NPa` sets of xi values over the span of each basis.

        Parameters
        ----------
        n_eval_per_elem : numpy.array of int or int, optional
            Number of values per element over each parametric axis, by default 10

        Returns
        -------
        XI : tuple of numpy.array of float
            Set of xi values over each span.
        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa # type: ignore
        XI = tuple([basis.linspace(n) for basis, n in zip(self.bases, n_eval_per_elem)]) # type: ignore
        return XI

    def linspace_for_integration(self, n_eval_per_elem=10, bounding_box=None):
        """
        Generate `NPa` sets of xi values over the span of the basis, 
        centerered on intervals of returned lengths.

        Parameters
        ----------
        n_eval_per_elem : numpy.array of int or int, optional
            Number of values per element over each parametric axis, by default 10
        bounding_box : numpy.array of float , optional
            Lower and upper bounds on each axis, by default [[xi0, xin], [eta0, etan], ...]

        Returns
        -------
        XI : tuple of numpy.array of float
            Set of xi values over each span.
        dXI : tuple of numpy.array of float
            Set of integration weight of each point in `XI`.
        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa # type: ignore
        if bounding_box is None:
            bounding_box = [b.span for b in self.bases] # type: ignore
        XI = []
        dXI = []
        for basis, (n, bb) in zip(self.bases, zip(n_eval_per_elem, bounding_box)): # type: ignore
            xi, dxi = basis.linspace_for_integration(n, bb)
            XI.append(xi)
            dXI.append(dxi)
        XI = tuple(XI)
        dXI = tuple(dXI)
        return XI, dXI
    
    def gauss_legendre_for_integration(self, n_eval_per_elem=None, bounding_box=None):
        """
        Generate a set of xi values with their coresponding weight acording to the Gauss Legendre 
        integration method over a given bounding box.

        Parameters
        ----------
        n_eval_per_elem : numpy.array of int, optional
            Number of values per element over each parametric axis, by default `self.getDegrees() + 1`
        bounding_box : numpy.array of float, optional
            Lower and upper bounds, by default `self`.`span`

        Returns
        -------
        XI : tuple of numpy.array of float
            Set of xi values over each span.
        dXI : tuple of numpy.array of float
            Set of integration weight of each point in `XI`.
        """
        if n_eval_per_elem is None:
            n_eval_per_elem = self.getDegrees() + 1
        if bounding_box is None:
            bounding_box = [None]*self.NPa # type: ignore
        XI = []
        dXI = []
        for basis, (n_eval_per_elem_axis, bb) in zip(self.bases, zip(n_eval_per_elem, bounding_box)): # type: ignore
            xi, dxi = basis.gauss_legendre_for_integration(n_eval_per_elem_axis, bb)
            XI.append(xi)
            dXI.append(dxi)
        XI = tuple(XI)
        dXI = tuple(dXI)
        return XI, dXI
    
    def normalize_knots(self):
        """
        Maps the knots vectors to [0, 1].
        """
        for basis in self.bases:
            basis.normalize_knots()
    
    def DN(self, XI, k=0):
        """
        Compute the `k`-th derivative of the B-spline basis at the points 
        in the parametric space given as input such that a dot product 
        with the reshaped and transposed control points evaluates the 
        B-spline.

        Parameters
        ----------
        XI : numpy.array of float or tuple of numpy.array of float
            If `numpy`.`array` of `float`, contains the `NPa`-uplets of 
            parametric coordinates as [[xi_0, ...], [eta_a, ...], ...].
            Else, if `tuple` of `numpy`.`array` of `float`, contains the `NPa` 
            parametric coordinates as [[xi_0, ...], [eta_0, ...], ...].
        k : list of int or int, optional
            If `numpy`.`array` of `int`, or if k is 0, compute the `k`-th 
            derivative of the B-spline basis evaluated on each axis of the 
            parametric space.
            If `int`, compute the `k`-th derivative along every axis. For 
            example, if `k` is 1, compute the gradient, if `k` is 2, compute 
            the hessian, and so on.
            , by default 0

        Returns
        -------
        DN : scipy.sparse.csr_matrix of float or numpy.array of scipy.sparse.csr_matrix of float
            Contains the basis of the B-spline.

        Examples
        --------
        Evaluation of a 2D BSpline basis on these `XI` values : [[0, 0.5], [1, 0]]
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> XI = np.array([[0, 0.5], [1, 0]], dtype='float')
        >>> spline.DN(XI, [0, 0]).A
        array([[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
        
        Evaluation of the 2D BSpline basis's derivative along the first axis :
        >>> XI = (np.array([0, 0.5], dtype='float'), 
                  np.array([1], dtype='float'))
        >>> spline.DN(XI, [1, 0]).A
        array([[ 0., -0., -0., -4.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        """
        
        if isinstance(XI, np.ndarray):
            fct = wide_product
            XI = XI.reshape((self.NPa, -1))
            s1 = XI.shape[1]
        else:
            fct = sps.kron
            s1 = 1
        
        if isinstance(k, int):
            if k==0:
                k = [0]*self.NPa # type: ignore
        
        if isinstance(k, int):
            dkbasis_dxik = np.empty((self.NPa, k + 1), dtype='object')
            for idx in range(self.NPa):
                basis = self.bases[idx]
                xi = XI[idx] - np.finfo('float').eps * (XI[idx]==basis.knot[-1])
                for k_querry in range(k + 1):
                    dkbasis_dxik[idx, k_querry] = basis.N(xi, k=k_querry)
            DN = np.empty([self.NPa]*k, dtype='object')
            dic = {}
            for axes in np.ndindex(*DN.shape):
                u, c = np.unique(axes, return_counts=True)
                k_arr = np.zeros(self.NPa, dtype='int')
                k_arr[u] = c
                key = tuple(k_arr)
                if key not in dic:
                    dic[key] = np.ones((s1, 1))
                    for idx in range(self.NPa):
                        k_querry = k_arr[idx]
                        dic[key] = fct(dic[key], dkbasis_dxik[idx, k_querry])
                DN[axes] = dic[key]
            return DN
        else:
            DN = np.ones((s1, 1))
            for idx in range(self.NPa):
                basis = self.bases[idx]
                k_idx = k[idx]
                xi = XI[idx] - np.finfo('float').eps * (XI[idx]==basis.knot[-1])
                DN_elem = basis.N(xi, k=k_idx)
                DN = fct(DN, DN_elem)
            return DN
    
    def __call__(self, ctrlPts, XI, k=0):
        """
        Evaluate the `k`-th derivative of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        XI : numpy.array of float or tuple of numpy.array of float
            If `numpy`.`array` of `float`, contains the `NPa`-uplets of 
            parametric coordinates as [[xi_0, ...], [eta_a, ...], ...].
            Else, if `tuple` of `numpy`.`array` of `float`, contains the `NPa` 
            parametric coordinates as [[xi_0, ...], [eta_0, ...], ...].
        k : numpy.array of int or int, optional
            If `numpy`.`array` of `int`, or if k is 0, compute the `k`-th 
            derivative of the B-spline evaluated on each axis of the parametric 
            space.
            If `int`, compute the `k`-th derivative along every axis. For 
            example, if `k` is 1, compute the gradient, if `k` is 2, compute 
            the hessian, and so on.
            , by default 0

        Returns
        -------
        values : numpy.array of float
            Evaluation of the `k`-th derivative of the B-spline at the 
            parametric space's coordinates given. This array contains the 
            physical space coordinates corresponding as [X, Y, ...]. 
            If `XI` is a tuple, its shape is 
            (NPh, shape of the derivative, len(xi), len(eta), ...).
            Else, its shape is 
            (NPh, shape of the derivative, *`XI`.shape[1:]).

        Examples
        --------
        Evaluation of a 2D BSpline on a grid :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> XI = (np.array([0, 0.5], dtype='float'), 
                  np.array([1], dtype='float'))
        >>> spline(ctrlPts, XI, [0, 0])
        array([[[0.0247786 ],
                [0.69670203]],
        
               [[0.80417637],
                [0.8896871 ]],
        
               [[0.8208673 ],
                [0.51415427]]])
        
        Evaluation of a 2D BSpline on an array of couples :
        >>> XI = np.array([[0, 0.5], [1, 0]], dtype='float')
        >>> spline(ctrlPts, XI, [0, 0])
        array([[0.0247786 , 0.42344584],
               [0.80417637, 0.42489571],
               [0.8208673 , 0.07327555]])

        """
        if isinstance(XI, np.ndarray):
            XI_shape = XI.shape[1:]
        else:
            XI_shape = [xi.size for xi in XI]
        DN = self.DN(XI, k)
        NPh = ctrlPts.shape[0]
        if isinstance(DN, np.ndarray):
            values = np.empty((*DN.shape, NPh, *XI_shape), dtype='float')
            for axes in np.ndindex(*DN.shape):
                values[axes] = (DN[axes] @ ctrlPts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        else:
            values = (DN @ ctrlPts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        return values
    
    def knotInsertion(self, ctrlPts, knots_to_add: Iterable[Union[npt.NDArray[np.float_], int]]):
        """
        Add the knots passed in parameter to the knot vector and modify the 
        attributes so that the evaluation of the spline stays the same.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        knots_to_add : Iterable[Union[npt.NDArray[np.float_], int]]
            Refinement on each axis :
            If `NDArray`, contains the knots to add on said axis. It must not 
            contain knots outside of the old knot vector's interval.
            If `int`, correspond to the number of knots to add in each B-spline 
            element.

        Returns
        -------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)

        Examples
        --------
        Knot insertion on a 2D BSpline in a 3D space :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> knots_to_add = [np.array([0.5, 0.75], dtype='float'), 
                            np.array([], dtype='float')]
        >>> ctrlPts = pline.knotInsertion(ctrlPts, knots_to_add)

        """
        true_knots_to_add = []
        for axis, knots_to_add_elem in enumerate(knots_to_add):
            if isinstance(knots_to_add_elem, int): # It is a number of knots to add in each element
                u_knot = np.unique(self.bases[axis].knot)
                a, b = u_knot[:-1, None], u_knot[1:, None]
                mu = np.linspace(0, 1, knots_to_add_elem + 1, endpoint=False)[None, 1:]
                true_knots_to_add.append((a + (b - a)*mu).ravel())
            else:
                true_knots_to_add.append(knots_to_add_elem)
        knots_to_add = true_knots_to_add
        
        pts_shape = np.empty(self.NPa, dtype='int')
        D = None
        for idx in range(self.NPa):
            basis = self.bases[idx]
            knots_to_add_elem = knots_to_add[idx]
            D_elem = basis.knotInsertion(knots_to_add_elem)
            pts_shape[idx] = D_elem.shape[0]
            if D is None:
                D = D_elem
            else:
                D = sps.kron(D, D_elem)
        NPh = ctrlPts.shape[0]
        pts = (D @ ctrlPts.reshape((NPh, -1)).T).T
        ctrlPts = pts.reshape((NPh, *pts_shape))
        return ctrlPts
    
    def orderElevation(self, ctrlPts, t):
        """
        Performs the order elevation algorithm on every B-spline basis and 
        apply the changes to the control points of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        t : numpy.array of int
            New degree of each B-spline basis will be its current degree plus 
            the value of `t` corresponding.

        Returns
        -------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)

        Examples
        --------
        Order elevation on a 2D BSpline in a 3D space :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> ctrlPts = spline.knotInsertion(ctrlPts, [1, 0])

        """
        pts_shape = np.empty(self.NPa, dtype='int')
        STD = None
        for idx in range(self.NPa):
            basis = self.bases[idx]
            t_basis = t[idx]
            STD_elem = basis.orderElevation(t_basis)
            pts_shape[idx] = STD_elem.shape[0]
            if STD is None:
                STD = STD_elem
            else:
                STD = sps.kron(STD, STD_elem)
        NPh = ctrlPts.shape[0]
        pts = (STD @ ctrlPts.reshape((NPh, -1)).T).T
        ctrlPts = pts.reshape((NPh, *pts_shape))
        return ctrlPts
    
    def greville_abscissa(self, return_weights=False):
        """
        Compute the Greville abscissa.
        
        Parameters
        ----------
        return_weights : bool, optional
            If `True`, return the weight, the length of the span of the basis 
            function corresponding to each abscissa, by default `False`

        Returns
        -------
        greville : list of np.array of float
            Greville abscissa on each parametric axis.
        weights : list of np.array of float
            Span of each basis function on each parametric axis.
        """
        greville = []
        weights = []
        for idx in range(self.NPa):
            basis = self.bases[idx]
            p = basis.p
            knot = basis.knot
            greville.append(np.convolve(knot[1:-1], np.ones(p, dtype=int), 'valid')/p)
            weights.append(knot[(p+1):] - knot[:-(p+1)])
        if return_weights:
            return greville, weights
        return greville
    
    def _saveControlPolyParaview(self, ctrlPts, file_prefix, n_step, fields={}):
        """
        Saves a paraview file containing all the data to plot the control 
        polygon of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        NPh = ctrlPts.shape[0]
        lines = np.empty((0, 2), dtype='int')
        size = np.prod(ctrlPts.shape[1:])
        inds = np.arange(size).reshape(ctrlPts.shape[1:])
        for idx in range(self.NPa):
            rng = np.arange(inds.shape[idx])
            lines = np.append(lines, 
                              np.concatenate((np.expand_dims(np.take(inds, rng[ :-1], axis=idx), axis=-1), 
                                              np.expand_dims(np.take(inds, rng[1:  ], axis=idx), axis=-1)), 
                                             axis=-1).reshape((-1, 2)), 
                             axis=0)
        cells = {'line': lines}
        points = np.moveaxis(ctrlPts, 0, -1).reshape((-1, NPh))
        greville = tuple(self.greville_abscissa())
        n = self.getNbFunc()
        point_data = {}
        for key, value in fields.items():
            if callable(value):
                point_data[key] = value(self, greville)
            else:
                point_data[key] = value.reshape((n_step, -1, n)).transpose(0, 2, 1)
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def _saveElemSeparatorParaview(self, ctrlPts, n_eval_per_elem, file_prefix, n_step=1, fields={}):
        """
        Saves a paraview file containing all the data to plot the limit of 
        every element of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        n_eval_per_elem : numpy.array of int
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        NPh = ctrlPts.shape[0]
        knots_uniq = []
        shape_uniq = []
        for basis in self.bases:
            knot = basis.knot
            knot_uniq = np.unique(knot[np.logical_and(knot>=0, knot<=1)])
            knots_uniq.append(knot_uniq)
            shape_uniq.append(knot_uniq.size)
        points = None
        point_data = {key: None for key in fields}
        lines = None
        Size = 0
        n = self.getNbFunc()
        for idx in range(self.NPa):
            basis = self.bases[idx]
            knot_uniq = knots_uniq[idx]
            n_eval = n_eval_per_elem[idx]
            couples = np.concatenate((np.expand_dims(knot_uniq[ :-1], axis=0), 
                                      np.expand_dims(knot_uniq[1:  ], axis=0)), 
                                     axis=0).reshape((2, -1)).T
            for a, b in couples:
                lin =  np.linspace(a, b, n_eval)
                XI = tuple(knots_uniq[:idx] + [lin] + knots_uniq[(idx+1):])
                shape = shape_uniq[:idx] + [lin.size] + shape_uniq[(idx+1):]
                N = self.DN(XI, [0]*self.NPa) # type: ignore
                pts = N @ ctrlPts.reshape((NPh, -1)).T
                points = pts if points is None else np.vstack((points, pts))
                for key, value in fields.items():
                    if callable(value):
                        to_store = value(self, XI)
                    else:
                        paraview_size = value.shape[1]
                        arr = value.reshape((n_step, paraview_size, n)).reshape((n_step*paraview_size, n))
                        to_store = (arr @ N.T).reshape((n_step, paraview_size, -1)).transpose(0, 2, 1)
                    if point_data[key] is None:
                        point_data[key] = to_store
                    else:
                        point_data[key] = np.concatenate((point_data[key], to_store), axis=1) # type: ignore
                size = np.prod(shape)
                lns = Size + np.arange(size).reshape(shape)
                lns = np.moveaxis(lns, idx, 0).reshape((shape[idx], -1))
                lns = np.concatenate((np.expand_dims(lns[ :-1], axis=-1), 
                                      np.expand_dims(lns[1:  ], axis=-1)), 
                                     axis=-1).reshape((-1, 2))
                lines = lns if lines is None else np.vstack((lines, lns))
                Size += size
        cells = {'line': lines}
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i] # type: ignore
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def _saveElementsInteriorParaview(self, ctrlPts, n_eval_per_elem, file_prefix, n_step=1, fields={}):
        """
        Saves a paraview file containing all the data to plot the interior of 
        the elements of the B-pline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        n_eval_per_elem : numpy.array of int
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        XI = self.linspace(n_eval_per_elem)
        # make points
        N = self.DN(XI)
        NPh = ctrlPts.shape[0]
        points = N @ ctrlPts.reshape((NPh, -1)).T
        # make connectivity
        elements = {2: "line", 4: "quad", 8: "hexahedron"}
        shape = [xi.size for xi in XI]
        NXI = np.prod(shape)
        inds = np.arange(NXI).reshape(shape)
        for idx in range(self.NPa):
            rng = np.arange(inds.shape[idx])
            inds = np.concatenate((np.expand_dims(np.take(inds, rng[ :-1], axis=idx), axis=-1), 
                                   np.expand_dims(np.take(inds, rng[1:  ], axis=idx), axis=-1)), axis=-1)
        if self.NPa>=2:
            for i in np.ndindex(*inds.shape[:-2]):
                inds[i] = [[inds[i+(0, 0)], inds[i+(0, 1)]], [inds[i+(1, 1)], inds[i+(1, 0)]]]
        inds = inds.reshape((*inds.shape[:self.NPa], -1))
        inds = inds.reshape((-1, inds.shape[-1]))
        cells = {elements[inds.shape[-1]]: inds}
        # make fields
        n = self.getNbFunc()
        point_data = {}
        for key, value in fields.items():
            if callable(value):
                point_data[key] = value(self, XI)
            else:
                paraview_size = value.shape[1]
                arr = value.reshape((n_step*paraview_size, n))
                point_data[key] = (arr @ N.T).reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
        # make fields
        # """
        # fields : dict of function or of tuple(int or numpy.array of int, numpy.array of float), default {}
        #     Fields to plot at each time step. The name of the field will 
        #     be the dict key. 
        #     If the value given is a `function`, it must take the spline 
        #     and a `tuple` of parametric points that could be given to 
        #     `self`.`DN` for example. It must return its value for 
        #     each time step and on each combination of parametric points.
        #     `function`(`BSpline` spline, 
        #                `tuple`(`numpy`.`array` of `float`) XI) 
        #     -> `numpy`.`array` of `float` of shape 
        #     (`n_step`, nb combinations of XI, paraview_size) 
        #     where paraview_size is the size of the vector at each control 
        #     point.
        #     If the value given is a tuple (k, pts) : 
        #     - pts : a `numpy`.`array` of `float` containing the value at 
        #         each control point, the shape must be 
        #         (`n_step`, paraview_size, *`self`.`ctrlPts`.`shape`[1:])
        #     - k : the value(s) of k given to the `DN` method. 
        #         If k is a `numpy`.`array`, its shape should be 
        #         (paraview_size, `self`.`NPa`).
        # """
        # NFct = self.getNbFunc()
        # point_data = {}
        # for key, (k, pts) in fields.items():
        #     if callable(value):
        #         point_data[key] = value(self, XI)
        #     else:
        #         paraview_size = pts.shape[1]
        #         if isinstance(k, int):
        #             if k==0:
        #                 arr = pts.reshape((n_step*paraview_size, NFct))
        #                 point_data[key] = (arr @ N.T).reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
        #             else:
        #                 DkN = self.DN(XI, k).ravel()
        #                 if paraview_size!=DkN.size:
        #                     if paraview_size==1:
        #                         arr = pts.reshape((n_step, NFct))
        #                         paraview_size = DkN.size
        #                         point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                         for i in range(paraview_size):
        #                             point_data[key][:, :, i] = (arr @ DkN[i].T).reshape((n_step, NXI))
        #                     else:
        #                         raise ValueError("paraview_size is " 
        #                                          + str(paraview_size) 
        #                                          + " but " 
        #                                          + str(DkN.size) 
        #                                          + " derivatives were acquiered !")
        #                 else:
        #                     arr = pts.reshape((n_step, paraview_size, NFct))
        #                     point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                     for i in range(paraview_size):
        #                         point_data[key][:, :, i] = (arr[:, i, :] @ DkN[i].T).reshape((n_step, NXI))
        #         else:
        #             DkN = np.array([self.DN(XI, ki) for ki in k], dtype='object')
        #             if paraview_size!=DkN.size:
        #                 if paraview_size==1:
        #                     arr = pts.reshape((n_step, NFct))
        #                     paraview_size = DkN.size
        #                     point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                     for i in range(paraview_size):
        #                         point_data[key][:, :, i] = (arr @ DkN[i].T).reshape((n_step, NXI))
        #                 else:
        #                     raise ValueError("paraview_size is " 
        #                                         + str(paraview_size) 
        #                                         + " but " 
        #                                         + str(DkN.size) 
        #                                         + " derivatives were acquiered !")
        #             else:
        #                 arr = pts.reshape((n_step, paraview_size, NFct))
        #                 point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                 for i in range(paraview_size):
        #                     point_data[key][:, :, i] = (arr[:, i, :] @ DkN[i].T).reshape((n_step, NXI))
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def saveParaview(self, ctrlPts, path, name, n_step=1, n_eval_per_elem=10, fields=None, groups=None, make_pvd=True, verbose=True):
        """
        Saves a plot as a set of .vtu files with a .pvd file.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        path : string
            Path of the directory in which all the files to show in Paraview 
            will be dumped.
        name : string
            Prefix of the files created.
        n_step : int
            Number of time steps to plot.
        n_eval_per_elem : numpy.array of int or int, default 10
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        fields : dict of function or of numpy.array of float, default None
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 
        groups : dict of dict, default None
            `dict` (out) of `dict` (in) as :
            - (out) : 
                * "interior" : (in) type of `dict`, 
                * "elements_borders" : (in) type of `dict`, 
                * "control_points" : (in) type of `dict`.
                * other keys from the input that are not checked
            - (in) : 
                * "ext" : name of the extention of the group, 
                * "npart" : number of parts to plot together,
                * "nstep" : number of time steps.
        make_pvd : bool, default True
            If True, create a PVD file for all the data in `groups`.
        verbose : bool, default True
            If True, print the advancement state to the standard output.

        Returns
        -------
        groups : dict of dict
            `dict` (out) of `dict` (in) as :
            - (out) : 
                * "interior" : (in) type of `dict`, 
                * "elements_borders" : (in) type of `dict`, 
                * "control_points" : (in) type of `dict`.
                * other keys from the input that are not checked
            - (in) : 
                * "ext" : name of the extention of the group, 
                * "npart" : number of parts to plot together,
                * "nstep" : number of time steps.

        Examples
        --------
        Save a 2D BSpline in a 3D space in the file file.pvd at the 
        location /path/to/file :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> spline.saveParaview(ctrlPts, "/path/to/file", "file")

        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if fields is None:
            fields = {}
        
        if groups is None:
            groups = {}
        
        interior = "interior"
        if interior in groups:
            groups[interior]["npart"] += 1
        else:
            groups[interior] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        elements_borders = "elements_borders"
        if elements_borders in groups:
            groups[elements_borders]["npart"] += 1
        else:
            groups[elements_borders] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        control_points = "control_points"
        if control_points in groups:
            groups[control_points]["npart"] += 1
        else:
            groups[control_points] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        
        interior_prefix = os.path.join(path, name+"_"+interior+"_"+str(groups[interior]["npart"] - 1))
        self._saveElementsInteriorParaview(ctrlPts, n_eval_per_elem, interior_prefix, n_step, fields)
        if verbose:
            print(interior, "done")

        elements_borders_prefix = os.path.join(path, name+"_"+elements_borders+"_"+str(groups[elements_borders]["npart"] - 1))
        self._saveElemSeparatorParaview(ctrlPts, n_eval_per_elem, elements_borders_prefix, n_step, fields)
        if verbose:
            print(elements_borders, "done")

        control_points_prefix = os.path.join(path, name+"_"+control_points+"_"+str(groups[control_points]["npart"] - 1))
        self._saveControlPolyParaview(ctrlPts, control_points_prefix, n_step, fields)
        if verbose:
            print(control_points, "done")
        
        if make_pvd:
            _writePVD(os.path.join(path, name), groups)
        
        return groups
    
    def getGeomdl(self, ctrl_pts):
        try:
            from geomdl import BSpline as geomdlBS
        except:
            raise 
        if self.NPa==1:
            curve = geomdlBS.Curve()
            curve.degree = self.bases[0].p
            curve.ctrlpts = ctrl_pts.T.tolist()
            curve.knotvector = self.bases[0].knot
            return curve
        elif self.NPa==2:
            surf = geomdlBS.Surface()
            surf.degree_u = self.bases[0].p
            surf.degree_v = self.bases[1].p
            surf.ctrlpts2d = ctrl_pts.transpose((1, 2, 0)).tolist()
            surf.knotvector_u = self.bases[0].knot
            surf.knotvector_v = self.bases[1].knot
            return surf
        elif self.NPa==3:
            vol = geomdlBS.Volume()
            vol.degree_u = self.bases[0].p
            vol.degree_v = self.bases[1].p
            vol.degree_w = self.bases[2].p
            vol.cpsize = ctrl_pts.shape[1:]
            # ctrl_pts format (zeta, xi, eta)
            vol.ctrlpts = ctrl_pts.transpose(3, 1, 2, 0).reshape((-1, ctrl_pts.shape[0])).tolist()
            vol.knotvector_u = self.bases[0].knot
            vol.knotvector_v = self.bases[1].knot
            vol.knotvector_w = self.bases[2].knot
            return vol
        else:
            raise NotImplementedError("Can only export curves, sufaces or volumes !")
    
    def plotPV(self, ctrl_pts):
        pass
    
    def plotMPL(self, ctrl_pts, n_eval_per_elem=10, ax=None, ctrl_color='#1b9e77', interior_color='#7570b3', elem_color='#666666', border_color='#d95f02'):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib import lines
        NPh = ctrl_pts.shape[0]
        fig = plt.figure() if ax is None else ax.get_figure()
        if NPh==2:
            ax = fig.add_subplot() if ax is None else ax
            if self.NPa==1:
                ax.plot(ctrl_pts[0], ctrl_pts[1], marker="o", c=ctrl_color, label="Control mesh", zorder=0)
                xi, = self.linspace(n_eval_per_elem=n_eval_per_elem)
                x, y = self.__call__(ctrl_pts, [xi])
                ax.plot(x, y, c=interior_color, label="B-spline", zorder=1)
                xi_elem, = self.linspace(n_eval_per_elem=1)
                x_elem, y_elem = self.__call__(ctrl_pts, [xi_elem])
                ax.scatter(x_elem, y_elem, marker='*', c=elem_color, label="Elements borders", zorder=2) # type: ignore
            elif self.NPa==2:
                xi, eta = self.linspace(n_eval_per_elem=n_eval_per_elem)
                xi_elem, eta_elem = self.linspace(n_eval_per_elem=1)
                x_xi, y_xi = self.__call__(ctrl_pts, [xi_elem, eta])
                x_eta, y_eta = self.__call__(ctrl_pts, [xi, eta_elem])
                x_pol = np.hstack((x_xi[ 0, :: 1], x_eta[:: 1, -1], x_xi[-1, ::-1], x_eta[::-1,  0]))
                y_pol = np.hstack((y_xi[ 0, :: 1], y_eta[:: 1, -1], y_xi[-1, ::-1], y_eta[::-1,  0]))
                xy_pol = np.hstack((x_pol[:, None], y_pol[:, None]))
                ax.add_patch(Polygon(xy_pol, fill=True, edgecolor=None, facecolor=interior_color, alpha=0.5, label="B-spline patch", zorder=0)) # type: ignore
                ax.plot(ctrl_pts[0, 0, 0], ctrl_pts[1, 0, 0], marker="o", c=ctrl_color, label="Control mesh", zorder=1, ms=plt.rcParams['lines.markersize']/np.sqrt(2))
                ax.add_collection(LineCollection(ctrl_pts.transpose(1, 2, 0), colors=ctrl_color, zorder=1)) # type: ignore
                ax.add_collection(LineCollection(ctrl_pts.transpose(2, 1, 0), colors=ctrl_color, zorder=1)) # type: ignore
                ax.scatter(ctrl_pts[0].ravel(), ctrl_pts[1].ravel(), marker="o", c=ctrl_color, zorder=1, s=0.5*plt.rcParams['lines.markersize']**2) # type: ignore
                ax.plot(x_xi[0, 0], y_xi[0, 0], linestyle="-", c=elem_color, label="Elements borders", zorder=2)
                ax.add_collection(LineCollection(np.array([x_xi, y_xi]).transpose(1, 2, 0)[1:-1], colors=elem_color, zorder=2)) # type: ignore
                ax.add_collection(LineCollection(np.array([x_eta, y_eta]).transpose(2, 1, 0)[1:-1], colors=elem_color, zorder=2)) # type: ignore
                ax.add_patch(Polygon(xy_pol, lw=1.25*plt.rcParams['lines.linewidth'], fill=False, edgecolor=border_color, label="Patch borders", zorder=2)) # type: ignore
            else:
                raise ValueError(f"Can't plot a {self.NPa}D shape in a 2D space.")
            ax.legend()
            ax.set_aspect(1)
        elif NPh==3:
            ax = fig.add_subplot(projection='3d') if ax is None else ax
            if self.NPa==1:
                pass
            elif self.NPa==2:
                xi, eta = self.linspace(n_eval_per_elem=n_eval_per_elem)
                xi_elem, eta_elem = self.linspace(n_eval_per_elem=1)
                x, y, z = self.__call__(ctrl_pts, [xi, eta])
                x_xi, y_xi, z_xi = self.__call__(ctrl_pts, [xi_elem, eta])
                x_eta, y_eta, z_eta = self.__call__(ctrl_pts, [xi, eta_elem])
                ax.plot_surface(x, y, z, rcount=1, ccount=1, edgecolor=None, facecolor=interior_color, alpha=0.5)
                ax.plot_wireframe(ctrl_pts[0], ctrl_pts[1], ctrl_pts[2], color=ctrl_color, zorder=2)
                ax.scatter(ctrl_pts[0], ctrl_pts[1], ctrl_pts[2], color=ctrl_color, zorder=2)
                ax.add_collection(Line3DCollection(np.array([x_xi, y_xi, z_xi]).transpose(1, 2, 0), colors=elem_color, zorder=1)) # type: ignore
                ax.add_collection(Line3DCollection(np.array([x_eta, y_eta, z_eta]).transpose(2, 1, 0), colors=elem_color, zorder=1)) # type: ignore
                ax.plot_surface(x, y, z, rcount=1, ccount=1, edgecolor=border_color, facecolor=None, alpha=0)
                ctrl_handle = lines.Line2D([], [], color=ctrl_color, marker='o', linestyle='-', label='Control mesh')
                elem_handle = lines.Line2D([], [], color=elem_color, linestyle='-', label='Elements borders')
                border_handle = lines.Line2D([], [], color=border_color, linestyle='-', label='Patch borders')
                ax.legend(handles=[ctrl_handle, elem_handle, border_handle])
                mid_param = [np.array([sum(self.bases[0].span)/2]), np.array([sum(self.bases[1].span)/2])]
                dxi, deta = self.__call__(ctrl_pts, mid_param, k=1)
                nx, ny, nz = np.cross(dxi.ravel(), deta.ravel())
                azim = np.degrees(np.arctan2(ny, nx))
                elev = np.degrees(np.arcsin(nz / np.sqrt(nx**2 + ny**2 + nz**2)))
                ax.view_init(elev=elev + 30, azim=azim)
            elif self.NPa==3:
                XI = self.linspace(n_eval_per_elem=n_eval_per_elem)
                XI_elem = self.linspace(n_eval_per_elem=1)
                for face in range(3):
                    for side in [-1, 0]:
                        XI_face = []
                        XI_1_face = []
                        XI_2_face = []
                        first = True
                        for i in range(3):
                            if i==face:
                                XI_face.append(np.array([XI[i][side]]))
                                XI_1_face.append(np.array([XI[i][side]]))
                                XI_2_face.append(np.array([XI[i][side]]))
                            else:
                                XI_face.append(XI[i])
                                XI_1_face.append(XI[i] if first else XI_elem[i])
                                XI_2_face.append(XI_elem[i] if first else XI[i])
                                first = False
                        X = np.squeeze(np.array(self.__call__(ctrl_pts, XI_face)))
                        X_1 = np.squeeze(np.array(self.__call__(ctrl_pts, XI_1_face)))
                        X_2 = np.squeeze(np.array(self.__call__(ctrl_pts, XI_2_face)))
                        ax.plot_surface(*X, rcount=1, ccount=1, edgecolor=None, color=interior_color, alpha=0.5)
                        ax.add_collection(Line3DCollection(X_1.transpose(2, 1, 0), colors=elem_color, zorder=1)) # type: ignore
                        ax.add_collection(Line3DCollection(X_2.transpose(1, 2, 0), colors=elem_color, zorder=1)) # type: ignore
                        ax.plot_surface(*X, rcount=1, ccount=1, edgecolor=border_color, facecolor=None, alpha=0)
                for axis in range(3):
                    ctrl_mesh_axis = np.rollaxis(ctrl_pts, axis + 1, 1).reshape((3, ctrl_pts.shape[axis + 1], -1))
                    ax.add_collection(Line3DCollection(ctrl_mesh_axis.transpose(2, 1, 0), colors=ctrl_color, zorder=2)) # type: ignore
                ax.scatter(*ctrl_pts.reshape((3, -1)), color=ctrl_color, zorder=2)
                ctrl_handle = lines.Line2D([], [], color=ctrl_color, marker='o', linestyle='-', label='Control mesh')
                elem_handle = lines.Line2D([], [], color=elem_color, linestyle='-', label='Elements borders')
                border_handle = lines.Line2D([], [], color=border_color, linestyle='-', label='Patch borders')
                ax.legend(handles=[ctrl_handle, elem_handle, border_handle])
            else:
                raise ValueError(f"Can't plot a {self.NPa}D shape in a 3D space.")
        else:
            raise ValueError(f"Can't plot in a {NPh}D space.")

def _writePVD(fileName, groups):
    """
    Write PVD file
    Usage: _writePVD("toto", "vtu", groups) 
    generated file: "toto.pvd" 
    
    VTK files must be named as follows:
    groups={"a": {"ext": "vtu", "npart": 1, "nstep": 2}, "b": {"ext": "vtr", "npart": 2, "nstep": 5}}
    =>  toto_b_2_5.vtr  (starts from toto_a_0_0.vtu)
    
    Parameters
    ----------
    fileName : STRING
        mesh files without numbers and extension
    groups : DICT of DICT
        Dict (out) of dict (in) as :
        - (out) : 
            * keys are the names of group to plot, 
            * values are (in) dict, 
        - (in) : 
            * "ext" : name of the extention of the group, 
            * "npart" : number of parts to plot together,
            * "nstep" : number of time steps.

    """
    rep,fname=os.path.split(fileName)
    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)
    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)    
    for name, grp in groups.items():
        for jp in range(grp["npart"]):
            for js in range(grp["nstep"]):
                dataSet = pvd.createElementNS("VTK", "DataSet")
                dataSet.setAttribute("timestep", str(js))
                dataSet.setAttribute("group", name)
                dataSet.setAttribute("part", str(jp))
                dataSet.setAttribute("file", f"{fname}_{name}_{jp}_{js}.{grp['ext']}")
                dataSet.setAttribute("name", f"{name}_{jp}")
                collection.appendChild(dataSet)
    outFile = open(fileName+".pvd", 'w')
    pvd.writexml(outFile, newl='\n')
    print("VTK: "+ fileName +".pvd written")
    outFile.close()