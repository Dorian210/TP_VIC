import numpy as np
import numba as nb
import scipy.sparse as sps
from scipy.special import comb
import matplotlib.pyplot as plt

class BSplineBasis:
    """
    BSpline basis in 1D.

    Attributes
    ----------
    p : int
        Degree of the polynomials composing the basis.
    knot : numpy.array of float
        Knot vector of the BSpline.
    m : int
        Last index of the knot vector.
    n : int
        Last index of the basis : when evaluated, returns an array of size 
        `n` + 1.
    span : tuple of 2 float
        Interval of definition of the basis.

    """
    
    def __init__(self, p, knot):
        """
        Create a `BSplineBasis` object that can compute its basis, and the 
        derivatives of these functions.

        Parameters
        ----------
        p : int
            Degree of the BSpline.
        knot : numpy.array of float
            Knot vector of the BSpline.

        Returns
        -------
        BSplineBasis : BSplineBasis instance
            Contains the `BSplineBasis` object created.

        Examples
        --------
        Creation of a `BSplineBasis` instance of degree 2 and knot vector 
        [0, 0, 0, 1, 1, 1] :
        >>> BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))

        """
        self.p = p
        self.knot = knot
        self.m = knot.size - 1
        self.n = self.m - self.p - 1
        self.span = (self.knot[self.p], self.knot[self.m - self.p])
    
    def linspace(self, n_eval_per_elem=10):
        """
        Generate a set of xi values over the span of the basis.

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of values per element, by default 10

        Returns
        -------
        numpy.array of float
            Set of xi values over the span.
        """
        knot_uniq = np.unique(self.knot[np.logical_and(self.knot>=0, self.knot<=1)])
        res = np.linspace(knot_uniq[-2], knot_uniq[-1], n_eval_per_elem + 1)
        for i in range(knot_uniq.size - 2, 0, -1):
            res = np.append(np.linspace(knot_uniq[i-1], knot_uniq[i], n_eval_per_elem, endpoint=False), 
                            res)
        return res
    
    def linspace_for_integration(self, n_eval_per_elem=10, bounding_box=None):
        """
        Generate a set of xi values over the span of the basis, centerered 
        on intervals of returned lengths.

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of values per element, by default 10
        bounding_box : numpy.array of float, optional
            Lower and upper bounds, by default `self`.`span`

        Returns
        -------
        xi : numpy.array of float
            Set of xi values over the span.
        dxi : numpy.array of float
            Integration weight of each point.
        """
        if bounding_box is None:
            lower, upper = self.span
        else:
            lower, upper = bounding_box
        knot_uniq = np.unique(self.knot[np.logical_and(self.knot>=self.span[0], self.knot<=self.span[1])])
        xi = []
        dxi = []
        for i in range(knot_uniq.size - 1):
            a = knot_uniq[i]
            b = knot_uniq[i + 1]
            if a<upper and b>lower:
                if a<lower and b>upper:
                    dxi_i_l = (upper - lower)/n_eval_per_elem
                    if (lower - 0.5*dxi_i_l)<a:
                        dxi_i_u = (upper - a)/n_eval_per_elem
                        if (upper + 0.5*dxi_i_u)>b:
                            dxi_i = (b - a)/n_eval_per_elem
                        else:
                            b = upper + 0.5*dxi_i_u
                            dxi_i = dxi_i_u
                    else:
                        a = lower - 0.5*dxi_i_l
                        dxi_i_u = dxi_i_l
                        if (upper + 0.5*dxi_i_u)>b:
                            dxi_i = (b - lower)/n_eval_per_elem
                        else:
                            dxi_i = dxi_i_u
                            b = upper + 0.5*dxi_i_u
                elif a<lower and b>lower:
                    dxi_i_l = (b - lower)/n_eval_per_elem
                    if (lower - 0.5*dxi_i_l)<a:
                        dxi_i = (b - a)/n_eval_per_elem
                    else:
                        a = lower - 0.5*dxi_i_l
                        dxi_i = dxi_i_l
                elif a<upper and b>upper:
                    dxi_i_u = (upper - a)/n_eval_per_elem
                    if (upper + 0.5*dxi_i_u)>b:
                        dxi_i = (b - a)/n_eval_per_elem
                    else:
                        b = upper + 0.5*dxi_i_u
                        dxi_i = dxi_i_u
                else:
                    dxi_i = (b - a)/n_eval_per_elem
                xi.append(np.linspace(a + 0.5*dxi_i, b - 0.5*dxi_i, n_eval_per_elem))
                dxi.append(dxi_i*np.ones(n_eval_per_elem))
        xi = np.hstack(xi)
        dxi = np.hstack(dxi)
        return xi, dxi
    
    def gauss_legendre_for_integration(self, n_eval_per_elem=None, bounding_box=None):
        """
        Generate a set of xi values with their coresponding weight acording to the Gauss Legendre 
        integration method over a given bounding box.

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of values per element, by default `self.p + 1`
        bounding_box : numpy.array of float, optional
            Lower and upper bounds, by default `self`.`span`

        Returns
        -------
        xi : numpy.array of float
            Set of xi values over the span.
        dxi : numpy.array of float
            Integration weight of each point.
        """
        if n_eval_per_elem is None:
            n_eval_per_elem = self.p + 1
        if bounding_box is None:
            lower, upper = self.span
        else:
            lower, upper = bounding_box
        knot_uniq = np.hstack(([lower], np.unique(self.knot[np.logical_and(self.knot>lower, self.knot<upper)]), [upper]))
        points, wheights = np.polynomial.legendre.leggauss(n_eval_per_elem)
        xi = np.hstack([(b - a)/2*points + (b + a)/2 for a, b in zip(knot_uniq[:-1], knot_uniq[1:])])
        dxi = np.hstack([(b - a)/2*wheights for a, b in zip(knot_uniq[:-1], knot_uniq[1:])])
        return xi, dxi
    
    def normalize_knots(self):
        """
        Maps the knots vector to [0, 1].
        """
        a, b = self.span
        self.knot = (self.knot - a)/(b - a)
        self.span = (0, 1)
    
    def N(self, XI, k=0):
        """
        Compute the `k`-th derivative of the BSpline basis functions for a set 
        of values in the parametric space.

        Parameters
        ----------
        XI : numpy.array of float
            Values in the parametric space at which the BSpline is evaluated.
        k : int, optional
            `k`-th derivative of the BSpline evaluated. The default is 0.

        Returns
        -------
        DN : scipy.sparse.coo_matrix of float
            Sparse matrix containing the values of the `k`-th derivative of the 
            BSpline basis functions in the rows for each value of `XI` in the 
            columns.

        Examples
        --------
        Evaluation of the BSpline basis on these `XI` values : [0, 0.5, 1]
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.N(np.array([0, 0.5, 1], dtype='float')).A
        array([[1.  , 0.  , 0.  ],
               [0.25, 0.5 , 0.25],
               [0.  , 0.  , 1.  ]])
        
        Evaluation of the 1st derivative of the BSpline basis on these `XI` 
        values : [0, 0.5, 1]
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.N(np.array([0, 0.5, 1], dtype='float'), k=1).A
        array([[-2.,  2.,  0.],
               [-1.,  0.,  1.],
               [ 0., -2.,  2.]])

        """
        vals, row, col = _DN(self.p, self.m, self.n, self.knot, XI, k)
        DN = sps.coo_matrix((vals, (row, col)), shape=(XI.size, self.n + 1))
        return DN
    
    def plotN(self, k=0, show=True):
        """
        Plots the basis functions over the span.

        Parameters
        ----------
        k : int, optional
            `k`-th derivative of the BSpline ploted. The default is 0.
        show : bool, optional
            Should the plot be displayed ? The default is True.

        Returns
        -------
        None.

        """
        n_eval_per_elem = 500//np.unique(self.knot).size
        for idx in range(self.n+1):
            XI = np.empty(0, dtype='float')
            for i in range(idx, idx+self.p+1):
                a = self.knot[i]
                b = self.knot[i+1]
                if a!=b:
                    b -= np.finfo('float').eps
                    XI = np.append(XI, np.linspace(a, b, n_eval_per_elem))
            DN_idx = np.empty(0, dtype='float')
            for ind in range(XI.size):
                DN_idx_ind = _funcDNElemOneXi(idx, self.p, self.knot, XI[ind], k)
                DN_idx = np.append(DN_idx, DN_idx_ind)
            label = "$N_{"+str(idx)+"}"+("'"*k)+"(\\xi)$"
            plt.plot(XI, DN_idx, label=label)
        plt.xlabel("$\\xi$")
        if self.n+1<=10:
            plt.legend()
        if show:
            plt.show()
    
    def _funcDElem(self, i, j, new_knot, p):
        """
        Compute the ij value of the knot insertion matrix D.

        Parameters
        ----------
        i : int
            Row index of D.
        j : int
            Column index of D.
        new_knot : numpy.array of float
            New knot vector to use.
        p : int
            Degree of the BSpline.

        Returns
        -------
        D_ij : float
            Value of D at the index ij.

        """
        if p==0:
            return int(new_knot[i]>=self.knot[j] and new_knot[i]<self.knot[j+1])
        if self.knot[j+p]!=self.knot[j]:
            rec_p = (new_knot[i+p] - self.knot[j])/(self.knot[j+p] - self.knot[j])
            rec_p *= self._funcDElem(i, j, new_knot, p-1)
        else:
            rec_p = 0
        if self.knot[j+p+1]!=self.knot[j+1]:
            rec_j = (self.knot[j+p+1] - new_knot[i+p])/(self.knot[j+p+1] - self.knot[j+1])
            rec_j *= self._funcDElem(i, j+1, new_knot, p-1)
        else:
            rec_j = 0
        D_ij = rec_p + rec_j
        return D_ij
    
    def _D(self, new_knot):
        """
        Compute the `D` matrix used to determine the position of the new control
        points for the knot insertion process. The instance of `BSplineBasis` 
        won't be modified here.

        Parameters
        ----------
        new_knot : numpy.array of float
            The new knot vector for the knot insertion.

        Returns
        -------
        D : scipy.sparse.coo_matrix of float
            The matrix `D` such that :
            newCtrlPtsCoordinate = `D` @ ancientCtrlPtsCoordinate.

        """
        new_m = new_knot.size - 1
        new_n = new_m - self.p - 1
        loop1 = new_n + 1
        loop2 = self.p + 1
        nb_val_max = loop1*loop2
        vals = np.empty(nb_val_max, dtype='float')
        row = np.empty(nb_val_max, dtype='int')
        col = np.empty(nb_val_max, dtype='int')
        nb_not_put = 0
        for ind1 in range(loop1):
            sparse_ind1 = ind1
            i = ind1
            new_knot_i = new_knot[i]
            # find {elem} so that new_knot_i \in [knot_{elem}, knot_{{elem} + 1}[
            elem = _findElem(self.p, self.m, self.n, self.knot, new_knot_i)
            # determine D_ij(new_knot_i) for the values of j where we know D_ij(new_knot_i) not equal to 0
            for ind2 in range(loop2):
                sparse_ind2 = sparse_ind1*loop2 + ind2
                j = ind2 + elem - self.p
                if j<0 or j>elem:
                    nb_not_put += 1
                else:
                    sparse_ind = sparse_ind2 - nb_not_put
                    vals[sparse_ind] = self._funcDElem(i, j, new_knot, self.p)
                    row[sparse_ind] = i
                    col[sparse_ind] = j
        if nb_not_put!=0:
            vals = vals[:-nb_not_put]
            row = row[:-nb_not_put]
            col = col[:-nb_not_put]
        D = sps.coo_matrix((vals, (row, col)), shape=(new_n + 1, self.n + 1))
        return D
    
    def knotInsertion(self, knots_to_add):
        """
        Performs the knot insersion process on the `BSplineBasis` instance and 
        returns the `D` matrix.

        Parameters
        ----------
        knots_to_add : numpy.array of float
            Array of knots to append to the knot vector.

        Returns
        -------
        D : scipy.sparse.coo_matrix of float
            The matrix `D` such that :
            newCtrlPtsCoordinate = `D` @ ancientCtrlPtsCoordinate.

        Examples
        --------
        Insert the knots [0.5, 0.5] to the `BSplineBasis` instance 
        and return the operator to apply on the control points.
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.knotInsertion(np.array([0.5, 0.5], dtype='float')).A
        array([[1.  , 0.  , 0.  ],
               [0.5 , 0.5 , 0.  ],
               [0.25, 0.5 , 0.25],
               [0.  , 0.5 , 0.5 ],
               [0.  , 0.  , 1.  ]])
        
        The knot vector is modified (as well as n and m) :
        >>> basis.knot
        array([0. , 0. , 0. , 0.5, 0.5, 1. , 1. , 1. ])

        """
        k = knots_to_add.size
        new_knot = np.sort(np.concatenate((self.knot, knots_to_add), dtype='float'))
        D = self._D(new_knot)
        self.m += k
        self.n += k
        self.knot = new_knot
        return D
    
    def orderElevation(self, t):
        """
        Performs the order elevation algorithm on the basis and return a 
        linear transformation to apply on the control points.

        Parameters
        ----------
        t : int
            New degree of the B-spline basis will be its current degree plus `t`.

        Returns
        -------
        STD : scipy.sparse.coo_matrix of float
            The matrix `STD` such that :
            newCtrlPtsCoordinate = `STD` @ ancientCtrlPtsCoordinate.

        Examples
        --------
        Elevate the orderof the `BSplineBasis` instance by 1 and return the operator 
        to apply on the control points.
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.orderElevation(1).A
        array([[1.        , 0.        , 0.        ],
               [0.33333333, 0.66666667, 0.        ],
               [0.        , 0.66666667, 0.33333333],
               [0.        , 0.        , 1.        ]])
        
        The knot vector and the degree are modified (as well as n and m) :
        >>> basis.knot
        array([0., 0., 0., 0., 1., 1., 1., 1.])
        >>> basis.p
        3

        """
        no_dup, counts = np.unique(self.knot, return_counts=True)
        missed = self.p + 1 - counts
        p0 = self.p
        p1 = p0
        p2 = p1 + t
        p3 = p2
        knot0 = self.knot
        knot1 = np.sort(np.concatenate((knot0, np.repeat(no_dup, missed)), axis=0))
        knot2 = np.sort(np.repeat(no_dup, p2 + 1))
        knot3 = np.sort(np.concatenate((knot0, np.repeat(no_dup, t)), axis=0))
        # step 1 : separate the B-spline in bezier curves by knot insertion
        D = self._D(knot1)
        # step 2 : perform the order elevation on every bezier curve
        num_bezier = no_dup.size - 1
        loop1 = num_bezier
        loop2 = p2 + 1
        loop3 = p1 + 1
        nb_val_max = loop1*loop2*loop3
        vals = np.empty(nb_val_max, dtype='float')
        row = np.empty(nb_val_max, dtype='int')
        col = np.empty(nb_val_max, dtype='int')
        nb_not_put = 0
        i_offset = 0
        j_offset = 0
        for ind1 in range(loop1):
            sparse_ind1 = ind1
            for ind2 in range(loop2):
                sparse_ind2 = sparse_ind1*loop2 + ind2
                i = ind2
                inv_denom = 1/comb(p2, i) # type: ignore
                for ind3 in range(loop3):
                    sparse_ind3 = sparse_ind2*loop3 + ind3
                    j = ind3
                    if j<(i - t) or j>i:
                        nb_not_put += 1
                    else:
                        sparse_ind = sparse_ind3 - nb_not_put
                        vals[sparse_ind] = comb(p1, j)*comb(t, i-j)*inv_denom # type: ignore
                        row[sparse_ind] = i_offset + i
                        col[sparse_ind] = j_offset + j
            i_offset += p2 + 1
            j_offset += p1 + 1
        if nb_not_put!=0:
            vals = vals[:-nb_not_put]
            row = row[:-nb_not_put]
            col = col[:-nb_not_put]
        T = sps.coo_matrix((vals, (row, col)), shape=((p2+1)*num_bezier, (p1+1)*num_bezier))
        # step 3 : come back to B-spline by removing useless knots
        self.__init__(p2, knot2)
        S = self._D(knot3)
        self.__init__(p3, knot3)
        STD = S@T@D
        return STD

# %% fast functions for evaluation

@nb.njit(cache=True)
def _funcNElemOneXi(i, p, knot, xi):
    """
    Evaluate the basis function N_i^p(xi) of the BSpline.

    Parameters
    ----------
    i : int
        Index of the basis function wanted.
    p : int
        Degree of the BSpline evaluated.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space at which the BSpline is evaluated.

    Returns
    -------
    N_i : float
        Value of the BSpline basis function N_i^p(xi).

    """
    if p==0:
        return int((xi>=knot[i] and xi<knot[i+1]) 
                    or (knot[i+1]==knot[-1] and xi==knot[i+1]))
    if knot[i+p]!=knot[i]:
        rec_p = (xi - knot[i])/(knot[i+p] - knot[i])
        rec_p *= _funcNElemOneXi(i, p-1, knot, xi)
    else:
        rec_p = 0
    if knot[i+p+1]!=knot[i+1]:
        rec_i = (knot[i+p+1] - xi)/(knot[i+p+1] - knot[i+1])
        rec_i *= _funcNElemOneXi(i+1, p-1, knot, xi)
    else:
        rec_i = 0
    N_i = rec_p + rec_i
    return N_i

@nb.njit(cache=True)
def _funcDNElemOneXi(i, p, knot, xi, k):
    """
    Evaluate the `k`-th derivative of the basis function N_i^p(xi) of the 
    BSpline.

    Parameters
    ----------
    i : int
        Index of the basis function wanted.
    p : int
        Degree of the BSpline evaluated.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space at which the BSpline is evaluated.
    k : int
        `k`-th derivative of the BSpline evaluated.

    Raises
    ------
    ValueError
        Can't compute the `k`-th derivative of a B-spline of degree strictly 
        less than `k` or if `k`<0.

    Returns
    -------
    DN_i : float
        Value of the `k`-th derivative of the BSpline basis function 
        N_i^p(xi).

    """
    if k==0:
        return _funcNElemOneXi(i, p, knot, xi)
    if p==0:
        if k>=0:
            raise ValueError("Impossible to determine the k-th derivative of a B-spline of degree strictly less than k !")
        raise ValueError("Impossible to determine the k-th derivative of a B-spline if k<0 !")
    if knot[i+p]!=knot[i]:
        rec_p = p/(knot[i+p] - knot[i])
        rec_p *= _funcDNElemOneXi(i, p-1, knot, xi, k-1)
    else:
        rec_p = 0
    if knot[i+p+1]!=knot[i+1]:
        rec_i = p/(knot[i+p+1] - knot[i+1])
        rec_i *= _funcDNElemOneXi(i+1, p-1, knot, xi, k-1)
    else:
        rec_i = 0
    N_i = rec_p - rec_i
    return N_i

@nb.njit(cache=True)
def _findElem(p, m, n, knot, xi):
    """
    Find `i` so that `xi` belongs to 
    [ `knot`[`i`], `knot`[`i` + 1] [.

    Parameters
    ----------
    p : int
        Degree of the polynomials composing the basis.
    m : int
        Last index of the knot vector.
    n : int
        Last index of the basis.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space.

    Raises
    ------
    ValueError
        If the value of `xi` is outside the definition interval 
        of the spline.

    Returns
    -------
    i : int
        Index of the first knot of the interval in which `xi` is bounded.

    """
    if xi==knot[m - p]:
        return n
    i = 0
    pastrouve = True
    while i<=n and pastrouve:
        pastrouve = xi<knot[i] or xi>=knot[i+1]
        i += 1
    if pastrouve:
        raise ValueError("xi is outside the definition interval of the spline !")
        # print("ValueError : xi=", xi, " is outside the definition interval [", knot[p], ", ", knot[m - p], "] of the spline !")
        # return None
    i -= 1
    return i

@nb.njit(cache=True)#, parallel=True)
def _DN(p, m, n, knot, XI, k):
    """
    Compute the `k`-th derivative of the BSpline basis functions for a set 
    of values in the parametric space.

    Parameters
    ----------
    p : int
        Degree of the polynomials composing the basis.
    m : int
        Last index of the knot vector.
    n : int
        Last index of the basis.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    XI : numpy.array of float
        Values in the parametric space at which the BSpline is evaluated.
    k : int
        `k`-th derivative of the BSpline evaluated. The default is 0.

    Returns
    -------
    (vals, row, col) : (numpy.array of float, numpy.array of int, numpy.array of int)
        Values and indices of the `k`-th derivative matrix of the BSpline 
        basis functions in the columns for each value of `XI` in the rows.

    """
    loop1 = XI.size
    loop2 = p + 1
    nb_val_max = loop1*loop2
    vals = np.empty(nb_val_max, dtype='float')
    row = np.empty(nb_val_max, dtype='int')
    col = np.empty(nb_val_max, dtype='int')
    nb_not_put = 0
    for ind1 in range(loop1):#nb.p
        sparse_ind1 = ind1
        i_xi = ind1
        xi = XI.flat[i_xi]
        # find {elem} so that \xi \in [\xi_{elem}, \xi_{{elem} + 1}[
        elem = _findElem(p, m, n, knot, xi)
        # determine DN_i(\xi) for the values of i where we know DN_i(\xi) not equal to 0
        for ind2 in range(loop2):
            sparse_ind2 = sparse_ind1*loop2 + ind2
            i = ind2 + elem - p
            if i<0:
                nb_not_put += 1
            else:
                sparse_ind = sparse_ind2 - nb_not_put
                vals[sparse_ind] = _funcDNElemOneXi(i, p, knot, xi, k)
                row[sparse_ind] = i_xi
                col[sparse_ind] = i
    if nb_not_put!=0:
        vals = vals[:-nb_not_put]
        row = row[:-nb_not_put]
        col = col[:-nb_not_put]
    return (vals, row, col)