"""
.. include:: ../README.md
"""
from .b_spline_basis import BSplineBasis
from .b_spline import BSpline
from .multi_patch_b_spline import MultiPatchBSplineConnectivity, CouplesBSplineBorder, MultiPatchBSpline
from .geometries_in_3D import (_scale_rotate_translate, 
                               new_quarter_circle, 
                               new_circle, 
                               new_disk, 
                               new_degenerated_disk, 
                               new_quarter_pipe, 
                               new_pipe, 
                               new_quarter_cylinder, 
                               new_cylinder, 
                               new_degenerated_cylinder, 
                               new_closed_circle, 
                               new_closed_disk, 
                               new_closed_pipe, 
                               new_closed_cylinder, 
                               new_quarter_strut, 
                               new_cube)
