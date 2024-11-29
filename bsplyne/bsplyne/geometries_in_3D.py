# %% Imports
from copy import deepcopy

import numpy as np

from .b_spline import BSpline

# %% 3D transformations
def _rotation_matrix(axis, angle):
    P = np.expand_dims(axis, axis=1)@np.expand_dims(axis, axis=0)
    I = np.eye(3)
    Q = np.cross(I, axis)
    R = P + np.cos(angle)*(I - P) + np.sin(angle)*Q
    return R

def _scale_matrix(scale_vector):
    S = np.diag(scale_vector)
    return S

def _scale_rotate_translate(pts, scale_vector, axis, angle, translation_vector):
    S = _scale_matrix(scale_vector)
    R = _rotation_matrix(axis, angle)
    new_pts = (np.tensordot(R@S, pts, 1).T + translation_vector).T
    return new_pts

# %% inner points computation
def _IDW(known_points, unknown_points, data, exposant=2):
    dim = known_points.shape[0]
    distances = np.empty((dim, unknown_points.shape[1], known_points.shape[1]))
    for d in range(dim):
        distances[d] = np.expand_dims(unknown_points[d], axis=-1) - known_points[d]
    distances = np.linalg.norm(distances, axis=0)
    ID = distances**(-exposant)
    ID = (ID.T/ID.sum(1)).T
    data_mean = np.mean(data, axis=-1)
    return (np.tensordot(ID, data.T - data_mean, 1) + data_mean).T

def _find_inner_ctrlPts(degrees, knotVects, ctrlPts, exposant=2):
    greville = []
    for idx in range(len(degrees)):
        p = degrees[idx]
        knot = knotVects[idx]
        n = ctrlPts.shape[1+idx]
        greville.append(np.array([sum([knot[i+k] for k in range(p)])/p for i in range(p-1, n-1+p)]))
    parametricPts = np.array(np.meshgrid(*greville, indexing='ij'))
    notFound = np.zeros(parametricPts.shape[1:], dtype='bool')
    notFound[tuple([slice(1, -1) for i in range(notFound.ndim)])] = True
    found = np.logical_not(notFound)
    known_points = parametricPts[:, found]
    unknown_points = parametricPts[:, notFound]
    data = ctrlPts[:, found]
    ctrlPts[:, notFound] = _IDW(known_points, unknown_points, data, exposant=exposant)
    return ctrlPts

# %% Constants for B-Spline circles
p = 2
knot = np.array([0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1], dtype='float')
C = np.array([-41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
              -41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
              -50557440/36473*np.sqrt(np.sqrt(2) + 2)/np.pi**3 - 92505600/36473*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (26527/2334272)*np.pi + (18163200/36473)/np.pi**3 + (103925760/36473)*np.sqrt(2)/np.pi**3, 
              -91238400/36473*np.sqrt(2)/np.pi**3 - (41034240/36473)/np.pi**3 - 65605/2334272*np.pi + (52646400/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (70632960/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3, 
              (1/16)*np.pi, 
              0], dtype='float')

# %% 1D parametric space
def base_quarter_circle():
    degrees = np.array([p], dtype='int')
    knots = [knot]
    Z = np.zeros_like(C)
    ctrlPts = np.array([C, C[::-1], Z])
    quarter_circle = BSpline(degrees, knots)
    return quarter_circle, ctrlPts
base_quarter_circle = base_quarter_circle()

def new_quarter_circle(center, normal, radius):
    quarter_circle, ctrlPts = deepcopy(base_quarter_circle)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_circle, ctrlPts

def base_circle():
    degrees = np.array([p], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in knot[:-p]] 
                    + [1/4 + float(k)/4 for k in knot[p:-p]] 
                    + [2/4 + float(k)/4 for k in knot[p:-p]] 
                    + [3/4 + float(k)/4 for k in knot[p:]], dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.concatenate((np.array([ C[:-1]      ,  C[::-1][:-1], Z[:-1]]), 
                              np.array([-C[::-1][:-1],  C[:-1]      , Z[:-1]]), 
                              np.array([-C[:-1]      , -C[::-1][:-1], Z[:-1]]), 
                              np.array([ C[::-1]     , -C           , Z     ])), axis=1)
    circle = BSpline(degrees, knots)
    return circle, ctrlPts
base_circle = base_circle()

def new_circle(center, normal, radius):
    circle, ctrlPts = deepcopy(base_circle)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return circle, ctrlPts

# %% 2D parametric space
def base_disk():
    q = 1
    degrees = np.array([p, q], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in knot[:-p]] 
                    + [1/4 + float(k)/4 for k in knot[p:-p]] 
                    + [2/4 + float(k)/4 for k in knot[p:-p]] 
                    + [3/4 + float(k)/4 for k in knot[p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.concatenate((np.array([ C[:-1]      ,  C[::-1][:-1], Z[:-1]]), 
                              np.array([-C[::-1][:-1],  C[:-1]      , Z[:-1]]), 
                              np.array([-C[:-1]      , -C[::-1][:-1], Z[:-1]]), 
                              np.array([ C[::-1]     , -C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts
base_disk = base_disk()

def new_disk(center, normal, radius):
    disk, ctrlPts = deepcopy(base_disk)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def base_degenerated_disk():
    degrees = np.array([p, p], dtype='int')
    knots = [knot, knot]
    Z = np.zeros_like(C)
    n = C.size
    ctrlPts = np.empty((3, n, n), dtype='float')
    ctrlPts[:,  0,  :] = [+C[:: 1], +C[::-1], Z]
    ctrlPts[:, -1,  :] = [-C[::-1], -C[:: 1], Z]
    ctrlPts[:,  :,  0] = [+C[:: 1], -C[::-1], Z]
    ctrlPts[:,  :, -1] = [-C[::-1], +C[:: 1], Z]
    ctrlPts = _find_inner_ctrlPts(degrees, knots, ctrlPts)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts
base_degenerated_disk = base_degenerated_disk()

def new_degenerated_disk(center, normal, radius):

    disk, ctrlPts = deepcopy(base_degenerated_disk)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def base_quarter_pipe():
    q = 1
    degrees = np.array([p, q], dtype='int')
    knots = [knot, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.array([C, C[::-1], Z])
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    quarter_pipe = BSpline(degrees, knots)
    return quarter_pipe, ctrlPts
base_quarter_pipe = base_quarter_pipe()

def new_quarter_pipe(center_front, orientation, radius, length):

    quarter_pipe, ctrlPts = deepcopy(base_quarter_pipe)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_pipe, ctrlPts

def base_pipe():
    q = 1
    degrees = np.array([p, q], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in knot[:-p]] 
                    + [1/4 + float(k)/4 for k in knot[p:-p]] 
                    + [2/4 + float(k)/4 for k in knot[p:-p]] 
                    + [3/4 + float(k)/4 for k in knot[p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.concatenate((np.array([ C[:-1]      ,  C[::-1][:-1], Z[:-1]]), 
                              np.array([-C[::-1][:-1],  C[:-1]      , Z[:-1]]), 
                              np.array([-C[:-1]      , -C[::-1][:-1], Z[:-1]]), 
                              np.array([ C[::-1]     , -C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    pipe = BSpline(degrees, knots)
    return pipe, ctrlPts
base_pipe = base_pipe()

def new_pipe(center_front, orientation, radius, length):

    pipe, ctrlPts = deepcopy(base_pipe)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return pipe, ctrlPts

# %% 3D parametric space
def base_quarter_cylinder():
    q = 1
    r = 1
    degrees = np.array([p, q, r], dtype='int')
    knots = [knot, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.array([C, C[::-1], Z])
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    quarter_cylinder = BSpline(degrees, knots)
    return quarter_cylinder, ctrlPts
base_quarter_cylinder = base_quarter_cylinder()

def new_quarter_cylinder(center_front, orientation, radius, length):

    quarter_cylinder, ctrlPts = deepcopy(base_quarter_cylinder)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_cylinder, ctrlPts

def base_cylinder():
    q = 1
    r = 1
    degrees = np.array([p, q, r], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in knot[:-p]] 
                    + [1/4 + float(k)/4 for k in knot[p:-p]] 
                    + [2/4 + float(k)/4 for k in knot[p:-p]] 
                    + [3/4 + float(k)/4 for k in knot[p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    Z = np.zeros_like(C)
    ctrlPts = np.concatenate((np.array([ C[:-1]      ,  C[::-1][:-1], Z[:-1]]), 
                              np.array([-C[::-1][:-1],  C[:-1]      , Z[:-1]]), 
                              np.array([-C[:-1]      , -C[::-1][:-1], Z[:-1]]), 
                              np.array([ C[::-1]     , -C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts
base_cylinder = base_cylinder()

def new_cylinder(center_front, orientation, radius, length):

    cylinder, ctrlPts = deepcopy(base_cylinder)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return cylinder, ctrlPts

def base_degenerated_cylinder():
    q = 1
    degrees = np.array([p, p, q], dtype='int')
    knots = [knot, knot, np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(C)
    n = C.size
    ctrlPts = np.empty((3, n, n), dtype='float')
    ctrlPts[:,  0,  :] = [+C[:: 1], +C[::-1], Z]
    ctrlPts[:, -1,  :] = [-C[::-1], -C[:: 1], Z]
    ctrlPts[:,  :,  0] = [+C[:: 1], -C[::-1], Z]
    ctrlPts[:,  :, -1] = [-C[::-1], +C[:: 1], Z]
    ctrlPts = _find_inner_ctrlPts(degrees[:-1], knots[:-1], ctrlPts)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, ctrlPts), axis=-1)
    ctrlPts[2, :, :, 1] = 1
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts
base_degenerated_cylinder = base_degenerated_cylinder()

def new_degenerated_cylinder(center_front, orientation, radius, length):

    cylinder, ctrlPts = deepcopy(base_degenerated_cylinder)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')
    
    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return cylinder, ctrlPts

# %% Constants for closed knot vector B-Spline circles
p_closed = 2
knot_closed = 1/8*np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float')
a = -(226560/751)/np.pi**3 + (176640/751)*np.sqrt(2)/np.pi**3
b = -(579840/751)/np.pi**3 + (403200/751)*np.sqrt(2)/np.pi**3
C_closed = np.array([[a, b], [a, -b], [-b, a], [b, a], [-a, -b], [-a, b], [b, -a], [-b, -a], [a, b], [a, -b]], dtype='float')

# %% 1D parametric space
def base_closed_circle():
    degrees = np.array([p_closed], dtype='int')
    knots = [knot_closed]
    Z = np.zeros_like(C)
    ctrlPts = C_closed
    circle = BSpline(degrees, knots)
    return circle, ctrlPts
base_closed_circle = base_closed_circle()

def new_closed_circle(center, normal, radius):
    circle, ctrlPts = deepcopy(base_closed_circle)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return circle, ctrlPts

# %% 2D parametric space
def base_closed_disk():
    q = 1
    degrees = np.array([p_closed, q], dtype='int')
    knots = [knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    ctrlPts = np.expand_dims(C_closed, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts
base_closed_disk = base_closed_disk()

def new_closed_disk(center, normal, radius):
    disk, ctrlPts = deepcopy(base_closed_disk)

    scale_vector = np.array([radius, radius, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def base_closed_pipe():
    q = 1
    degrees = np.array([p_closed, q], dtype='int')
    knots = [knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    ctrlPts = np.expand_dims(C_closed, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    pipe = BSpline(degrees, knots)
    return pipe, ctrlPts
base_closed_pipe = base_closed_pipe()

def new_closed_pipe(center_front, orientation, radius, length):

    pipe, ctrlPts = deepcopy(base_closed_pipe)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return pipe, ctrlPts

# %% 3D parametric space
def base_closed_cylinder():
    q = 1
    r = 1
    degrees = np.array([p_closed, q, r], dtype='int')
    knots = [knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    ctrlPts = np.expand_dims(C_closed, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts
base_closed_cylinder = base_closed_cylinder()

def new_closed_cylinder(center_front, orientation, radius, length):

    cylinder, ctrlPts = deepcopy(base_closed_cylinder)

    scale_vector = np.array([radius, radius, length], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return cylinder, ctrlPts

# %% strut elements

def new_quarter_strut(center_front, orientation, radius, length):
    degrees = np.array([2, 1, 1], dtype='int')
    knots = [np.array([0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1], dtype='float'), 
            np.array([0, 0, 1, 1], dtype='float'), 
            np.array([0, 0, 1, 1], dtype='float')]
    C = np.array([-41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
                  -41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
                  -50557440/36473*np.sqrt(np.sqrt(2) + 2)/np.pi**3 - 92505600/36473*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (26527/2334272)*np.pi + (18163200/36473)/np.pi**3 + (103925760/36473)*np.sqrt(2)/np.pi**3, 
                  -91238400/36473*np.sqrt(2)/np.pi**3 - (41034240/36473)/np.pi**3 - 65605/2334272*np.pi + (52646400/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (70632960/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3, 
                  (1/16)*np.pi, 
                  0], dtype='float')*radius

    tmp = radius*np.ones_like(C)
    front = np.array([C, C[::-1], tmp])
    tmp = np.zeros_like(front)
    front = np.concatenate((front[..., None], tmp[..., None]), axis=-1)

    tmp = length - radius*np.ones_like(C)
    back = np.array([C, C[::-1], tmp])
    tmp = np.zeros_like(back)
    tmp[-1] = length
    back = np.concatenate((back[..., None], tmp[..., None]), axis=-1)

    ctrlPts = np.concatenate((front[..., None], back[..., None]), axis=-1)

    spline = BSpline(degrees, knots)
    ctrlPts = spline.orderElevation(ctrlPts, [0, 0, 1])
    ctrlPts = spline.knotInsertion(ctrlPts, [np.array([]), np.array([]), np.array([0.25, 0.5, 0.75], dtype='float')])
    
    p = np.arctan(3 - 2*np.sqrt(2))
    k = np.linspace(p, np.pi/2 - p, C.size)
    x = radius*np.sqrt(2)*((0.5 + 0.25*np.sqrt(2))*np.cos(k) + (0.25*np.sqrt(2) - 0.5)*np.sin(k))
    y = radius*np.sqrt(2)*((0.5 + 0.25*np.sqrt(2))*np.sin(k) + (0.25*np.sqrt(2) - 0.5)*np.cos(k))
    z = radius*(0.5*np.cos(k) + 0.5*np.sin(k))
    
    p = 0
    k = np.linspace(p, np.pi/2 - p, C.size)
    x = radius*np.cos(k)
    y = radius*np.sin(k)
    z = radius/np.sqrt(2)*(np.cos(k) + np.sin(k))
    
    # x = radius*np.linspace(np.sqrt(2), 0, C.size)
    # y = radius*np.linspace(0, np.sqrt(2), C.size)
    # ctrlPts[0, :, 0,  0] = x
    # ctrlPts[1, :, 0,  0] = y
    ctrlPts[2, :, 0,  0] = z
    # ctrlPts[0, :, 0, -1] = x
    # ctrlPts[1, :, 0, -1] = y
    ctrlPts[2, :, 0, -1] = length - z

    scale_vector = np.array([1, 1, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)
    
    return spline, ctrlPts

# %% cube

def new_cube(center, orientation, side_length):
    degrees = np.array([1]*3, dtype='int')
    knots = [np.array([0, 0, 1, 1], dtype='float')]*3
    ctrlPts = np.array(np.meshgrid(*([np.array([-0.5, 0.5])]*3), indexing='ij'))

    spline = BSpline(degrees, knots)
    
    scale_vector = np.array([side_length]*3, dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center, dtype='float')

    ctrlPts = _scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)
    
    return spline, ctrlPts